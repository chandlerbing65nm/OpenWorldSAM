from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .tent import Tent


class Entropy(nn.Module):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def __call__(self, x: torch.Tensor, x_ema: torch.Tensor) -> torch.Tensor:
        loss_a = -(x_ema.softmax(dim=1) * x.log_softmax(dim=1)).sum(dim=1)
        loss_b = -(x.softmax(dim=1) * x_ema.log_softmax(dim=1)).sum(dim=1)
        return (1 - self.alpha) * loss_a + self.alpha * loss_b


class SoftLikelihoodRatio(nn.Module):
    def __init__(self, clip: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.clip = clip

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=1)
        probs = torch.clamp(probs, min=0.0, max=self.clip)
        return -(probs * torch.log((probs / (torch.ones_like(probs) - probs)) + self.eps)).sum(dim=1)


@torch.no_grad()
def update_model_probs(x_ema: torch.Tensor, x: torch.Tensor, momentum: float = 0.9) -> torch.Tensor:
    return momentum * x_ema + (1 - momentum) * x


@torch.no_grad()
def ema_update_model(model_to_update: nn.Module, model_to_merge: nn.Module, momentum: float):
    for p_upd, p_src in zip(model_to_update.parameters(), model_to_merge.parameters()):
        p_upd.data.copy_(momentum * p_upd.data + (1 - momentum) * p_src.data)
    return model_to_update


class ROID(Tent):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        self.num_classes = int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
        self.use_weighting = bool(cfg.TTA.ROID.USE_WEIGHTING)
        self.use_prior_correction = bool(cfg.TTA.ROID.USE_PRIOR_CORRECTION)
        self.use_consistency = bool(cfg.TTA.ROID.USE_CONSISTENCY)
        self.momentum_src = float(cfg.TTA.ROID.MOMENTUM_SRC)
        self.momentum_probs = float(cfg.TTA.ROID.MOMENTUM_PROBS)
        self.temperature = float(cfg.TTA.ROID.TEMPERATURE)
        self.batch_size = max(1, int(cfg.TTA.BATCH_SIZE))
        self.device = next(self.model.parameters()).device

        self.class_probs_ema = (1.0 / self.num_classes) * torch.ones(self.num_classes, device=self.device)
        self.tta_transform = self._build_tta_transform()

        self.slr = SoftLikelihoodRatio()
        self.symmetric_cross_entropy = SymmetricCrossEntropy()
        self.softmax_entropy = Entropy()

        self.src_model = deepcopy(self.model)
        self.src_model.eval()
        self.src_model.requires_grad_(False)
        for param in self.src_model.parameters():
            param.detach_()

        self.model_states = [deepcopy(self.src_model.state_dict()), deepcopy(self.model.state_dict())]
        self.optimizer_state = deepcopy(self.optimizer.state_dict()) if self.optimizer is not None else None

    def reset(self):
        if self.optimizer is None:
            self.model.load_state_dict(self.model_states[1], strict=True)
            self.src_model.load_state_dict(self.model_states[0], strict=True)
            self.configure_model()
        else:
            self.src_model.load_state_dict(self.model_states[0], strict=True)
            self.model.load_state_dict(self.model_states[1], strict=True)
            self.optimizer.load_state_dict(self.optimizer_state)
            self.configure_model()
        self.src_model.eval()
        self.src_model.requires_grad_(False)
        self.class_probs_ema = (1.0 / self.num_classes) * torch.ones(self.num_classes, device=self.device)

    def forward_and_adapt(self, batched_inputs):
        if self.optimizer is None:
            with torch.no_grad():
                return self.model(batched_inputs)

        self.optimizer.zero_grad(set_to_none=True)
        outputs, loss = self.loss_calculation(batched_inputs)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.model = ema_update_model(
            model_to_update=self.model,
            model_to_merge=self.src_model,
            momentum=self.momentum_src,
        )

        with torch.no_grad():
            if self.use_prior_correction and len(outputs) > 0:
                corrected_outputs = []
                priors = torch.stack([output["sem_seg"].float().softmax(dim=0).mean(dim=(1, 2)) for output in outputs])
                for output, prior in zip(outputs, priors):
                    smooth = max(1 / max(len(outputs), 1), 1 / output["sem_seg"].shape[0]) / torch.max(prior)
                    smoothed_prior = (prior + smooth) / (1 + smooth * output["sem_seg"].shape[0])
                    corrected_output = dict(output)
                    corrected_output["sem_seg"] = output["sem_seg"] * smoothed_prior.view(-1, 1, 1)
                    corrected_outputs.append(corrected_output)
                outputs = corrected_outputs

        return outputs

    def loss_calculation(self, batched_inputs):
        outputs = self.model(batched_inputs)
        sem_seg_batch = torch.stack([output["sem_seg"].float() for output in outputs])

        weights = None
        mask = None
        if self.use_weighting:
            with torch.no_grad():
                probs = sem_seg_batch.softmax(dim=1)
                probs_img = probs.mean(dim=(2, 3))

                weights_div = 1 - F.cosine_similarity(self.class_probs_ema.unsqueeze(0), probs_img, dim=1)
                denom = (weights_div.max() - weights_div.min()).clamp(min=1e-12)
                weights_div = (weights_div - weights_div.min()) / denom
                mask = weights_div < weights_div.mean()

                ent_img = self.softmax_entropy(sem_seg_batch).mean(dim=(1, 2))
                weights_cert = -ent_img
                denom2 = (weights_cert.max() - weights_cert.min()).clamp(min=1e-12)
                weights_cert = (weights_cert - weights_cert.min()) / denom2

                weights = torch.exp(weights_div * weights_cert / max(self.temperature, 1e-6))
                weights[mask] = 0.0

                self.class_probs_ema = update_model_probs(
                    x_ema=self.class_probs_ema,
                    x=probs_img.mean(dim=0),
                    momentum=self.momentum_probs,
                )

        loss_out = self.slr(sem_seg_batch).mean(dim=(1, 2))

        if self.use_weighting:
            loss_out = loss_out * weights
            loss_out = loss_out[~mask]

        loss = loss_out.sum() / max(1, self.batch_size)

        if self.use_consistency:
            if self.use_weighting:
                selected_indices = (~mask).nonzero(as_tuple=False).flatten().tolist()
                selected_inputs = [batched_inputs[idx] for idx in selected_indices]
                out_ref = sem_seg_batch[~mask].detach()
                w_sel = weights[~mask]
            else:
                selected_inputs = batched_inputs
                out_ref = sem_seg_batch.detach()
                w_sel = None

            if len(selected_inputs) > 0:
                augmented_inputs = [self._augment_sample(sample) for sample in selected_inputs]
                outputs_aug = self.model(augmented_inputs)
                outputs_aug_sem_seg = torch.stack([output["sem_seg"].float() for output in outputs_aug])
                loss_cons = self.symmetric_cross_entropy(outputs_aug_sem_seg, out_ref).mean(dim=(1, 2))
                if w_sel is not None:
                    loss_cons = loss_cons * w_sel
                loss = loss + loss_cons.sum() / max(1, self.batch_size)

        return outputs, loss

    def _build_tta_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(size=(1024, 1024), scale=(0.3, 1.0), ratio=(1.0, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

    def _augment_sample(self, sample):
        augmented = dict(sample)
        image = sample["image"].float()
        augmented["image"] = self.tta_transform(image).to(sample["image"].dtype)

        if "evf_image" in sample:
            evf_image = sample["evf_image"].float()
            augmented["evf_image"] = transforms.Compose([
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.3, 1.0), ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])(evf_image).to(sample["evf_image"].dtype)

        if "padding_mask" in sample:
            padding_mask = sample["padding_mask"].float().unsqueeze(0)
            padding_mask = transforms.Compose([
                transforms.RandomResizedCrop(size=(1024, 1024), scale=(0.3, 1.0), ratio=(1.0, 1.0), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomHorizontalFlip(p=0.5),
            ])(padding_mask).squeeze(0)
            augmented["padding_mask"] = padding_mask.to(sample["padding_mask"].dtype)

        return augmented
