import copy
import random
from copy import deepcopy

import torch
import torch.nn.functional as F

from .tent import Tent


@torch.no_grad()
def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha_teacher).add_(param.data, alpha=1 - alpha_teacher)
    return ema_model


class CoTTA(Tent):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        self.n_augmentations = int(cfg.TTA.COTTA.N_AUGMENTATIONS)
        self.mt = float(cfg.TTA.COTTA.MT_ALPHA)
        self.rst = float(cfg.TTA.COTTA.RST)
        self.ap = float(cfg.TTA.COTTA.AP)
        self.student_crops = max(1, int(cfg.TTA.COTTA.STUDENT_CROPS))
        self.student_crop_scale = float(cfg.TTA.COTTA.STUDENT_CROP_SCALE)
        self.scale_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75][: max(0, self.n_augmentations)]

        self.model_ema = deepcopy(self.model)
        self.model_ema.eval()
        self.model_ema.requires_grad_(False)
        for param in self.model_ema.parameters():
            param.detach_()

        self.model_anchor = deepcopy(self.model)
        self.model_anchor.eval()
        self.model_anchor.requires_grad_(False)
        for param in self.model_anchor.parameters():
            param.detach_()

        self.model_ema_state = deepcopy(self.model_ema.state_dict())
        self.model_anchor_state = deepcopy(self.model_anchor.state_dict())

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        if self.optimizer is not None and self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)
        self.model_ema.load_state_dict(self.model_ema_state, strict=True)
        self.model_anchor.load_state_dict(self.model_anchor_state, strict=True)
        self.configure_model()
        self.model_ema.eval()
        self.model_ema.requires_grad_(False)
        self.model_anchor.eval()
        self.model_anchor.requires_grad_(False)

    def forward(self, batched_inputs):
        if self.episodic:
            self.reset()

        if self.optimizer is None:
            with torch.no_grad():
                return self.model(batched_inputs)

        for _ in range(max(1, self.steps)):
            student_inputs = self._build_student_inputs(batched_inputs)
            self.forward_and_adapt(student_inputs)

        with torch.no_grad():
            return self.model_ema(batched_inputs)

    def forward_and_adapt(self, batched_inputs):
        if self.optimizer is None:
            with torch.no_grad():
                return self.model_ema(batched_inputs)

        self.optimizer.zero_grad(set_to_none=True)
        student_outputs = self.model(batched_inputs)

        with torch.no_grad():
            anchor_outputs = self.model_anchor(batched_inputs)
            anchor_conf = self._anchor_confidence(anchor_outputs)
            ema_outputs = self.model_ema(batched_inputs)
            if anchor_conf < self.ap and len(self.scale_ratios) > 0:
                ema_outputs = self.create_ensemble_pred(batched_inputs, ema_outputs)

        loss = self._distillation_loss(student_outputs, ema_outputs)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.model_ema = update_ema_variables(self.model_ema, self.model, self.mt)
        self._stochastic_restore()
        return ema_outputs

    def create_ensemble_pred(self, batched_inputs, ema_outputs):
        ensemble_outputs = copy.deepcopy(ema_outputs)
        sem_seg_sums = [output["sem_seg"].detach().clone() for output in ema_outputs]

        for scale_ratio in self.scale_ratios:
            augmented_inputs, flip_flags = self._build_teacher_augmented_inputs(batched_inputs, scale_ratio)
            augmented_outputs = self.model_ema(augmented_inputs)
            for idx, output in enumerate(augmented_outputs):
                sem_seg = output["sem_seg"].detach()
                if flip_flags[idx]:
                    sem_seg = sem_seg.flip(dims=(-1,))
                sem_seg_sums[idx] = sem_seg_sums[idx] + sem_seg

        normalizer = float(len(self.scale_ratios) + 1)
        for idx, output in enumerate(ensemble_outputs):
            output["sem_seg"] = sem_seg_sums[idx] / normalizer
        return ensemble_outputs

    def _build_student_inputs(self, batched_inputs):
        augmented_inputs = []
        for _ in range(self.student_crops):
            for sample in batched_inputs:
                augmented_inputs.append(self._random_crop_resize_sample(sample))
        return augmented_inputs

    def _build_teacher_augmented_inputs(self, batched_inputs, scale_ratio):
        augmented_inputs = []
        flip_flags = []
        for sample in batched_inputs:
            flip = random.random() <= 0.5
            augmented_inputs.append(self._scale_jitter_sample(sample, scale_ratio, flip))
            flip_flags.append(flip)
        return augmented_inputs, flip_flags

    def _random_crop_resize_sample(self, sample):
        augmented = dict(sample)
        augmented["image"] = self._random_resized_crop(augmented["image"], self.student_crop_scale)
        augmented["evf_image"] = self._random_resized_crop(augmented["evf_image"], self.student_crop_scale)
        if "padding_mask" in augmented:
            augmented["padding_mask"] = self._random_resized_crop(
                augmented["padding_mask"].float().unsqueeze(0),
                self.student_crop_scale,
                mode="nearest",
            ).squeeze(0).to(sample["padding_mask"].dtype)
        return augmented

    def _scale_jitter_sample(self, sample, scale_ratio, flip):
        augmented = dict(sample)
        augmented["image"] = self._scale_jitter_tensor(augmented["image"], scale_ratio, flip)
        augmented["evf_image"] = self._scale_jitter_tensor(augmented["evf_image"], scale_ratio, flip)
        if "padding_mask" in augmented:
            augmented["padding_mask"] = self._scale_jitter_tensor(
                augmented["padding_mask"].float().unsqueeze(0),
                scale_ratio,
                flip,
                mode="nearest",
            ).squeeze(0).to(sample["padding_mask"].dtype)
        return augmented

    def _scale_jitter_tensor(self, tensor, scale_ratio, flip, mode="bilinear"):
        original_dtype = tensor.dtype
        work_tensor = tensor.float()
        if flip:
            work_tensor = work_tensor.flip(dims=(-1,))
        resized = self._resize_with_ratio(work_tensor, scale_ratio, mode=mode)
        return resized.to(original_dtype)

    def _random_resized_crop(self, tensor, crop_scale, mode="bilinear"):
        original_dtype = tensor.dtype
        work_tensor = tensor.float()
        if work_tensor.dim() == 2:
            work_tensor = work_tensor.unsqueeze(0)
        _, height, width = work_tensor.shape
        crop_h = max(1, int(height * crop_scale))
        crop_w = max(1, int(width * crop_scale))
        if crop_h >= height and crop_w >= width:
            return tensor
        top = random.randint(0, max(0, height - crop_h))
        left = random.randint(0, max(0, width - crop_w))
        cropped = work_tensor[:, top:top + crop_h, left:left + crop_w]
        resized = F.interpolate(
            cropped.unsqueeze(0),
            size=(height, width),
            mode=mode,
            align_corners=False if mode != "nearest" else None,
        ).squeeze(0)
        if tensor.dim() == 2:
            resized = resized.squeeze(0)
        return resized.to(original_dtype)

    def _resize_with_ratio(self, tensor, scale_ratio, mode="bilinear"):
        original_shape = tensor.shape
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        _, height, width = tensor.shape
        scaled_h = max(1, int(height * scale_ratio))
        scaled_w = max(1, int(width * scale_ratio))
        scaled = F.interpolate(
            tensor.unsqueeze(0),
            size=(scaled_h, scaled_w),
            mode=mode,
            align_corners=False if mode != "nearest" else None,
        )
        restored = F.interpolate(
            scaled,
            size=(height, width),
            mode=mode,
            align_corners=False if mode != "nearest" else None,
        ).squeeze(0)
        if len(original_shape) == 2:
            restored = restored.squeeze(0)
        return restored

    def _anchor_confidence(self, outputs):
        confidences = []
        for output in outputs:
            sem_seg = output["sem_seg"].float()
            confidences.append(sem_seg.softmax(dim=0).max(dim=0)[0].mean())
        if len(confidences) == 0:
            return 1.0
        return float(torch.stack(confidences).mean().item())

    def _distillation_loss(self, student_outputs, teacher_outputs):
        losses = []
        for student_output, teacher_output in zip(student_outputs, teacher_outputs):
            student_sem_seg = student_output["sem_seg"].float()
            teacher_sem_seg = teacher_output["sem_seg"].float()
            losses.append(self._softmax_entropy(student_sem_seg, teacher_sem_seg).mean())
        return torch.stack(losses).mean()

    def _softmax_entropy(self, student_logits, teacher_logits):
        return -(teacher_logits.softmax(dim=0) * student_logits.log_softmax(dim=0)).sum(dim=0)

    def _stochastic_restore(self):
        if self.rst <= 0.0:
            return
        source_state = self.model_state
        for module_name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                if not param.requires_grad or full_name not in source_state:
                    continue
                mask = (torch.rand_like(param, dtype=torch.float32) < self.rst).to(param.dtype)
                with torch.no_grad():
                    param.data.copy_(source_state[full_name] * mask + param.data * (1.0 - mask))
