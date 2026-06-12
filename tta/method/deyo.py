import math

import torch
import torch.nn.functional as F

from .tent import Tent, pixelwise_softmax_entropy


class DeYO(Tent):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        num_classes = int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
        self.reweight_ent = bool(cfg.TTA.DEYO.REWEIGHT_ENT)
        self.reweight_plpd = bool(cfg.TTA.DEYO.REWEIGHT_PLPD)
        self.plpd_threshold = float(cfg.TTA.DEYO.PLPD)
        self.deyo_margin = float(cfg.TTA.DEYO.MARGIN) * math.log(max(num_classes, 2))
        self.margin_e0 = float(cfg.TTA.DEYO.MARGIN_E0) * math.log(max(num_classes, 2))
        self.aug_type = str(cfg.TTA.DEYO.AUG_TYPE)
        self.occlusion_size = int(cfg.TTA.DEYO.OCCLUSION_SIZE)
        self.row_start = int(cfg.TTA.DEYO.ROW_START)
        self.column_start = int(cfg.TTA.DEYO.COLUMN_START)
        self.patch_len = int(cfg.TTA.DEYO.PATCH_LEN)

    def forward_and_adapt(self, batched_inputs):
        if self.optimizer is None:
            with torch.no_grad():
                return self.model(batched_inputs)

        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(batched_inputs)
        loss = self._deyo_loss(batched_inputs, outputs)
        if loss is not None:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return outputs

    def _deyo_loss(self, batched_inputs, outputs):
        ent = torch.stack([
            pixelwise_softmax_entropy(output["sem_seg"].float()).mean()
            for output in outputs
        ])

        filter_ids_1 = torch.where(ent < self.deyo_margin)[0]
        if filter_ids_1.numel() == 0:
            return None

        ent1 = ent[filter_ids_1]
        augmented_inputs = [self._augment_sample(batched_inputs[int(idx)]) for idx in filter_ids_1.tolist()]

        with torch.no_grad():
            outputs_prime = self.model(augmented_inputs)

        prob = torch.stack([outputs[int(idx)]["sem_seg"].float().softmax(dim=0) for idx in filter_ids_1.tolist()])
        prob_prime = torch.stack([output_prime["sem_seg"].float().softmax(dim=0) for output_prime in outputs_prime])

        max_prob = prob.max(dim=1).values.mean(dim=(1, 2))
        max_prob_prime = prob_prime.max(dim=1).values.mean(dim=(1, 2))

        plpd = (max_prob - max_prob_prime).reshape(-1)
        filter_ids_2 = torch.where(plpd > self.plpd_threshold)[0]
        if filter_ids_2.numel() == 0:
            return None

        ent2 = ent1[filter_ids_2]
        plpd2 = plpd[filter_ids_2]

        if self.reweight_ent or self.reweight_plpd:
            coeff = (
                float(self.reweight_ent) * (1.0 / torch.exp(ent2.detach() - self.margin_e0))
                + float(self.reweight_plpd) * (1.0 / torch.exp(-plpd2.detach()))
            )
            ent2 = ent2 * coeff

        return ent2.mean()

    def _augment_sample(self, sample):
        augmented = dict(sample)
        aug_state = self._build_aug_state(sample["image"])

        augmented["image"] = self._apply_augmentation(sample["image"], aug_state)
        if "evf_image" in sample:
            augmented["evf_image"] = self._apply_augmentation(sample["evf_image"], aug_state)
        if "padding_mask" in sample:
            augmented["padding_mask"] = self._apply_augmentation(sample["padding_mask"], aug_state, mode="nearest")
        return augmented

    def _build_aug_state(self, tensor):
        if self.aug_type == "pixel":
            height, width = tensor.shape[-2], tensor.shape[-1]
            return {"perm": torch.randperm(height * width, device=tensor.device)}

        if self.aug_type == "patch":
            height, width = tensor.shape[-2], tensor.shape[-1]
            target_h = (height // max(self.patch_len, 1)) * max(self.patch_len, 1)
            target_w = (width // max(self.patch_len, 1)) * max(self.patch_len, 1)
            if target_h == 0 or target_w == 0:
                return None
            return {
                "target_h": target_h,
                "target_w": target_w,
                "perm": torch.randperm(self.patch_len * self.patch_len, device=tensor.device),
            }

        return None

    def _apply_augmentation(self, tensor, aug_state, mode="bilinear"):
        if self.aug_type == "occ":
            return self._occlude_tensor(tensor)
        if self.aug_type == "patch":
            return self._patch_shuffle_tensor(tensor, aug_state, mode=mode)
        if self.aug_type == "pixel":
            return self._pixel_shuffle_tensor(tensor, aug_state)
        return tensor.clone()

    def _occlude_tensor(self, tensor):
        augmented = tensor.clone()
        if augmented.dim() == 2:
            work_tensor = augmented.unsqueeze(0).float()
            squeeze = True
        else:
            work_tensor = augmented.float()
            squeeze = False

        _, height, width = work_tensor.shape
        occ_h = min(self.occlusion_size, height)
        occ_w = min(self.occlusion_size, width)
        row_start = min(max(self.row_start, 0), max(height - occ_h, 0))
        col_start = min(max(self.column_start, 0), max(width - occ_w, 0))
        fill_value = work_tensor.view(work_tensor.shape[0], -1).mean(dim=1, keepdim=True).view(-1, 1, 1)
        work_tensor[:, row_start:row_start + occ_h, col_start:col_start + occ_w] = fill_value

        if squeeze:
            return work_tensor.squeeze(0).to(tensor.dtype)
        return work_tensor.to(tensor.dtype)

    def _pixel_shuffle_tensor(self, tensor, aug_state):
        if aug_state is None or "perm" not in aug_state:
            return tensor.clone()

        original_dtype = tensor.dtype
        if tensor.dim() == 2:
            work_tensor = tensor.unsqueeze(0)
            squeeze = True
        else:
            work_tensor = tensor
            squeeze = False

        channels, height, width = work_tensor.shape
        shuffled = work_tensor.reshape(channels, height * width)[:, aug_state["perm"]].reshape(channels, height, width)
        if squeeze:
            shuffled = shuffled.squeeze(0)
        return shuffled.to(original_dtype)

    def _patch_shuffle_tensor(self, tensor, aug_state, mode="bilinear"):
        if aug_state is None:
            return tensor.clone()

        original_dtype = tensor.dtype
        if tensor.dim() == 2:
            work_tensor = tensor.unsqueeze(0).float()
            squeeze = True
        else:
            work_tensor = tensor.float()
            squeeze = False

        channels, height, width = work_tensor.shape
        target_h = aug_state["target_h"]
        target_w = aug_state["target_w"]
        resized = F.interpolate(
            work_tensor.unsqueeze(0),
            size=(target_h, target_w),
            mode=mode,
            align_corners=False if mode != "nearest" else None,
        ).squeeze(0)

        patch_len = max(self.patch_len, 1)
        patch_h = target_h // patch_len
        patch_w = target_w // patch_len
        patches = resized.view(channels, patch_len, patch_h, patch_len, patch_w)
        patches = patches.permute(1, 3, 0, 2, 4).contiguous().view(patch_len * patch_len, channels, patch_h, patch_w)
        patches = patches[aug_state["perm"]]
        shuffled = patches.view(patch_len, patch_len, channels, patch_h, patch_w)
        shuffled = shuffled.permute(2, 0, 3, 1, 4).contiguous().view(channels, target_h, target_w)
        restored = F.interpolate(
            shuffled.unsqueeze(0),
            size=(height, width),
            mode=mode,
            align_corners=False if mode != "nearest" else None,
        ).squeeze(0)

        if squeeze:
            restored = restored.squeeze(0)
        return restored.to(original_dtype)
