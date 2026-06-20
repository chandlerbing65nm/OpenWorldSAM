import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

from datasets.dataset_mappers.open_world_sam_semantic_dataset_mapper import beit3_preprocess, sam_preprocess
from .base import SegTTAMethod


class AlphaBatchNorm(nn.Module):
    @staticmethod
    def find_bns(parent, alpha):
        replace_mods = []
        if parent is None:
            return replace_mods
        if isinstance(parent, AlphaBatchNorm):
            return replace_mods
        for name, child in parent.named_children():
            if isinstance(child, nn.BatchNorm2d):
                module = AlphaBatchNorm(child, alpha).to(device=child.weight.device)
                replace_mods.append((parent, name, module))
            elif not isinstance(child, AlphaBatchNorm):
                replace_mods.extend(AlphaBatchNorm.find_bns(child, alpha))
        return replace_mods

    @staticmethod
    def adapt_model(model, alpha):
        replace_mods = AlphaBatchNorm.find_bns(model, alpha)
        for parent, name, child in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, alpha):
        super().__init__()
        self.layer = layer
        self.layer.eval()
        self.alpha = float(alpha)
        self.norm = nn.BatchNorm2d(self.layer.num_features, affine=False, momentum=1.0)

    def forward(self, inputs):
        self.norm(inputs)
        running_mean = (1.0 - self.alpha) * self.layer.running_mean + self.alpha * self.norm.running_mean
        running_var = (1.0 - self.alpha) * self.layer.running_var + self.alpha * self.norm.running_var
        return F.batch_norm(
            inputs,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0.0,
            self.layer.eps,
        )


class MEMO(SegTTAMethod):
    def __init__(self, cfg, model):
        self._base_trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
        self.n_augmentations = max(1, int(cfg.TTA.MEMO.N_AUGMENTATIONS))
        self.aug_batch_size = max(1, int(cfg.TTA.MEMO.AUG_BATCH_SIZE))
        self.bn_alpha = float(cfg.TTA.MEMO.BN_ALPHA)
        super().__init__(cfg, model)

    def _use_selected_module_adaptation(self):
        return any(
            bool(getattr(self.cfg.TTA.ADAPT, name))
            for name in (
                "SAM_VISUAL_ENCODER",
                "SAM_MASK_DECODER",
                "SAM_PROMPT_ENCODER",
                "VLM_ENCODER",
                "SOFT_PROMPTING_TRANSFORMER",
            )
        )

    def _selected_modules(self):
        modules = []
        adapt_cfg = self.cfg.TTA.ADAPT

        if bool(adapt_cfg.SAM_VISUAL_ENCODER) and hasattr(self.model.visual_model, "image_encoder"):
            modules.append(self.model.visual_model.image_encoder)
        if bool(adapt_cfg.SAM_MASK_DECODER) and hasattr(self.model.visual_model, "sam_mask_decoder"):
            modules.append(self.model.visual_model.sam_mask_decoder)
        if bool(adapt_cfg.SAM_PROMPT_ENCODER) and hasattr(self.model.visual_model, "sam_prompt_encoder"):
            modules.append(self.model.visual_model.sam_prompt_encoder)
        if bool(adapt_cfg.VLM_ENCODER) and hasattr(self.model, "mm_extractor"):
            modules.append(self.model.mm_extractor)
        if bool(adapt_cfg.SOFT_PROMPTING_TRANSFORMER):
            if hasattr(self.model, "text_hidden_fcs"):
                modules.append(self.model.text_hidden_fcs)
            if hasattr(self.model, "cross_attention_transformer"):
                modules.append(self.model.cross_attention_transformer)
        return modules

    def configure_model(self):
        if self.bn_alpha > 0.0:
            self.model = AlphaBatchNorm.adapt_model(self.model, self.bn_alpha)

        self.model.eval()
        self.model.requires_grad_(False)

        if self._use_selected_module_adaptation():
            for module in self._selected_modules():
                module.train()
                module.requires_grad_(True)
            if bool(self.cfg.TTA.ADAPT.SOFT_PROMPTING_TRANSFORMER) and hasattr(self.model, "positional_tokens"):
                self.model.positional_tokens.requires_grad_(True)
            return

        for name, param in self.model.named_parameters():
            canonical_name = name.replace(".layer.", ".")
            if name in self._base_trainable_names or canonical_name in self._base_trainable_names:
                param.requires_grad_(True)

    def collect_params(self):
        params = []
        names = []
        seen = set()
        for name, param in self.model.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            params.append(param)
            names.append(name)
            seen.add(id(param))
        return params, names

    def forward(self, batched_inputs):
        if self.episodic:
            self.reset()

        if self.optimizer is not None:
            for _ in range(max(1, self.steps)):
                self.forward_and_adapt(batched_inputs)

        with torch.no_grad():
            return self.model(batched_inputs)

    def forward_and_adapt(self, batched_inputs):
        if self.optimizer is None:
            with torch.no_grad():
                return self.model(batched_inputs)

        self.optimizer.zero_grad(set_to_none=True)
        total = 0
        for start in range(0, self.n_augmentations, self.aug_batch_size):
            current_batch = min(self.aug_batch_size, self.n_augmentations - start)
            augmented_inputs = self._build_augmented_batch(batched_inputs, current_batch)
            outputs = self.model(augmented_inputs)
            sem_seg = torch.stack([output["sem_seg"].float() for output in outputs], dim=0)
            loss = marginal_entropy(sem_seg)
            loss.backward()
            total += 1

        if total > 0:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            return self.model(batched_inputs)

    def _build_augmented_batch(self, batched_inputs, n_repeat):
        augmented = []
        for _ in range(n_repeat):
            for sample in batched_inputs:
                augmented.append(self._augment_sample(sample))
        return augmented

    def _augment_sample(self, sample):
        augmented = dict(sample)
        image = Image.open(sample["file_name"]).convert("RGB")
        image = self._augmix_aug(image)
        augmented["image"] = sam_preprocess(np.array(image)).to(sample["image"].dtype)
        if "evf_image" in sample:
            augmented["evf_image"] = beit3_preprocess(np.array(image)).to(sample["evf_image"].dtype)
        if "padding_mask" in sample:
            augmented["padding_mask"] = sample["padding_mask"].clone()
        return augmented

    def _augmix_aug(self, image):
        original = self._preaugment(image)
        processed = self._to_tensor(original)
        weights = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
        mix_ratio = np.float32(np.random.beta(1.0, 1.0))

        mix = torch.zeros_like(processed)
        for branch_idx in range(3):
            branch = original.copy()
            for _ in range(np.random.randint(1, 4)):
                branch = random.choice(self._augmentations())(branch)
            mix = mix + weights[branch_idx] * self._to_tensor(branch)

        mixed = mix_ratio * processed + (1.0 - mix_ratio) * mix
        mixed = mixed.clamp(0.0, 1.0)
        mixed_np = (mixed.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(mixed_np)

    def _preaugment(self, image):
        width, height = image.size
        scale = random.uniform(0.3, 1.0)
        crop_w = max(1, int(width * scale))
        crop_h = max(1, int(height * scale))
        if crop_w < width or crop_h < height:
            left = random.randint(0, max(0, width - crop_w))
            top = random.randint(0, max(0, height - crop_h))
            image = image.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.BILINEAR)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def _to_tensor(self, image):
        array = np.asarray(image).astype(np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1))

    def _augmentations(self):
        return [
            self._autocontrast,
            self._equalize,
            lambda x: self._rotate(x, 1),
            lambda x: self._solarize(x, 1),
            lambda x: self._shear_x(x, 1),
            lambda x: self._shear_y(x, 1),
            lambda x: self._translate_x(x, 1),
            lambda x: self._translate_y(x, 1),
            lambda x: self._posterize(x, 1),
        ]

    def _autocontrast(self, image):
        return ImageOps.autocontrast(image)

    def _equalize(self, image):
        return ImageOps.equalize(image)

    def _rotate(self, image, level):
        degrees = self._int_parameter(self._rand_level(level), 30)
        if random.random() > 0.5:
            degrees = -degrees
        return image.rotate(degrees, resample=Image.BILINEAR, fillcolor=128)

    def _solarize(self, image, level):
        level = self._int_parameter(self._rand_level(level), 256)
        return ImageOps.solarize(image, 256 - level)

    def _shear_x(self, image, level):
        amount = self._float_parameter(self._rand_level(level), 0.3)
        if random.random() > 0.5:
            amount = -amount
        width, height = image.size
        return image.transform((width, height), Image.AFFINE, (1, amount, 0, 0, 1, 0), resample=Image.BILINEAR, fillcolor=128)

    def _shear_y(self, image, level):
        amount = self._float_parameter(self._rand_level(level), 0.3)
        if random.random() > 0.5:
            amount = -amount
        width, height = image.size
        return image.transform((width, height), Image.AFFINE, (1, 0, 0, amount, 1, 0), resample=Image.BILINEAR, fillcolor=128)

    def _translate_x(self, image, level):
        width, height = image.size
        amount = self._int_parameter(self._rand_level(level), width / 3)
        if random.random() > 0.5:
            amount = -amount
        return image.transform((width, height), Image.AFFINE, (1, 0, amount, 0, 1, 0), resample=Image.BILINEAR, fillcolor=128)

    def _translate_y(self, image, level):
        width, height = image.size
        amount = self._int_parameter(self._rand_level(level), height / 3)
        if random.random() > 0.5:
            amount = -amount
        return image.transform((width, height), Image.AFFINE, (1, 0, 0, 0, 1, amount), resample=Image.BILINEAR, fillcolor=128)

    def _posterize(self, image, level):
        amount = self._int_parameter(self._rand_level(level), 4)
        return ImageOps.posterize(image, max(1, 4 - amount))

    def _int_parameter(self, level, maxval):
        return int(level * maxval / 10)

    def _float_parameter(self, level, maxval):
        return float(level) * maxval / 10.0

    def _rand_level(self, n):
        return np.random.uniform(low=0.1, high=n)


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=1, keepdim=True)
    avg_logits = logits.logsumexp(dim=(0, 2, 3))
    avg_logits = avg_logits - math.log(logits.shape[0]) - math.log(logits.shape[2]) - math.log(logits.shape[3])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
