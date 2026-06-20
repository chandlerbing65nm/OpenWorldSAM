import os
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader

from datasets import OpenWorldSAM2SemanticDatasetMapper
from tta.conf import get_tta_init_weights
from .tent import Tent


class AugCrossEntropy(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = float(alpha)

    def __call__(self, x: torch.Tensor, x_aug: torch.Tensor, x_anchor: torch.Tensor) -> torch.Tensor:
        loss_a = -(x.softmax(dim=1) * x_anchor.log_softmax(dim=1)).sum(dim=1)
        loss_b = -(x_aug.softmax(dim=1) * x_anchor.log_softmax(dim=1)).sum(dim=1)
        return (1.0 - self.alpha) * loss_a + self.alpha * loss_b


class SANTA(Tent):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        self.device = next(self.model.parameters()).device
        self.num_classes = int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
        self.temperature = float(cfg.TTA.SANTA.TEMPERATURE)
        self.base_temperature = self.temperature
        self.projection_dim = int(cfg.TTA.SANTA.PROJECTION_DIM)
        self.contrast_mode = str(cfg.TTA.SANTA.CONTRAST_MODE).lower()
        self.lambda_ce_trg = float(cfg.TTA.SANTA.LAMBDA_CE_TRG)
        self.lambda_cont = float(cfg.TTA.SANTA.LAMBDA_CONT)
        self.aug_alpha = float(cfg.TTA.SANTA.AUG_ALPHA)
        self.source_dataset = self._get_source_dataset_name()
        self.source_batch_size = max(1, int(cfg.TTA.SANTA.SOURCE_BATCH_SIZE))
        self.prototype_max_samples = max(0, int(cfg.TTA.SANTA.PROTOTYPE_MAX_SAMPLES))
        self.aug_crop_min = float(cfg.TTA.SANTA.AUG_CROP_MIN)
        self.aug_crop_max = float(cfg.TTA.SANTA.AUG_CROP_MAX)
        self.ignore_label = self._get_ignore_label()

        self.aug_entropy = AugCrossEntropy(alpha=self.aug_alpha)

        self.anchor_model = deepcopy(self.model)
        DetectionCheckpointer(self.anchor_model).resume_or_load(get_tta_init_weights(cfg), resume=False)
        self.anchor_model.eval()
        self.anchor_model.requires_grad_(False)
        for param in self.anchor_model.parameters():
            param.detach_()

        embed_dim = self._infer_embed_dim()
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, self.projection_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim),
        ).to(self.device)

        if self.optimizer is not None:
            self.optimizer.add_param_group({"params": self.projector.parameters(), "lr": self.optimizer.param_groups[0]["lr"]})
            self.optimizer_state = deepcopy(self.optimizer.state_dict())

        self.anchor_model_state = deepcopy(self.anchor_model.state_dict())
        self.projector_state = deepcopy(self.projector.state_dict())

        self.prototype_labels_src, self.prototypes_src = self._load_or_build_source_prototypes()

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        if self.optimizer is not None and self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)
        self.anchor_model.load_state_dict(self.anchor_model_state, strict=True)
        self.projector.load_state_dict(self.projector_state, strict=True)
        self.configure_model()
        self.anchor_model.eval()
        self.anchor_model.requires_grad_(False)
        for param in self.anchor_model.parameters():
            param.detach_()
        self.projector.train()

    def forward_and_adapt(self, batched_inputs):
        if self.optimizer is None or self.prototypes_src is None or self.prototypes_src.numel() == 0:
            with torch.no_grad():
                return self.model(batched_inputs)

        self.optimizer.zero_grad(set_to_none=True)
        outputs, loss = self.loss_calculation(batched_inputs)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return outputs

    def loss_calculation(self, batched_inputs):
        outputs, intermediates = self._forward_with_intermediates(self.model, batched_inputs)
        augmented_inputs = [self._augment_sample(sample) for sample in batched_inputs]
        outputs_aug, intermediates_aug = self._forward_with_intermediates(self.model, augmented_inputs)

        with torch.no_grad():
            outputs_anchor = self.anchor_model(batched_inputs)

        sem_seg = torch.stack([output["sem_seg"].float() for output in outputs], dim=0)
        sem_seg_aug = torch.stack([output["sem_seg"].float() for output in outputs_aug], dim=0)
        sem_seg_anchor = torch.stack([output["sem_seg"].float() for output in outputs_anchor], dim=0)

        features = self._stack_global_features(intermediates)
        features_aug = self._stack_global_features(intermediates_aug)
        nearest_src = self._match_source_prototypes(features)
        contrastive_features = torch.stack([nearest_src, features, features_aug], dim=1)

        loss_contrastive = self.contrastive_loss(features=contrastive_features, labels=None)
        loss_self_training = self.aug_entropy(sem_seg, sem_seg_aug, sem_seg_anchor).mean()
        loss = self.lambda_ce_trg * loss_self_training + self.lambda_cont * loss_contrastive
        return outputs, loss

    def contrastive_loss(self, features, labels=None, mask=None):
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both labels and mask")
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = self.projector(contrast_feature)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)

        if self.contrast_mode == "one":
            anchor_feature = contrast_feature[:batch_size]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast mode: {self.contrast_mode}")

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), max(self.temperature, 1e-6))
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(1, torch.arange(batch_size * anchor_count, device=self.device).view(-1, 1), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp_min(1e-12))

        positive_counts = mask.sum(1).clamp_min(1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / positive_counts
        loss = -(self.temperature / max(self.base_temperature, 1e-6)) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()

    def _infer_embed_dim(self):
        with torch.no_grad():
            dummy = self._build_source_loader(self.source_dataset)
            if dummy is None:
                raise ValueError("SANTA requires a valid source dataset to infer feature dimensionality")
            first_batch = next(iter(dummy))
            _, intermediates = self._forward_with_intermediates(self.anchor_model, first_batch)
            return int(intermediates[0]["image_embed"].shape[0])

    def _forward_with_intermediates(self, model, batched_inputs):
        outputs = []
        intermediates = []
        for sample in batched_inputs:
            sample_outputs, sample_intermediates = model([sample], return_intermediate=True)
            outputs.append(sample_outputs[0])
            intermediates.append(sample_intermediates[0])
        return outputs, intermediates

    def _stack_global_features(self, intermediates):
        features = []
        for item in intermediates:
            image_embed = item["image_embed"].float().unsqueeze(0)
            pooled = F.adaptive_avg_pool2d(image_embed, (1, 1)).flatten(1).squeeze(0)
            features.append(pooled)
        return torch.stack(features, dim=0)

    def _match_source_prototypes(self, features):
        normalized_features = F.normalize(features, dim=1)
        normalized_prototypes = F.normalize(self.prototypes_src, dim=1)
        similarities = normalized_features @ normalized_prototypes.T
        indices = similarities.argmax(dim=1)
        return self.prototypes_src[indices]

    def _augment_sample(self, sample):
        augmented = dict(sample)
        crop_scale = random.uniform(self.aug_crop_min, self.aug_crop_max)
        offset_h = random.random()
        offset_w = random.random()
        flip = random.random() < 0.5

        augmented["image"] = self._apply_aug(sample["image"], crop_scale, offset_h, offset_w, flip, mode="bilinear")
        if "evf_image" in sample:
            augmented["evf_image"] = self._apply_aug(sample["evf_image"], crop_scale, offset_h, offset_w, flip, mode="bilinear")
        if "padding_mask" in sample:
            augmented["padding_mask"] = self._apply_aug(
                sample["padding_mask"].float().unsqueeze(0),
                crop_scale,
                offset_h,
                offset_w,
                flip,
                mode="nearest",
            ).squeeze(0).to(sample["padding_mask"].dtype)
        return augmented

    def _apply_aug(self, tensor, crop_scale, offset_h, offset_w, flip, mode="bilinear"):
        original_dtype = tensor.dtype
        work = tensor.float()
        squeeze = False
        if work.dim() == 2:
            work = work.unsqueeze(0)
            squeeze = True

        if flip:
            work = work.flip(dims=(-1,))

        _, height, width = work.shape
        crop_h = min(height, max(1, int(height * crop_scale)))
        crop_w = min(width, max(1, int(width * crop_scale)))
        max_top = max(0, height - crop_h)
        max_left = max(0, width - crop_w)
        top = int(round(offset_h * max_top)) if max_top > 0 else 0
        left = int(round(offset_w * max_left)) if max_left > 0 else 0
        cropped = work[:, top:top + crop_h, left:left + crop_w]
        restored = F.interpolate(
            cropped.unsqueeze(0),
            size=(height, width),
            mode=mode,
            align_corners=False if mode != "nearest" else None,
        ).squeeze(0)

        if squeeze:
            restored = restored.squeeze(0)
        return restored.to(original_dtype)

    def _get_ignore_label(self):
        if hasattr(self.model, "metadata") and hasattr(self.model.metadata, "ignore_label"):
            return int(self.model.metadata.ignore_label)
        return 255

    def _get_source_dataset_name(self):
        if str(self.cfg.TTA.SANTA.SOURCE_DATASET):
            return str(self.cfg.TTA.SANTA.SOURCE_DATASET)

        dataset_key = str(self.cfg.TTA.DATASET).lower()
        dataset_name_map = {
            "suim_c_sem_seg": "suim_sem_seg_val",
            "suim_sem_seg": "suim_sem_seg_val",
            "suim_sem_seg_val": "suim_sem_seg_val",
            "dutuseg_c_sem_seg": "dutuseg_sem_seg_val",
            "dutuseg_sem_seg": "dutuseg_sem_seg_val",
            "dutuseg_sem_seg_val": "dutuseg_sem_seg_val",
        }
        return dataset_name_map.get(dataset_key, "")

    def _build_source_loader(self, dataset_name):
        if not dataset_name:
            return None

        loader_cfg = self.cfg.clone()
        loader_cfg.defrost()
        loader_cfg.DATASETS.TEST = (dataset_name,)
        loader_cfg.DATALOADER.NUM_WORKERS = int(self.cfg.TTA.NUM_WORKERS)
        loader_cfg.freeze()
        mapper = OpenWorldSAM2SemanticDatasetMapper(loader_cfg, is_train=False)
        return build_detection_test_loader(
            loader_cfg,
            dataset_name=dataset_name,
            mapper=mapper,
            batch_size=self.source_batch_size,
        )

    def _prototype_cache_path(self):
        weights_name = os.path.splitext(os.path.basename(str(get_tta_init_weights(self.cfg))))[0]
        dataset_name = self.source_dataset or str(self.cfg.TTA.DATASET).lower()
        cache_dir = os.path.join(self.cfg.OUTPUT_DIR, "tta", "santa", "prototype_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{dataset_name}_{weights_name}_samples{self.prototype_max_samples}.pt")

    def _load_or_build_source_prototypes(self):
        cache_path = self._prototype_cache_path()
        if os.path.isfile(cache_path):
            payload = torch.load(cache_path, map_location=self.device)
            return payload["labels"].to(self.device), payload["prototypes"].to(self.device)

        loader = self._build_source_loader(self.source_dataset)
        if loader is None:
            return None, None

        class_sums = None
        class_counts = None
        processed = 0

        with torch.no_grad():
            for batch in loader:
                _, intermediates = self._forward_with_intermediates(self.anchor_model, batch)
                feature_maps = torch.stack([item["image_embed"].float() for item in intermediates], dim=0)
                labels = torch.stack([sample["semseg"].long().to(self.device) for sample in batch], dim=0)
                labels = F.interpolate(labels.unsqueeze(1).float(), size=feature_maps.shape[-2:], mode="nearest").squeeze(1).long()

                if class_sums is None:
                    class_sums = torch.zeros(self.num_classes, feature_maps.shape[1], device=self.device)
                    class_counts = torch.zeros(self.num_classes, device=self.device)

                for feat_map, label_map in zip(feature_maps, labels):
                    flat_feat = feat_map.permute(1, 2, 0).reshape(-1, feat_map.shape[0])
                    flat_label = label_map.reshape(-1)
                    valid = (flat_label != self.ignore_label) & (flat_label >= 0) & (flat_label < self.num_classes)
                    if not torch.any(valid):
                        continue
                    valid_feat = flat_feat[valid]
                    valid_label = flat_label[valid]
                    unique_labels = torch.unique(valid_label)
                    for class_id in unique_labels.tolist():
                        mask = valid_label == class_id
                        class_sums[class_id] += valid_feat[mask].sum(dim=0)
                        class_counts[class_id] += float(mask.sum().item())

                processed += len(batch)
                if self.prototype_max_samples > 0 and processed >= self.prototype_max_samples:
                    break

        if class_sums is None:
            return None, None

        valid_classes = class_counts > 0
        if not torch.any(valid_classes):
            return None, None

        prototypes = class_sums[valid_classes] / class_counts[valid_classes].unsqueeze(1)
        labels = torch.nonzero(valid_classes, as_tuple=False).flatten()
        torch.save({"labels": labels.cpu(), "prototypes": prototypes.cpu()}, cache_path)
        return labels.to(self.device), prototypes.to(self.device)
