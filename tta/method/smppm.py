import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.data import build_detection_test_loader

from datasets import OpenWorldSAM2SemanticDatasetMapper
from .base import SegTTAMethod


class UncCELoss(nn.Module):
    def __init__(self, ignore_label=255):
        super().__init__()
        self.ignore_label = ignore_label

    def forward(self, logits, target, conf):
        b, c, h, w = logits.shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, c)
        target = target.reshape(-1)
        conf = conf.reshape(-1)

        valid = target != self.ignore_label
        if not torch.any(valid):
            return logits.sum() * 0.0

        logits = logits[valid]
        target = target[valid]
        conf = conf[valid]

        log_probs = F.log_softmax(logits, dim=1)
        target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        weights = 1.0 + conf
        return -(weights * target_log_probs).mean()


class SMPPM(SegTTAMethod):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        self.device = next(model.parameters()).device
        self.ignore_label = self._get_ignore_label()
        self.prototype_grid_size = max(1, int(cfg.TTA.SMPPM.PROTOTYPE_GRID_SIZE))
        self.seg_loss = UncCELoss(ignore_label=self.ignore_label).to(self.device)
        self.src_loader = self._build_source_loader(self._get_source_dataset_name())
        self.src_loader_iter = iter(self.src_loader) if self.src_loader is not None else None

    def configure_model(self):
        self.model.eval()

    def forward_and_adapt(self, batched_inputs):
        if self.optimizer is None or self.src_loader is None:
            with torch.no_grad():
                return self.model(batched_inputs)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            _, target_intermediates = self.model(batched_inputs, return_intermediate=True)
            target_prototypes = self._build_target_prototypes(target_intermediates)

        source_batch = self._next_source_batch()
        source_outputs, source_intermediates = self.model(source_batch, return_intermediate=True)

        source_logits = torch.stack([output["sem_seg"].float() for output in source_outputs], dim=0)
        source_labels = torch.stack([sample["semseg"].long().to(self.device) for sample in source_batch], dim=0)
        source_features = torch.stack([item["image_embed"].float() for item in source_intermediates], dim=0)

        entropy_map = self._compute_entropy_map(source_logits)
        conf = self._compute_similarity(source_features, target_prototypes).max(dim=1, keepdim=True)[0]
        conf = F.interpolate(conf, size=source_logits.shape[-2:], mode="bilinear", align_corners=False)
        weighting = conf * (1.0 - entropy_map.unsqueeze(1))

        loss = self.seg_loss(source_logits, source_labels, weighting.squeeze(1))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            return self.model(batched_inputs)

    def _compute_entropy_map(self, logits):
        probs = logits.softmax(dim=1)
        entropy = -(probs * logits.log_softmax(dim=1)).sum(dim=1)
        return entropy / math.log(max(logits.shape[1], 2))

    def _build_target_prototypes(self, target_intermediates):
        prototypes = []
        for item in target_intermediates:
            image_embed = item["image_embed"].float().unsqueeze(0)
            pooled = F.adaptive_avg_pool2d(image_embed, (self.prototype_grid_size, self.prototype_grid_size))
            pooled = pooled.squeeze(0).permute(1, 2, 0).reshape(-1, pooled.shape[1])
            prototypes.append(pooled)
        if len(prototypes) == 0:
            raise ValueError("SM_PPM requires target intermediate features but none were returned")
        return torch.cat(prototypes, dim=0)

    def _compute_similarity(self, source_features, target_prototypes):
        normalized_source = F.normalize(source_features, dim=1)
        normalized_prototypes = F.normalize(target_prototypes, dim=1)
        return torch.einsum("bchw,kc->bkhw", normalized_source, normalized_prototypes)

    def _next_source_batch(self):
        try:
            return next(self.src_loader_iter)
        except StopIteration:
            self.src_loader_iter = iter(self.src_loader)
            return next(self.src_loader_iter)

    def _get_ignore_label(self):
        if hasattr(self.model, "metadata") and hasattr(self.model.metadata, "ignore_label"):
            return int(self.model.metadata.ignore_label)
        return 255

    def _get_source_dataset_name(self):
        if str(self.cfg.TTA.SMPPM.SOURCE_DATASET):
            return str(self.cfg.TTA.SMPPM.SOURCE_DATASET)

        dataset_key = str(self.cfg.TTA.DATASET).lower()
        if dataset_key == "suim_c_sem_seg":
            return "suim_sem_seg_val"
        if dataset_key == "dutuseg_c_sem_seg":
            return "dutuseg_sem_seg_val"
        return ""

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
            batch_size=max(1, int(self.cfg.TTA.SMPPM.SOURCE_BATCH_SIZE)),
        )
