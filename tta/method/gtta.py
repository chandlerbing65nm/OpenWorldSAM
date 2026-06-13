import logging

import numpy as np
import torch
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.structures import BitMasks, Instances

from datasets import OpenWorldSAM2SemanticDatasetMapper

from .tent import Tent


logger = logging.getLogger("open-world-sam2-tta")


class GTTA(Tent):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        self.device = next(self.model.parameters()).device
        self.ignore_label = int(getattr(getattr(self.model, "metadata", None), "ignore_label", 255))
        self.lambda_ce_trg = float(cfg.TTA.GTTA.LAMBDA_CE_TRG)
        self.source_batch_size = max(1, int(cfg.TTA.GTTA.SOURCE_BATCH_SIZE))
        self.pseudo_momentum = float(cfg.TTA.GTTA.PSEUDO_MOMENTUM)
        self.avg_conf = torch.tensor(0.9, device=self.device)
        self.requested_style_transfer = bool(cfg.TTA.GTTA.USE_STYLE_TRANSFER)
        if self.requested_style_transfer:
            logger.warning(
                "GTTA style transfer was requested, but the AdaIN/VGG style-transfer subsystem "
                "from the semantic TTA codebase is not available in OpenWorldSAM. "
                "Proceeding with source-supervised + target pseudo-label GTTA only."
            )

        self.source_dataset = self._resolve_source_dataset()
        self.source_loader = self._build_source_loader(self.source_dataset)
        self.source_loader_iter = iter(self.source_loader)

    def reset(self):
        super().reset()
        self.avg_conf = torch.tensor(0.9, device=self.device)
        self.source_loader_iter = iter(self.source_loader)

    def _resolve_source_dataset(self):
        configured_dataset = str(getattr(self.cfg.TTA.GTTA, "SOURCE_DATASET", "")).strip()
        if configured_dataset:
            return configured_dataset

        dataset_key = str(self.cfg.TTA.DATASET).lower()
        source_datasets = {
            "suim_c_sem_seg": "suim_sem_seg_val",
            "dutuseg_c_sem_seg": "dutuseg_sem_seg_val",
        }
        if dataset_key not in source_datasets:
            raise ValueError(
                f"Unable to infer GTTA source dataset for TTA.DATASET={self.cfg.TTA.DATASET}. "
                "Set TTA.GTTA.SOURCE_DATASET explicitly."
            )
        return source_datasets[dataset_key]

    def _build_source_loader(self, dataset_name):
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

    def _next_source_batch(self):
        try:
            return next(self.source_loader_iter)
        except StopIteration:
            self.source_loader_iter = iter(self.source_loader)
            return next(self.source_loader_iter)

    def _forward_loss_dict(self, batched_inputs):
        previous_training = self.model.training
        self.model.training = True
        try:
            loss_dict = self.model(batched_inputs)
        finally:
            self.model.training = previous_training
        return loss_dict

    def _optimize_batch(self, batched_inputs, loss_weight=1.0):
        if self.optimizer is None or len(batched_inputs) == 0:
            return

        self.optimizer.zero_grad(set_to_none=True)
        loss_dict = self._forward_loss_dict(batched_inputs)
        if not isinstance(loss_dict, dict) or len(loss_dict) == 0:
            self.optimizer.zero_grad(set_to_none=True)
            return

        loss = None
        for value in loss_dict.values():
            loss = value if loss is None else loss + value

        if loss is None:
            self.optimizer.zero_grad(set_to_none=True)
            return

        loss = loss * float(loss_weight)
        if loss.requires_grad:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _create_pseudo_batch(self, batched_inputs):
        outputs = self.model(batched_inputs)

        class_names = None
        if hasattr(self.model, "metadata") and hasattr(self.model.metadata, "stuff_classes"):
            class_names = self.model.metadata.stuff_classes
        if class_names is None:
            class_names = MetadataCatalog.get(self.source_dataset).stuff_classes

        confidence_means = []
        pseudo_batches = []
        per_sample_confidences = []
        per_sample_labels = []

        for output in outputs:
            sem_seg = output["sem_seg"].float()
            confidences, pseudo_labels = torch.max(sem_seg.softmax(dim=0), dim=0)
            confidence_means.append(confidences.mean())
            per_sample_confidences.append(confidences)
            per_sample_labels.append(pseudo_labels)

        if confidence_means:
            batch_conf = torch.stack(confidence_means).mean()
            self.avg_conf = self.pseudo_momentum * self.avg_conf + (1.0 - self.pseudo_momentum) * batch_conf
        threshold = torch.sqrt(self.avg_conf).clamp(max=1.0)

        for sample, confidences, pseudo_labels in zip(batched_inputs, per_sample_confidences, per_sample_labels):
            filtered_labels = pseudo_labels.clone()
            filtered_labels[confidences < threshold] = self.ignore_label
            filtered_cpu = filtered_labels.to(dtype=torch.int32).detach().cpu()
            unique_categories = [
                int(class_id)
                for class_id in torch.unique(filtered_cpu)
                if int(class_id) != self.ignore_label
            ]

            if len(unique_categories) == 0:
                continue

            pseudo_sample = dict(sample)
            pseudo_sample["semseg"] = filtered_cpu
            pseudo_sample["unique_categories"] = unique_categories
            pseudo_sample["prompt"] = [class_names[class_id] for class_id in unique_categories]

            semseg_np = filtered_cpu.numpy()
            instances = []
            height, width = semseg_np.shape[-2], semseg_np.shape[-1]
            for class_id in unique_categories:
                mask = semseg_np == class_id
                if not np.any(mask):
                    continue
                instance = Instances((height, width))
                instance.gt_classes = torch.as_tensor([class_id], dtype=torch.int64)
                instance.gt_masks = BitMasks(torch.from_numpy(mask[None, ...].astype("bool")))
                instance.gt_boxes = instance.gt_masks.get_bounding_boxes()
                instances.append(instance)
            if len(instances) == 0:
                continue
            pseudo_sample["instances"] = instances
            pseudo_batches.append(pseudo_sample)

        return outputs, pseudo_batches

    def forward_and_adapt(self, batched_inputs):
        if self.optimizer is None:
            with torch.no_grad():
                return self.model(batched_inputs)

        outputs, pseudo_batches = self._create_pseudo_batch(batched_inputs)
        source_batch = self._next_source_batch()

        self._optimize_batch(source_batch, loss_weight=1.0)
        if len(pseudo_batches) > 0 and self.lambda_ce_trg > 0.0:
            self._optimize_batch(pseudo_batches, loss_weight=self.lambda_ce_trg)

        return outputs
