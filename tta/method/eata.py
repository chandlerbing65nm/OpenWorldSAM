import logging
import math

import torch
import torch.nn.functional as F
from detectron2.data import build_detection_test_loader

from datasets import OpenWorldSAM2SemanticDatasetMapper
from .tent import Tent, pixelwise_softmax_entropy

logger = logging.getLogger(__name__)


@torch.no_grad()
def update_model_probs(current_model_probs: torch.Tensor, new_probs: torch.Tensor):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        return new_probs.mean(0)

    if new_probs.size(0) == 0:
        return current_model_probs

    return 0.9 * current_model_probs + 0.1 * new_probs.mean(0)


class EATA(Tent):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        num_classes = int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
        self.e_margin = float(cfg.TTA.EATA.MARGIN_E0) * math.log(max(num_classes, 2))
        self.d_margin = float(cfg.TTA.EATA.D_MARGIN)
        self.fisher_alpha = float(cfg.TTA.EATA.FISHER_ALPHA)
        self.reset_after_num_updates = int(cfg.TTA.EATA.RESET_AFTER_NUM_UPDATES)
        self.current_model_probs = None
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0
        self.performed_updates = 0
        self.fishers = self._compute_fishers()

    def forward(self, batched_inputs):
        if self.episodic:
            self.reset()

        outputs = None
        should_reset_after_forward = False
        for _ in range(max(1, self.steps)):
            outputs = self.forward_and_adapt(batched_inputs)
            self.performed_updates += 1
            if self.reset_after_num_updates > 0 and self.performed_updates % self.reset_after_num_updates == 0:
                should_reset_after_forward = True

        if should_reset_after_forward:
            self.reset()
        return outputs

    def reset(self):
        super().reset()
        self.current_model_probs = None
        self.performed_updates = 0

    def forward_and_adapt(self, batched_inputs):
        if self.optimizer is None:
            with torch.no_grad():
                return self.model(batched_inputs)

        self.optimizer.zero_grad(set_to_none=True)
        outputs, loss, perform_update = self.loss_calculation(batched_inputs)
        if perform_update:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return outputs

    def loss_calculation(self, batched_inputs):
        outputs = self.model(batched_inputs)

        entropies = []
        probs_kept1 = []
        for output in outputs:
            sem_seg = output["sem_seg"].float()
            entropies.append(pixelwise_softmax_entropy(sem_seg).mean())
            probs_kept1.append(sem_seg.softmax(dim=0).mean(dim=(1, 2)))

        if len(entropies) == 0:
            return outputs, None, False

        entropies = torch.stack(entropies)
        probs_kept1 = torch.stack(probs_kept1)

        filter_ids_1 = torch.where(entropies < self.e_margin)
        entropies = entropies[filter_ids_1]
        probs_kept1 = probs_kept1[filter_ids_1]

        if self.current_model_probs is not None:
            cosine_similarities = F.cosine_similarity(
                self.current_model_probs.unsqueeze(dim=0),
                probs_kept1,
                dim=1,
            )
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropies = entropies[filter_ids_2]
            updated_probs = update_model_probs(self.current_model_probs, probs_kept1[filter_ids_2])
        else:
            updated_probs = update_model_probs(self.current_model_probs, probs_kept1)

        self.num_samples_update_1 += int(filter_ids_1[0].numel())
        self.num_samples_update_2 += int(entropies.numel())
        self.current_model_probs = updated_probs

        if entropies.numel() == 0:
            return outputs, None, False

        coeff = 1 / torch.exp(entropies.detach() - self.e_margin)
        loss = entropies.mul(coeff).mean()

        if self.fishers is not None:
            ewc_loss = 0.0
            for name, param in self.model.named_parameters():
                if name in self.fishers:
                    fisher, reference = self.fishers[name]
                    ewc_loss = ewc_loss + self.fisher_alpha * (fisher * (param - reference) ** 2).sum()
            loss = loss + ewc_loss

        return outputs, loss, True

    def _compute_fishers(self):
        if self.fisher_alpha <= 0.0:
            logger.info("EATA Fisher disabled; using entropy-only EATA behavior")
            return None

        source_dataset = self._get_source_dataset_name()
        if not source_dataset:
            logger.warning("Skipping EATA Fisher computation because no source dataset is configured")
            return None

        try:
            fisher_loader = self._build_source_loader(source_dataset)
        except Exception as exc:
            logger.warning("Skipping EATA Fisher computation for %s: %s", source_dataset, exc)
            return None

        train_loss_fn = torch.nn.CrossEntropyLoss().to(next(self.model.parameters()).device)
        fishers = {}
        processed_samples = 0
        processed_batches = 0
        max_samples = int(self.cfg.TTA.EATA.NUM_SAMPLES)

        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)

        for batch in fisher_loader:
            outputs = self.model(batch)
            losses = []
            batch_size = len(outputs)
            processed_samples += batch_size
            processed_batches += 1
            for output in outputs:
                sem_seg = output["sem_seg"].float().unsqueeze(0)
                targets = sem_seg.detach().argmax(dim=1)
                losses.append(train_loss_fn(sem_seg, targets))
            if len(losses) == 0:
                continue
            loss = torch.stack(losses).mean()
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                fisher = param.grad.detach().clone() ** 2
                if name in fishers:
                    fisher = fisher + fishers[name][0]
                fishers[name] = [fisher, param.detach().clone()]
            self.optimizer.zero_grad(set_to_none=True)
            if max_samples > 0 and processed_samples >= max_samples:
                break

        if not fishers:
            logger.warning("Skipping EATA Fisher regularization because no gradients were accumulated")
            return None

        num_batches = max(1, processed_batches)
        for name in list(fishers.keys()):
            fishers[name][0] = fishers[name][0] / num_batches

        logger.info("Finished computing EATA fisher matrices from %s", source_dataset)
        return fishers

    def _get_source_dataset_name(self):
        if str(self.cfg.TTA.EATA.SOURCE_DATASET):
            return str(self.cfg.TTA.EATA.SOURCE_DATASET)

        dataset_key = str(self.cfg.TTA.DATASET).lower()
        if dataset_key == "suim_c_sem_seg":
            return "suim_sem_seg_val"
        if dataset_key == "dutuseg_c_sem_seg":
            return "dutuseg_sem_seg_val"
        return ""

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
            batch_size=max(1, int(self.cfg.TTA.EATA.FISHER_BATCH_SIZE)),
        )
