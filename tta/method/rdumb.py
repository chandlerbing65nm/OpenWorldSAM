import math

import torch
import torch.nn.functional as F

from .tent import Tent, pixelwise_softmax_entropy


@torch.no_grad()
def update_model_probs(current_model_probs: torch.Tensor, new_probs: torch.Tensor, momentum: float):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        return new_probs.mean(0)

    if new_probs.size(0) == 0:
        return current_model_probs

    return momentum * current_model_probs + (1 - momentum) * new_probs.mean(0)


class RDumb(Tent):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        num_classes = int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
        self.e_margin = float(cfg.TTA.RDUMB.MARGIN_E0) * math.log(max(num_classes, 2))
        self.d_margin = float(cfg.TTA.RDUMB.D_MARGIN)
        self.prob_ema = float(cfg.TTA.RDUMB.PROB_EMA)
        self.reset_after_num_samples = int(cfg.TTA.RDUMB.RESET_AFTER_NUM_SAMPLES)
        self.current_model_probs = None
        self.processed_samples = 0
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0

    def forward(self, batched_inputs):
        if self.episodic:
            self.reset()

        outputs = None
        for _ in range(max(1, self.steps)):
            outputs = self.forward_and_adapt(batched_inputs)
            self.processed_samples += len(batched_inputs)

        if self.reset_after_num_samples > 0 and self.processed_samples >= self.reset_after_num_samples:
            self.reset()

        return outputs

    def reset(self):
        super().reset()
        self.current_model_probs = None
        self.processed_samples = 0

    def forward_and_adapt(self, batched_inputs):
        if self.optimizer is None:
            with torch.no_grad():
                return self.model(batched_inputs)

        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(batched_inputs)
        loss, perform_update = self._rdumb_loss(outputs)
        if perform_update:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return outputs

    def _rdumb_loss(self, outputs):
        entropies = []
        probs = []
        for output in outputs:
            sem_seg = output["sem_seg"].float()
            entropies.append(pixelwise_softmax_entropy(sem_seg).mean())
            probs.append(sem_seg.softmax(dim=0).mean(dim=(1, 2)))

        if len(entropies) == 0:
            return None, False

        entropies = torch.stack(entropies)
        probs = torch.stack(probs)

        filter_ids_1 = torch.where(entropies < self.e_margin)
        entropies = entropies[filter_ids_1]
        probs = probs[filter_ids_1]

        if self.current_model_probs is not None:
            cosine_similarities = F.cosine_similarity(
                self.current_model_probs.unsqueeze(0),
                probs,
                dim=1,
            )
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            entropies = entropies[filter_ids_2]
            probs = probs[filter_ids_2]
            updated_probs = update_model_probs(self.current_model_probs, probs, self.prob_ema)
        else:
            updated_probs = update_model_probs(self.current_model_probs, probs, self.prob_ema)

        self.num_samples_update_1 += int(filter_ids_1[0].numel())
        self.num_samples_update_2 += int(entropies.numel())
        self.current_model_probs = updated_probs

        if entropies.numel() == 0:
            return None, False

        coeff = 1 / torch.exp(entropies.detach() - self.e_margin)
        entropies = entropies * coeff
        return entropies.mean(), True
