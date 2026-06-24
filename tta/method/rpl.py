import torch

from .tent import Tent


class GeneralizedCrossEntropy:
    def __init__(self, q=0.8, eps=1e-8):
        self.q = float(q)
        self.eps = float(eps)

    def __call__(self, logits, targets=None):
        probs = logits.softmax(dim=1)
        if targets is None:
            targets = probs.argmax(dim=1)
        if targets.dtype.is_floating_point:
            probs_with_targets = (probs * targets).sum(dim=1).clamp_min(self.eps)
            if abs(self.q) < self.eps:
                return -(targets * probs.clamp_min(self.eps).log()).sum(dim=1)
            return (targets * (1.0 - probs.clamp_min(self.eps).pow(self.q)) / self.q).sum(dim=1)

        probs_with_targets = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(self.eps)
        if abs(self.q) < self.eps:
            return -probs_with_targets.log()
        return (1.0 - probs_with_targets.pow(self.q)) / self.q


class RPL(Tent):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        self.gce = GeneralizedCrossEntropy(q=float(cfg.TTA.RPL.Q))
        self.pseudo_label_type = str(getattr(cfg.TTA.RPL, "PSEUDO_LABEL_TYPE", "hard")).lower()
        if self.pseudo_label_type not in {"hard", "soft"}:
            raise ValueError(
                f"Unsupported TTA.RPL.PSEUDO_LABEL_TYPE: {cfg.TTA.RPL.PSEUDO_LABEL_TYPE}. Expected 'hard' or 'soft'."
            )
        self.temperature = float(getattr(cfg.TTA.RPL, "TEMPERATURE", 1.0))
        if self.temperature <= 0:
            raise ValueError(f"TTA.RPL.TEMPERATURE must be > 0, got {self.temperature}")
        self.confidence_filter = str(getattr(cfg.TTA.RPL, "CONFIDENCE_FILTER", "off")).lower()
        if self.confidence_filter not in {"off", "max_prob"}:
            raise ValueError(
                f"Unsupported TTA.RPL.CONFIDENCE_FILTER: {cfg.TTA.RPL.CONFIDENCE_FILTER}. Expected 'off' or 'max_prob'."
            )
        self.confidence_threshold = float(getattr(cfg.TTA.RPL, "CONFIDENCE_THRESHOLD", 0.0))
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"TTA.RPL.CONFIDENCE_THRESHOLD must be in [0, 1], got {self.confidence_threshold}"
            )

    def _build_pseudo_labels_and_confidence(self, sem_seg):
        pseudo_probs = (sem_seg.detach() / self.temperature).softmax(dim=1) if self.pseudo_label_type == "soft" else sem_seg.detach().softmax(dim=1)
        confidence = pseudo_probs.max(dim=1).values
        if self.pseudo_label_type == "soft":
            pseudo_labels = pseudo_probs
        else:
            pseudo_labels = pseudo_probs.argmax(dim=1)
        return pseudo_labels, confidence

    def forward_and_adapt(self, batched_inputs):
        if self.optimizer is None:
            with torch.no_grad():
                return self.model(batched_inputs)

        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(batched_inputs)
        losses = []
        for output in outputs:
            sem_seg = output["sem_seg"].float().unsqueeze(0)
            pseudo_labels, confidence = self._build_pseudo_labels_and_confidence(sem_seg)
            loss_map = self.gce(sem_seg, pseudo_labels)
            if self.confidence_filter == "max_prob":
                valid_mask = confidence >= self.confidence_threshold
                if valid_mask.any():
                    losses.append(loss_map[valid_mask].mean())
            else:
                losses.append(loss_map.mean())

        if losses:
            loss = torch.stack(losses).mean()
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        return outputs
