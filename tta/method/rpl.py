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
        probs_with_targets = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(self.eps)
        if abs(self.q) < self.eps:
            return -probs_with_targets.log()
        return (1.0 - probs_with_targets.pow(self.q)) / self.q


class RPL(Tent):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        self.gce = GeneralizedCrossEntropy(q=float(cfg.TTA.RPL.Q))

    def forward_and_adapt(self, batched_inputs):
        if self.optimizer is None:
            with torch.no_grad():
                return self.model(batched_inputs)

        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(batched_inputs)
        losses = []
        for output in outputs:
            sem_seg = output["sem_seg"].float().unsqueeze(0)
            pseudo_labels = sem_seg.detach().argmax(dim=1)
            losses.append(self.gce(sem_seg, pseudo_labels).mean())

        if losses:
            loss = torch.stack(losses).mean()
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        return outputs
