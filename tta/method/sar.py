import logging
import math

import torch

from .tent import Tent, pixelwise_softmax_entropy

logger = logging.getLogger(__name__)


@torch.no_grad()
def update_ema(ema, new_data, alpha=0.9):
    if ema is None:
        return new_data
    return alpha * ema + (1 - alpha) * new_data


class SAM(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0, nesterov=False, rho=0.05, adaptive=False):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            rho=rho,
            adaptive=adaptive,
        )
        super().__init__(params, defaults)
        self.base_optimizer = torch.optim.SGD(
            self.param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for param in group["params"]:
                if param.grad is None:
                    continue
                self.state[param]["old_p"] = param.data.clone()
                e_w = ((torch.pow(param, 2) if group["adaptive"] else 1.0) * param.grad * scale.to(param))
                param.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.data = self.state[param]["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                scale = torch.abs(param) if group["adaptive"] else 1.0
                norms.append((scale * param.grad).norm(p=2).to(shared_device))
        if len(norms) == 0:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class SAR(Tent):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        num_classes = int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
        self.margin_e0 = float(cfg.TTA.SAR.MARGIN_E0) * math.log(max(num_classes, 2))
        self.reset_constant_em = float(cfg.TTA.SAR.RESET_CONSTANT_EM)
        self.ema = None

    def setup_optimizer(self):
        return SAM(
            self.params,
            lr=self.cfg.TTA.OPTIM.LR,
            momentum=self.cfg.TTA.OPTIM.MOMENTUM,
            weight_decay=self.cfg.TTA.OPTIM.WD,
            nesterov=self.cfg.TTA.OPTIM.NESTEROV,
            rho=self.cfg.TTA.SAR.RHO,
            adaptive=bool(self.cfg.TTA.SAR.ADAPTIVE),
        )

    def reset(self):
        super().reset()
        self.ema = None

    def forward_and_adapt(self, batched_inputs):
        if self.optimizer is None:
            with torch.no_grad():
                return self.model(batched_inputs)

        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(batched_inputs)
        entropies = self._mean_entropies(outputs)
        reliable_mask = entropies < self.margin_e0
        if not reliable_mask.any():
            return outputs

        loss = entropies[reliable_mask].mean()
        loss.backward()
        self.optimizer.first_step(zero_grad=True)

        outputs_second = self.model(batched_inputs)
        entropies_second = self._mean_entropies(outputs_second)
        entropies_second_filtered = entropies_second[reliable_mask]
        reliable_second_mask = entropies_second_filtered < self.margin_e0
        if not reliable_second_mask.any():
            self.optimizer.second_step(zero_grad=True)
            return outputs

        loss_second = entropies_second_filtered[reliable_second_mask].mean()
        if not torch.isnan(loss_second):
            self.ema = update_ema(self.ema, float(loss_second.item()))

        loss_second.backward()
        self.optimizer.second_step(zero_grad=True)

        if self.ema is not None and self.ema < self.reset_constant_em:
            logger.info("ema < %.4f, resetting SAR model state", self.reset_constant_em)
            self.reset()

        return outputs

    def _mean_entropies(self, outputs):
        entropies = []
        for output in outputs:
            sem_seg = output["sem_seg"].float()
            entropy = pixelwise_softmax_entropy(sem_seg.unsqueeze(0)).mean()
            entropies.append(entropy)
        if len(entropies) == 0:
            return torch.zeros(0, device=self.model_state[next(iter(self.model_state))].device)
        return torch.stack(entropies)
