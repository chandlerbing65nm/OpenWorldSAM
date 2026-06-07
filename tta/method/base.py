from copy import deepcopy

import torch
import torch.nn as nn


class SegTTAMethod(nn.Module):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.episodic = bool(cfg.TTA.EPISODIC)
        self.steps = int(cfg.TTA.OPTIM.STEPS)
        self.configure_model()
        self.params, self.param_names = self.collect_params()
        self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict()) if self.optimizer is not None else None

    def forward(self, batched_inputs):
        if self.episodic:
            self.reset()

        outputs = None
        for _ in range(max(1, self.steps)):
            outputs = self.forward_and_adapt(batched_inputs)
        return outputs

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        if self.optimizer is not None and self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)
        self.configure_model()

    def collect_params(self):
        params = []
        names = []
        for module_name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                params.append(param)
                names.append(f"{module_name}.{param_name}" if module_name else param_name)
        return params, names

    def setup_optimizer(self):
        method = str(self.cfg.TTA.OPTIM.METHOD).lower()
        if method == "adam":
            return torch.optim.Adam(
                self.params,
                lr=self.cfg.TTA.OPTIM.LR,
                betas=(self.cfg.TTA.OPTIM.BETA, 0.999),
                weight_decay=self.cfg.TTA.OPTIM.WD,
            )
        if method == "adamw":
            return torch.optim.AdamW(
                self.params,
                lr=self.cfg.TTA.OPTIM.LR,
                betas=(self.cfg.TTA.OPTIM.BETA, 0.999),
                weight_decay=self.cfg.TTA.OPTIM.WD,
            )
        if method == "sgd":
            return torch.optim.SGD(
                self.params,
                lr=self.cfg.TTA.OPTIM.LR,
                momentum=self.cfg.TTA.OPTIM.MOMENTUM,
                dampening=self.cfg.TTA.OPTIM.DAMPENING,
                weight_decay=self.cfg.TTA.OPTIM.WD,
                nesterov=self.cfg.TTA.OPTIM.NESTEROV,
            )
        raise ValueError(f"Unsupported optimizer: {self.cfg.TTA.OPTIM.METHOD}")

    def configure_model(self):
        raise NotImplementedError

    def forward_and_adapt(self, batched_inputs):
        raise NotImplementedError
