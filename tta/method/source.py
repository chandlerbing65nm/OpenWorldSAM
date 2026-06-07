import torch

from .base import SegTTAMethod


class Source(SegTTAMethod):
    def collect_params(self):
        return [], []

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

    def forward_and_adapt(self, batched_inputs):
        with torch.no_grad():
            return self.model(batched_inputs)
