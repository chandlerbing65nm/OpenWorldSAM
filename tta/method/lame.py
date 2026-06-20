import copy

import torch
import torch.nn.functional as F

from .base import SegTTAMethod


class AffinityMatrix:
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class KNNAffinity(AffinityMatrix):
    def __init__(self, knn: int, **kwargs):
        self.knn = max(1, int(knn))

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        num_tokens = features.size(0)
        distances = torch.cdist(features, features, p=2)
        n_neighbors = min(self.knn + 1, num_tokens)
        knn_index = distances.topk(n_neighbors, dim=-1, largest=False).indices[:, 1:]
        weights = torch.zeros(num_tokens, num_tokens, device=features.device, dtype=features.dtype)
        weights.scatter_(dim=-1, index=knn_index, value=1.0)
        return weights


class RBFAffinity(AffinityMatrix):
    def __init__(self, sigma: float, knn: int, **kwargs):
        self.sigma = float(sigma)
        self.knn = max(1, int(knn))

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(features, features, p=2)
        n_neighbors = min(self.knn, features.size(0))
        kth_dist = distances.topk(k=n_neighbors, dim=-1, largest=False).values[:, -1]
        sigma = kth_dist.mean().clamp_min(1e-6)
        return torch.exp(-(distances ** 2) / (2.0 * sigma ** 2))


class LinearAffinity(AffinityMatrix):
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return torch.matmul(features, features.t())


def entropy_energy(probabilities, unary, pairwise, bound_lambda):
    return (unary * probabilities - bound_lambda * pairwise * probabilities + probabilities * torch.log(probabilities.clamp_min(1e-20))).sum()


def laplacian_optimization(unary, kernel, bound_lambda=1.0, max_steps=100):
    old_energy = float("inf")
    probabilities = (-unary).softmax(dim=-1)
    for step in range(max_steps):
        pairwise = bound_lambda * kernel.matmul(probabilities)
        probabilities = (-unary + pairwise).softmax(dim=-1)
        energy = entropy_energy(probabilities, unary, pairwise, bound_lambda).item()
        if step > 1 and abs(energy - old_energy) <= 1e-8 * max(abs(old_energy), 1.0):
            break
        old_energy = energy
    return probabilities


_AFFINITY_FACTORIES = {
    "knn": KNNAffinity,
    "rbf": RBFAffinity,
    "linear": LinearAffinity,
}


class LAME(SegTTAMethod):
    def __init__(self, cfg, model):
        self.affinity_name = str(cfg.TTA.LAME.AFFINITY).lower()
        if self.affinity_name not in _AFFINITY_FACTORIES:
            raise ValueError(f"Unsupported LAME affinity: {cfg.TTA.LAME.AFFINITY}")
        self.knn = int(cfg.TTA.LAME.KNN)
        self.sigma = float(cfg.TTA.LAME.SIGMA)
        self.force_symmetry = bool(cfg.TTA.LAME.FORCE_SYMMETRY)
        self.bound_lambda = float(cfg.TTA.LAME.BOUND_LAMBDA)
        self.max_steps = int(cfg.TTA.LAME.MAX_STEPS)
        super().__init__(cfg, model)
        self.affinity = _AFFINITY_FACTORIES[self.affinity_name](sigma=self.sigma, knn=self.knn)

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

    def collect_params(self):
        return [], []

    def forward_and_adapt(self, batched_inputs):
        outputs, intermediates = self._forward_with_intermediates(batched_inputs)
        refined_outputs = []
        for output, intermediate in zip(outputs, intermediates):
            refined_output = copy.deepcopy(output)
            refined_output["sem_seg"] = self._refine_sem_seg(output["sem_seg"].float(), intermediate["image_embed"].float())
            refined_outputs.append(refined_output)
        return refined_outputs

    def _forward_with_intermediates(self, batched_inputs):
        outputs = []
        intermediates = []
        with torch.no_grad():
            for sample in batched_inputs:
                sample_outputs, sample_intermediates = self.model([sample], return_intermediate=True)
                outputs.append(sample_outputs[0])
                intermediates.append(sample_intermediates[0])
        return outputs, intermediates

    def _refine_sem_seg(self, sem_seg: torch.Tensor, image_embed: torch.Tensor) -> torch.Tensor:
        _, out_h, out_w = sem_seg.shape
        embed_c, embed_h, embed_w = image_embed.shape

        sem_seg_lowres = F.interpolate(
            sem_seg.unsqueeze(0),
            size=(embed_h, embed_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        unary = -torch.log(sem_seg_lowres.softmax(dim=0).clamp_min(1e-10))
        unary = unary.permute(1, 2, 0).reshape(embed_h * embed_w, -1)

        features = image_embed.permute(1, 2, 0).reshape(embed_h * embed_w, embed_c)
        features = F.normalize(features, p=2, dim=-1)
        kernel = self.affinity(features)
        if self.force_symmetry:
            kernel = 0.5 * (kernel + kernel.t())

        refined = laplacian_optimization(
            unary=unary,
            kernel=kernel,
            bound_lambda=self.bound_lambda,
            max_steps=self.max_steps,
        )
        refined = refined.reshape(embed_h, embed_w, -1).permute(2, 0, 1)
        refined = torch.log(refined.clamp_min(1e-10))
        refined = F.interpolate(
            refined.unsqueeze(0),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return refined
