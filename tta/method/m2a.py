import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tent import Tent
from model.segment_anything_2.sam2.modeling.sam2_utils import LayerNorm2d


@torch.jit.script
def softmax_entropy_seg(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1)


def _gaussian_kernel1d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    center = (kernel_size - 1) / 2.0
    xs = torch.arange(kernel_size, device=device, dtype=dtype) - center
    kernel = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
    kernel = kernel / (kernel.sum() + 1e-12)
    return kernel


def gaussian_blur2d(x: torch.Tensor, kernel_size: int = 11, sigma: float = None) -> torch.Tensor:
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    if sigma is None:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    _, c, _, _ = x.shape
    device = x.device
    dtype = x.dtype
    k1d = _gaussian_kernel1d(kernel_size, sigma, device, dtype)
    k2d = torch.outer(k1d, k1d)
    kernel = k2d.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.to(device=device, dtype=dtype)
    kernel = kernel.expand(c, 1, kernel_size, kernel_size).contiguous()
    padding = kernel_size // 2
    return F.conv2d(x, kernel, bias=None, stride=1, padding=padding, groups=c)


def _build_conjugate_pairs(h: int, w: int, device: torch.device):
    us = torch.arange(h, device=device)
    vs = torch.arange(w, device=device)
    uu, vv = torch.meshgrid(us, vs, indexing="ij")
    uu_flat = uu.reshape(-1)
    vv_flat = vv.reshape(-1)
    cu = (-uu_flat) % h
    cv = (-vv_flat) % w
    idx = uu_flat * w + vv_flat
    cidx = cu * w + cv
    keep = idx < cidx
    pair_uv = torch.stack([uu_flat[keep], vv_flat[keep]], dim=1)
    pair_conj = torch.stack([cu[keep], cv[keep]], dim=1)
    return pair_uv, pair_conj


def _radial_distance(h: int, w: int, device: torch.device) -> torch.Tensor:
    fu = torch.arange(h, device=device, dtype=torch.float32)
    fv = torch.arange(w, device=device, dtype=torch.float32)
    fu = torch.where(fu <= h // 2, fu, fu - h).float()
    fv = torch.where(fv <= w // 2, fv, fv - w).float()
    gu, gv = torch.meshgrid(fu, fv, indexing="ij")
    max_r = math.sqrt((h // 2) ** 2 + (w // 2) ** 2)
    if max_r == 0:
        max_r = 1.0
    return torch.sqrt(gu * gu + gv * gv) / max_r


def apply_frequency_mask(x: torch.Tensor, mask_percent: float, spectral_type: str = "all") -> torch.Tensor:
    mask_percent = float(mask_percent)
    mask_percent = max(0.0, min(100.0, mask_percent))
    b, _, h, w = x.shape
    if mask_percent <= 0.0:
        return x

    x_fft = torch.fft.fft2(x.to(torch.float32), dim=(-2, -1), norm="ortho")
    st = str(spectral_type).lower()
    if st not in ("all", "low", "high"):
        st = "all"

    device = x.device
    pair_uv, pair_conj = _build_conjugate_pairs(h, w, device)
    r = _radial_distance(h, w, device)
    cutoff = 0.5

    if st == "all":
        pair_in_band = torch.ones(pair_uv.shape[0], device=device, dtype=torch.bool)
    else:
        r_pair = r[pair_uv[:, 0], pair_uv[:, 1]]
        pair_in_band = r_pair <= cutoff if st == "low" else r_pair > cutoff

    eligible_idx = pair_in_band.nonzero(as_tuple=False).squeeze(1)
    num_eligible = int(eligible_idx.numel())
    if num_eligible <= 0:
        x_masked = x_fft
    else:
        k = int(math.ceil((mask_percent / 100.0) * num_eligible))
        k = min(k, num_eligible)
        if k <= 0:
            x_masked = x_fft
        else:
            mask_batch = []
            for _ in range(b):
                perm = torch.randperm(num_eligible, device=device)[:k]
                chosen = eligible_idx[perm]
                flat_mask = torch.ones(h * w, device=device, dtype=x_fft.dtype)
                uv = pair_uv[chosen]
                cv = pair_conj[chosen]
                flat_mask[uv[:, 0] * w + uv[:, 1]] = 0
                flat_mask[cv[:, 0] * w + cv[:, 1]] = 0
                mask_batch.append(flat_mask.view(1, h, w))
            mask = torch.stack(mask_batch, dim=0)
            x_masked = x_fft * mask

    return torch.fft.ifft2(x_masked, dim=(-2, -1), norm="ortho").real


def build_random_square_mask(h: int, w: int, ratio: float, num_squares: int = 1, generator: torch.Generator = None) -> torch.Tensor:
    total_area = int(round(ratio * h * w))
    if total_area <= 0 or num_squares <= 0:
        return torch.zeros((h, w), dtype=torch.float32)

    side = int(round(math.sqrt(total_area / float(max(num_squares, 1)))))
    side = max(1, min(side, min(h, w)))
    max_y0 = max(0, h - side)
    max_x0 = max(0, w - side)
    mask = torch.zeros((h, w), dtype=torch.float32)
    placed = []

    def overlaps(y0, x0, s, others):
        for yy, xx, ss in others:
            if not (x0 + s <= xx or xx + ss <= x0 or y0 + s <= yy or yy + ss <= y0):
                return True
        return False

    attempts = 0
    max_attempts = 2000
    while len(placed) < num_squares and attempts < max_attempts:
        y0 = int(torch.randint(low=0, high=max_y0 + 1, size=(1,), generator=generator).item()) if max_y0 > 0 else 0
        x0 = int(torch.randint(low=0, high=max_x0 + 1, size=(1,), generator=generator).item()) if max_x0 > 0 else 0
        if not overlaps(y0, x0, side, placed):
            placed.append((y0, x0, side))
        attempts += 1

    while len(placed) < num_squares:
        y0 = int(torch.randint(low=0, high=max_y0 + 1, size=(1,), generator=generator).item()) if max_y0 > 0 else 0
        x0 = int(torch.randint(low=0, high=max_x0 + 1, size=(1,), generator=generator).item()) if max_x0 > 0 else 0
        placed.append((y0, x0, side))

    for y0, x0, s in placed:
        mask[y0:y0 + s, x0:x0 + s] = 1.0
    return mask


class Entropy(nn.Module):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        ent_map = -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)
        return ent_map.mean(dim=(1, 2))


class M2A(Tent):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        num_classes = int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
        self.entropy = Entropy()
        self.m = float(cfg.TTA.M2A.M)
        self.n = int(cfg.TTA.M2A.N)
        self.mn = [i * self.m for i in range(self.n)]
        self.lambda_erl = float(cfg.TTA.M2A.LAMBDA_ERL)
        self.lambda_eml = float(cfg.TTA.M2A.LAMBDA_EML)
        self.margin = float(cfg.TTA.M2A.MARGIN) * math.log(max(float(num_classes), 2.0))
        self.disable_mcl = bool(cfg.TTA.M2A.DISABLE_MCL)
        self.disable_erl = bool(cfg.TTA.M2A.DISABLE_ERL)
        self.disable_eml = bool(cfg.TTA.M2A.DISABLE_EML)

        random_masking = str(cfg.TTA.M2A.RANDOM_MASKING).lower()
        self.random_masking = random_masking if random_masking in ["spatial", "spectral"] else "spatial"
        self.num_squares = max(1, int(cfg.TTA.M2A.NUM_SQUARES))

        mask_type = str(cfg.TTA.M2A.MASK_TYPE).lower()
        self.mask_type = mask_type if mask_type in ["binary", "gaussian", "mean"] else "binary"

        spatial_type = str(cfg.TTA.M2A.SPATIAL_TYPE).lower()
        self.spatial_type = spatial_type if spatial_type in ["patch", "pixel"] else "patch"

        spectral_type = str(cfg.TTA.M2A.SPECTRAL_TYPE).lower()
        self.spectral_type = spectral_type if spectral_type in ["all", "low", "high"] else "all"

        self._rng = torch.Generator(device="cpu")
        seed = int(cfg.TTA.M2A.SEED)
        if seed >= 0:
            self._rng.manual_seed(seed)

    def collect_params(self):
        params = []
        names = []
        for module_name, module in self.model.named_modules():
            lowered_name = module_name.lower()
            if any(key in lowered_name for key in ("layer4", "blocks.9", "blocks.10", "blocks.11")):
                continue
            if lowered_name in {"norm"} or "norm." in lowered_name:
                continue
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, LayerNorm2d)):
                for param_name, param in module.named_parameters(recurse=False):
                    if param_name not in {"weight", "bias"} or not param.requires_grad:
                        continue
                    params.append(param)
                    names.append(f"{module_name}.{param_name}" if module_name else param_name)
        return params, names

    def forward_and_adapt(self, batched_inputs):
        if self.optimizer is None:
            with torch.no_grad():
                return self.model(batched_inputs)

        self.optimizer.zero_grad(set_to_none=True)
        outputs, loss = self.loss_calculation(batched_inputs)
        if loss is not None and loss.requires_grad:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return outputs

    def loss_calculation(self, batched_inputs):
        images = torch.stack([sample["image"].float() for sample in batched_inputs], dim=0)
        evf_images = None
        if all("evf_image" in sample for sample in batched_inputs):
            evf_images = torch.stack([sample["evf_image"].float() for sample in batched_inputs], dim=0)

        outputs0 = self.model(batched_inputs)
        logits0 = torch.stack([output["sem_seg"].float() for output in outputs0], dim=0)
        teacher_logits = [logits0.detach()]

        mcl_loss = None
        erl_loss = None
        eml_sum = None
        entropy_history = []

        entropy0 = None
        if not self.disable_erl or not self.disable_eml:
            entropy0 = self.entropy(logits0)
        if not self.disable_erl and entropy0 is not None:
            entropy_history.append(entropy0)
        if not self.disable_eml and entropy0 is not None:
            eml_sum = entropy0.mean()

        b, _, h, w = images.shape
        for m_val in self.mn[1:]:
            mfrac = float(m_val)
            xb = self._mask_tensor_batch(images, mfrac)
            evf_xb = self._mask_tensor_batch(evf_images, mfrac) if evf_images is not None else None

            masked_inputs = []
            for idx, sample in enumerate(batched_inputs):
                masked_sample = dict(sample)
                masked_sample["image"] = xb[idx].to(sample["image"].dtype)
                if evf_xb is not None and "evf_image" in sample:
                    masked_sample["evf_image"] = evf_xb[idx].to(sample["evf_image"].dtype)
                masked_inputs.append(masked_sample)
            out_m = self.model(masked_inputs)
            logits_m = torch.stack([output["sem_seg"].float() for output in out_m], dim=0)

            if not self.disable_mcl:
                mcl_term = softmax_entropy_seg(logits_m, teacher_logits[0]).mean()
                for prev_teacher in teacher_logits[1:]:
                    mcl_term = mcl_term + softmax_entropy_seg(logits_m, prev_teacher).mean()
                mcl_loss = mcl_term if mcl_loss is None else mcl_loss + mcl_term

            entropy_m = None
            if not self.disable_erl or not self.disable_eml:
                entropy_m = self.entropy(logits_m)

            if not self.disable_erl and entropy_m is not None:
                for prev_entropy in entropy_history:
                    erl_term = F.relu(prev_entropy - entropy_m.detach() + self.margin).mean()
                    erl_loss = erl_term if erl_loss is None else erl_loss + erl_term
                entropy_history.append(entropy_m)

            if not self.disable_eml and entropy_m is not None:
                eml_term = entropy_m.mean()
                eml_sum = eml_term if eml_sum is None else eml_sum + eml_term

            teacher_logits.append(logits_m.detach())

        total_loss_terms = []

        if mcl_loss is not None and mcl_loss.requires_grad:
            total_loss_terms.append(mcl_loss)

        if erl_loss is not None and erl_loss.requires_grad:
            total_loss_terms.append(self.lambda_erl * erl_loss)

        if eml_sum is not None and eml_sum.requires_grad:
            total_loss_terms.append(self.lambda_eml * (eml_sum / float(len(self.mn))))

        if len(total_loss_terms) > 0:
            loss = total_loss_terms[0]
            for loss_term in total_loss_terms[1:]:
                loss = loss + loss_term
        else:
            loss = None

        return outputs0, loss

    def _mask_tensor_batch(self, tensor_batch, mfrac):
        if tensor_batch is None:
            return None

        b, _, h, w = tensor_batch.shape
        if self.random_masking == "spectral":
            return apply_frequency_mask(tensor_batch, mask_percent=(mfrac * 100.0), spectral_type=self.spectral_type)

        xb = tensor_batch.clone()
        x_blur = gaussian_blur2d(xb, kernel_size=11, sigma=None) if self.mask_type == "gaussian" else None
        if self.spatial_type == "patch":
            for bi in range(b):
                mask_bw = build_random_square_mask(h, w, ratio=mfrac, num_squares=self.num_squares, generator=self._rng).to(tensor_batch.device)
                self._apply_spatial_mask(xb, tensor_batch, bi, mask_bw, x_blur)
            return xb

        total_pixels = h * w
        k_pix = int(round(mfrac * total_pixels))
        k_pix = max(0, min(k_pix, total_pixels))
        for bi in range(b):
            if k_pix > 0:
                flat = torch.zeros((total_pixels,), device=tensor_batch.device, dtype=torch.float32)
                idx = torch.randperm(total_pixels, device=tensor_batch.device)[:k_pix]
                flat[idx] = 1.0
                mask_bw = flat.view(h, w)
            else:
                mask_bw = torch.zeros((h, w), device=tensor_batch.device, dtype=torch.float32)
            self._apply_spatial_mask(xb, tensor_batch, bi, mask_bw, x_blur)
        return xb

    def _apply_spatial_mask(self, output_batch, source_batch, batch_index, mask_bw, blurred_batch):
        mask_c = mask_bw.unsqueeze(0)
        if self.mask_type == "binary":
            output_batch[batch_index] = source_batch[batch_index] * (1.0 - mask_c)
            return
        if self.mask_type == "mean":
            mean_val = source_batch[batch_index].mean(dim=(1, 2), keepdim=True)
            output_batch[batch_index] = source_batch[batch_index] * (1.0 - mask_c) + mean_val * mask_c
            return
        if self.mask_type == "gaussian" and blurred_batch is not None:
            output_batch[batch_index] = source_batch[batch_index] * (1.0 - mask_c) + blurred_batch[batch_index] * mask_c
