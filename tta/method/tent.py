import torch
import torch.nn as nn

from .base import SegTTAMethod
from model.segment_anything_2.sam2.modeling.sam2_utils import LayerNorm2d


def pixelwise_softmax_entropy(score_map: torch.Tensor) -> torch.Tensor:
    if score_map.dim() == 3:
        score_map = score_map.unsqueeze(0)
    return -(score_map.softmax(dim=1) * score_map.log_softmax(dim=1)).sum(dim=1)


class Tent(SegTTAMethod):
    def _is_adaptable_norm_module(self, module):
        return isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, LayerNorm2d))

    def _use_full_trainable_except_excluded(self):
        return bool(getattr(self.cfg.TTA.ADAPT, "FULL_TRAINABLE_EXCEPT_EXCLUDED", False))

    def _frozen_parameter_prefixes(self):
        return (
            "mm_extractor.",
            "visual_model.image_encoder.",
            "visual_model.sam_prompt_encoder.",
            "visual_model.sam_mask_decoder.",
        )

    def _is_excluded_parameter(self, name):
        return name in {"mm_extractor", "visual_model.image_encoder", "visual_model.sam_prompt_encoder", "visual_model.sam_mask_decoder"} or name.startswith(self._frozen_parameter_prefixes())

    def _enable_full_trainable_except_excluded(self):
        for name, param in self.model.named_parameters():
            if self._is_excluded_parameter(name):
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

        for module_name, module in self.model.named_modules():
            if self._is_excluded_parameter(module_name):
                continue
            if self._is_adaptable_norm_module(module):
                self._configure_bn_for_adaptation(module)

    def _collect_all_trainable_params(self):
        params = []
        names = []
        seen = set()
        for name, param in self.model.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            params.append(param)
            names.append(name)
            seen.add(id(param))
        return params, names

    def _use_selected_module_adaptation(self):
        return any(
            bool(getattr(self.cfg.TTA.ADAPT, name))
            for name in (
                "SAM_VISUAL_ENCODER",
                "SAM_MASK_DECODER",
                "SAM_PROMPT_ENCODER",
                "VLM_ENCODER",
                "SOFT_PROMPTING_TRANSFORMER",
            )
        )

    def _selected_modules_and_params(self):
        modules = []
        adapt_cfg = self.cfg.TTA.ADAPT

        if bool(adapt_cfg.SAM_VISUAL_ENCODER) and hasattr(self.model.visual_model, "image_encoder"):
            modules.append(("visual_model.image_encoder", self.model.visual_model.image_encoder))

        if bool(adapt_cfg.SAM_MASK_DECODER) and hasattr(self.model.visual_model, "sam_mask_decoder"):
            modules.append(("visual_model.sam_mask_decoder", self.model.visual_model.sam_mask_decoder))

        if bool(adapt_cfg.SAM_PROMPT_ENCODER) and hasattr(self.model.visual_model, "sam_prompt_encoder"):
            modules.append(("visual_model.sam_prompt_encoder", self.model.visual_model.sam_prompt_encoder))

        if bool(adapt_cfg.VLM_ENCODER) and hasattr(self.model, "mm_extractor"):
            modules.append(("mm_extractor", self.model.mm_extractor))

        if bool(adapt_cfg.SOFT_PROMPTING_TRANSFORMER):
            if hasattr(self.model, "text_hidden_fcs"):
                modules.append(("text_hidden_fcs", self.model.text_hidden_fcs))
            if hasattr(self.model, "cross_attention_transformer"):
                modules.append(("cross_attention_transformer", self.model.cross_attention_transformer))

        return modules

    def _enable_norm_adaptation(self, module):
        for submodule in module.modules():
            if not self._is_adaptable_norm_module(submodule):
                continue
            for param_name, param in submodule.named_parameters(recurse=False):
                if param_name in {"weight", "bias"}:
                    param.requires_grad_(True)
            self._configure_bn_for_adaptation(submodule)

    def _collect_norm_params_from_module(self, module_name, module, seen):
        params = []
        names = []
        for submodule_name, submodule in module.named_modules():
            if not self._is_adaptable_norm_module(submodule):
                continue
            full_module_name = f"{module_name}.{submodule_name}" if submodule_name else module_name
            for param_name, param in submodule.named_parameters(recurse=False):
                if param_name not in {"weight", "bias"} or not param.requires_grad or id(param) in seen:
                    continue
                params.append(param)
                names.append(f"{full_module_name}.{param_name}" if full_module_name else param_name)
                seen.add(id(param))
        return params, names

    def _configure_bn_for_adaptation(self, module):
        for submodule in module.modules():
            if isinstance(submodule, nn.BatchNorm2d):
                submodule.train()
                submodule.track_running_stats = False
                submodule.running_mean = None
                submodule.running_var = None
            elif isinstance(submodule, nn.BatchNorm1d):
                submodule.train()

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

        if self._use_full_trainable_except_excluded():
            self._enable_full_trainable_except_excluded()
            return

        if self._use_selected_module_adaptation():
            modules = self._selected_modules_and_params()
            for _, module in modules:
                self._enable_norm_adaptation(module)
            return

        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()
                module.requires_grad_(True)
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
            elif isinstance(module, nn.BatchNorm1d):
                module.train()
                module.requires_grad_(True)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, LayerNorm2d)):
                module.requires_grad_(True)

    def collect_params(self):
        if self._use_full_trainable_except_excluded():
            return self._collect_all_trainable_params()

        if self._use_selected_module_adaptation():
            params = []
            names = []
            seen = set()
            modules = self._selected_modules_and_params()

            for module_name, module in modules:
                module_params, module_names = self._collect_norm_params_from_module(module_name, module, seen)
                params.extend(module_params)
                names.extend(module_names)

            return params, names

        params = []
        names = []
        for module_name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, LayerNorm2d)):
                for param_name, param in module.named_parameters(recurse=False):
                    if param_name not in {"weight", "bias"}:
                        continue
                    params.append(param)
                    names.append(f"{module_name}.{param_name}" if module_name else param_name)
        return params, names

    def forward_and_adapt(self, batched_inputs):
        if self.optimizer is None:
            with torch.no_grad():
                return self.model(batched_inputs)

        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(batched_inputs)
        loss = self._entropy_loss(outputs)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return outputs

    def _entropy_loss(self, outputs):
        losses = []
        for output in outputs:
            sem_seg = output["sem_seg"].float()
            entropy = pixelwise_softmax_entropy(sem_seg)
            losses.append(entropy.mean())
        return torch.stack(losses).mean()
