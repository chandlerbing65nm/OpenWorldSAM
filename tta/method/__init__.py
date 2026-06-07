from .source import Source
from .cotta import CoTTA
from .tent import Tent


_METHODS = {
    "cotta": CoTTA,
    "source": Source,
    "tent": Tent,
}


def build_tta_method(cfg, model):
    method_name = str(cfg.TTA.METHOD).lower()
    if method_name not in _METHODS:
        raise ValueError(f"Unsupported TTA method: {cfg.TTA.METHOD}")
    return _METHODS[method_name](cfg, model)
