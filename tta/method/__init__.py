from .source import Source
from .cotta import CoTTA
from .deyo import DeYO
from .eata import EATA
from .gtta import GTTA
from .m2a import M2A
from .rdumb import RDumb
from .roid import ROID
from .sar import SAR
from .tent import Tent


_METHODS = {
    "cotta": CoTTA,
    "deyo": DeYO,
    "eata": EATA,
    "gtta": GTTA,
    "m2a": M2A,
    "rdumb": RDumb,
    "roid": ROID,
    "sar": SAR,
    "source": Source,
    "tent": Tent,
}


def build_tta_method(cfg, model):
    method_name = str(cfg.TTA.METHOD).lower()
    if method_name not in _METHODS:
        raise ValueError(f"Unsupported TTA method: {cfg.TTA.METHOD}")
    return _METHODS[method_name](cfg, model)
