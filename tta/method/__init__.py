from .source import Source
from .cotta import CoTTA
from .deyo import DeYO
from .eata import EATA
from .lame import LAME
from .m2a import M2A
from .memo import MEMO
from .rpl import RPL
from .rdumb import RDumb
from .roid import ROID
from .santa import SANTA
from .sar import SAR
from .smppm import SMPPM
from .tent import Tent


_METHODS = {
    "cotta": CoTTA,
    "deyo": DeYO,
    "eata": EATA,
    "lame": LAME,
    "m2a": M2A,
    "memo": MEMO,
    "rpl": RPL,
    "rdumb": RDumb,
    "roid": ROID,
    "santa": SANTA,
    "sar": SAR,
    "sm_ppm": SMPPM,
    "source": Source,
    "tent": Tent,
}


def build_tta_method(cfg, model):
    method_name = str(cfg.TTA.METHOD).lower()
    if method_name not in _METHODS:
        raise ValueError(f"Unsupported TTA method: {cfg.TTA.METHOD}")
    return _METHODS[method_name](cfg, model)
