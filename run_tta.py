import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

from model import add_open_world_sam2_config
from tta import add_tta_config, run_tta


def setup(args):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_open_world_sam2_config(cfg)
    add_tta_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TTA.ENABLED = True
    if args.tta_method:
        cfg.TTA.METHOD = args.tta_method.lower()
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f"run_{args.run_idx}")
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="open-world-sam2-tta")
    return cfg


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    set_seed()
    cfg = setup(args)
    model = build_model(cfg)
    return run_tta(cfg, model)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--run_idx", default=0, type=int, metavar="N")
    parser.add_argument("--tta_method", default="source", type=str)
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
