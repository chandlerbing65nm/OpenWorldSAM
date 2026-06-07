import os

from detectron2.config import CfgNode as CN


DEFAULT_SUIM_C_ROOT = "/scratch/project_465002853/datasets/suim/SUIM-C"
DEFAULT_SUIM_C_WEIGHTS = "/flash/project_465002853/projects/openworldsam/checkpoints/suim_model_final.pth"
DEFAULT_SUIM_C_CORRUPTIONS = [
    "brightness",
    "contrast",
    "defocus_blur",
    "fog",
    "gaussian_noise",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "shot_noise",
]
DEFAULT_SUIM_C_SEVERITIES = [1, 3, 5]


def add_tta_config(cfg):
    cfg.TTA = CN()
    cfg.TTA.ENABLED = False
    cfg.TTA.METHOD = "source"
    cfg.TTA.DATASET = "suim_c_sem_seg"
    cfg.TTA.DATA_ROOT = DEFAULT_SUIM_C_ROOT
    cfg.TTA.DOMAIN_ROOT = os.path.join(DEFAULT_SUIM_C_ROOT, "domains")
    cfg.TTA.MASK_ROOT = os.path.join(DEFAULT_SUIM_C_ROOT, "masks")
    cfg.TTA.CORRUPTIONS = list(DEFAULT_SUIM_C_CORRUPTIONS)
    cfg.TTA.SEVERITIES = list(DEFAULT_SUIM_C_SEVERITIES)
    cfg.TTA.NUM_WORKERS = 4
    cfg.TTA.BATCH_SIZE = 1
    cfg.TTA.TTA_MODE = "normal_tta"
    cfg.TTA.EPISODIC = False
    cfg.TTA.INIT_WEIGHTS = DEFAULT_SUIM_C_WEIGHTS

    cfg.TTA.ADAPT = CN()
    cfg.TTA.ADAPT.SAM_VISUAL_ENCODER = False
    cfg.TTA.ADAPT.SAM_MASK_DECODER = False
    cfg.TTA.ADAPT.SAM_PROMPT_ENCODER = False
    cfg.TTA.ADAPT.VLM_ENCODER = False
    cfg.TTA.ADAPT.SOFT_PROMPTING_TRANSFORMER = False

    cfg.TTA.COTTA = CN()
    cfg.TTA.COTTA.N_AUGMENTATIONS = 6
    cfg.TTA.COTTA.MT_ALPHA = 0.999
    cfg.TTA.COTTA.RST = 0.01
    cfg.TTA.COTTA.AP = 0.9
    cfg.TTA.COTTA.STUDENT_CROPS = 2
    cfg.TTA.COTTA.STUDENT_CROP_SCALE = 0.5

    cfg.TTA.OPTIM = CN()
    cfg.TTA.OPTIM.STEPS = 1
    cfg.TTA.OPTIM.LR = 1e-5
    cfg.TTA.OPTIM.METHOD = "SGD"
    cfg.TTA.OPTIM.BETA = 0.9
    cfg.TTA.OPTIM.MOMENTUM = 0.9
    cfg.TTA.OPTIM.DAMPENING = 0.0
    cfg.TTA.OPTIM.NESTEROV = False
    cfg.TTA.OPTIM.WD = 0.0


def get_tta_init_weights(cfg):
    if getattr(cfg.TTA, "INIT_WEIGHTS", ""):
        return cfg.TTA.INIT_WEIGHTS
    return cfg.MODEL.WEIGHTS
