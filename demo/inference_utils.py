import logging
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.data import MetadataCatalog, detection_utils as utils
from detectron2.modeling import build_model

import datasets  # noqa: F401 - ensures dataset registration
from model import add_open_world_sam2_config


_SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
_SAM_PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
_SAM_IMAGE_SIZE = 1024
_BEIT_RESIZE = 224
_UNWANTED_SUFFIXES = ("-other", "-merged", "-stuff")

logger = logging.getLogger(__name__)


def setup_cfg(
    config_file: str,
    weights: Optional[str] = None,
    device: Optional[str] = None,
    opts: Optional[Sequence[str]] = None,
) -> CfgNode:
    """
    Create a Detectron2 config for inference.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)  # Add this line before merging the file
    add_open_world_sam2_config(cfg)
    cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(list(opts))
    if weights:
        cfg.MODEL.WEIGHTS = weights
    if device:
        cfg.MODEL.DEVICE = device
    cfg.INPUT.FORMAT = cfg.INPUT.FORMAT or "BGR"
    return cfg


def load_model(cfg: CfgNode):
    """
    Build and load the OpenWorldSAM model.
    """
    model = build_model(cfg)
    model.eval()
    if cfg.MODEL.WEIGHTS:
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return model


def sam_preprocess(image: np.ndarray) -> torch.Tensor:
    """
    Normalize and resize input image for the SAM image encoder.
    """
    tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
    tensor = F.interpolate(
        tensor.unsqueeze(0), (_SAM_IMAGE_SIZE, _SAM_IMAGE_SIZE), mode="bilinear", align_corners=False
    ).squeeze(0)
    tensor = (tensor - _SAM_PIXEL_MEAN) / _SAM_PIXEL_STD
    return tensor


_beit_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((_BEIT_RESIZE, _BEIT_RESIZE), interpolation=3, antialias=None),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


def beit3_preprocess(image: np.ndarray) -> torch.Tensor:
    """
    Resize and normalize the input image for the BEiT-3 encoder.
    """
    return _beit_transform(image)


def prepare_image_inputs(image_path: str, image_format: str = "BGR"):
    """
    Load an image from disk and produce tensors for SAM and BEiT-3.
    """
    original_image = utils.read_image(image_path, format=image_format)
    sam_tensor = sam_preprocess(original_image)
    beit_tensor = beit3_preprocess(original_image)
    height, width = original_image.shape[:2]
    return original_image, sam_tensor, beit_tensor, height, width


def _normalize_text(value: str) -> str:
    text = value.lower().strip()
    for suffix in _UNWANTED_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    replacements = {
        "-": " ",
        "_": " ",
        "/": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = " ".join(text.split())
    return text


def _build_class_lookup(metadata) -> Dict[str, int]:
    """
    Build a mapping from normalized class names to contiguous IDs using metadata.
    """
    lookup: Dict[str, int] = {}

    def _add_entry(name: str, class_id: int):
        if name is None:
            return
        for variant in {_normalize_text(name), _normalize_text(name.replace(" ", "")),
                        _normalize_text(name.replace('-other','').replace('-merged','').replace('-stuff',''))}:
            if not variant:
                continue
            lookup.setdefault(variant, class_id)

    if hasattr(metadata, "thing_dataset_id_to_contiguous_id") and hasattr(metadata, "thing_classes"):
        for dataset_id, contiguous_id in metadata.thing_dataset_id_to_contiguous_id.items():
            if contiguous_id < len(metadata.thing_classes):
                _add_entry(metadata.thing_classes[contiguous_id], contiguous_id)

    if hasattr(metadata, "stuff_dataset_id_to_contiguous_id") and hasattr(metadata, "stuff_classes"):
        for dataset_id, contiguous_id in metadata.stuff_dataset_id_to_contiguous_id.items():
            if contiguous_id < len(metadata.stuff_classes):
                _add_entry(metadata.stuff_classes[contiguous_id], contiguous_id)
    elif hasattr(metadata, "stuff_classes"):
        for contiguous_id, name in enumerate(metadata.stuff_classes):
            _add_entry(name, contiguous_id)

    return lookup


def resolve_category_ids(prompts: Iterable[str], metadata) -> List[int]:
    """
    Map user-provided prompts to contiguous category IDs using dataset metadata.
    """
    lookup = _build_class_lookup(metadata)
    resolved_ids: List[int] = []
    missing: List[str] = []

    for prompt in prompts:
        normalized = _normalize_text(prompt)
        if normalized in lookup:
            resolved_ids.append(lookup[normalized])
        else:
            missing.append(prompt)

    if missing:
        raise ValueError(
            f"Could not match the following prompts to dataset classes: {missing}. "
            "Please ensure the prompts correspond to the dataset taxonomy."
        )
    return resolved_ids


def build_inference_inputs(
    sam_tensor: torch.Tensor,
    beit_tensor: torch.Tensor,
    height: int,
    width: int,
    prompts: Sequence[str],
    unique_categories: Sequence[int],
):
    """
    Package tensors and metadata into the structure expected by the model.
    """
    return [
        {
            "image": sam_tensor,
            "evf_image": beit_tensor,
            "height": height,
            "width": width,
            "prompt": list(prompts),
            "unique_categories": list(unique_categories),
        }
    ]


def get_metadata(cfg: CfgNode):
    """
    Fetch Detectron2 metadata for the configured evaluation dataset.
    """
    dataset_name = cfg.DATASETS.TEST[0]
    return MetadataCatalog.get(dataset_name)


def ensure_dir(path: str):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
