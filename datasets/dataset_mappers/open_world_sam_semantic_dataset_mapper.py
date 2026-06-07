# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import numpy as np
import torch
from PIL import Image
import scipy.io
import logging
from torch.nn import functional as F

from typing import Optional, Union

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Boxes, Instances
from torchvision import transforms
from detectron2.utils.file_io import PathManager

# tokenizing the prompts
from transformers import AutoTokenizer

__all__ = ["OpenWorldSAM2SemanticDatasetMapper"]

def filter_empty_instances_by_box(
        instances, by_box=True, by_mask=False, box_threshold=1e-5, return_mask=False
):
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    if return_mask:
        return instances[m], m
    return instances[m]


def sam_preprocess(x: np.ndarray) -> torch.Tensor:
    """
    Preprocess for the Segment Anything Model (SAM), including scaling, normalization, and padding.
    """
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024

    x = torch.as_tensor(np.ascontiguousarray(x.transpose(2, 0, 1)))
    x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
    x = (x - pixel_mean) / pixel_std

    return x

def beit3_preprocess(x: np.ndarray) -> torch.Tensor:
    """
    Preprocess for BEIT-3 model.
    """
    img_size = 224
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=3, antialias=None),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(np.array(x))

def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    augmentation = []
    return augmentation

def get_class_name_from_id_hack(metadata, class_id):
    """Get class name from class_id using metadata"""
    for k, v in metadata.stuff_dataset_id_to_contiguous_id.items():
        if v == class_id:
            return metadata.stuff_classes[v]
        

def load_image_into_numpy_array(
    filename: str,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    with PathManager.open(filename, "rb") as f:
        array = np.asarray(Image.open(f), dtype=dtype)
    return array


def decode_rgb_semantic_mask(semseg: np.ndarray, colors, ignore_label: int) -> np.ndarray:
    if semseg.ndim == 2:
        return semseg

    decoded = np.full(semseg.shape[:2], ignore_label, dtype=np.int32)
    for class_id, color in enumerate(colors):
        color_arr = np.asarray(color, dtype=semseg.dtype)
        decoded[np.all(semseg == color_arr, axis=-1)] = class_id
    return decoded


def decode_rgb_semantic_mask_with_mapping(semseg: np.ndarray, color_to_class_id, ignore_label: int) -> np.ndarray:
    if semseg.ndim == 2:
        return semseg

    decoded = np.full(semseg.shape[:2], ignore_label, dtype=np.int32)
    for color, class_id in color_to_class_id.items():
        color_arr = np.asarray(color, dtype=semseg.dtype)
        decoded[np.all(semseg == color_arr, axis=-1)] = class_id
    return decoded


def normalize_semantic_mask(semseg: np.ndarray, metadata, ignore_label: int) -> np.ndarray:
    if getattr(metadata, "suim_rgb_mask", False):
        color_to_class_id = getattr(metadata, "suim_color_to_class_id", None)
        if color_to_class_id is not None:
            semseg = decode_rgb_semantic_mask_with_mapping(semseg, color_to_class_id, ignore_label)
        else:
            semseg = decode_rgb_semantic_mask(semseg, metadata.stuff_colors, ignore_label)
    if getattr(metadata, "dutuseg_rgb_mask", False):
        semseg = decode_rgb_semantic_mask(semseg, metadata.stuff_colors, ignore_label)
    if getattr(metadata, "coralscapes_label_shift", False):
        semseg = semseg.copy().astype(np.int32)
        semseg[semseg == 0] = ignore_label
        valid_mask = semseg != ignore_label
        semseg[valid_mask] = semseg[valid_mask] - 1
    if getattr(metadata, "ciona17_jpeg_void_remap", False):
        semseg = semseg.copy()
        semseg[semseg == 3] = 0
        semseg[semseg > 3] = ignore_label
    return semseg


class OpenWorldSAM2SemanticDatasetMapper:
    """
    A callable which takes a BDD dataset dict in Detectron2 Dataset format,
    and maps it for use with both SAM2 and BeiT3 models.
    
    The callable performs the following:
    1. Read the image from "file_name"
    2. Read the semantic segmentation annotation from "sem_seg_file_name"
    3. Prepare SAM-specific image preprocessing
    4. Prepare BeiT-specific image preprocessing
    5. Extract unique labels and generate text prompts
    6. Return a dataset dictionary with all required inputs
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        stuff_ids,
        metadata,
        tokenizer,
        dataset_name
    ):
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.stuff_ids = stuff_ids
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        augs = build_transform_gen(cfg, is_train)

        dataset_name = cfg.DATASETS.TRAIN[0] if is_train else cfg.DATASETS.TEST[0]
        meta = MetadataCatalog.get(dataset_name)
        
        ignore_label = meta.ignore_label if hasattr(meta, "ignore_label") else 255
        stuff_ids = list(meta.stuff_dataset_id_to_contiguous_id.values())
        
        tokenizer_config = cfg.MODEL.OpenWorldSAM2.TOKENIZER_CONFIG
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config, padding_side="right", use_fast=False)

        return {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY if hasattr(cfg.INPUT, "SIZE_DIVISIBILITY") else 0,
            "stuff_ids": stuff_ids,
            "metadata": meta,
            "tokenizer": tokenizer,
            "dataset_name": dataset_name
        }

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that OpenWorldSAM2 can consume
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # Get padding mask for transformer's attention
        padding_mask = np.ones(image.shape[:2])
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~padding_mask.astype(bool)

        # Apply SAM preprocessing
        dataset_dict["image"] = sam_preprocess(image)
        dataset_dict["evf_image"] = beit3_preprocess(image)
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        dataset_dict["height"] = 1024
        dataset_dict["width"] = 1024

        # read sem seg file
        gt_filename = dataset_dict["sem_seg_file_name"]
        semseg = load_image_into_numpy_array(gt_filename, dtype=int)
        semseg = normalize_semantic_mask(semseg, self.metadata, self.ignore_label)
        semseg = F.interpolate(
            torch.from_numpy(semseg.astype(np.float32)).unsqueeze(0).unsqueeze(0),
            size=(1024, 1024),
            mode="nearest",
        ).squeeze(0).squeeze(0).to(torch.int32).numpy()
        dataset_dict['semseg'] = torch.from_numpy(semseg.astype(np.int32))

        # get unique ids
        unique_categories = list(np.unique(semseg))
        # for ignore_id in self.ignore_label:
        #     if ignore_id in unique_categories:
        #         unique_categories.remove(ignore_id)
        if self.ignore_label in unique_categories:
            unique_categories.remove(self.ignore_label)
        dataset_dict["unique_categories"] = unique_categories
        # print(f"Unique categories: {unique_categories}")
        class_names = self.metadata.stuff_classes
        dataset_dict["prompt"] = [class_names[id] for id in unique_categories]

        if len(unique_categories) == 0:
            dataset_dict["unique_categories"] = [0]
            dataset_dict["prompt"] = ["object"]

        # Create pseudo-instance annotations per semantic class so that the
        # OpenWorldSAM2 training code, which expects an "instances" field,
        # can compute losses without requiring changes to the meta-arch.
        height, width = semseg.shape[:2]
        instances_list = []
        for class_id in dataset_dict["unique_categories"]:
            mask = semseg == class_id
            if not np.any(mask):
                continue
            inst = Instances((height, width))
            inst.gt_classes = torch.as_tensor([class_id], dtype=torch.int64)
            inst.gt_masks = BitMasks(torch.from_numpy(mask[None, ...].astype("bool")))
            inst.gt_boxes = inst.gt_masks.get_bounding_boxes()
            instances_list.append(inst)

        # Even if no valid masks are found (rare), provide an empty list so
        # that downstream code finds an "instances" key instead of failing
        # with a KeyError.
        dataset_dict["instances"] = instances_list

        return dataset_dict

    