import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.utils.file_io import PathManager


_ALL_GROUPS = {f"group_{idx:02d}" for idx in range(1, 26)}
_VAL_GROUPS = {"group_10", "group_14", "group_20", "group_21", "group_22"}
_TRAIN_GROUPS = _ALL_GROUPS - _VAL_GROUPS
_AUTOFISH_THING_CLASSES = ["whiting", "cod", "haddock", "hake", "horse_mackerel", "other"]
_RAW_TO_AUTOFISH_CLASS = {
    "horse_mackerel": "horse_mackerel",
    "whiting": "whiting",
    "haddock": "haddock",
    "cod": "cod",
    "hake": "hake",
    "saithe": "other",
    "other": "other",
}


def _get_raw_category_id_to_contiguous_id(annotation_json):
    with PathManager.open(annotation_json) as f:
        json_info = json.load(f)

    contiguous_id_by_name = {class_name: idx for idx, class_name in enumerate(_AUTOFISH_THING_CLASSES)}
    raw_category_id_to_contiguous_id = {}
    for category in sorted(json_info["categories"], key=lambda x: x["id"]):
        raw_name = category["name"]
        if raw_name not in _RAW_TO_AUTOFISH_CLASS:
            raise ValueError(f"Unexpected AutoFish category '{raw_name}' in {annotation_json}")
        mapped_name = _RAW_TO_AUTOFISH_CLASS[raw_name]
        raw_category_id_to_contiguous_id[category["id"]] = contiguous_id_by_name[mapped_name]

    return raw_category_id_to_contiguous_id


def _get_autofish_instances_meta(annotation_json):
    _get_raw_category_id_to_contiguous_id(annotation_json)

    return {
        "thing_classes": _AUTOFISH_THING_CLASSES,
        "thing_dataset_id_to_contiguous_id": {idx: idx for idx in range(len(_AUTOFISH_THING_CLASSES))},
    }


def load_autofish_json(image_root, annotation_json, dataset_name, include_groups, exclude_groups=None):
    category_id_map = _get_raw_category_id_to_contiguous_id(annotation_json)
    dataset_dicts = load_coco_json(annotation_json, image_root, None)
    filtered_dicts = []
    missing_files = []
    unexpected_groups = set()

    for dataset_dict in dataset_dicts:
        relative_path = os.path.relpath(dataset_dict["file_name"], image_root)
        group_name = relative_path.split(os.sep, 1)[0]
        if group_name not in _ALL_GROUPS:
            unexpected_groups.add(group_name)
            continue
        if include_groups is not None and group_name not in include_groups:
            continue
        if exclude_groups is not None and group_name in exclude_groups:
            continue
        if not PathManager.isfile(dataset_dict["file_name"]):
            missing_files.append(dataset_dict["file_name"])
            continue
        remapped_annotations = []
        for annotation in dataset_dict.get("annotations", []):
            annotation = annotation.copy()
            category_id = annotation.get("category_id")
            if category_id not in category_id_map:
                raise ValueError(f"Unexpected AutoFish category id {category_id} in dataset {dataset_name}")
            annotation["category_id"] = category_id_map[category_id]
            remapped_annotations.append(annotation)
        dataset_dict = dataset_dict.copy()
        dataset_dict["annotations"] = remapped_annotations
        filtered_dicts.append(dataset_dict)

    if unexpected_groups:
        preview = ", ".join(sorted(unexpected_groups)[:3])
        raise ValueError(f"Unexpected AutoFish groups in annotations for dataset {dataset_name}: {preview}")

    if missing_files:
        preview = ", ".join(missing_files[:3])
        raise FileNotFoundError(f"Missing AutoFish images, e.g. {preview}")

    if not filtered_dicts:
        raise ValueError(f"No AutoFish images found for dataset {dataset_name}")

    return filtered_dicts


def register_autofish_instance(name, metadata, image_root, annotation_json, include_groups, exclude_groups=None):
    DatasetCatalog.register(
        name,
        lambda image_root=image_root, annotation_json=annotation_json, dataset_name=name, include_groups=include_groups, exclude_groups=exclude_groups: load_autofish_json(
            image_root,
            annotation_json,
            dataset_name,
            include_groups,
            exclude_groups,
        ),
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        evaluator_type="coco",
        **metadata,
    )


def register_all_autofish_instance(root):
    image_root = os.getenv("AUTOFISH_DATASET_ROOT", os.path.join(root, "autofish", "AutoFish"))
    annotation_json = os.getenv("AUTOFISH_ANNOTATIONS_JSON", os.path.join(image_root, "annotations.json"))
    if not os.path.isfile(annotation_json) or not os.path.isdir(image_root):
        return

    metadata = _get_autofish_instances_meta(annotation_json)
    register_autofish_instance(
        "autofish_train",
        metadata,
        image_root,
        annotation_json,
        include_groups=_TRAIN_GROUPS,
        exclude_groups=None,
    )
    register_autofish_instance(
        "autofish_val",
        metadata,
        image_root,
        annotation_json,
        include_groups=_VAL_GROUPS,
        exclude_groups=None,
    )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_autofish_instance(_root)
