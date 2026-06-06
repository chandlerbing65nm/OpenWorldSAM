import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.utils.file_io import PathManager


_FOD_CATEGORY_NAMES = ["part", "whole", "fragment"]
_FOD_CATEGORY_MODES = {
    "part": ["part"],
    "whole": ["whole"],
    "fragment": ["fragment"],
    "all": _FOD_CATEGORY_NAMES,
}


def _get_category_name_to_dataset_id(annotation_json):
    with PathManager.open(annotation_json) as f:
        json_info = json.load(f)

    category_name_to_dataset_id = {}
    for category in sorted(json_info["categories"], key=lambda x: x["id"]):
        category_name = category["name"]
        if category_name in _FOD_CATEGORY_NAMES:
            category_name_to_dataset_id[category_name] = category["id"]

    missing_category_names = [name for name in _FOD_CATEGORY_NAMES if name not in category_name_to_dataset_id]
    if missing_category_names:
        raise ValueError(f"Missing FOD categories {missing_category_names} in {annotation_json}")

    return category_name_to_dataset_id


def _get_category_names_for_mode(category_mode):
    if category_mode not in _FOD_CATEGORY_MODES:
        raise ValueError(
            f"Unsupported FOD_CATEGORY_MODE '{category_mode}'. Expected one of {sorted(_FOD_CATEGORY_MODES)}"
        )

    return list(_FOD_CATEGORY_MODES[category_mode])


def _get_fod_instances_meta(annotation_json, category_mode):
    category_name_to_dataset_id = _get_category_name_to_dataset_id(annotation_json)
    selected_category_names = _get_category_names_for_mode(category_mode)
    thing_dataset_id_to_contiguous_id = {
        category_name_to_dataset_id[category_name]: category_idx
        for category_idx, category_name in enumerate(selected_category_names)
    }

    return {
        "thing_classes": selected_category_names,
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "fod_category_mode": category_mode,
    }


def load_fod_json(image_root, annotation_json, dataset_name, category_mode="fragment"):
    category_name_to_dataset_id = _get_category_name_to_dataset_id(annotation_json)
    selected_category_names = _get_category_names_for_mode(category_mode)
    selected_category_id_to_contiguous_id = {
        category_name_to_dataset_id[category_name]: category_idx
        for category_idx, category_name in enumerate(selected_category_names)
    }
    dataset_dicts = load_coco_json(annotation_json, image_root, None)
    filtered_dicts = []
    missing_files = []

    for dataset_dict in dataset_dicts:
        if not PathManager.isfile(dataset_dict["file_name"]):
            missing_files.append(dataset_dict["file_name"])
            continue

        filtered_annotations = []
        for annotation in dataset_dict.get("annotations", []):
            category_id = annotation.get("category_id")
            if category_id in selected_category_id_to_contiguous_id:
                annotation = annotation.copy()
                annotation["category_id"] = selected_category_id_to_contiguous_id[category_id]
                filtered_annotations.append(annotation)

        dataset_dict = dataset_dict.copy()
        dataset_dict["annotations"] = filtered_annotations
        filtered_dicts.append(dataset_dict)

    if missing_files:
        preview = ", ".join(missing_files[:3])
        raise FileNotFoundError(f"Missing FOD images, e.g. {preview}")

    if not filtered_dicts:
        raise ValueError(f"No FOD images found for dataset {dataset_name}")

    return filtered_dicts


def register_fod_instance(name, metadata, image_root, annotation_json, category_mode):
    DatasetCatalog.register(
        name,
        lambda image_root=image_root, annotation_json=annotation_json, dataset_name=name, category_mode=category_mode: load_fod_json(
            image_root,
            annotation_json,
            dataset_name,
            category_mode,
        ),
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        evaluator_type="coco",
        **metadata,
    )


def register_all_fod_instance(root):
    fod_root = os.getenv("FOD_DATASET_ROOT", os.path.join(root, "fod", "data"))
    train_image_root = os.getenv("FOD_TRAIN_IMAGE_ROOT", os.path.join(fod_root, "images", "train"))
    val_image_root = os.getenv("FOD_VAL_IMAGE_ROOT", os.path.join(fod_root, "images", "test"))
    train_annotation_json = os.getenv(
        "FOD_TRAIN_ANNOTATIONS_JSON", os.path.join(fod_root, "annotations", "instances_train.json")
    )
    val_annotation_json = os.getenv(
        "FOD_VAL_ANNOTATIONS_JSON", os.path.join(fod_root, "annotations", "instances_test.json")
    )
    category_mode = os.getenv("FOD_CATEGORY_MODE", "fragment")

    required_paths = [
        train_image_root,
        val_image_root,
        train_annotation_json,
        val_annotation_json,
    ]
    if not all(os.path.isdir(path) or os.path.isfile(path) for path in required_paths):
        return

    train_metadata = _get_fod_instances_meta(train_annotation_json, category_mode)
    val_metadata = _get_fod_instances_meta(val_annotation_json, category_mode)

    register_fod_instance("fod_train", train_metadata, train_image_root, train_annotation_json, category_mode)
    register_fod_instance("fod_val", val_metadata, val_image_root, val_annotation_json, category_mode)


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_fod_instance(_root)
