import os
import json

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager


_CORALSCAPES_SPLITS = {
    "coralscapes_sem_seg_train": "train",
    "coralscapes_sem_seg_val": "test",
}


def _load_coralscapes_metadata(root):
    classes_path = os.path.join(root, "classes.json")
    colors_path = os.path.join(root, "colors.json")

    with PathManager.open(classes_path, "r") as f:
        classes_data = json.load(f)
    with PathManager.open(colors_path, "r") as f:
        colors_data = json.load(f)

    sorted_items = sorted(classes_data.items(), key=lambda item: item[1])
    stuff_classes = [name for name, _ in sorted_items]
    stuff_colors = [tuple(colors_data[name]) for name, _ in sorted_items]

    return {
        "stuff_classes": stuff_classes,
        "stuff_dataset_id_to_contiguous_id": {raw_id: raw_id - 1 for _, raw_id in sorted_items},
        "stuff_colors": stuff_colors,
    }


def load_coralscapes_sem_seg(root, dataset_name, split_name):
    image_root = os.path.join(root, "leftImg8bit", split_name)
    mask_root = os.path.join(root, "gtFine", split_name)

    dataset_dicts = []
    missing_files = []

    if not os.path.isdir(image_root) or not os.path.isdir(mask_root):
        raise ValueError(f"Missing Coralscapes split directories for dataset {dataset_name}: {image_root} {mask_root}")

    for site_name in sorted(os.listdir(image_root)):
        image_site_dir = os.path.join(image_root, site_name)
        mask_site_dir = os.path.join(mask_root, site_name)
        if not os.path.isdir(image_site_dir) or not os.path.isdir(mask_site_dir):
            continue

        image_filenames = sorted(
            f for f in os.listdir(image_site_dir) if f.lower().endswith("_leftimg8bit.png")
        )

        for filename in image_filenames:
            image_path = os.path.join(image_site_dir, filename)
            mask_filename = filename.replace("_leftImg8bit.png", "_gtFine.png")
            mask_path = os.path.join(mask_site_dir, mask_filename)

            if not PathManager.isfile(image_path):
                missing_files.append(image_path)
                continue
            if not PathManager.isfile(mask_path):
                missing_files.append(mask_path)
                continue

            dataset_dicts.append(
                {
                    "file_name": image_path,
                    "sem_seg_file_name": mask_path,
                }
            )

    if missing_files:
        preview = ", ".join(missing_files[:3])
        raise FileNotFoundError(f"Missing Coralscapes semantic files, e.g. {preview}")

    if not dataset_dicts:
        raise ValueError(f"No Coralscapes semantic segmentation images found for dataset {dataset_name}")

    return dataset_dicts


def register_coralscapes_sem_seg(name, metadata, root, split_name):
    DatasetCatalog.register(
        name,
        lambda root=root, dataset_name=name, split_name=split_name: load_coralscapes_sem_seg(
            root,
            dataset_name,
            split_name,
        ),
    )
    MetadataCatalog.get(name).set(
        image_root=os.path.join(root, "leftImg8bit", split_name),
        sem_seg_root=os.path.join(root, "gtFine", split_name),
        evaluator_type="sem_seg",
        ignore_label=255,
        coralscapes_label_shift=True,
        **metadata,
    )


def register_all_coralscapes_sem_seg(root):
    coralscapes_root = os.getenv("CORALSCAPES_DATASET_ROOT", os.path.join(root, "coralscapes", "Coralscapes"))
    if not os.path.isdir(coralscapes_root):
        return

    metadata = _load_coralscapes_metadata(coralscapes_root)
    for dataset_name, split_name in _CORALSCAPES_SPLITS.items():
        register_coralscapes_sem_seg(dataset_name, metadata, coralscapes_root, split_name)


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coralscapes_sem_seg(_root)
