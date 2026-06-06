import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager


_SUIM_STUFF_CLASSES = [
    "background",
    "human_divers",
    "wrecks_and_ruins",
    "robots_and_instruments",
    "reefs_and_invertebrates",
    "fish_and_vertebrates",
]

_SUIM_STUFF_COLORS = [
    (0, 0, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 0),
    (255, 0, 255),
    (255, 255, 0),
]

_SUIM_RAW_COLOR_TO_CLASS_ID = {
    (0, 0, 0): 0,
    (0, 0, 255): 1,
    (0, 255, 0): 0,
    (0, 255, 255): 2,
    (255, 0, 0): 3,
    (255, 0, 255): 4,
    (255, 255, 0): 5,
    (255, 255, 255): 0,
}

_SUIM_SPLITS = {
    "suim_sem_seg_train": "train",
    "suim_sem_seg_val": "test",
}


def _get_suim_sem_seg_meta():
    return {
        "stuff_classes": _SUIM_STUFF_CLASSES,
        "stuff_dataset_id_to_contiguous_id": {idx: idx for idx in range(len(_SUIM_STUFF_CLASSES))},
        "stuff_colors": _SUIM_STUFF_COLORS,
        "suim_color_to_class_id": _SUIM_RAW_COLOR_TO_CLASS_ID,
    }


def load_suim_sem_seg(root, dataset_name, split_name):
    image_root = os.path.join(root, split_name, "images")
    mask_root = os.path.join(root, split_name, "masks")

    dataset_dicts = []
    missing_files = []

    if not os.path.isdir(image_root) or not os.path.isdir(mask_root):
        raise ValueError(f"Missing SUIM split directories for dataset {dataset_name}: {image_root} {mask_root}")

    image_filenames = sorted(
        f for f in os.listdir(image_root) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    for filename in image_filenames:
        image_path = os.path.join(image_root, filename)
        stem, _ = os.path.splitext(filename)
        mask_path = os.path.join(mask_root, f"{stem}.bmp")

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
        raise FileNotFoundError(f"Missing SUIM semantic files, e.g. {preview}")

    if not dataset_dicts:
        raise ValueError(f"No SUIM semantic segmentation images found for dataset {dataset_name}")

    return dataset_dicts


def register_suim_sem_seg(name, metadata, root, split_name):
    DatasetCatalog.register(
        name,
        lambda root=root, dataset_name=name, split_name=split_name: load_suim_sem_seg(
            root,
            dataset_name,
            split_name,
        ),
    )
    MetadataCatalog.get(name).set(
        image_root=os.path.join(root, split_name, "images"),
        sem_seg_root=os.path.join(root, split_name, "masks"),
        evaluator_type="sem_seg",
        ignore_label=255,
        suim_rgb_mask=True,
        **metadata,
    )


def register_all_suim_sem_seg(root):
    suim_root = os.getenv("SUIM_DATASET_ROOT", os.path.join(root, "suim", "SUIM"))
    if not os.path.isdir(suim_root):
        return

    metadata = _get_suim_sem_seg_meta()
    for dataset_name, split_name in _SUIM_SPLITS.items():
        register_suim_sem_seg(dataset_name, metadata, suim_root, split_name)


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_suim_sem_seg(_root)
