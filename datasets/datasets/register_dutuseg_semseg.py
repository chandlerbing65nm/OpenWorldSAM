import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager


_DUTUSEG_STUFF_CLASSES = [
    "background",
    "sea_cucumber",
    "sea_urchin",
    "scallop",
    "starfish",
]

_DUTUSEG_STUFF_COLORS = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
]

_DUTUSEG_SPLITS = {
    "dutuseg_sem_seg_train": "labeled_train.txt",
    "dutuseg_sem_seg_val": "val.txt",
}


def _get_dutuseg_sem_seg_meta():
    return {
        "stuff_classes": _DUTUSEG_STUFF_CLASSES,
        "stuff_dataset_id_to_contiguous_id": {idx: idx for idx in range(len(_DUTUSEG_STUFF_CLASSES))},
        "stuff_colors": _DUTUSEG_STUFF_COLORS,
    }


def load_dutuseg_sem_seg(root, dataset_name, split_file):
    image_root = os.path.join(root, "JPEGImages")
    mask_root = os.getenv("DUTUSEG_MASKS_ROOT", os.path.join(root, "SegmentationClassVisual"))
    split_path = os.path.join(root, "ImageSets", split_file)

    if not os.path.isdir(image_root) or not os.path.isdir(mask_root):
        raise ValueError(f"Missing DUT-USEG image or mask directories for dataset {dataset_name}: {image_root} {mask_root}")
    if not PathManager.isfile(split_path):
        raise ValueError(f"Missing DUT-USEG split file for dataset {dataset_name}: {split_path}")

    dataset_dicts = []
    missing_files = []
    with PathManager.open(split_path, "r") as f:
        image_ids = [line.strip() for line in f if line.strip()]

    for image_id in image_ids:
        image_path = os.path.join(image_root, f"{image_id}.jpg")
        mask_path = os.path.join(mask_root, f"{image_id}.png")

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
                "image_id": image_id,
            }
        )

    if missing_files:
        preview = ", ".join(missing_files[:3])
        raise FileNotFoundError(f"Missing DUT-USEG semantic files, e.g. {preview}")

    if not dataset_dicts:
        raise ValueError(f"No DUT-USEG semantic segmentation images found for dataset {dataset_name}")

    return dataset_dicts


def register_dutuseg_sem_seg(name, metadata, root, split_file):
    mask_root = os.getenv("DUTUSEG_MASKS_ROOT", os.path.join(root, "SegmentationClassVisual"))
    DatasetCatalog.register(
        name,
        lambda root=root, dataset_name=name, split_file=split_file: load_dutuseg_sem_seg(
            root,
            dataset_name,
            split_file,
        ),
    )
    MetadataCatalog.get(name).set(
        image_root=os.path.join(root, "JPEGImages"),
        sem_seg_root=mask_root,
        evaluator_type="sem_seg",
        ignore_label=255,
        dutuseg_rgb_mask=True,
        **metadata,
    )


def register_all_dutuseg_sem_seg(root):
    dutuseg_root = os.getenv("DUTUSEG_DATASET_ROOT", os.path.join(root, "dut-useg", "DUT-USEG"))
    if not os.path.isdir(dutuseg_root):
        return

    metadata = _get_dutuseg_sem_seg_meta()
    for dataset_name, split_file in _DUTUSEG_SPLITS.items():
        register_dutuseg_sem_seg(dataset_name, metadata, dutuseg_root, split_file)


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_dutuseg_sem_seg(_root)
