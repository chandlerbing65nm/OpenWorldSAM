import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager


_CIONA17_STUFF_CLASSES = ["other", "mussel", "ciona"]
_CIONA17_SPLITS = {
    "ciona17_sem_seg_train": [
        ("farm1_train1", "_mask.jpg"),
        ("farm1_train2", ".jpg"),
    ],
    "ciona17_sem_seg_val": [
        ("farm1_val", "_mask.jpg"),
        ("farm2_val", ".jpg"),
    ],
}


def _get_ciona17_sem_seg_meta():
    return {
        "stuff_classes": _CIONA17_STUFF_CLASSES,
        "stuff_dataset_id_to_contiguous_id": {idx: idx for idx in range(len(_CIONA17_STUFF_CLASSES))},
        # Fixed RGB colors (0-255) for visualization of each semantic class:
        # other, mussel, ciona (styela merged into other)
        "stuff_colors": [
            (128, 128, 128),  # other - gray
            (0, 0, 255),      # mussel - blue
            (0, 255, 0),      # ciona - green
        ],
    }


def load_ciona17_sem_seg(root, dataset_name, split_specs):
    dataset_dicts = []
    missing_files = []

    for split_name, mask_suffix in split_specs:
        split_root = os.path.join(root, split_name)
        image_root = os.path.join(split_root, "images")
        mask_root = os.path.join(split_root, "masks")

        if not os.path.isdir(image_root) or not os.path.isdir(mask_root):
            continue

        image_filenames = sorted(
            f for f in os.listdir(image_root) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        for filename in image_filenames:
            image_path = os.path.join(image_root, filename)
            stem, _ = os.path.splitext(filename)
            mask_path = os.path.join(mask_root, f"{stem}{mask_suffix}")

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
        raise FileNotFoundError(f"Missing Ciona17 semantic files, e.g. {preview}")

    if not dataset_dicts:
        raise ValueError(f"No Ciona17 semantic segmentation images found for dataset {dataset_name}")

    return dataset_dicts


def register_ciona17_sem_seg(name, metadata, root, split_specs):
    DatasetCatalog.register(
        name,
        lambda root=root, dataset_name=name, split_specs=split_specs: load_ciona17_sem_seg(
            root,
            dataset_name,
            split_specs,
        ),
    )
    MetadataCatalog.get(name).set(
        image_root=root,
        sem_seg_root=root,
        evaluator_type="sem_seg",
        ignore_label=255,
        ciona17_jpeg_void_remap=True,
        **metadata,
    )


def register_all_ciona17_sem_seg(root):
    ciona17_root = os.getenv("CIONA17_DATASET_ROOT", os.path.join(root, "ciona17", "Ciona17"))
    if not os.path.isdir(ciona17_root):
        return

    metadata = _get_ciona17_sem_seg_meta()
    for dataset_name, split_specs in _CIONA17_SPLITS.items():
        register_ciona17_sem_seg(dataset_name, metadata, ciona17_root, split_specs)


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ciona17_sem_seg(_root)
