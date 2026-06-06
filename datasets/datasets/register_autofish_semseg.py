import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager


_ALL_GROUPS = {f"group_{idx:02d}" for idx in range(1, 26)}
_VAL_GROUPS = {"group_10", "group_14", "group_20", "group_21", "group_22"}
_TRAIN_GROUPS = _ALL_GROUPS - _VAL_GROUPS
_AUTOFISH_STUFF_CLASSES = ["whiting", "cod", "haddock", "hake", "horse_mackerel", "other"]


def _get_autofish_sem_seg_meta():
    return {
        "stuff_classes": _AUTOFISH_STUFF_CLASSES,
        "stuff_dataset_id_to_contiguous_id": {idx: idx for idx in range(len(_AUTOFISH_STUFF_CLASSES))},
    }


def load_autofish_sem_seg(image_root, gt_root, dataset_name, include_groups, exclude_groups=None):
    filtered_dicts = []
    missing_files = []
    # AutoFish images and semantic masks live in per-group subdirectories, e.g.:
    #   image_root/group_01/00001.png
    #   gt_root/group_01/00001.png
    # Detectron2's builtin load_sem_seg assumes a flat directory, so we build
    # the (image, mask) pairs explicitly by walking the known group folders.
    for group_name in sorted(_ALL_GROUPS):
        if include_groups is not None and group_name not in include_groups:
            continue
        if exclude_groups is not None and group_name in exclude_groups:
            continue

        image_group_dir = os.path.join(image_root, group_name)
        mask_group_dir = os.path.join(gt_root, group_name)
        if not os.path.isdir(image_group_dir) or not os.path.isdir(mask_group_dir):
            continue

        try:
            image_filenames = sorted(
                f for f in os.listdir(image_group_dir) if f.lower().endswith(".png")
            )
        except FileNotFoundError:
            continue

        for filename in image_filenames:
            image_path = os.path.join(image_group_dir, filename)
            mask_filename = os.path.splitext(filename)[0] + ".png"
            mask_path = os.path.join(mask_group_dir, mask_filename)

            if not PathManager.isfile(image_path):
                missing_files.append(image_path)
                continue
            if not PathManager.isfile(mask_path):
                missing_files.append(mask_path)
                continue

            filtered_dicts.append(
                {
                    "file_name": image_path,
                    "sem_seg_file_name": mask_path,
                }
            )

    if missing_files:
        preview = ", ".join(missing_files[:3])
        raise FileNotFoundError(f"Missing AutoFish semantic files, e.g. {preview}")

    if not filtered_dicts:
        raise ValueError(f"No AutoFish semantic segmentation images found for dataset {dataset_name}")

    return filtered_dicts


def register_autofish_sem_seg(name, metadata, image_root, gt_root, include_groups, exclude_groups=None):
    DatasetCatalog.register(
        name,
        lambda image_root=image_root, gt_root=gt_root, dataset_name=name, include_groups=include_groups, exclude_groups=exclude_groups: load_autofish_sem_seg(
            image_root,
            gt_root,
            dataset_name,
            include_groups,
            exclude_groups,
        ),
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        sem_seg_root=gt_root,
        evaluator_type="sem_seg",
        ignore_label=255,
        **metadata,
    )


def register_all_autofish_sem_seg(root):
    image_root = os.getenv("AUTOFISH_DATASET_ROOT", os.path.join(root, "autofish", "AutoFish"))
    gt_root = os.getenv("AUTOFISH_MASKS_ROOT", os.path.join(image_root, "masks"))
    if not os.path.isdir(image_root) or not os.path.isdir(gt_root):
        return

    metadata = _get_autofish_sem_seg_meta()
    register_autofish_sem_seg(
        "autofish_sem_seg_train",
        metadata,
        image_root,
        gt_root,
        include_groups=_TRAIN_GROUPS,
        exclude_groups=None,
    )
    register_autofish_sem_seg(
        "autofish_sem_seg_val",
        metadata,
        image_root,
        gt_root,
        include_groups=_VAL_GROUPS,
        exclude_groups=None,
    )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_autofish_sem_seg(_root)
