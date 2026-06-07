import copy
import json
import logging
import os

import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader

from datasets import OpenWorldSAM2SemanticDatasetMapper
from datasets.datasets.register_dutuseg_semseg import _get_dutuseg_sem_seg_meta
from datasets.datasets.register_suim_semseg import _get_suim_sem_seg_meta
from evaluation import SemSegEvaluator
from tta.conf import get_tta_init_weights
from tta.method import build_tta_method

logger = logging.getLogger("open-world-sam2-tta")


def build_tta_dataset_name(dataset_key, corruption, severity):
    return f"{dataset_key}_{corruption}_{severity}"


def load_corrupted_sem_seg(domain_root, mask_root, corruption, severity, mask_extension):
    image_root = os.path.join(domain_root, corruption, str(severity))
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Missing corrupted image directory: {image_root}")
    if not os.path.isdir(mask_root):
        raise FileNotFoundError(f"Missing semantic mask directory: {mask_root}")

    dataset_dicts = []
    image_filenames = sorted(
        filename
        for filename in os.listdir(image_root)
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    for filename in image_filenames:
        image_path = os.path.join(image_root, filename)
        stem, _ = os.path.splitext(filename)
        mask_path = os.path.join(mask_root, f"{stem}{mask_extension}")
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Missing semantic mask for {image_path}: {mask_path}")
        dataset_dicts.append(
            {
                "file_name": image_path,
                "sem_seg_file_name": mask_path,
            }
        )

    if not dataset_dicts:
        raise ValueError(f"No corrupted semantic segmentation images found in {image_root}")
    return dataset_dicts


def get_tta_dataset_spec(cfg):
    dataset_key = str(cfg.TTA.DATASET).lower()
    if dataset_key == "suim_c_sem_seg":
        return {
            "dataset_key": dataset_key,
            "mask_extension": ".bmp",
            "metadata": _get_suim_sem_seg_meta(),
            "metadata_flags": {"suim_rgb_mask": True},
        }
    if dataset_key == "dutuseg_c_sem_seg":
        return {
            "dataset_key": dataset_key,
            "mask_extension": ".png",
            "metadata": _get_dutuseg_sem_seg_meta(),
            "metadata_flags": {"dutuseg_rgb_mask": True},
        }
    raise ValueError(f"Unsupported TTA dataset: {cfg.TTA.DATASET}")


def register_tta_dataset(cfg, corruption, severity):
    dataset_spec = get_tta_dataset_spec(cfg)
    dataset_name = build_tta_dataset_name(dataset_spec["dataset_key"], corruption, severity)
    if dataset_name not in DatasetCatalog.list():
        DatasetCatalog.register(
            dataset_name,
            lambda domain_root=cfg.TTA.DOMAIN_ROOT, mask_root=cfg.TTA.MASK_ROOT, corruption=corruption, severity=severity, mask_extension=dataset_spec["mask_extension"]: load_corrupted_sem_seg(
                domain_root,
                mask_root,
                corruption,
                severity,
                mask_extension,
            ),
        )
        MetadataCatalog.get(dataset_name).set(
            image_root=os.path.join(cfg.TTA.DOMAIN_ROOT, corruption, str(severity)),
            sem_seg_root=cfg.TTA.MASK_ROOT,
            evaluator_type="sem_seg",
            ignore_label=255,
            **dataset_spec["metadata_flags"],
            **dataset_spec["metadata"],
        )
    return dataset_name


def build_tta_loader(cfg, dataset_name):
    loader_cfg = cfg.clone()
    loader_cfg.defrost()
    loader_cfg.DATASETS.TEST = (dataset_name,)
    loader_cfg.DATALOADER.NUM_WORKERS = int(cfg.TTA.NUM_WORKERS)
    loader_cfg.freeze()
    mapper = OpenWorldSAM2SemanticDatasetMapper(loader_cfg, is_train=False)
    return build_detection_test_loader(
        loader_cfg,
        dataset_name=dataset_name,
        mapper=mapper,
        batch_size=int(cfg.TTA.BATCH_SIZE),
    )


def evaluate_loader(model, data_loader, evaluator):
    evaluator.reset()
    for inputs in data_loader:
        outputs = model(inputs)
        evaluator.process(inputs, outputs)
    return evaluator.evaluate()


def _format_sem_seg_metrics(results):
    sem_seg = results.get("sem_seg", {}) if isinstance(results, dict) else {}
    metrics = []
    for metric_name in ("mIoU", "fwIoU", "mACC", "pACC"):
        if metric_name in sem_seg:
            metrics.append(f"{metric_name}={float(sem_seg[metric_name]):.4f}")
    return " ".join(metrics)


def run_tta(cfg, base_model):
    weights_path = get_tta_init_weights(cfg)
    DetectionCheckpointer(base_model).resume_or_load(weights_path, resume=False)
    source_state = copy.deepcopy(base_model.state_dict())
    tta_mode = str(cfg.TTA.TTA_MODE).lower()
    if tta_mode not in {"normal_tta", "cont_tta"}:
        raise ValueError(f"Unsupported TTA mode: {cfg.TTA.TTA_MODE}")

    all_results = {}
    summary_scores = []

    for corruption in cfg.TTA.CORRUPTIONS:
        corruption_scores = []
        for severity in cfg.TTA.SEVERITIES:
            dataset_name = register_tta_dataset(cfg, corruption, severity)
            data_loader = build_tta_loader(cfg, dataset_name)
            output_dir = os.path.join(
                cfg.OUTPUT_DIR,
                "tta",
                str(cfg.TTA.METHOD).lower(),
                corruption,
                str(severity),
            )
            evaluator = SemSegEvaluator(dataset_name, distributed=False, output_dir=output_dir)

            if tta_mode == "normal_tta":
                base_model.load_state_dict(source_state, strict=True)
            base_model.metadata = MetadataCatalog.get(dataset_name)
            adapt_model = build_tta_method(cfg, base_model)
            results = evaluate_loader(adapt_model, data_loader, evaluator)

            key = f"{corruption}/severity_{severity}"
            all_results[key] = results
            miou = float(results["sem_seg"]["mIoU"])
            metrics_str = _format_sem_seg_metrics(results)
            corruption_scores.append(miou)
            summary_scores.append(miou)
            logger.info(
                "[TTA][DOMAIN] mode=%s method=%s domain=%s severity=%s %s",
                tta_mode,
                cfg.TTA.METHOD,
                corruption,
                severity,
                metrics_str,
            )

        if corruption_scores:
            corruption_mean_miou = float(np.mean(corruption_scores))
            all_results[f"{corruption}/mean_mIoU"] = corruption_mean_miou
            logger.info(
                "[TTA][CORRUPTION] mode=%s method=%s corruption=%s mean_mIoU=%.4f",
                tta_mode,
                cfg.TTA.METHOD,
                corruption,
                corruption_mean_miou,
            )

    summary = {
        "tta_mode": tta_mode,
        "method": str(cfg.TTA.METHOD).lower(),
        "weights": weights_path,
        "mean_mIoU": float(np.mean(summary_scores)) if summary_scores else 0.0,
        "results": all_results,
    }

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(cfg.OUTPUT_DIR, f"tta_{str(cfg.TTA.METHOD).lower()}_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info(
        "[TTA][SUMMARY] mode=%s method=%s mean_mIoU=%.4f",
        tta_mode,
        cfg.TTA.METHOD,
        summary["mean_mIoU"],
    )
    logger.info("[TTA] summary saved to %s", summary_path)
    return summary
