import copy
import json
import logging
import os
import random

import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader

from datasets import OpenWorldSAM2SemanticDatasetMapper
from datasets.datasets.register_dutuseg_semseg import _get_dutuseg_sem_seg_meta
from datasets.datasets.register_suim_semseg import _get_suim_sem_seg_meta
from tta.conf import get_tta_init_weights
from tta.method import build_tta_method
from tta.sem_seg_tta_evaluator import TTASemSegEvaluator

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


def use_clean_tta_data(cfg):
    return bool(getattr(cfg.TTA, "USE_CLEAN_DATA", False))


def include_clean_as_first_domain(cfg):
    return bool(getattr(cfg.TTA, "INCLUDE_CLEAN_AS_FIRST_DOMAIN", False))


def get_clean_tta_dataset_name(cfg):
    dataset_key = str(getattr(cfg.TTA, "DATASET", "")).lower()
    dataset_name_map = {
        "suim_c_sem_seg": "suim_sem_seg_val",
        "suim_sem_seg": "suim_sem_seg_val",
        "suim_sem_seg_val": "suim_sem_seg_val",
        "dutuseg_c_sem_seg": "dutuseg_sem_seg_val",
        "dutuseg_sem_seg": "dutuseg_sem_seg_val",
        "dutuseg_sem_seg_val": "dutuseg_sem_seg_val",
    }
    if dataset_key in dataset_name_map:
        return dataset_name_map[dataset_key]

    dataset_candidates = list(getattr(cfg.DATASETS, "TEST", ()))
    if dataset_candidates:
        return str(dataset_candidates[0])

    raise ValueError(f"Unable to resolve clean TTA dataset for TTA.DATASET={cfg.TTA.DATASET}")


def evaluate_loader(model, data_loader, evaluator):
    evaluator.reset()
    for inputs in data_loader:
        outputs = model(inputs)
        evaluator.process(inputs, outputs)
    return evaluator.evaluate()


def _format_sem_seg_metrics(results):
    sem_seg = results.get("sem_seg", {}) if isinstance(results, dict) else {}
    metrics = []
    for metric_name in ("mIoU", "fwIoU", "mDice", "BoundaryF1", "TrimapIoU", "mACC", "pACC", "ECE", "BrierScore"):
        if metric_name in sem_seg:
            metrics.append(f"{metric_name}={float(sem_seg[metric_name]):.4f}")
    return " ".join(metrics)


def _evaluate_domain(cfg, base_model, corruption, severity, tta_mode, output_dir_suffix=None):
    dataset_name = register_tta_dataset(cfg, corruption, severity)
    data_loader = build_tta_loader(cfg, dataset_name)
    output_dir_parts = [
        cfg.OUTPUT_DIR,
        "tta",
        str(cfg.TTA.METHOD).lower(),
    ]
    if output_dir_suffix is not None:
        output_dir_parts.append(output_dir_suffix)
    output_dir_parts.extend([corruption, str(severity)])
    output_dir = os.path.join(*output_dir_parts)
    evaluator = TTASemSegEvaluator(dataset_name, distributed=False, output_dir=output_dir)

    base_model.metadata = MetadataCatalog.get(dataset_name)
    adapt_model = build_tta_method(cfg, base_model)
    return evaluate_loader(adapt_model, data_loader, evaluator)


def _evaluate_clean_dataset(cfg, base_model, output_dir_suffix=None):
    dataset_name = get_clean_tta_dataset_name(cfg)
    data_loader = build_tta_loader(cfg, dataset_name)
    output_dir_parts = [
        cfg.OUTPUT_DIR,
        "tta",
        str(cfg.TTA.METHOD).lower(),
    ]
    if output_dir_suffix is not None:
        output_dir_parts.append(output_dir_suffix)
    output_dir_parts.append("clean")
    output_dir = os.path.join(*output_dir_parts)
    evaluator = TTASemSegEvaluator(dataset_name, distributed=False, output_dir=output_dir)

    base_model.metadata = MetadataCatalog.get(dataset_name)
    adapt_model = build_tta_method(cfg, base_model)
    return evaluate_loader(adapt_model, data_loader, evaluator)


def run_tta(cfg, base_model):
    weights_path = get_tta_init_weights(cfg)
    DetectionCheckpointer(base_model).resume_or_load(weights_path, resume=False)
    source_state = copy.deepcopy(base_model.state_dict())
    tta_mode = str(cfg.TTA.TTA_MODE).lower()
    if tta_mode not in {"normal_tta", "cont_tta", "lifelong_rand_cont_tta"}:
        raise ValueError(f"Unsupported TTA mode: {cfg.TTA.TTA_MODE}")

    all_results = {}
    summary_scores = []
    prepend_clean = include_clean_as_first_domain(cfg) and not use_clean_tta_data(cfg)

    def _mean_metric(metric_name):
        values = []
        for result in all_results.values():
            if not isinstance(result, dict):
                continue
            sem_seg = result.get("sem_seg", {})
            if metric_name in sem_seg:
                values.append(float(sem_seg[metric_name]))
        return float(np.mean(values)) if values else 0.0

    def _run_clean_first_if_enabled(output_dir_suffix=None):
        if not prepend_clean or "clean" in all_results:
            return

        if tta_mode == "normal_tta":
            base_model.load_state_dict(source_state, strict=True)

        results = _evaluate_clean_dataset(cfg, base_model, output_dir_suffix=output_dir_suffix)
        all_results["clean"] = results
        miou = float(results["sem_seg"]["mIoU"])
        metrics_str = _format_sem_seg_metrics(results)
        summary_scores.append(miou)
        logger.info(
            "[TTA][CLEAN-FIRST] mode=%s method=%s dataset=%s %s",
            tta_mode,
            cfg.TTA.METHOD,
            get_clean_tta_dataset_name(cfg),
            metrics_str,
        )

    if use_clean_tta_data(cfg):
        if tta_mode == "normal_tta":
            base_model.load_state_dict(source_state, strict=True)

        results = _evaluate_clean_dataset(cfg, base_model)
        all_results["clean"] = results
        miou = float(results["sem_seg"]["mIoU"])
        metrics_str = _format_sem_seg_metrics(results)
        summary_scores.append(miou)
        logger.info(
            "[TTA][CLEAN] mode=%s method=%s dataset=%s %s",
            tta_mode,
            cfg.TTA.METHOD,
            get_clean_tta_dataset_name(cfg),
            metrics_str,
        )
    elif tta_mode == "lifelong_rand_cont_tta":
        _run_clean_first_if_enabled()
        corruption_list = list(cfg.TTA.CORRUPTIONS)
        num_rounds = max(1, int(cfg.TTA.TTA_ROUNDS))
        rng = random.Random(int(getattr(cfg, "SEED", 0)))
        round_corruption_scores = {}

        for round_idx in range(num_rounds):
            round_corruptions = list(corruption_list)
            rng.shuffle(round_corruptions)
            logger.info(
                "[TTA][ROUND] mode=%s method=%s round=%d corruption_order=%s",
                tta_mode,
                cfg.TTA.METHOD,
                round_idx + 1,
                round_corruptions,
            )

            for corruption in round_corruptions:
                round_key = f"round_{round_idx + 1}/{corruption}"
                corruption_scores = []
                for severity in cfg.TTA.SEVERITIES:
                    results = _evaluate_domain(
                        cfg,
                        base_model,
                        corruption,
                        severity,
                        tta_mode,
                        output_dir_suffix=f"round_{round_idx + 1}",
                    )

                    key = f"{round_key}/severity_{severity}"
                    all_results[key] = results
                    miou = float(results["sem_seg"]["mIoU"])
                    metrics_str = _format_sem_seg_metrics(results)
                    corruption_scores.append(miou)
                    summary_scores.append(miou)
                    logger.info(
                        "[TTA][DOMAIN] mode=%s method=%s round=%d domain=%s severity=%s %s",
                        tta_mode,
                        cfg.TTA.METHOD,
                        round_idx + 1,
                        corruption,
                        severity,
                        metrics_str,
                    )

                if corruption_scores:
                    corruption_mean_miou = float(np.mean(corruption_scores))
                    all_results[f"{round_key}/mean_mIoU"] = corruption_mean_miou
                    round_corruption_scores.setdefault(corruption, []).append(corruption_mean_miou)
                    logger.info(
                        "[TTA][CORRUPTION] mode=%s method=%s round=%d corruption=%s mean_mIoU=%.4f",
                        tta_mode,
                        cfg.TTA.METHOD,
                        round_idx + 1,
                        corruption,
                        corruption_mean_miou,
                    )

        for corruption, scores in round_corruption_scores.items():
            if scores:
                all_results[f"{corruption}/mean_mIoU_across_rounds"] = float(np.mean(scores))
    else:
        _run_clean_first_if_enabled()
        for corruption in cfg.TTA.CORRUPTIONS:
            corruption_scores = []
            for severity in cfg.TTA.SEVERITIES:
                if tta_mode == "normal_tta":
                    base_model.load_state_dict(source_state, strict=True)

                results = _evaluate_domain(cfg, base_model, corruption, severity, tta_mode)

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
        "mean_mDice": _mean_metric("mDice"),
        "mean_BoundaryF1": _mean_metric("BoundaryF1"),
        "mean_TrimapIoU": _mean_metric("TrimapIoU"),
        "mean_mACC": _mean_metric("mACC"),
        "mean_pACC": _mean_metric("pACC"),
        "mean_ECE": _mean_metric("ECE"),
        "mean_BrierScore": _mean_metric("BrierScore"),
        "results": all_results,
    }

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(cfg.OUTPUT_DIR, f"tta_{str(cfg.TTA.METHOD).lower()}_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info(
        "[TTA][SUMMARY] mode=%s method=%s mean_mIoU=%.4f mean_mDice=%.4f mean_BoundaryF1=%.4f mean_TrimapIoU=%.4f mean_mACC=%.4f mean_pACC=%.4f mean_ECE=%.4f mean_BrierScore=%.4f",
        tta_mode,
        cfg.TTA.METHOD,
        summary["mean_mIoU"],
        summary["mean_mDice"],
        summary["mean_BoundaryF1"],
        summary["mean_TrimapIoU"],
        summary["mean_mACC"],
        summary["mean_pACC"],
        summary["mean_ECE"],
        summary["mean_BrierScore"],
    )
    logger.info("[TTA] summary saved to %s", summary_path)
    return summary
