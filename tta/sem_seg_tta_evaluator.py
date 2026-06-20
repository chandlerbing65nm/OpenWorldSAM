from collections import OrderedDict
import json
import os

import numpy as np
import scipy.ndimage
import torch
from detectron2.utils.file_io import PathManager

from evaluation.segmentation_evaluation import (
    SemSegEvaluator,
    decode_rgb_semantic_mask,
    decode_rgb_semantic_mask_with_mapping,
    load_semseg,
)


class TTASemSegEvaluator(SemSegEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed=False,
        output_dir=None,
        boundary_tolerance=2,
        trimap_width=3,
        calibration_bins=15,
        *,
        num_classes=None,
        ignore_label=None,
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        self._boundary_tolerance = max(1, int(boundary_tolerance))
        self._trimap_width = max(1, int(trimap_width))
        self._calibration_bins = max(2, int(calibration_bins))

    def reset(self):
        super().reset()
        self._boundary_pred_matches = np.zeros(self._num_classes, dtype=np.float64)
        self._boundary_gt_matches = np.zeros(self._num_classes, dtype=np.float64)
        self._boundary_pred_pixels = np.zeros(self._num_classes, dtype=np.float64)
        self._boundary_gt_pixels = np.zeros(self._num_classes, dtype=np.float64)
        self._trimap_intersection = np.zeros(self._num_classes, dtype=np.float64)
        self._trimap_union = np.zeros(self._num_classes, dtype=np.float64)
        self._ece_bin_counts = np.zeros(self._calibration_bins, dtype=np.float64)
        self._ece_bin_confidence = np.zeros(self._calibration_bins, dtype=np.float64)
        self._ece_bin_accuracy = np.zeros(self._calibration_bins, dtype=np.float64)
        self._brier_sum = 0.0
        self._brier_count = 0

    def _prepare_ground_truth(self, input_record):
        gt_from_input = "semseg" in input_record

        if gt_from_input:
            gt = np.array(input_record["semseg"].to(self._cpu_device), dtype=np.int64)
        else:
            with PathManager.open(self.input_file_to_gt_file[input_record["file_name"]], "rb") as handle:
                gt = load_semseg(handle, self._semseg_loader) - self._class_offset

        if self._suim_rgb_mask and not gt_from_input:
            if self._suim_color_to_class_id is not None:
                gt = decode_rgb_semantic_mask_with_mapping(gt, self._suim_color_to_class_id, self._num_classes)
            else:
                gt = decode_rgb_semantic_mask(gt, self._stuff_colors, self._num_classes)
        if self._dutuseg_rgb_mask and not gt_from_input:
            gt = decode_rgb_semantic_mask(gt, self._stuff_colors, self._num_classes)
        if self._coralscapes_label_shift:
            gt = gt.copy().astype(np.int64)
            gt[gt == 0] = self._num_classes
            valid_mask = gt != self._num_classes
            gt[valid_mask] = gt[valid_mask] - 1
        if self._jpeg_void_remap:
            gt = gt.copy()
            gt[gt == 3] = 0
            gt[gt > 3] = self._num_classes
        elif not self._coralscapes_label_shift:
            if isinstance(self._ignore_label, int):
                gt[gt == self._ignore_label] = self._num_classes
            elif isinstance(self._ignore_label, list):
                for ignore_label in self._ignore_label:
                    gt[gt == ignore_label] = self._num_classes

        return gt

    def _binary_boundary(self, mask):
        if mask.ndim != 2 or not np.any(mask):
            return np.zeros_like(mask, dtype=bool)
        mask = mask.astype(bool)
        eroded = scipy.ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
        return np.logical_and(mask, np.logical_not(eroded))

    def _label_boundary(self, labels, valid_mask):
        boundary = np.zeros_like(valid_mask, dtype=bool)
        boundary[1:, :] |= np.logical_and(valid_mask[1:, :], valid_mask[:-1, :]) & (labels[1:, :] != labels[:-1, :])
        boundary[:-1, :] |= np.logical_and(valid_mask[:-1, :], valid_mask[1:, :]) & (labels[:-1, :] != labels[1:, :])
        boundary[:, 1:] |= np.logical_and(valid_mask[:, 1:], valid_mask[:, :-1]) & (labels[:, 1:] != labels[:, :-1])
        boundary[:, :-1] |= np.logical_and(valid_mask[:, :-1], valid_mask[:, 1:]) & (labels[:, :-1] != labels[:, 1:])
        return boundary

    def _update_boundary_metrics(self, pred, gt, valid_mask):
        structure = np.ones((2 * self._boundary_tolerance + 1, 2 * self._boundary_tolerance + 1), dtype=bool)
        for class_idx in range(self._num_classes):
            gt_mask = np.logical_and(gt == class_idx, valid_mask)
            pred_mask = np.logical_and(pred == class_idx, valid_mask)
            if not np.any(gt_mask) and not np.any(pred_mask):
                continue
            gt_boundary = self._binary_boundary(gt_mask)
            pred_boundary = self._binary_boundary(pred_mask)
            gt_count = float(gt_boundary.sum())
            pred_count = float(pred_boundary.sum())
            self._boundary_gt_pixels[class_idx] += gt_count
            self._boundary_pred_pixels[class_idx] += pred_count
            if gt_count > 0 and np.any(pred_boundary):
                matched_pred = np.logical_and(
                    pred_boundary,
                    scipy.ndimage.binary_dilation(gt_boundary, structure=structure),
                )
                self._boundary_pred_matches[class_idx] += float(matched_pred.sum())
            if pred_count > 0 and np.any(gt_boundary):
                matched_gt = np.logical_and(
                    gt_boundary,
                    scipy.ndimage.binary_dilation(pred_boundary, structure=structure),
                )
                self._boundary_gt_matches[class_idx] += float(matched_gt.sum())

    def _update_trimap_metrics(self, pred, gt, valid_mask):
        structure = np.ones((2 * self._trimap_width + 1, 2 * self._trimap_width + 1), dtype=bool)
        trimap_mask = np.logical_and(
            scipy.ndimage.binary_dilation(self._label_boundary(gt, valid_mask), structure=structure),
            valid_mask,
        )
        if not np.any(trimap_mask):
            return
        for class_idx in range(self._num_classes):
            gt_class = np.logical_and(gt == class_idx, trimap_mask)
            pred_class = np.logical_and(pred == class_idx, trimap_mask)
            if not np.any(gt_class) and not np.any(pred_class):
                continue
            self._trimap_intersection[class_idx] += float(np.logical_and(gt_class, pred_class).sum())
            self._trimap_union[class_idx] += float(np.logical_or(gt_class, pred_class).sum())

    def _update_calibration_metrics(self, sem_seg, pred, gt, valid_mask):
        logits = sem_seg.detach().float().to(self._cpu_device)
        probs = torch.softmax(logits, dim=0)
        confidence, _ = torch.max(probs, dim=0)
        confidence = confidence.numpy()
        probs = probs.numpy()
        valid_confidence = confidence[valid_mask]
        valid_pred = pred[valid_mask]
        valid_gt = gt[valid_mask]
        if valid_gt.size == 0:
            return
        correctness = (valid_pred == valid_gt).astype(np.float64)
        bin_indices = np.minimum((valid_confidence * self._calibration_bins).astype(np.int64), self._calibration_bins - 1)
        for bin_idx in range(self._calibration_bins):
            in_bin = bin_indices == bin_idx
            if not np.any(in_bin):
                continue
            self._ece_bin_counts[bin_idx] += float(in_bin.sum())
            self._ece_bin_confidence[bin_idx] += float(valid_confidence[in_bin].sum())
            self._ece_bin_accuracy[bin_idx] += float(correctness[in_bin].sum())
        valid_probs = probs[:, valid_mask].transpose(1, 0)
        one_hot = np.eye(self._num_classes, dtype=np.float32)[valid_gt]
        self._brier_sum += float(np.sum((valid_probs - one_hot) ** 2))
        self._brier_count += int(valid_gt.size)

    def process(self, inputs, outputs):
        for input_record, output in zip(inputs, outputs):
            sem_seg = output["sem_seg"]
            pred_tensor = sem_seg.argmax(dim=0).to(self._cpu_device)
            pred = np.array(pred_tensor, dtype=np.int64)
            gt = self._prepare_ground_truth(input_record)

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            valid_mask = gt != self._num_classes
            self._update_boundary_metrics(pred, gt, valid_mask)
            self._update_trimap_metrics(pred, gt, valid_mask)
            self._update_calibration_metrics(sem_seg, pred, gt, valid_mask)
            self._predictions.extend(self.encode_json_sem_seg(pred, input_record["file_name"]))

    def evaluate(self):
        self._distributed = False

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            predictions_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(predictions_path, "w") as handle:
                handle.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float64)
        iou = np.full(self._num_classes, np.nan, dtype=np.float64)
        dice = np.full(self._num_classes, np.nan, dtype=np.float64)

        tp = self._conf_matrix.diagonal()[:-1].astype(np.float64)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float64)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float64)
        total_gt = np.sum(pos_gt)
        class_weights = pos_gt / total_gt if total_gt > 0 else np.zeros_like(pos_gt)

        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]

        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        dice[iou_valid] = (2.0 * tp[iou_valid]) / (pos_gt[iou_valid] + pos_pred[iou_valid])

        macc = np.nanmean(acc) if np.any(acc_valid) else np.nan
        miou = np.nanmean(iou) if np.any(iou_valid) else np.nan
        mdice = np.nanmean(dice) if np.any(iou_valid) else np.nan
        fiou = np.nansum(iou * class_weights)
        pacc = np.sum(tp) / total_gt if total_gt > 0 else np.nan

        boundary_precision = np.full(self._num_classes, np.nan, dtype=np.float64)
        boundary_recall = np.full(self._num_classes, np.nan, dtype=np.float64)
        boundary_f1 = np.full(self._num_classes, np.nan, dtype=np.float64)
        pred_boundary_valid = self._boundary_pred_pixels > 0
        gt_boundary_valid = self._boundary_gt_pixels > 0
        boundary_precision[pred_boundary_valid] = (
            self._boundary_pred_matches[pred_boundary_valid] / self._boundary_pred_pixels[pred_boundary_valid]
        )
        boundary_recall[gt_boundary_valid] = (
            self._boundary_gt_matches[gt_boundary_valid] / self._boundary_gt_pixels[gt_boundary_valid]
        )
        boundary_sum = np.nan_to_num(boundary_precision, nan=0.0) + np.nan_to_num(boundary_recall, nan=0.0)
        boundary_valid = np.logical_and(np.logical_or(pred_boundary_valid, gt_boundary_valid), boundary_sum > 0)
        boundary_f1[boundary_valid] = (
            2.0 * boundary_precision[boundary_valid] * boundary_recall[boundary_valid] / boundary_sum[boundary_valid]
        )
        mean_boundary_f1 = np.nanmean(boundary_f1) if np.any(boundary_valid) else np.nan

        trimap_iou = np.full(self._num_classes, np.nan, dtype=np.float64)
        trimap_valid = self._trimap_union > 0
        trimap_iou[trimap_valid] = self._trimap_intersection[trimap_valid] / self._trimap_union[trimap_valid]
        mean_trimap_iou = np.nanmean(trimap_iou) if np.any(trimap_valid) else np.nan

        total_calibration = np.sum(self._ece_bin_counts)
        ece = np.nan
        if total_calibration > 0:
            ece = 0.0
            for bin_idx in range(self._calibration_bins):
                count = self._ece_bin_counts[bin_idx]
                if count <= 0:
                    continue
                avg_conf = self._ece_bin_confidence[bin_idx] / count
                avg_acc = self._ece_bin_accuracy[bin_idx] / count
                ece += abs(avg_acc - avg_conf) * (count / total_calibration)
        brier_score = self._brier_sum / self._brier_count if self._brier_count > 0 else np.nan

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mDice"] = 100 * mdice
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        res["BoundaryF1"] = 100 * mean_boundary_f1
        res["TrimapIoU"] = 100 * mean_trimap_iou
        res["ECE"] = ece
        res["BrierScore"] = brier_score

        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
            res[f"Dice-{name}"] = 100 * dice[i]
            res[f"ACC-{name}"] = 100 * acc[i]
            res[f"BoundaryF1-{name}"] = 100 * boundary_f1[i]
            res[f"TrimapIoU-{name}"] = 100 * trimap_iou[i]

        if self._output_dir:
            eval_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(eval_path, "wb") as handle:
                torch.save(res, handle)

        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
