"""
metrics.py — 评估指标与训练辅助
===============================
- mAP (mean Average Precision) 计算
- AverageMeter 训练日志辅助
- PR 曲线数据收集
"""

import torch
import numpy as np
from typing import List, Tuple
from collections import defaultdict


class AverageMeter:
    """追踪和平均每个 metric 的值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.avg:.4f}"


def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Compute Average Precision (AP) from precision-recall curve.
    使用 PR 曲线下面积 (11-point interpolation)。
    """
    # 在末尾添加哨兵
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # 保证 precision 单调递减
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # 计算 PR 曲线下面积
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap)


def compute_map(all_predictions: List[Tuple[torch.Tensor, torch.Tensor]],
                all_groundtruths: List[Tuple[torch.Tensor, torch.Tensor]],
                iou_threshold: float = 0.5,
                num_points: int = 101) -> dict:
    """
    计算 mAP (mean Average Precision)。

    Args:
        all_predictions: List of (boxes, scores)
            boxes:  [K_i, 4] 预测框 [x1, y1, x2, y2] (绝对坐标)
            scores: [K_i]    置信度
        all_groundtruths: List of (gt_boxes, gt_labels)
            gt_boxes:  [M_i, 4] GT 框
            gt_labels: [M_i]    GT 标签

    Returns:
        metrics: dict with keys {'mAP@0.5', 'num_images', 'num_gt', 'num_pred'}
    """
    # 收集所有图像的所有检测和 GT
    all_detections = []   # [(image_id, box, score)]
    all_gts = []          # [(image_id, box)]
    gt_counts = {}        # image_id → num_gt

    for img_id, (pred, gt) in enumerate(zip(all_predictions, all_groundtruths)):
        boxes, scores = pred
        gt_boxes, _ = gt

        gt_counts[img_id] = len(gt_boxes)
        for b in range(len(gt_boxes)):
            all_gts.append((img_id, gt_boxes[b]))

        if boxes.numel() == 0:
            continue
        for d in range(len(boxes)):
            all_detections.append((img_id, boxes[d], scores[d].item()))

    if len(all_detections) == 0 or len(all_gts) == 0:
        return {"mAP@0.5": 0.0, "num_images": len(all_predictions),
                "num_gt": len(all_gts), "num_pred": len(all_detections)}

    # 按置信度降序排列所有检测
    all_detections.sort(key=lambda x: x[2], reverse=True)

    # 初始化匹配状态
    # TP: 正确的检测 (与 GT 的 IoU ≥ threshold 且该 GT 未被使用)
    # FP: 错误的检测 (IoU < threshold 或 GT 已被匹配)
    tp = np.zeros(len(all_detections))
    fp = np.zeros(len(all_detections))

    # 跟踪每个图像中哪些 GT 已被匹配
    gt_matched = defaultdict(set)  # image_id → set of matched gt indices

    for d_idx, (img_id, det_box, _) in enumerate(all_detections):
        # 找到该图像中的所有 GT
        img_gts = [(gt_idx, gt_box) for gt_idx, (gid, gt_box)
                   in enumerate(all_gts) if gid == img_id]

        if len(img_gts) == 0:
            fp[d_idx] = 1
            continue

        # 计算与所有 GT 的 IoU
        gt_indices, gt_boxes = zip(*img_gts)
        gt_boxes = torch.stack(gt_boxes)
        ious = compute_iou_torch(det_box.unsqueeze(0), gt_boxes)[0]

        # 找到最大 IoU
        max_iou, max_idx = ious.max(dim=0)
        gt_id = gt_indices[max_idx.item()]

        if max_iou >= iou_threshold and gt_id not in gt_matched[img_id]:
            tp[d_idx] = 1
            gt_matched[img_id].add(gt_id)
        else:
            fp[d_idx] = 1

    # 计算累积
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recall = tp_cumsum / max(len(all_gts), 1)
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-10)

    # 插值到固定点数
    ap = compute_ap(precision, recall)

    return {
        "mAP@0.5": float(ap),
        "num_images": len(all_predictions),
        "num_gt": len(all_gts),
        "num_pred": len(all_detections),
    }


def compute_iou_torch(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """计算两组框之间的 IoU (同 detector.py 中实现)"""
    lt = torch.max(box_a[:, None, :2], box_b[:, :2])
    rb = torch.min(box_a[:, None, 2:], box_b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    union = area_a[:, None] + area_b - inter

    return inter / union.clamp(min=1e-10)
