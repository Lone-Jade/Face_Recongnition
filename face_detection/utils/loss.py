"""
loss.py — 多任务损失函数 (MultiBox Loss)
==========================================
SSD 风格损失:
  - 分类损失: Focal Loss (处理正负样本极度不平衡)
  - 回归损失: Smooth L1 Loss (仅对正样本计算)

锚框匹配策略:
  1. 对每个 GT 框, 选择与其 IoU 最大的锚框 (保证至少匹配一次)
  2. 对每个锚框, 如果与任意 GT 框 IoU > 0.5, 标记为正样本
  3. Hard Negative Mining: 负样本按置信度排序, 取 top-K (NEG_POS_RATIO × 正样本数)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from config import NEG_POS_RATIO, LOSS_CLS_WEIGHT, LOSS_REG_WEIGHT
from models.detector import compute_iou


def encode_boxes(anchors: torch.Tensor, gt_boxes: torch.Tensor,
                 variance: Tuple[float, float] = (0.1, 0.2)) -> torch.Tensor:
    """
    将 GT 框编码为相对于锚框的偏移量 (训练目标)。

    Args:
        anchors:  [N, 4] 锚框 [cx, cy, w, h]
        gt_boxes: [N, 4] GT 框 [cx, cy, w, h] (匹配后与 anchors 一一对应)
        variance: 编码方差

    Returns:
        targets: [N, 4] 编码偏移 [Δcx, Δcy, Δw, Δh]
    """
    # 中心偏移
    g_cx = (gt_boxes[:, 0] - anchors[:, 0]) / (anchors[:, 2] * variance[0])
    g_cy = (gt_boxes[:, 1] - anchors[:, 1]) / (anchors[:, 3] * variance[0])

    # 宽高偏移 (对数尺度)
    g_w = torch.log(gt_boxes[:, 2] / anchors[:, 2].clamp(min=1e-10)) / variance[1]
    g_h = torch.log(gt_boxes[:, 3] / anchors[:, 3].clamp(min=1e-10)) / variance[1]

    return torch.stack([g_cx, g_cy, g_w, g_h], dim=-1)


def xywh_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """[x, y, w, h] → [cx, cy, w, h]"""
    result = boxes.clone()
    result[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
    return result


class MultiBoxLoss(nn.Module):
    """
    SSD MultiBox Loss (Focal Loss + Smooth L1)
    ==========================================

    对每个锚框:
      - 与 GT 匹配 → 正样本 (label=1), 计算分类 + 回归损失
      - 与 GT 不匹配 → 负样本 (label=0), 仅计算分类损失 (hard mining)

    分类使用 Focal Loss (γ=2, α=0.25) 解决正负样本不平衡。
    回归使用 Smooth L1, 仅对正样本。
    """

    def __init__(self, neg_pos_ratio: float = NEG_POS_RATIO,
                 cls_weight: float = LOSS_CLS_WEIGHT,
                 reg_weight: float = LOSS_REG_WEIGHT,
                 overlap_thresh: float = 0.5):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.overlap_thresh = overlap_thresh

    def forward(self, cls_preds: torch.Tensor, reg_preds: torch.Tensor,
                anchors: torch.Tensor, gt_boxes_list: list,
                gt_labels_list: list) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            cls_preds:  [B, N, 1] 分类预测 (原始 logits)
            reg_preds:  [B, N, 4] 回归预测 (编码偏移)
            anchors:    [N, 4]    锚框 [cx, cy, w, h] (归一化)
            gt_boxes_list: List of [num_gt_i, 4]   GT 框 [x, y, w, h] (归一化)
            gt_labels_list: List of [num_gt_i]      GT 标签 (全为 1=face)

        Returns:
            total_loss: 标量总损失
            loss_info:  dict 各项损失详情 (用于日志)
        """
        batch_size = cls_preds.shape[0]
        cls_losses = []
        reg_losses = []
        num_pos_total = 0

        for b in range(batch_size):
            gt_boxes = gt_boxes_list[b]   # [M, 4] xywh
            if gt_boxes.numel() == 0:
                # 无 GT 框的图片: 所有锚框都是负样本
                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_preds[b].squeeze(-1),
                    torch.zeros_like(cls_preds[b].squeeze(-1)))
                cls_losses.append(cls_loss)
                reg_losses.append(torch.tensor(0.0, device=cls_preds.device))
                continue

            # 转换 GT 为 [cx, cy, w, h]
            gt_cxcywh = xywh_to_cxcywh(gt_boxes)

            # 1. 锚框匹配
            pos_mask, matched_gt = self._match_anchors(anchors, gt_cxcywh)

            num_pos = pos_mask.sum().item()
            num_pos_total += max(num_pos, 1)

            # 2. 分类损失 (Focal Loss)
            cls_target = pos_mask.float().unsqueeze(-1)  # [N, 1]
            cls_loss = self._focal_loss(
                cls_preds[b], cls_target, alpha=0.25, gamma=2.0)

            # 3. Hard Negative Mining
            if num_pos > 0:
                neg_mask = ~pos_mask
                cls_loss_neg = self._focal_loss(
                    cls_preds[b][neg_mask],
                    torch.zeros(neg_mask.sum(), 1, device=cls_preds.device),
                    alpha=0.25, gamma=2.0)

                # 取负样本中损失最大的 top-K
                max_neg = min(int(num_pos * self.neg_pos_ratio), neg_mask.sum().item())
                if max_neg > 0:
                    top_neg_loss, _ = cls_loss_neg.view(-1).topk(max_neg)
                    cls_loss_neg = top_neg_loss.mean()
                else:
                    cls_loss_neg = torch.tensor(0.0, device=cls_preds.device)

                cls_pos_loss = self._focal_loss(
                    cls_preds[b][pos_mask],
                    torch.ones(num_pos, 1, device=cls_preds.device),
                    alpha=0.25, gamma=2.0)

                final_cls_loss = (cls_pos_loss.mean() + cls_loss_neg) / 2
            else:
                final_cls_loss = cls_loss.mean()

            cls_losses.append(final_cls_loss)

            # 4. 回归损失 (Smooth L1, 仅正样本)
            if num_pos > 0:
                reg_target = encode_boxes(anchors[pos_mask], matched_gt[pos_mask])
                reg_loss = F.smooth_l1_loss(reg_preds[b][pos_mask], reg_target,
                                            beta=1.0 / 9, reduction='mean')
                reg_losses.append(reg_loss)
            else:
                reg_losses.append(torch.tensor(0.0, device=cls_preds.device))

        # 汇总
        cls_loss_total = torch.stack(cls_losses).mean()
        reg_loss_total = torch.stack(reg_losses).mean()
        total_loss = self.cls_weight * cls_loss_total + self.reg_weight * reg_loss_total

        return total_loss, {
            "cls_loss": cls_loss_total.item(),
            "reg_loss": reg_loss_total.item(),
            "total_loss": total_loss.item(),
            "num_pos": num_pos_total / batch_size,
        }

    @torch.no_grad()
    def _match_anchors(self, anchors: torch.Tensor,
                       gt_boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        锚框与 GT 框的匹配。

        Returns:
            pos_mask:   [N]     布尔张量, 正样本锚框
            matched_gt: [N, 4]  每个锚框对应的 GT 框 (负样本对应的值未定义)
        """
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        if num_gt == 0:
            pos_mask = torch.zeros(num_anchors, dtype=torch.bool,
                                   device=anchors.device)
            matched_gt = torch.zeros(num_anchors, 4, device=anchors.device)
            return pos_mask, matched_gt

        # 计算 IoU 矩阵 [num_anchors, num_gt]
        anchors_xyxy = torch.stack([
            anchors[:, 0] - anchors[:, 2] / 2,
            anchors[:, 1] - anchors[:, 3] / 2,
            anchors[:, 0] + anchors[:, 2] / 2,
            anchors[:, 1] + anchors[:, 3] / 2,
        ], dim=-1)

        gt_xyxy = torch.stack([
            gt_boxes[:, 0] - gt_boxes[:, 2] / 2,
            gt_boxes[:, 1] - gt_boxes[:, 3] / 2,
            gt_boxes[:, 0] + gt_boxes[:, 2] / 2,
            gt_boxes[:, 1] + gt_boxes[:, 3] / 2,
        ], dim=-1)

        iou = compute_iou(anchors_xyxy, gt_xyxy)

        # 最佳匹配
        best_gt_iou, best_gt_idx = iou.max(dim=1)  # [N]

        # 正样本: IoU > threshold
        pos_mask = best_gt_iou > self.overlap_thresh

        # 为每个 GT 框至少匹配一个锚框 (best match anchor)
        best_anchor_per_gt = iou.max(dim=0).indices  # [num_gt]
        pos_mask[best_anchor_per_gt] = True

        matched_gt = gt_boxes[best_gt_idx]

        return pos_mask, matched_gt

    def _focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor,
                    alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """
        Focal Loss for binary classification.

        FL(pt) = -α_t * (1 - p_t)^γ * log(p_t)

        Args:
            inputs:  [N, 1] 预测 logits
            targets: [N, 1] 目标标签 (0 or 1)
            alpha:   class balance 因子
            gamma:   focusing 参数

        Returns:
            loss: [N, 1]
        """
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - pt) ** gamma
        return focal_weight * bce
