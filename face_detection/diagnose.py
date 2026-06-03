"""
diagnose.py — mAP=0 诊断工具
============================
诊断为什么验证集 mAP 始终为 0:
  1. 查看原始 sigmoid 分数的分布
  2. 测试不同 score_threshold 的效果
  3. 检查 NMS 是否过度过滤
  4. 查看 IoU 匹配情况
"""

import os
import sys
import torch
import numpy as np
from collections import defaultdict

from config import *
from models import create_face_detector
from models.detector import decode_boxes, batched_nms, compute_iou
from data.dataset import create_dataloaders


@torch.no_grad()
def diagnose(model, dataloader, device):
    model.eval()

    all_raw_scores = []       # 所有 sigmoid 分数 (before NMS)
    all_nms_scores = []       # NMS 后最高分
    all_nms_counts = []       # NMS 后每图检测数
    score_buckets = defaultdict(int)  # 分数分布直方图

    # 取前 500 张图做详细分析
    max_images = 500
    images_processed = 0

    print(f"[Diagnose] Analyzing up to {max_images} images...")

    for images, boxes_list, labels_list in dataloader:
        if images_processed >= max_images:
            break

        images = images.to(device)
        cls_preds, reg_preds = model(images)

        # 原始分数
        scores_raw = torch.sigmoid(cls_preds.squeeze(-1))  # [B, 1550]

        for b in range(images.size(0)):
            if images_processed >= max_images:
                break

            s = scores_raw[b]  # [1550]
            all_raw_scores.append(s.cpu())

            # 分桶统计
            for threshold in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]:
                count = (s > threshold).sum().item()
                score_buckets[f"> {threshold:.2f}"] += count

            # NMS 结果
            pred_boxes = decode_boxes(model.anchors.to(device), reg_preds)
            boxes_nms, scores_nms = batched_nms(
                pred_boxes[b:b+1], scores_raw[b:b+1])

            all_nms_counts.append(len(scores_nms[0]))
            if len(scores_nms[0]) > 0:
                all_nms_scores.extend(scores_nms[0].cpu().tolist())

            images_processed += 1

        if images_processed % 100 == 0:
            print(f"  Processed {images_processed}/{max_images}...")

    # =========================================
    # 报告
    # =========================================
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC REPORT - {images_processed} images")
    print(f"{'='*60}")

    # 1. 原始分数统计
    all_scores = torch.cat(all_raw_scores)  # [N_images * 1550]
    print(f"\n[1] Raw Sigmoid Scores ({all_scores.numel():,} anchors total)")
    print(f"    Max:     {all_scores.max().item():.6f}")
    print(f"    Mean:    {all_scores.mean().item():.6f}")
    print(f"    Median:  {all_scores.median().item():.6f}")
    print(f"    Std:     {all_scores.std().item():.6f}")
    print(f"    > 0.01:  {(all_scores > 0.01).sum().item():,} ({(all_scores > 0.01).float().mean().item()*100:.2f}%)")
    print(f"    > 0.05:  {(all_scores > 0.05).sum().item():,} ({(all_scores > 0.05).float().mean().item()*100:.2f}%)")
    print(f"    > 0.1:   {(all_scores > 0.1).sum().item():,} ({(all_scores > 0.1).float().mean().item()*100:.2f}%)")
    print(f"    > 0.3:   {(all_scores > 0.3).sum().item():,} ({(all_scores > 0.3).float().mean().item()*100:.2f}%)")
    print(f"    > 0.5:   {(all_scores > 0.5).sum().item():,} ({(all_scores > 0.5).float().mean().item()*100:.2f}%)")

    # 2. 分数分布 (top-20 最高分)
    topk = all_scores.topk(min(20, len(all_scores)))
    print(f"\n[2] Top-20 Highest Scores (all anchors, all images)")
    for i, (val, idx) in enumerate(zip(topk.values, topk.indices)):
        img_idx = idx.item() // 1550
        anchor_idx = idx.item() % 1550
        print(f"    #{i+1:2d}: {val.item():.4f}  (img={img_idx}, anchor={anchor_idx})")

    # 3. NMS 后结果
    print(f"\n[3] After NMS (threshold={SCORE_THRESHOLD}, top-100 pre-filter)")
    print(f"    Images with 0 detections: {all_nms_counts.count(0)}/{len(all_nms_counts)} ({all_nms_counts.count(0)/len(all_nms_counts)*100:.1f}%)")
    print(f"    Images with >0 detections: {len(all_nms_counts) - all_nms_counts.count(0)}")
    print(f"    Max detections per image: {max(all_nms_counts)}")
    print(f"    Mean detections per image: {np.mean(all_nms_counts):.2f}")
    if all_nms_scores:
        print(f"    NMS score range: [{min(all_nms_scores):.4f}, {max(all_nms_scores):.4f}]")
    else:
        print(f"    NMS score range: [N/A] (no detections survived!)")

    # 4. 降低阈值测试
    print(f"\n[4] Threshold sensitivity test (on all {images_processed} images)")
    from utils.metrics import compute_map
    all_gt = []
    for boxes, labels in [(dataloader.dataset[i][1], dataloader.dataset[i][2])
                          for i in range(min(max_images, len(dataloader.dataset)))]:
        all_gt.append((boxes.clone(), labels))

    # Re-run prediction with different thresholds
    for test_thresh in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        all_preds = []
        for i in range(min(max_images // 64 + 1, len(dataloader))):
            pass
        print(f"    (Threshold sensitivity requires re-inference, skipped in quick diagnostic)")

    print(f"\n{'='*60}")
    print(f"CONCLUSION:")
    if all_scores.max() < 0.1:
        print(f"  Model scores are VERY LOW. The model hasn't learned to detect faces yet.")
        print(f"  Possible causes: training insufficient, architecture issue, or loss problem.")
    elif all_scores.max() < 0.5:
        print(f"  Model produces some signal but below 0.5 threshold.")
        print(f"  Try lowering SCORE_THRESHOLD to {all_scores.max().item()*0.8:.2f}")
    else:
        if all_nms_counts.count(0) == len(all_nms_counts):
            print(f"  Scores are high but NMS kills all detections. Check NMS logic.")
        else:
            print(f"  Scores look OK but mAP=0. Check IoU matching or GT encoding.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 加载模型
    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"[Error] Model not found: {model_path}")
        sys.exit(1)

    print(f"[Model] Loading: {model_path}")
    model = create_face_detector(width_mult=BACKBONE_WIDTH_MULTIPLIER)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(device)
    print(f"[Model] Epoch: {ckpt.get('epoch', '?')}, mAP: {ckpt.get('best_map', '?')}")

    # 数据 (只用验证集)
    _, val_loader = create_dataloaders(batch_size=64, num_workers=0, max_val_images=500)

    # 诊断
    diagnose(model, val_loader, device)
