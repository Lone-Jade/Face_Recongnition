"""
train.py — 全精度训练脚本
==========================
训练轻量级人脸检测模型, 包含:
  - 自动创建 DataLoader
  - 完整训练循环 (warmup + cosine annealing)
  - 每 epoch 验证 (mAP + loss)
  - 自动保存最佳模型
  - 训练曲线输出
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无 GUI 后端
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import *
from models import create_face_detector, get_model_info
from models.detector import decode_boxes, batched_nms
from utils.loss import MultiBoxLoss
from utils.metrics import AverageMeter, compute_map
from data.dataset import create_dataloaders


# ============================================================
# 训练脚本主体
# ============================================================
def train_one_epoch(
    model, dataloader, criterion, optimizer, device, epoch, total_epochs
):
    """训练一个 epoch"""
    model.train()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()

    for batch_idx, (images, boxes_list, labels_list) in enumerate(dataloader):
        images = images.to(device)

        # 前向传播
        cls_preds, reg_preds = model(images)

        # 将 GT 框归一化到 [0, 1]
        gt_boxes_norm = []
        for boxes in boxes_list:
            if boxes.numel() > 0:
                boxes_norm = boxes.clone()
                boxes_norm[:, [0, 2]] /= IMAGE_SIZE  # x, w
                boxes_norm[:, [1, 3]] /= IMAGE_SIZE  # y, h
                gt_boxes_norm.append(boxes_norm.to(device))
            else:
                gt_boxes_norm.append(torch.zeros((0, 4), device=device))

        # 损失计算
        anchors = model.anchors.to(device)
        loss, loss_info = criterion(
            cls_preds, reg_preds, anchors, gt_boxes_norm, labels_list
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # 更新统计
        loss_meter.update(loss_info["total_loss"], images.size(0))
        cls_loss_meter.update(loss_info["cls_loss"], images.size(0))
        reg_loss_meter.update(loss_info["reg_loss"], images.size(0))

        if batch_idx % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch [{epoch}/{total_epochs}] "
                f"Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss_meter.val:.4f} "
                f"Avg: {loss_meter.avg:.4f} "
                f"LR: {lr:.6f}"
            )

    return {
        "loss": loss_meter.avg,
        "cls_loss": cls_loss_meter.avg,
        "reg_loss": reg_loss_meter.avg,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """在验证集上评估"""
    model.eval()
    loss_meter = AverageMeter()

    all_predictions = []
    all_groundtruths = []

    for images, boxes_list, labels_list in dataloader:
        images = images.to(device)

        # 损失
        cls_preds, reg_preds = model(images)
        gt_boxes_norm = []
        for boxes in boxes_list:
            if boxes.numel() > 0:
                boxes_norm = boxes.clone()
                boxes_norm[:, [0, 2]] /= IMAGE_SIZE
                boxes_norm[:, [1, 3]] /= IMAGE_SIZE
                gt_boxes_norm.append(boxes_norm.to(device))
            else:
                gt_boxes_norm.append(torch.zeros((0, 4), device=device))

        anchors = model.anchors.to(device)
        _, loss_info = criterion(
            cls_preds, reg_preds, anchors, gt_boxes_norm, labels_list
        )
        loss_meter.update(loss_info["total_loss"], images.size(0))

        # 推理预测 (用于 mAP 计算)
        scores = torch.sigmoid(cls_preds.squeeze(-1))
        pred_boxes = decode_boxes(model.anchors.to(device), reg_preds)
        pred_boxes_list, pred_scores_list = batched_nms(pred_boxes, scores)

        for b in range(images.size(0)):
            # 预测结果 (反归一化到像素坐标)
            pred_boxes_abs = pred_boxes_list[b] * IMAGE_SIZE
            all_predictions.append((pred_boxes_abs, pred_scores_list[b]))

            # GT 结果 (像素坐标)
            gt_boxes_abs = boxes_list[b].clone()
            all_groundtruths.append((gt_boxes_abs, labels_list[b]))

    # 计算 mAP
    map_results = compute_map(all_predictions, all_groundtruths)

    return {
        "loss": loss_meter.avg,
        "mAP": map_results["mAP@0.5"],
        "num_gt": map_results["num_gt"],
    }


def plot_training_curves(history: dict, save_path: str):
    """绘制训练曲线并保存"""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # mAP
    axes[1].plot(epochs, history["val_map"], "g-", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP@0.5")
    axes[1].set_title("Validation mAP")
    axes[1].grid(True)

    # LR Schedule
    axes[2].plot(epochs, history["lr"], "m-")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_yscale("log")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"[Plot] Training curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Face Detector")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument(
        "--lr", type=float, default=BASE_LR, help="Initial learning rate"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: auto / cuda / cpu"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Max training images (for quick testing)",
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=None,
        help="Max validation images (for quick testing)",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Device] Using: {device}")

    # 创建模型
    model = create_face_detector(width_mult=BACKBONE_WIDTH_MULTIPLIER)
    model = model.to(device)

    info = get_model_info(model)
    print(f"[Model] Total params: {info['total_params']:,}")
    print(f"[Model] FP32 size: ~{info['fp32_weight_size_kb']:.1f} KB")
    print(f"[Model] INT8 size: ~{info['int8_weight_size_kb']:.1f} KB")
    print(f"[Model] Num anchors: {info['num_anchors']}")

    # 数据加载
    print("\n[Data] Loading WIDER Face dataset...")
    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_images=args.max_train,
        max_val_images=args.max_val,
    )
    print(f"[Data] Train batches: {len(train_loader)}")
    print(f"[Data] Val batches:   {len(val_loader)}")

    # 损失函数
    criterion = MultiBoxLoss()

    # 优化器
    if OPTIMIZER.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
        )
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
        )

    # 学习率调度: Linear Warmup → Cosine Annealing
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=LR_WARMUP_EPOCHS
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - LR_WARMUP_EPOCHS, eta_min=MIN_LR
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[LR_WARMUP_EPOCHS],
    )

    # 恢复训练
    start_epoch = 1
    best_map = 0.0
    history = {"train_loss": [], "val_loss": [], "val_map": [], "lr": []}

    if args.resume and os.path.exists(args.resume):
        print(f"[Resume] Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_map = checkpoint.get("best_map", 0.0)
        history = checkpoint.get("history", history)
        print(f"[Resume] Resuming from epoch {start_epoch}")

    # ============================================================
    # 训练循环
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs, {args.batch_size} batch_size")
    print(f"Initial LR: {args.lr}, Optimizer: {OPTIMIZER}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )

        # 验证 (每 5 个 epoch 或第一个/最后一个)
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            val_metrics = validate(model, val_loader, criterion, device)
        else:
            val_metrics = {"loss": 0.0, "mAP": 0.0}

        # 学习率更新
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # 记录历史
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_map"].append(val_metrics["mAP"])
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        # 日志输出
        if epoch % 5 == 0 or epoch == 1:
            print(f"\n{'='*40}")
            print(f"Epoch {epoch}/{args.epochs} completed in {epoch_time:.1f}s")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Val mAP:    {val_metrics['mAP']:.4f}")
            print(f"  LR:         {current_lr:.6f}")
            print(f"{'='*40}\n")

        # 保存最佳模型
        if val_metrics["mAP"] > best_map:
            best_map = val_metrics["mAP"]
            best_path = os.path.join(MODEL_DIR, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_map": best_map,
                    "history": history,
                    "config": {
                        "image_size": IMAGE_SIZE,
                        "width_mult": BACKBONE_WIDTH_MULTIPLIER,
                        "num_classes": 1,
                    },
                },
                best_path,
            )
            print(f"  ✓ Best model saved (mAP={best_map:.4f}) → {best_path}")

        # 定期保存 checkpoint (每 30 epoch)
        if epoch % 30 == 0:
            ckpt_path = os.path.join(MODEL_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_map": best_map,
                    "history": history,
                },
                ckpt_path,
            )
            print(f"  Checkpoint saved → {ckpt_path}")

    # ============================================================
    # 训练结束: 保存最终模型 + 绘制曲线
    # ============================================================
    final_path = os.path.join(MODEL_DIR, "final_model.pth")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "best_map": best_map,
            "history": history,
            "config": {
                "image_size": IMAGE_SIZE,
                "width_mult": BACKBONE_WIDTH_MULTIPLIER,
                "num_classes": 1,
            },
        },
        final_path,
    )
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best mAP:     {best_map:.4f}")
    print(f"  Final model:  {final_path}")
    print(f"  Best model:   {MODEL_DIR}/best_model.pth")

    # 绘制训练曲线
    plot_training_curves(history, os.path.join(OUTPUT_DIR, "training_curves.png"))

    # 保存历史记录为 JSON
    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History:      {history_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
