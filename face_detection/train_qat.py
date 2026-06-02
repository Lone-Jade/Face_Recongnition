"""
train_qat.py — 量化感知训练 (Quantization-Aware Training)
===========================================================
在 INT8 量化约束下微调模型，最小化量化精度损失。

流程:
  1. 加载全精度最佳模型
  2. 插入 FakeQuantization 节点 (模拟 INT8 推理)
  3. 微调 10-20 epochs
  4. 保存 QAT 后模型 (可导出为 INT8 ONNX/TFLite)

目标:
  - 量化后 mAP 下降 < 2%
  - 模型大小 ~150KB (INT8)
"""

import os
import sys
import time
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.ao.quantization import (
    prepare_qat,
    convert,
    get_default_qat_qconfig,
    default_observer,
    default_weight_observer,
    QConfig,
)

from config import *
from models import create_face_detector, get_model_info
from models.detector import decode_boxes, batched_nms
from utils.loss import MultiBoxLoss
from utils.metrics import AverageMeter, compute_map
from data.dataset import create_dataloaders


# ============================================================
# QAT 准备: 配置 FakeQuantization
# ============================================================
def prepare_model_for_qat(model: nn.Module) -> nn.Module:
    """
    为模型插入 QAT 节点。

    PyTorch QAT 流程:
      1. 设置 qconfig (观察器 + 伪量化器)
      2. torch.ao.quantization.prepare_qat(model)
      3. 进行 QAT 训练
      4. torch.ao.quantization.convert(model) → INT8 模型

    注意:
      - 检测头最后的 Conv 层不量化 (输出精度敏感)
      - BatchNorm 与 Conv 融合后观察
    """

    # 配置 INT8 量化方案
    backend = QAT_CONFIG.get("backend", "fbgemm")
    qconfig = get_default_qat_qconfig(backend)

    # 或使用自定义 qconfig (更精细控制)
    # qconfig = QConfig(
    #     activation=default_observer.with_args(
    #         quant_min=0, quant_max=127, dtype=torch.quint8),
    #     weight=default_weight_observer.with_args(
    #         quant_min=-128, quant_max=127, dtype=torch.qint8),
    # )

    # 递归设置 qconfig
    def _set_qconfig(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                # 最后的检测头 Conv 不量化 (保持 FP32 精度)
                if any(
                    skip in full_name
                    for skip in [
                        "cls_conv",
                        "reg_conv",  # 检测头
                    ]
                ):
                    continue
                child.qconfig = qconfig
            elif isinstance(child, (nn.ReLU, nn.ReLU6)):
                child.qconfig = qconfig
            elif isinstance(child, nn.BatchNorm2d):
                # BN 在 QAT 中会与前面的 Conv 融合
                child.qconfig = qconfig
            _set_qconfig(child, full_name)

    _set_qconfig(model)

    # 全局设置默认 qconfig (未明确设置的模块)
    model.qconfig = qconfig

    # 插入伪量化观测器
    torch.ao.quantization.prepare_qat(model, inplace=True)

    return model


def calibrate_model(model: nn.Module, dataloader, device, num_batches: int = 10):
    """
    校准量化参数 (收集激活值范围)。
    在 QAT 训练前运行以加快收敛。
    """
    model.eval()
    print(f"[Calibrate] Running {num_batches} batches for quantization calibration...")

    with torch.no_grad():
        for batch_idx, (images, _, _) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            _ = model(images)

    print("[Calibrate] Done.")


# ============================================================
# QAT 训练主循环
# ============================================================
def train_qat_epoch(
    model, dataloader, criterion, optimizer, device, epoch, total_epochs
):
    """QAT 训练一个 epoch"""
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, (images, boxes_list, labels_list) in enumerate(dataloader):
        images = images.to(device)

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

        loss, loss_info = criterion(
            cls_preds, reg_preds, model.anchors.to(device), gt_boxes_norm, labels_list
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        loss_meter.update(loss_info["total_loss"], images.size(0))

        if batch_idx % 50 == 0:
            print(
                f"  QAT Epoch [{epoch}/{total_epochs}] "
                f"Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss_meter.val:.4f}"
            )

    return loss_meter.avg


@torch.no_grad()
def validate_qat(model, dataloader, criterion, device):
    """QAT 验证 (同全精度验证逻辑)"""
    model.eval()
    loss_meter = AverageMeter()
    all_predictions = []
    all_groundtruths = []

    for images, boxes_list, labels_list in dataloader:
        images = images.to(device)

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

        _, loss_info = criterion(
            cls_preds, reg_preds, model.anchors.to(device), gt_boxes_norm, labels_list
        )
        loss_meter.update(loss_info["total_loss"], images.size(0))

        scores = torch.sigmoid(cls_preds.squeeze(-1))
        pred_boxes = decode_boxes(model.anchors.to(device), reg_preds)
        pred_boxes_list, pred_scores_list = batched_nms(pred_boxes, scores)

        for b in range(images.size(0)):
            all_predictions.append(
                (pred_boxes_list[b] * IMAGE_SIZE, pred_scores_list[b])
            )
            all_groundtruths.append((boxes_list[b].clone(), labels_list[b]))

    map_results = compute_map(all_predictions, all_groundtruths)
    return loss_meter.avg, map_results["mAP@0.5"]


def main():
    parser = argparse.ArgumentParser(description="QAT Fine-tuning")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=os.path.join(MODEL_DIR, "best_model.pth"),
        help="Path to FP32 pretrained model",
    )
    parser.add_argument(
        "--epochs", type=int, default=QAT_EPOCHS, help="QAT fine-tuning epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="QAT learning rate (smaller than FP training)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (smaller for QAT stability)",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-calibrate", action="store_true")
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Device] Using: {device}")

    # 加载预训练模型
    if not os.path.exists(args.pretrained):
        print(f"[Error] Pretrained model not found: {args.pretrained}")
        print("        Run train.py first.")
        sys.exit(1)

    print(f"[Model] Loading pretrained: {args.pretrained}")
    model = create_face_detector(width_mult=BACKBONE_WIDTH_MULTIPLIER)
    checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model = model.to(device)

    fp32_info = get_model_info(model)
    print(
        f"[Model] FP32 params: {fp32_info['total_params']:,} "
        f"({fp32_info['fp32_weight_size_kb']:.1f} KB)"
    )

    # 数据
    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_images=args.max_train,
        max_val_images=args.max_val,
    )

    # 评估 FP32 基线
    print("\n[Eval] FP32 baseline:")
    criterion = MultiBoxLoss()
    fp32_loss, fp32_map = validate_qat(model, val_loader, criterion, device)
    print(f"  Val Loss: {fp32_loss:.4f}")
    print(f"  mAP@0.5:  {fp32_map:.4f}")

    # 准备 QAT
    print("\n[QAT] Preparing model for quantization-aware training...")
    model = prepare_model_for_qat(model)
    model = model.to(device)

    # 校准 (收集激活值统计)
    if not args.skip_calibrate:
        calibrate_model(model, train_loader, device)

    # QAT 优化器 (低学习率)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY * 0.1
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # QAT 训练
    print(f"\n{'='*60}")
    print(f"QAT Fine-tuning: {args.epochs} epochs, LR={args.lr}")
    print(f"{'='*60}\n")

    best_qat_map = 0.0
    for epoch in range(1, args.epochs + 1):
        qat_loss = train_qat_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        scheduler.step()

        qat_val_loss, qat_val_map = validate_qat(model, val_loader, criterion, device)

        print(
            f"  QAT Epoch {epoch}: Loss={qat_loss:.4f}, "
            f"Val Loss={qat_val_loss:.4f}, mAP={qat_val_map:.4f}"
        )

        if qat_val_map > best_qat_map:
            best_qat_map = qat_val_map
            # 保存 QAT 状态模型
            qat_path = os.path.join(MODEL_DIR, "best_model_qat.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "fp32_map": fp32_map,
                    "qat_map": qat_val_map,
                    "config": {
                        "image_size": IMAGE_SIZE,
                        "width_mult": BACKBONE_WIDTH_MULTIPLIER,
                    },
                },
                qat_path,
            )

        if epoch == args.epochs:
            qat_path = os.path.join(MODEL_DIR, "best_model_qat.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "fp32_map": fp32_map,
                    "qat_map": qat_val_map,
                },
                qat_path,
            )

    # 转换为 INT8 模型
    print("\n[Convert] Converting QAT model to INT8...")
    model.eval()
    model_int8 = torch.ao.quantization.convert(model, inplace=False)

    # 保存 INT8 模型
    int8_path = os.path.join(MODEL_DIR, "model_int8.pth")
    torch.save(model_int8.state_dict(), int8_path)

    # 统计
    int8_params = sum(p.numel() for p in model_int8.parameters())
    int8_size_kb = int8_params  # INT8: 1 byte/param
    print(f"[Result] INT8 model params: {int8_params:,} (~{int8_size_kb:.1f} KB)")

    # 精度对比
    map_drop = fp32_map - qat_val_map
    print(f"\n{'='*60}")
    print(f"QAT Complete!")
    print(f"  FP32 mAP:   {fp32_map:.4f}")
    print(f"  QAT mAP:    {qat_val_map:.4f}")
    print(
        f"  mAP delta:  {map_drop:.4f} ({'+' if map_drop < 0 else ''}{map_drop*100:.1f}%)"
    )
    print(f"  INT8 model: {int8_path}")
    print(f"  QAT model:  {MODEL_DIR}/best_model_qat.pth")
    print(f"  Target:     mAP drop < 2%")
    status = "✓ PASS" if abs(map_drop) < 0.02 else "✗ FAIL (need more QAT epochs)"
    print(f"  Status:     {status}")
    print(f"{'='*60}\n")

    # 保存报告
    report = {
        "fp32_map": fp32_map,
        "qat_map": qat_val_map,
        "map_delta": map_drop,
        "fp32_params": fp32_info["total_params"],
        "int8_params": int8_params,
        "fp32_size_kb": fp32_info["fp32_weight_size_kb"],
        "int8_size_kb": int8_size_kb,
    }
    with open(os.path.join(OUTPUT_DIR, "qat_report.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
