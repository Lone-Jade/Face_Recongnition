"""
dataset.py — WIDER Face 数据集加载器
=====================================
解析 wider_face_train_bbx_gt.txt / wider_face_val_bbx_gt.txt
提供 PyTorch Dataset 接口, 支持:
  - 增量解析 (支持超大数据集)
  - 按难度级别过滤 (Easy / Medium / Hard)
  - 在线数据增强
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict

from config import (WIDER_TRAIN_IMG, WIDER_VAL_IMG,
                    WIDER_TRAIN_ANN, WIDER_VAL_ANN,
                    IMAGE_SIZE, BATCH_SIZE, DIFFICULTY)
from utils.augment import TrainAugmentation, ValTransform


# ============================================================
# WIDER Face 注解解析
# ============================================================
def parse_wider_annotation(ann_path: str,
                           img_dir: str,
                           difficulty: str = "easy_medium",
                           max_images: int = None) -> List[Dict]:
    """
    解析 WIDER Face 标注文件, 返回图片列表。

    每张图片的标注格式:
    {
        'path': str,              # 图片完整路径
        'boxes': [[x, y, w, h], ...],   # 边界框 (绝对像素)
        'num_faces': int,          # 人脸数量
    }

    WIDER Face 标注文件格式:
    <图片路径>
    <人脸数量 N>
    <x y w h blur expression illumination invalid occlusion pose>  (×N 行)
    重复...

    Args:
        ann_path:    标注文件路径 (如 wider_face_train_bbx_gt.txt)
        img_dir:     图片根目录
        difficulty:  难度过滤 "all" / "easy" / "medium" / "hard" / "easy_medium"
        max_images:  最大图片数 (None=全部, 用于调试)

    Returns:
        samples: 图片标注列表
    """
    samples = []

    with open(ann_path, 'r') as f:
        lines = f.readlines()

    idx = 0
    total_lines = len(lines)
    skipped_no_face = 0
    skipped_small = 0

    while idx < total_lines:
        # 解析图片路径
        img_rel_path = lines[idx].strip()
        idx += 1

        if idx >= total_lines:
            break

        # 解析人脸数量
        try:
            num_faces = int(lines[idx].strip())
        except ValueError:
            # 可能是空行或格式异常
            idx += 1
            continue
        idx += 1

        img_full_path = os.path.join(img_dir, img_rel_path)

        boxes = []
        for face_idx in range(num_faces):
            if idx >= total_lines:
                break
            parts = lines[idx].strip().split()
            idx += 1

            if len(parts) < 4:
                continue

            x, y, w, h = map(int, parts[:4])
            blur = int(parts[4]) if len(parts) > 4 else 0

            # 按难度过滤
            keep = _filter_by_difficulty(w, h, blur, difficulty)
            if keep:
                boxes.append([x, y, w, h])

        # 跳过没有有效框的图片
        if len(boxes) == 0:
            skipped_no_face += 1
            if num_faces == 0:
                continue  # 原始标注就没有人脸的图片

        samples.append({
            'path': img_full_path,
            'boxes': boxes,
        })

        if max_images and len(samples) >= max_images:
            break

    print(f"[Dataset] Parsed {len(samples)} images from {ann_path}")
    print(f"          Skipped {skipped_no_face} images with no valid faces")
    print(f"          Difficulty filter: {difficulty}")
    return samples


def _filter_by_difficulty(w: int, h: int, blur: int,
                           difficulty: str) -> bool:
    """
    根据 WIDER Face 难度定义过滤脸孔。

    WIDER Face 难度定义:
      - Easy:   高度 > 50px, 清晰 (blur=0)
      - Medium: 高度 > 50px, 有遮挡/姿态变化 (blur 0 or 1)
      - Hard:   高度 > 10px, 极端遮挡/模糊

    我们的合并定义:
      - easy_medium: 接受 blur 0-1, 高度 > 30px (适应 160×160 输入)
      - all: 高度 > 16px (160×160 输入的 1/10)
    """
    if difficulty == "all":
        return h >= 16 and w >= 16

    if difficulty == "easy_medium":
        return h >= 30 and w >= 30 and blur <= 1

    if difficulty == "easy":
        return h >= 50 and w >= 50 and blur == 0

    if difficulty == "medium":
        return h >= 50 and w >= 50 and blur <= 1

    if difficulty == "hard":
        return h >= 10 and w >= 10

    # 默认接受所有
    return h >= 16 and w >= 16


# ============================================================
# PyTorch Dataset
# ============================================================
class WiderFaceDataset(Dataset):
    """
    WIDER Face PyTorch Dataset。

    返回格式:
      image: [C, H, W] 张量, float32, [0, 1]
      boxes: [N, 5]    张量 [x, y, w, h, label]
    """

    def __init__(self, ann_path: str, img_dir: str,
                 transform=None, difficulty: str = "easy_medium",
                 max_images: int = None):
        """
        Args:
            ann_path:   标注 txt 路径
            img_dir:    图片目录
            transform:  数据变换 (TrainAugmentation / ValTransform)
            difficulty: 难度过滤
            max_images: 最大图片数
        """
        self.img_dir = img_dir
        self.transform = transform

        self.samples = parse_wider_annotation(
            ann_path, img_dir, difficulty, max_images)

        print(f"[Dataset] {len(self.samples)} images ready")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图片
        image = cv2.imread(sample['path'])
        if image is None:
            # 偶尔有损坏的图片文件, 返回下一张
            print(f"[Warning] Failed to load: {sample['path']}")
            return self.__getitem__((idx + 1) % len(self))

        image = image.astype(np.float32)

        # 准备标注: [x, y, w, h] → [x, y, w, h, label]
        boxes = np.array(sample['boxes'], dtype=np.float32)
        if len(boxes) > 0:
            labels = np.ones((len(boxes), 1), dtype=np.float32)
            boxes = np.hstack([boxes, labels])
        else:
            boxes = np.zeros((0, 5), dtype=np.float32)

        # 数据变换
        if self.transform:
            image, boxes = self.transform(image, boxes)

        return image, boxes


# ============================================================
# DataLoader 工厂函数
# ============================================================
def create_dataloaders(batch_size: int = BATCH_SIZE,
                       num_workers: int = 4,
                       max_train_images: int = None,
                       max_val_images: int = None) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证 DataLoader。

    Args:
        batch_size:       批大小
        num_workers:      DataLoader 工作进程数
        max_train_images: 训练集最大图片数 (None=全部, 用于快速调试)
        max_val_images:   验证集最大图片数

    Returns:
        train_loader, val_loader
    """
    # 训练集
    train_transform = TrainAugmentation(size=IMAGE_SIZE)
    train_dataset = WiderFaceDataset(
        ann_path=WIDER_TRAIN_ANN,
        img_dir=WIDER_TRAIN_IMG,
        transform=train_transform,
        difficulty=DIFFICULTY,
        max_images=max_train_images,
    )

    # 验证集
    val_transform = ValTransform(size=IMAGE_SIZE)
    val_dataset = WiderFaceDataset(
        ann_path=WIDER_VAL_ANN,
        img_dir=WIDER_VAL_IMG,
        transform=val_transform,
        difficulty=DIFFICULTY,
        max_images=max_val_images,
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader


def _collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]
                ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    自定义 batch collate: 每张图的框数不同, 不使用默认 stack。

    Returns:
        images:        [B, C, H, W] 堆叠的图像张量
        boxes_list:    List of [num_boxes_i, 5]  每张图的框 (前4列: xywh, 第5列: label)
        labels_list:   List of [num_boxes_i]     每张图的标签
    """
    images = []
    boxes_list = []
    labels_list = []

    for img, boxes in batch:
        images.append(img)
        if len(boxes) > 0:
            boxes_list.append(boxes[:, :4])     # [x, y, w, h]
            labels_list.append(boxes[:, 4])     # label (全1 = face)
        else:
            boxes_list.append(torch.zeros((0, 4)))
            labels_list.append(torch.zeros(0))

    images = torch.stack(images, dim=0)

    return images, boxes_list, labels_list


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    print("Testing WIDER Face dataset loader...")
    print(f"Train annotation: {WIDER_TRAIN_ANN}")
    print(f"Train images:     {WIDER_TRAIN_IMG}")
    print(f"Val annotation:   {WIDER_VAL_ANN}")
    print(f"Val images:       {WIDER_VAL_IMG}")

    # 检查文件是否存在
    for path in [WIDER_TRAIN_ANN, WIDER_VAL_ANN]:
        if os.path.exists(path):
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ NOT FOUND: {path}")
            print(f"    Please download WIDER Face dataset first.")
            print(f"    See dataset_requirement.txt for instructions.")

    if os.path.exists(WIDER_TRAIN_ANN):
        # 快速测试: 只加载 100 张图
        loader, _ = create_dataloaders(
            batch_size=4, num_workers=0,
            max_train_images=100, max_val_images=10)

        images, boxes_list, labels_list = next(iter(loader))
        print(f"\nSample batch:")
        print(f"  Images shape: {images.shape}")
        print(f"  Boxes per image: {[b.shape[0] for b in boxes_list]}")
        print(f"  Image value range: [{images.min():.4f}, {images.max():.4f}]")
        print("  ✓ DataLoader OK")
