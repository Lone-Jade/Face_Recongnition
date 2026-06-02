"""
augment.py — 数据增强管道
==========================
训练增强:
  - 随机水平翻转
  - 随机裁剪 + 缩放
  - 色彩抖动 (亮度/对比度/饱和度/色调)
  - 随机旋转

验证变换:
  - Resize 到固定尺寸
  - 归一化
"""

import random
import cv2
import numpy as np
import torch
from torchvision import transforms as T
from typing import Tuple, List, Optional
from config import IMAGE_SIZE, AUGMENTATION


class TrainAugmentation:
    """
    训练数据增强。
    所有操作同步应用到图像和边界框。
    """

    def __init__(self, size: int = IMAGE_SIZE, cfg: dict = None):
        self.size = size
        self.cfg = cfg or AUGMENTATION

    def __call__(self, image: np.ndarray,
                 boxes: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: [H, W, C] BGR (OpenCV 格式)
            boxes: [N, 5] 每行 [x, y, w, h, label]

        Returns:
            image: [C, H, W] RGB 张量, 归一化 [0, 1]
            boxes: [N, 5] 变换后的框 (超出边界的已裁剪)
        """
        H, W = image.shape[:2]
        boxes_out = boxes.copy()

        # ---------- 随机翻转 ----------
        if self.cfg.get("random_flip", False) and random.random() < 0.5:
            image = cv2.flip(image, 1)  # 水平翻转
            # 更新 x 坐标
            boxes_out[:, 0] = W - boxes_out[:, 0] - boxes_out[:, 2]

        # ---------- 随机裁剪 + 缩放 ----------
        if self.cfg.get("random_crop", False):
            scale = random.uniform(*self.cfg.get("random_scale", (0.8, 1.2)))
            crop_h = int(H * scale)
            crop_w = int(W * scale)

            # 随机偏移
            max_dy = max(0, H - crop_h)
            max_dx = max(0, W - crop_w)
            dy = random.randint(0, max_dy) if max_dy > 0 else 0
            dx = random.randint(0, max_dx) if max_dx > 0 else 0

            image = image[dy:dy + crop_h, dx:dx + crop_w]
            boxes_out[:, 0] -= dx
            boxes_out[:, 1] -= dy

            # 缩放回固定尺寸
            image = cv2.resize(image, (self.size, self.size))
            scale_x = self.size / crop_w
            scale_y = self.size / crop_h
            boxes_out[:, 0] *= scale_x
            boxes_out[:, 1] *= scale_y
            boxes_out[:, 2] *= scale_x
            boxes_out[:, 3] *= scale_y

        # ---------- 色彩抖动 ----------
        color_cfg = self.cfg.get("color_jitter", {})
        if color_cfg:
            image = self._color_jitter(image, color_cfg)

        # ---------- 随机旋转 ----------
        if self.cfg.get("random_rotate", 0) > 0:
            angle = random.uniform(-self.cfg["random_rotate"],
                                   self.cfg["random_rotate"])
            if abs(angle) > 1.0:
                image, boxes_out = self._rotate(image, boxes_out, angle)

        # ---------- Resize 到固定尺寸 ----------
        if image.shape[0] != self.size or image.shape[1] != self.size:
            scale_x = self.size / image.shape[1]
            scale_y = self.size / image.shape[0]
            image = cv2.resize(image, (self.size, self.size))
            boxes_out[:, 0] *= scale_x
            boxes_out[:, 1] *= scale_y
            boxes_out[:, 2] *= scale_x
            boxes_out[:, 3] *= scale_y

        # ---------- 裁剪越界框 ----------
        boxes_out = self._clip_boxes(boxes_out, self.size, self.size)

        # ---------- 转换为张量 ----------
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        boxes_out = torch.from_numpy(boxes_out).float()

        return image, boxes_out

    def _color_jitter(self, image: np.ndarray, cfg: dict) -> np.ndarray:
        """随机色彩抖动 (HSV 空间)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # 亮度
        if cfg.get("brightness", 0) > 0:
            delta = random.uniform(-cfg["brightness"], cfg["brightness"]) * 255
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + delta, 0, 255)

        # 对比度
        if cfg.get("contrast", 0) > 0:
            factor = random.uniform(1 - cfg["contrast"], 1 + cfg["contrast"])
            hsv[:, :, 2] = np.clip(127 + factor * (hsv[:, :, 2] - 127), 0, 255)

        # 饱和度
        if cfg.get("saturation", 0) > 0:
            factor = random.uniform(1 - cfg["saturation"], 1 + cfg["saturation"])
            hsv[:, :, 1] = np.clip(factor * hsv[:, :, 1], 0, 255)

        # 色调
        if cfg.get("hue", 0) > 0:
            delta = random.uniform(-cfg["hue"], cfg["hue"]) * 180
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + delta, 0, 180)

        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return image

    def _rotate(self, image: np.ndarray, boxes: np.ndarray,
                angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """旋转图像+框"""
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h),
                               borderMode=cv2.BORDER_REPLICATE)

        if len(boxes) == 0:
            return image, boxes

        # 转换框的四个角点, 旋转后取外接矩形
        boxes_rotated = []
        for box in boxes:
            x, y, bw, bh = box[:4]
            corners = np.array([
                [x, y], [x + bw, y], [x, y + bh], [x + bw, y + bh]
            ])
            ones = np.ones((4, 1))
            corners = np.hstack([corners, ones])
            rotated = (M @ corners.T).T
            x1, y1 = rotated.min(axis=0)
            x2, y2 = rotated.max(axis=0)
            boxes_rotated.append([x1, y1, x2 - x1, y2 - y1] + list(box[4:]))

        return image, np.array(boxes_rotated)

    def _clip_boxes(self, boxes: np.ndarray, W: int, H: int) -> np.ndarray:
        """裁剪框到图像边界内, 移除无效框"""
        if len(boxes) == 0:
            return boxes

        # 裁剪
        boxes[:, 0] = np.clip(boxes[:, 0], 0, W)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, H)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, W - boxes[:, 0])
        boxes[:, 3] = np.clip(boxes[:, 3], 0, H - boxes[:, 1])

        # 移除过小的框 (面积 < 4px²)
        valid = (boxes[:, 2] >= 2) & (boxes[:, 3] >= 2)
        return boxes[valid]


class ValTransform:
    """验证/测试数据变换 (无数据增强)"""

    def __init__(self, size: int = IMAGE_SIZE):
        self.size = size

    def __call__(self, image: np.ndarray,
                 boxes: Optional[np.ndarray] = None
                 ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            image: [H, W, C] BGR
            boxes: [N, 5] 或 None

        Returns:
            image: [C, H, W] RGB 张量
            boxes: [N, 5] 或 None
        """
        H, W = image.shape[:2]

        # Resize
        image = cv2.resize(image, (self.size, self.size))

        if boxes is not None and len(boxes) > 0:
            scale_x = self.size / W
            scale_y = self.size / H
            boxes_out = boxes.copy()
            boxes_out[:, 0] *= scale_x
            boxes_out[:, 1] *= scale_y
            boxes_out[:, 2] *= scale_x
            boxes_out[:, 3] *= scale_y

            # 裁剪
            boxes_out[:, 0] = np.clip(boxes_out[:, 0], 0, self.size)
            boxes_out[:, 1] = np.clip(boxes_out[:, 1], 0, self.size)
            boxes_out[:, 2] = np.clip(boxes_out[:, 2], 0, self.size - boxes_out[:, 0])
            boxes_out[:, 3] = np.clip(boxes_out[:, 3], 0, self.size - boxes_out[:, 1])

            valid = (boxes_out[:, 2] >= 1) & (boxes_out[:, 3] >= 1)
            boxes_out = boxes_out[valid]
        else:
            boxes_out = boxes.copy() if boxes is not None else None

        # 转换
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if boxes_out is not None and len(boxes_out) > 0:
            boxes_out = torch.from_numpy(boxes_out).float()

        return image, boxes_out
