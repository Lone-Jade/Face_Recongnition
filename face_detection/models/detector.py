"""
detector.py — 轻量级 SSD 人脸检测器
=====================================
- 多尺度检测头 (3层)
- 锚框生成与匹配
- 训练/推理双模式
- 导出友好 (标准算子, 无复杂控制流)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .backbone import LightweightBackbone
from config import ANCHOR_CONFIGS, IMAGE_SIZE, SCORE_THRESHOLD, NMS_THRESHOLD, MAX_DETECTIONS


# ============================================================
# 锚框生成器
# ============================================================
class AnchorGenerator:
    """
    为每个检测层生成先验框 (anchor boxes / priors)。

    每个锚框表示为 [cx, cy, w, h], 归一化到 [0,1]。
    锚框按检测层依次排列, 形状 [num_anchors_total, 4]。
    """

    def __init__(self, image_size: int = IMAGE_SIZE):
        self.image_size = image_size
        self.configs = ANCHOR_CONFIGS
        self.anchors = self._generate()  # [N, 4]

    def _generate(self) -> torch.Tensor:
        anchors = []
        for cfg in self.configs:
            fsize = cfg["feat_size"]
            stride = cfg["stride"]
            sizes = cfg["sizes"]

            for y in range(fsize):
                cy = (y + 0.5) * stride / self.image_size
                for x in range(fsize):
                    cx = (x + 0.5) * stride / self.image_size
                    for s in sizes:
                        w = s / self.image_size
                        h = s / self.image_size
                        anchors.append([cx, cy, w, h])

        return torch.tensor(anchors, dtype=torch.float32)

    @property
    def num_anchors(self) -> int:
        return self.anchors.shape[0]

    def get_anchors_per_layer(self) -> List[int]:
        """返回每层锚框数量"""
        return [cfg["feat_size"] ** 2 * len(cfg["sizes"])
                for cfg in self.configs]


# ============================================================
# 检测头
# ============================================================
class DetectionHead(nn.Module):
    """
    SSD 风格检测头 — 在单个特征图上预测。
    每个空间位置有 K 个锚框, 每个预测:
      - 4 个边界框偏移 (Δcx, Δcy, Δw, Δh)
      - 1 个人脸置信度 (分类)
    """

    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.cls_conv = nn.Conv2d(in_channels, num_anchors * 1,   # 1个二分类 = 人脸概率
                                  kernel_size=3, padding=1)
        self.reg_conv = nn.Conv2d(in_channels, num_anchors * 4,   # 4 个偏移量
                                  kernel_size=3, padding=1)
        self._init_weights()

    def _init_weights(self):
        for m in [self.cls_conv, self.reg_conv]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] 特征图
        Returns:
            cls: [B, num_anchors*1, H, W]
            reg: [B, num_anchors*4, H, W]
        """
        cls = self.cls_conv(x)
        reg = self.reg_conv(x)
        return cls, reg


# ============================================================
# 解码工具函数
# ============================================================
def decode_boxes(anchors: torch.Tensor, reg_preds: torch.Tensor,
                 variance: Tuple[float, float] = (0.1, 0.2)) -> torch.Tensor:
    """
    将锚框 + 预测偏移 → 解码为绝对边界框坐标。

    Args:
        anchors:    [N, 4]  锚框 [cx, cy, w, h] (归一化)
        reg_preds:  [B, N, 4] 预测偏移 [Δcx, Δcy, Δw, Δh]
        variance:   编码方差

    Returns:
        boxes: [B, N, 4] 解码后的框 [x1, y1, x2, y2] (归一化)
    """
    # 中心点解码
    cx = anchors[:, 0] + reg_preds[..., 0] * variance[0] * anchors[:, 2]
    cy = anchors[:, 1] + reg_preds[..., 1] * variance[0] * anchors[:, 3]

    # 宽高解码
    w = anchors[:, 2] * torch.exp(reg_preds[..., 2] * variance[1])
    h = anchors[:, 3] * torch.exp(reg_preds[..., 3] * variance[1])

    # 转换为角点坐标
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=-1)


# ============================================================
# NMS (非极大值抑制)
# ============================================================
def batched_nms(boxes: torch.Tensor, scores: torch.Tensor,
                iou_threshold: float = NMS_THRESHOLD,
                score_threshold: Optional[float] = None,
                max_output: int = MAX_DETECTIONS) -> List[torch.Tensor]:
    """
    批量 NMS 后处理。

    Args:
        boxes:  [B, N, 4] 边界框 [x1, y1, x2, y2] (归一化)
        scores: [B, N]    置信度
        iou_threshold:    IoU 阈值
        score_threshold:  分数阈值 (None=使用全局 SCORE_THRESHOLD)
        max_output:       每张图最大保留框数

    Returns:
        List of [keep_N_i, 4], List of [keep_N_i]  (每张图结果)
    """
    if score_threshold is None:
        score_threshold = SCORE_THRESHOLD

    batch_size = boxes.shape[0]
    result_boxes = []
    result_scores = []

    for i in range(batch_size):
        mask = scores[i] > score_threshold
        if mask.sum() == 0:
            result_boxes.append(torch.zeros((0, 4), device=boxes.device))
            result_scores.append(torch.zeros(0, device=boxes.device))
            continue

        # Pre-filter: keep only top-200 scores for speed
        masked_scores = scores[i][mask]
        masked_boxes = boxes[i][mask]
        if masked_scores.numel() > 100:
            topk = masked_scores.topk(100).indices
            masked_scores = masked_scores[topk]
            masked_boxes = masked_boxes[topk]

        keep = torchvision_nms(
            masked_boxes, masked_scores, iou_threshold)

        if len(keep) > max_output:
            keep = keep[:max_output]

        result_boxes.append(masked_boxes[keep])
        result_scores.append(masked_scores[keep])

    return result_boxes, result_scores


def torchvision_nms(boxes: torch.Tensor, scores: torch.Tensor,
                    iou_threshold: float) -> torch.Tensor:
    """
    简洁的 NMS 实现 (不依赖 torchvision.ops)。

    Args:
        boxes:  [N, 4] 边界框 [x1, y1, x2, y2]
        scores: [N]    置信度 (按降序排列)
        iou_threshold: IoU 阈值

    Returns:
        keep: 保留的索引
    """
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    # 按 scores 降序排列
    order = scores.argsort(descending=True)
    boxes = boxes[order]

    keep = []
    while order.numel() > 0:
        if len(keep) == 0:
            keep.append(order[0].item())
        else:
            iou = compute_iou(boxes[0:1], boxes[1:])[0]
            mask = iou <= iou_threshold
            order = order[1:][mask]
            boxes = boxes[1:][mask]
            if order.numel() == 0:
                break
            keep.append(order[0].item())
            order = order[1:]
            boxes = boxes[1:]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def compute_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """
    计算两组框之间的 IoU。

    Args:
        box_a: [M, 4] 框 [x1, y1, x2, y2]
        box_b: [N, 4] 框 [x1, y1, x2, y2]

    Returns:
        iou: [M, N]
    """
    M, N = box_a.shape[0], box_b.shape[0]

    # 计算交集
    lt = torch.max(box_a[:, None, :2], box_b[:, :2])     # [M, N, 2]
    rb = torch.min(box_a[:, None, 2:], box_b[:, 2:])     # [M, N, 2]
    wh = (rb - lt).clamp(min=0)                           # [M, N, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]                    # [M, N]

    # 计算并集
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    union = area_a[:, None] + area_b - inter

    return inter / union.clamp(min=1e-10)


# ============================================================
# 完整人脸检测器
# ============================================================
class FaceDetector(nn.Module):
    """
    轻量级人脸检测器 — 骨干 + 多尺度 SSD 检测头

    训练模式 (self.training=True):
      forward(x) → (cls_preds_list, reg_preds_list)
      返回每个检测层的原始预测, 用于计算损失。

    推理模式 (self.training=False):
      forward(x) → (boxes_list, scores_list)
      返回 NMS 后的检测结果。
    """

    def __init__(self, num_classes: int = 1, width_mult: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = LightweightBackbone(width_mult=width_mult)
        self.anchor_gen = AnchorGenerator()

        # 为每个检测层创建分类+回归头
        out_channels = self.backbone.out_channels
        anchors_per_layer = self.anchor_gen.get_anchors_per_layer()
        num_per_location = [apl // (cfg["feat_size"] ** 2)
                           for apl, cfg in zip(anchors_per_layer, ANCHOR_CONFIGS)]

        self.det_heads = nn.ModuleList([
            DetectionHead(oc, npl) for oc, npl in zip(out_channels, num_per_location)
        ])

        # 注册锚框为 buffer (随模型保存/移动)
        anchors = self.anchor_gen.anchors
        self.register_buffer("anchors", anchors)

        self._init_weights()

    def _init_weights(self):
        # backbone 已在 __init__ 中初始化
        # detection heads 已在 DetectionHead.__init__ 中初始化
        pass

    def forward(self, x: torch.Tensor):
        """
        统一前向传播。

        Returns (训练模式):
          cls_preds: [B, N, 1] 每锚框的人脸概率
          reg_preds: [B, N, 4] 每锚框的偏移量

        Returns (推理模式):
          boxes:  List of [det_i, 4]  NMS后框
          scores: List of [det_i]     NMS后得分
        """
        features = self.backbone(x)
        batch_size = x.shape[0]

        cls_preds_list = []
        reg_preds_list = []

        for feat, head in zip(features, self.det_heads):
            cls_raw, reg_raw = head(feat)
            B, _, H, W = cls_raw.shape

            # Reshape: [B, C, H, W] → [B, H*W*num_per_loc, 1 or 4]
            cls = cls_raw.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            reg = reg_raw.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)

            cls_preds_list.append(cls)
            reg_preds_list.append(reg)

        # 拼接所有检测层
        cls_preds = torch.cat(cls_preds_list, dim=1)   # [B, N_total, 1]
        reg_preds = torch.cat(reg_preds_list, dim=1)   # [B, N_total, 4]

        return cls_preds, reg_preds

    def predict(self, image: torch.Tensor,
                score_threshold: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单张图片推理.

        Args:
            image: [C, H, W] 或 [1, C, H, W] 输入图像
            score_threshold: 置信度阈值 (默认 0.1 以显示低分检测)

        Returns:
            boxes:  [K, 4]  xyxy 坐标 (像素)
            scores: [K]     置信度
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            cls_preds, reg_preds = self.forward(image)
        # sigmoid → decode → NMS (with lowered threshold)
        scores = torch.sigmoid(cls_preds.squeeze(-1))  # [1, N]
        pred_boxes = decode_boxes(self.anchors, reg_preds)
        boxes_list, scores_list = batched_nms(
            pred_boxes, scores,
            iou_threshold=NMS_THRESHOLD,
            score_threshold=score_threshold)
        boxes = boxes_list[0]
        scores = scores_list[0]
        boxes = boxes * IMAGE_SIZE
        return boxes, scores


# ============================================================
# 模型工厂函数
# ============================================================
def create_face_detector(pretrained_path: Optional[str] = None,
                         width_mult: float = 0.5) -> FaceDetector:
    """
    创建人脸检测器实例并可选加载预训练权重。

    Args:
        pretrained_path: 预训练 .pth 路径 (可选)
        width_mult:      通道倍率

    Returns:
        FaceDetector 实例
    """
    model = FaceDetector(num_classes=1, width_mult=width_mult)

    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location="cpu",
                                weights_only=True)
        # 兼容不同格式的 checkpoint
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)

    return model


# ============================================================
# 模型信息工具
# ============================================================
def get_model_info(model: FaceDetector) -> dict:
    """返回模型统计数据: 参数量、模型大小估算"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)

    # INT8 估算 (权重 1 byte/param)
    int8_weight_bytes = total_params

    # FP32 估算 (权重 4 bytes/param)
    fp32_weight_bytes = total_params * 4

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "fp32_weight_size_kb": fp32_weight_bytes / 1024,
        "int8_weight_size_kb": int8_weight_bytes / 1024,
        "num_anchors": AnchorGenerator().num_anchors,
    }
