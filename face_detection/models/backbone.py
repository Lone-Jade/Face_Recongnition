"""
backbone.py — 轻量级骨干网络
=============================
基于 Modified MobileNetV2 结构:
  - 深度可分离卷积 (Depthwise Separable Convolution)
  - 倒残差模块 (Inverted Residual Block)
  - ReLU6 激活 (Cube.AI 原生支持)
  - 不使用 BatchNorm (部署时 folded into Conv)

输出:
  features[0]: 20×20 低层特征 (小脸检测)
  features[1]: 10×10 中层特征 (中脸检测)
  features[2]: 5×5  高层特征 (大脸检测)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v: float, divisor: int = 8) -> int:
    """确保通道数是 divisor 的整数倍 (硬件友好)"""
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


class ConvBNReLU(nn.Sequential):
    """标准卷积 + BatchNorm + ReLU6"""
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, groups=1):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        ]
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """
    MobileNetV2 倒残差模块
    =======================
    1. 1×1 点卷积扩展通道 (expand_ratio × in_channels)
    2. 3×3 深度卷积 (groups = expanded_channels)
    3. 1×1 点卷积投射回输出通道
    4. 残差连接 (当 stride=1 且 in_channels==out_channels 时)

    全部使用 ReLU6 激活。
    """

    def __init__(self, in_channels, out_channels, stride=1,
                 expand_ratio=6):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = _make_divisible(in_channels * expand_ratio)

        layers = []
        # 扩展层
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1,
                                     stride=1, padding=0))

        # 深度卷积
        layers.append(ConvBNReLU(
            hidden_dim, hidden_dim, kernel_size=3,
            stride=stride, padding=1, groups=hidden_dim))

        # 投射层 (线性, 无激活)
        layers.append(nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class LightweightBackbone(nn.Module):
    """
    轻量级骨干网络 — 为 160×160 嵌入式人脸检测优化
    ====================================================

    架构表:
    | 层       | 输入尺寸    | 操作          | 输出通道 | 输出尺寸   | 步长 |
    |----------|------------|---------------|---------|-----------|------|
    | stem     | 160×160×3  | Conv 3×3      | 16      | 80×80     | 2   |
    | irb1     | 80×80×16   | InvRes(×1)    | 24      | 80×80     | 1   |
    | irb2     | 80×80×24   | InvRes(×6)    | 32      | 40×40     | 2   |
    | irb3     | 40×40×32   | InvRes(×6)    | 48      | 20×20     | 2   | ← feat[0]
    | irb4     | 20×20×48   | InvRes(×6)    | 64      | 10×10     | 2   | ← feat[1]
    | irb5     | 10×10×64   | InvRes(×6)    | 96      | 5×5       | 2   | ← feat[2]

    总参数量: ~35K (FP32 ≈ 140KB, INT8 ≈ 35KB)
    """

    def __init__(self, width_mult=0.5):
        super().__init__()

        def wm(c): return _make_divisible(c * width_mult)

        # Stem
        self.stem = ConvBNReLU(3, wm(16), kernel_size=3, stride=2, padding=1)

        # Stage 1: 80×80
        self.stage1 = InvertedResidual(wm(16), wm(24), stride=1, expand_ratio=1)

        # Stage 2: 80→40
        self.stage2 = InvertedResidual(wm(24), wm(32), stride=2, expand_ratio=6)

        # Stage 3: 40→20 (feat[0])
        self.stage3 = InvertedResidual(wm(32), wm(48), stride=2, expand_ratio=6)

        # Stage 4: 20→10 (feat[1])
        self.stage4 = InvertedResidual(wm(48), wm(64), stride=2, expand_ratio=6)

        # Stage 5: 10→5 (feat[2])
        self.stage5 = InvertedResidual(wm(64), wm(96), stride=2, expand_ratio=6)

        self._out_channels = [wm(48), wm(64), wm(96)]
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def out_channels(self):
        """返回各检测层输出通道数"""
        return self._out_channels

    def forward(self, x):
        """返回多尺度特征图列表 [feat_20, feat_10, feat_5]"""
        x = self.stem(x)       # 80×80
        x = self.stage1(x)     # 80×80
        x = self.stage2(x)     # 40×40
        feat0 = self.stage3(x) # 20×20
        feat1 = self.stage4(feat0)  # 10×10
        feat2 = self.stage5(feat1)  # 5×5
        return [feat0, feat1, feat2]
