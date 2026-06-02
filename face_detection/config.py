"""
config.py — 全局配置
=====================
所有路径、超参数、模型结构参数集中管理。
修改此文件即可调整训练策略, 无需改动其他代码。
"""

import os

# ============================================================
# 路径配置
# ============================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))         # face_detection/
DATA_DIR = os.path.join(ROOT_DIR, "WIDERFace")                # 数据集根目录
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")                 # 模型+日志输出
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")                    # TensorBoard 日志
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")                # 模型保存

# WIDER Face 子路径
WIDER_TRAIN_IMG = os.path.join(DATA_DIR, "WIDER_train", "images")
WIDER_VAL_IMG   = os.path.join(DATA_DIR, "WIDER_val",   "images")
WIDER_TRAIN_ANN = os.path.join(DATA_DIR, "wider_face_split", "wider_face_train_bbx_gt.txt")
WIDER_VAL_ANN   = os.path.join(DATA_DIR, "wider_face_split", "wider_face_val_bbx_gt.txt")

# 创建必要的目录
for d in [OUTPUT_DIR, LOG_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# 数据集配置
# ============================================================
IMAGE_SIZE      = 160          # 模型输入 (正方形)
INPUT_CHANNELS  = 3            # RGB
TRAIN_SPLIT     = 0.8          # 训练/验证 8:2
DIFFICULTY      = "easy_medium"  # 使用的 WIDER Face 子集: easy_medium / all

# ============================================================
# 模型配置 — 轻量级 SSD 检测器
# ============================================================
NUM_CLASSES     = 1            # 检测类别: face (不含背景)

# 骨干网络 — 深度可分离卷积 + 倒残差结构
BACKBONE_WIDTH_MULTIPLIER = 0.5  # 通道倍率 (0.25/0.5/1.0)

# 锚框 (先验框) 配置 — 3 个检测层
# 格式: (feature_map_size, strides, sizes)
#   sizes 是该层锚框的边长列表 (像素坐标)
ANCHOR_CONFIGS = [
    # (feat_size, stride,  sizes_list)
    {"feat_size": 20, "stride": 8,   "sizes": [16, 24, 32]},
    {"feat_size": 10, "stride": 16,  "sizes": [48, 64, 96]},
    {"feat_size": 5,  "stride": 32,  "sizes": [128, 192]},
]
# 总计锚框数 = 20²×3 + 10²×3 + 5²×2 = 1200 + 300 + 50 = 1550

# 后处理
SCORE_THRESHOLD = 0.5          # 置信度阈值
NMS_THRESHOLD   = 0.45         # NMS IoU 阈值
MAX_DETECTIONS  = 50           # 单图最大检测数

# ============================================================
# 训练超参数
# ============================================================
BATCH_SIZE      = 32           # 批大小 (可根据 GPU 内存调整)
EPOCHS          = 150          # 全精度训练轮数
QAT_EPOCHS      = 15           # QAT 微调轮数
BASE_LR         = 1e-3         # 初始学习率
MIN_LR          = 1e-6         # 最小学习率
WEIGHT_DECAY    = 5e-4         # L2 正则化系数
MOMENTUM        = 0.9          # SGD 动量 (使用 SGD 时)
OPTIMIZER       = "adamw"      # 优化器: adamw / sgd

# 学习率调度
LR_SCHEDULER    = "cosine"     # cosine / step / plateau
LR_WARMUP_EPOCHS = 3           # 前 N 轮线性 warmup

# 损失权重
LOSS_CLS_WEIGHT = 1.0          # 分类损失权重
LOSS_REG_WEIGHT = 2.0          # 回归损失权重

# 负样本挖掘
NEG_POS_RATIO   = 3            # 负:正样本比例 (hard negative mining)

# ============================================================
# 数据增强
# ============================================================
AUGMENTATION = {
    "random_flip": True,         # 随机水平翻转
    "random_crop": True,         # 随机裁剪 + 缩放
    "color_jitter": {
        "brightness": 0.3,
        "contrast": 0.3,
        "saturation": 0.3,
        "hue": 0.1
    },
    "random_rotate": 10,         # 随机旋转 (±10°)
    "random_scale": (0.8, 1.2),  # 随机缩放范围
}

# ============================================================
# 量化配置 (QAT)
# ============================================================
QAT_CONFIG = {
    "backend": "fbgemm",         # x86 模拟量化 (实际部署用 qnnpack 或 STM32)
    "activation_precision": 8,   # 激活值 INT8
    "weight_precision": 8,       # 权重 INT8
    "per_channel": False,        # 逐张量量化 (Cube.AI 推荐)
}

# ============================================================
# ONNX 导出配置
# ============================================================
ONNX_CONFIG = {
    "opset_version": 11,         # Cube.AI 推荐 opset 11
    "input_name": "input",
    "output_names": ["boxes", "scores"],
    "dynamic_batch": False,      # 固定 batch=1 (嵌入式推理)
    "simplify": True,            # 使用 onnx-simplifier 优化图
}

# ============================================================
# TFLite 导出配置
# ============================================================
TFLITE_CONFIG = {
    "quantization": "int8",      # int8 / float16 / full_integer
    "representative_dataset_size": 200,  # 校准数据集大小
    "supported_ops": "TFLITE_BUILTINS",  # 仅使用内置算子 (Cube.AI 兼容)
}

# ============================================================
# 硬件信息 (用于日志/报告)
# ============================================================
DEVICE = "cpu"                  # 训练设备: cpu / cuda / mps
TARGET_MCU = "STM32H747XI"
TARGET_CLOCK = 400              # MHz (Cortex-M7)
TARGET_FLASH = 1024             # KB
TARGET_RAM   = 256              # KB (模型部分)
"""============================================================
Embedded System - BigHomework - Face Recognition
Phase B: 轻量级人脸检测模型训练
============================================================"""
