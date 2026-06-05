# Plan A：使用 ST Model Zoo 预训练人脸检测模型 — 完整实施方案

> **目标**：在 STM32H747I-DISCO 上部署人脸检测系统
> **策略**：使用 ST Model Zoo 的预训练人脸检测模型 + WIDER Face 微调 + Cube.AI 部署
> **与当前目录的关系**：本计划在**独立文件夹**中执行，当前 `face_detection/` 中的训练代码保留作参考，Model Zoo 的直接替代其训练流程。当前 Keil 基线工程（LCD 显示）继续使用。

---

## 零、前提条件

| 需求 | 说明 |
|------|------|
| 磁盘空间 | ~30GB（WIDER Face 数据集 ~3.4GB + 解压 ~5GB + Model Zoo 模型 ~2GB + 工作空间） |
| Python | 3.10+ |
| Git LFS | 安装 `git-lfs`（Model Zoo 的模型文件用 LFS 存储） |
| GPU（推荐） | NVIDIA GPU 8GB+ VRAM（微调需要，纯 CPU 也可以但很慢） |
| 当前基线工程 | `LCD_DSI_CMD_mode_Single_Buffer/` Keil 工程（已编译通过，LCD 显示正常） |

---

## 一、环境与仓库准备

### 1.1 在新文件夹中创建工作空间

```bash
# 在项目根目录外（或同级）创建新工作目录
mkdir STM32_FaceDetection
cd STM32_FaceDetection

# 目录结构规划
mkdir -p model_zoo        # ST Model Zoo 仓库
mkdir -p data/wider_face  # WIDER Face 数据集
mkdir -p workspace        # 训练工作目录
mkdir -p output           # 输出模型 + 报告
mkdir -p mcu_deploy       # MCU 部署文件（Cube.AI 生成 + 后处理代码）
```

### 1.2 克隆 ST Model Zoo 仓库

```bash
# 安装 Git LFS（如果还没有）
# Windows: 下载安装 https://git-lfs.com/
# Linux: sudo apt-get install git-lfs

git lfs install

# 克隆 Model Zoo（预训练模型在 LFS 中）
cd model_zoo
git clone https://github.com/STMicroelectronics/stm32ai-modelzoo.git
cd stm32ai-modelzoo
git checkout v4.1.0   # 最新稳定版

# 查看人脸检测模型目录
ls face_detection/
# 预期内容：models/  scripts/  config_file_examples/

# 查看有哪些预训练模型
ls face_detection/models/
```

### 1.3 克隆 Model Zoo Services（训练/量化脚本）

```bash
cd ../..  # 回到 STM32_FaceDetection/
cd model_zoo
git clone https://github.com/STMicroelectronics/stm32ai-modelzoo-services.git
cd stm32ai-modelzoo-services

# 查看人脸检测训练脚本
ls face_detection/
# 预期：README.md  training/  evaluation/  quantization/  config_file_examples/
```

### 1.4 安装 Python 依赖

```bash
pip install -r stm32ai-modelzoo-services/requirements.txt

# 额外依赖
pip install tensorflow==2.18.0     # TFLite 转换
pip install onnx==1.16.1           # ONNX 工具
pip install onnxruntime            # ONNX 推理验证
pip install opencv-python          # 图像处理
pip install matplotlib seaborn     # 可视化
pip install pycocotools            # COCO 格式工具
pip install tqdm                   # 进度条
```

---

## 二、模型选择

### 2.1 进入 Model Zoo 查看人脸检测模型列表

```bash
cd model_zoo/stm32ai-modelzoo/face_detection/models/
ls -la
```

### 2.2 预期可用的模型

基于 ST Model Zoo v4.0+ 的信息，人脸检测用例应包括：

| 模型 | 框架 | 输入 | Flash | RAM | 推荐场景 |
|------|------|------|-------|-----|---------|
| **SSD MobileNetV1 0.25** | TF/PT/ONNX | 192×192 | ~438KB | ~196KB | 优先选择：Cube.AI 原生支持 |
| **SSD MobileNetV1 0.25** | TF/PT/ONNX | 224×224 | ~596KB | ~333KB | 精度更高但 RAM 略超 |
| **ST Yolo LC v1** | TF | 192×192 | ~277KB | ~157KB | 最优 Flash/RAM，但需联系 ST |

### 2.3 决策矩阵（针对 STM32H747I：Flash 1MB, RAM 256KB 给模型）

```
推荐首选：SSD MobileNetV1 0.25 @ 192×192
  - Flash 438KB < 512KB ✓
  - RAM 196KB < 256KB ✓
  - Cube.AI 对 SSD MobileNet 的算子支持最成熟
  - COCO Person 预训练 → 人脸微调迁移效果好

备选：ST Yolo LC v1 @ 192×192
  - Flash 277KB、RAM 157KB，余量更大
  - 需要发邮件到 Edge.ai@st.com 获取预训练权重
```

### 2.4 记录选定模型的配置

```
模型：SSD MobileNetV1 0.25
输入：192×192×3 (RGB)
锚框：由 Model Zoo 配置自动生成
输出：检测框 [N, 4] + 分数 [N]（或 SSD 原始输出 [N, num_classes+4]）
预训练数据集：COCO 2017 Person（20 万+ 张含人图片）
```

---

## 三、数据准备

### 3.1 下载 WIDER Face 数据集

WIDER Face 是最大的人脸检测数据集，包含 32,203 张图片和 393,703 个标注人脸。

```bash
cd data/wider_face

# 方式 1：从官网下载（推荐）
# http://shuoyang1213.me/WIDERFACE/
# 下载以下文件：
#   WIDER_train.zip
#   WIDER_val.zip
#   wider_face_split.zip（标注文件）

# 方式 2：使用 Google Drive 镜像
# https://drive.google.com/file/d/...

# 解压
unzip WIDER_train.zip
unzip WIDER_val.zip
unzip wider_face_split.zip

# 最终目录结构
# data/wider_face/
#   WIDER_train/images/    # 训练图片
#   WIDER_val/images/      # 验证图片
#   wider_face_split/      # 标注 txt 文件
```

### 3.2 WIDER Face → COCO 格式转换

Model Zoo 的 SSD 训练脚本需要 COCO 格式。WIDER Face 的标注格式是 `[x, y, w, h]`，需要转换。

**创建 `workspace/convert_wider_to_coco.py`**：

```python
"""
将 WIDER Face 标注转换为 COCO JSON 格式。
供 ST Model Zoo 的 SSD MobileNet 训练脚本使用。
"""
import os
import json
import cv2
from tqdm import tqdm

# ============================================================
# 配置（根据实际情况修改）
# ============================================================
WIDER_ROOT = "../data/wider_face"
WIDER_TRAIN_IMG = os.path.join(WIDER_ROOT, "WIDER_train", "images")
WIDER_VAL_IMG = os.path.join(WIDER_ROOT, "WIDER_val", "images")
WIDER_TRAIN_ANN = os.path.join(WIDER_ROOT, "wider_face_split", "wider_face_train_bbx_gt.txt")
WIDER_VAL_ANN = os.path.join(WIDER_ROOT, "wider_face_split", "wider_face_val_bbx_gt.txt")
OUTPUT_DIR = "../data/coco_format"
FACE_CATEGORY_ID = 1  # COCO 格式的类别 ID

# 难度过滤：只保留可用的人脸
MIN_FACE_SIZE = 16     # 192×192 输入的 1/12
MAX_BLUR = 1           # 0=清晰, 1=轻微模糊, 2=严重模糊


def parse_wider_annotation(ann_path, img_dir, split_name):
    """解析 WIDER Face 标注 → COCO 格式"""
    images = []
    annotations = []
    ann_id = 0  # annotation id 递增

    with open(ann_path, 'r') as f:
        lines = f.readlines()

    idx = 0
    img_id = 0
    total_lines = len(lines)

    while idx < total_lines:
        img_rel_path = lines[idx].strip()
        idx += 1

        if idx >= total_lines:
            break

        try:
            num_faces = int(lines[idx].strip())
        except ValueError:
            idx += 1
            continue
        idx += 1

        img_full_path = os.path.join(img_dir, img_rel_path)

        # 读取图片获取真实尺寸
        image = cv2.imread(img_full_path)
        if image is None:
            idx += num_faces  # 跳过标注行
            continue

        h, w = image.shape[:2]

        # COCO image entry
        images.append({
            "id": img_id,
            "file_name": img_rel_path,
            "width": w,
            "height": h,
        })

        valid_face_count = 0
        for _ in range(num_faces):
            if idx >= total_lines:
                break
            parts = lines[idx].strip().split()
            idx += 1

            if len(parts) < 4:
                continue

            x, y, bw, bh = map(int, parts[:4])
            blur = int(parts[4]) if len(parts) > 4 else 0

            # 过滤
            if bh < MIN_FACE_SIZE or bw < MIN_FACE_SIZE:
                continue
            if blur > MAX_BLUR:
                continue

            # 裁剪到图片边界
            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            if bw <= 0 or bh <= 0:
                continue

            area = bw * bh
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": FACE_CATEGORY_ID,
                "bbox": [x, y, bw, bh],       # COCO: [x, y, w, h]
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1
            valid_face_count += 1

        if valid_face_count > 0 or num_faces == 0:
            # 保留有人脸标注的图片（或无人的图片用于负样本）
            img_id += 1

    # 构建 COCO JSON
    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": FACE_CATEGORY_ID, "name": "face", "supercategory": "person"}
        ],
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, f"wider_face_{split_name}.json")
    with open(output_path, 'w') as f:
        json.dump(coco_json, f)

    print(f"[{split_name}] {len(images)} images, {len(annotations)} faces → {output_path}")
    return output_path


if __name__ == "__main__":
    print("Converting WIDER Face to COCO format...")
    parse_wider_annotation(WIDER_TRAIN_ANN, WIDER_TRAIN_IMG, "train")
    parse_wider_annotation(WIDER_VAL_ANN, WIDER_VAL_IMG, "val")
    print("\nDone! COCO JSON files are in:", OUTPUT_DIR)
```

运行转换：

```bash
cd workspace
python convert_wider_to_coco.py
# 预期输出:
# [train] ~12000 images, ~150000 faces
# [val]   ~3000 images, ~40000 faces
```

### 3.3 验证 COCO 数据

```bash
python -c "
import json
with open('../data/coco_format/wider_face_train.json') as f:
    data = json.load(f)
print(f'Images: {len(data[\"images\"])}')
print(f'Annotations: {len(data[\"annotations\"])}')
print(f'Categories: {data[\"categories\"]}')
"
```

---

## 四、模型微调（Fine-tuning）

### 4.1 创建 ST Model Zoo 训练配置文件

Model Zoo Services 使用 YAML 文件驱动训练。在 `workspace/` 下创建 `face_detection_config.yaml`：

```yaml
# face_detection_config.yaml
# ST Model Zoo 格式的训练配置 —— 人脸检测微调
# 参考: model_zoo/stm32ai-modelzoo-services/face_detection/config_file_examples/

# ============================================================
# 操作模式: chained train + quantize + evaluate + benchmark
# ============================================================
operation_mode: chain_tqeb

# ============================================================
# 通用配置
# ============================================================
general:
  model_path: ../model_zoo/stm32ai-modelzoo/face_detection/models/ssd_mobilenet_v1/ST_pretrainedmodel_public_dataset/coco_2017_person/ssd_mobilenet_v1_025_192
  model_type: ssd_mobilenet_v1
  project_name: face_detection_stm32h7
  saved_models_dir: ../output/saved_models
  logs_dir: ../output/logs
  display_figures: true

# ============================================================
# 数据集
# ============================================================
dataset:
  name: wider_face
  class_names: [face]           # 1 类
  training_path: ../data/coco_format/wider_face_train.json
  validation_path: ../data/coco_format/wider_face_val.json
  test_path: null

preprocessing:
  rescaling:
    scale: 1/127.5
    offset: -1         # 归一化到 [-1, 1]
  resizing:
    aspect_ratio: fit
    interpolation: bilinear
  color_mode: rgb

# ============================================================
# 训练配置
# ============================================================
training:
  model:
    input_shape: [192, 192, 3]
    alpha: 0.25        # MobileNet 宽度乘数
    num_classes: 1     # 只有 face
    pretrained_weights: imagenet  # 或 coco

  batch_size: 32       # 根据 GPU 内存调整
  epochs: 50           # 预训练权重上微调，不需要太多轮
  dropout: 0.0

  optimizer:
    adamw:
      learning_rate: 0.0001   # 微调用小学习率
      weight_decay: 0.0005

  callbacks:
    ReduceLROnPlateau:
      monitor: val_loss
      factor: 0.5
      patience: 5
      min_lr: 1e-6
    EarlyStopping:
      monitor: val_loss
      patience: 15
      restore_best_weights: true

  # 数据增强（人脸检测专用）
  data_augmentation:
    random_flip:
      mode: horizontal
    random_crop:
      aspect_ratio_range: [0.8, 1.2]
      area_range: [0.3, 1.0]
    random_brightness:
      max_delta: 0.1
    random_contrast:
      lower: 0.8
      upper: 1.2

# ============================================================
# 量化配置
# ============================================================
quantization:
  quantizer: TFlite_converter
  quantization_type: PTQ           # Post-Training Quantization
  quantization_input_type: uint8
  quantization_output_type: float
  export_dir: ../output/quantized_models

# ============================================================
# 评估配置
# ============================================================
evaluation:
  test_split: validation

# ============================================================
# Benchmark（在 STM32H747I-DISCO 实际硬件上）
# ============================================================
tools:
  stedgeai:
    path_to_stedgeai: C:/ST/STEdgeAI/Utilities/windows/stedgeai.exe
    target: STM32H747I-DISCO
    output_dir: ../output/benchmark
```

### 4.2 更正训练脚本入口

由于 Model Zoo Services 的 Python 训练脚本通常是通用接口，执行方式如下：

```bash
cd workspace

# ST Model Zoo 的入口通常是 run_training.py 或类似脚本
# 具体路径取决于 Model Zoo Services 的实际结构
python ../model_zoo/stm32ai-modelzoo-services/face_detection/training/run_training.py \
    --config-path ./face_detection_config.yaml
```

### 4.3 监控训练

训练过程中关注：
- **Loss** 应该在 5-10 个 epoch 内显著下降（预训练权重效果）
- **Validation mAP@0.5** 目标：> 0.70（WIDER Face Easy）
- 如果 loss 不下降：检查数据加载是否正确，降低学习率，检查 GT 格式

### 4.4 预期训练时间

| 配置 | 每 epoch 时间 | 总时间 |
|------|-------------|--------|
| NVIDIA GPU 8GB | ~5 min | ~4 hours (50 epochs) |
| CPU only | ~30 min | ~25 hours |
| Colab T4 free | ~8 min | ~7 hours |

---

## 五、量化与导出

### 5.1 INT8 量化

Model Zoo 的 chain 模式会在训练完成后自动执行 PTQ（Post-Training Quantization）：

```bash
# 如果链式执行，这一步自动完成
# 手动执行量化：
python ../model_zoo/stm32ai-modelzoo-services/face_detection/quantization/quantize.py \
    --model_path ../output/saved_models/best_model.h5 \
    --output_path ../output/quantized_models/face_detector_int8.tflite \
    --quantization_type PTQ \
    --input_type uint8
```

### 5.2 量化精度验证

```bash
# 评估量化后模型的 mAP
python ../model_zoo/stm32ai-modelzoo-services/face_detection/evaluation/evaluate.py \
    --model_path ../output/quantized_models/face_detector_int8.tflite \
    --dataset_path ../data/coco_format/wider_face_val.json

# 量化前后精度对比：
# FP32 mAP@0.5: ~0.75+
# INT8 mAP@0.5: ~0.73+ (损失 < 2% 可接受)
```

### 5.3 最终输出物

```
output/
├── saved_models/
│   ├── best_model.h5              # Keras/TF 完整模型
│   └── last_model.h5
├── quantized_models/
│   └── face_detector_int8.tflite   # ★ 这是部署到 Cube.AI 的文件
├── logs/
│   └── training_log.csv
└── benchmark/
    └── stm32h747i_report.txt       # Cube.AI 资源分析报告
```

### 5.4 备选导出格式（ONNX）

如果 Cube.AI 对 TFLite 的支持有问题，也可以导出 ONNX：

```bash
# 如果 Model Zoo 支持 ONNX 导出
python ../model_zoo/stm32ai-modelzoo-services/face_detection/export/export_onnx.py \
    --model_path ../output/saved_models/best_model.h5 \
    --output_path ../output/face_detector.onnx \
    --opset 11
```

---

## 六、Cube.AI / ST Edge AI 部署

### 6.1 安装 ST Edge AI（Cube.AI 继任者）

```bash
# 选项 1：下载 ST Edge AI Core CLI 工具
# https://www.st.com/en/development-tools/stedgeai-core.html
# 安装到 C:/ST/STEdgeAI/

# 选项 2：使用 STM32CubeMX 内置的 X-CUBE-AI 插件
# 在 CubeMX 中：Software Packs → Select Components → X-CUBE-AI
```

### 6.2 分析模型

```bash
# 使用 stedgeai CLI 分析模型
cd mcu_deploy

stedgeai analyze \
    --model ../output/quantized_models/face_detector_int8.tflite \
    --target STM32H747I-DISCO

# 预期输出：
# ========================================
# Model Analysis Report
# ========================================
# Flash occupation (weights + code):  ~438 KB
# RAM occupation (activation buffer):  ~196 KB
# Inference time (M7 @ 400MHz):        ~150-300 ms
# Complexity (MACC):                   ~50M
# Compatible: YES
# ========================================
```

### 6.3 生成 C 代码

```bash
# 生成 Cube.AI 兼容的 C 代码
stedgeai generate \
    --model ../output/quantized_models/face_detector_int8.tflite \
    --target STM32H747I-DISCO \
    --output ./generated

# 生成的文件：
# generated/
#   network.c          # 模型包装函数
#   network.h          # 头文件
#   network_data.c     # 权重数据（Flash 常量）
#   network_data.h     # 权重头文件
#   network_config.h   # 配置宏定义
```

### 6.4 集成到 Keil 工程

将生成的文件复制到 Keil 工程：

```bash
# 假设 Keil 工程在：
# ../LCD_DSI_CMD_mode_Single_Buffer/

# 复制 AI 文件到工程
cp generated/network*.c ../LCD_DSI_CMD_mode_Single_Buffer/CM7/Core/Src/
cp generated/network*.h ../LCD_DSI_CMD_mode_Single_Buffer/CM7/Core/Inc/
```

在 Keil uVision 中：
1. 添加 `network.c`、`network_data.c` 到 CM7 Target
2. 添加 Include Path: `../CM7/Core/Inc`（已存在）
3. 启用 CRC 外设：`stm32h7xx_hal_conf.h` 中 `#define HAL_CRC_MODULE_ENABLED`

### 6.5 修改 main.c — 核心集成架构

这是最关键的部分。在 Keil 工程的 `main.c` 中加入 AI 推理流程。

#### 6.5.1 新增 include 和全局变量

```c
// ===== 在 main.c 顶部新增 =====
#include "network.h"
#include "network_data.h"

// 模型输入输出配置（从 network_config.h 获取）
#define AI_INPUT_WIDTH    192
#define AI_INPUT_HEIGHT   192
#define AI_INPUT_CHANNELS 3
#define AI_INPUT_SIZE     (AI_INPUT_WIDTH * AI_INPUT_HEIGHT * AI_INPUT_CHANNELS)

// 检测后处理参数
#define SCORE_THRESHOLD   0.5f
#define NMS_THRESHOLD     0.45f
#define MAX_DETECTIONS    20

// SDRAM 缓冲区规划
// 0xD0000000  LCD_FRAME_BUFFER (800×480×4 = 1.5MB)
// 0xD0180000  AI 输入图像缓冲 (192×192×3 ≈ 110KB)
// 0xD01C0000  预处理临时缓冲
#define AI_INPUT_BUFFER    ((uint8_t*)0xD0180000)
#define AI_TEMP_BUFFER     ((uint8_t*)0xD01C0000)

// AI 运行时缓冲
static ai_handle network_handle = AI_HANDLE_NULL;
static ai_network_report report;

// 输入/输出量化参数（INT8 模型）
static uint8_t ai_input[AI_INPUT_SIZE];
static float ai_output_boxes[MAX_DETECTIONS * 4];    // [y1, x1, y2, x2]
static float ai_output_scores[MAX_DETECTIONS];
```

#### 6.5.2 AI 初始化函数

```c
/**
 * @brief  初始化 Cube.AI 网络
 * @retval 0=成功, -1=失败
 */
static int ai_init(void)
{
    ai_error err;

    // 创建网络实例
    err = ai_network_create(&network_handle, AI_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE) {
        printf("[AI] Failed to create network: code=%d\r\n", err.code);
        return -1;
    }

    // 获取激活缓冲区大小并分配
    ai_size act_size = ai_network_activations_size_get(network_handle);

    // 初始化网络（激活缓冲在 AXI SRAM 0x24000000）
    const ai_network_params params = {
        AI_NETWORK_DATA_WEIGHTS(ai_network_data_weights_get()),
        AI_NETWORK_DATA_ACTIVATIONS((ai_float*)(0x24000000))
    };

    if (ai_network_init(network_handle, &params) != 0) {
        printf("[AI] Failed to init network\r\n");
        return -1;
    }

    // 获取网络报告（Flash/RAM 占用等）
    ai_network_get_report(network_handle, &report);
    printf("[AI] Model: %lu weights, %lu activations\r\n",
           report.n_weights, report.n_activations);
    printf("[AI] Flash: %lu B, RAM: %lu B\r\n",
           report.n_weights_size, report.n_activations_size);

    return 0;
}
```

#### 6.5.3 预处理函数（DMA2D 加速，可选）

```c
/**
 * @brief  将图片缩放到 192×192 并填入 AI 输入缓冲
 * @param  src: 源图像缓冲（RGB888, 任意尺寸 w×h）
 * @param  w, h: 源图像尺寸
 */
static void preprocess_image(uint8_t* src, int w, int h)
{
    // 简单实现：CPU 双线性缩放
    // 优化实现：使用 DMA2D 或 STM32_ImageProcessing_Library
    //
    // 缩放比例
    float scale_x = (float)w / AI_INPUT_WIDTH;
    float scale_y = (float)h / AI_INPUT_HEIGHT;

    for (int dy = 0; dy < AI_INPUT_HEIGHT; dy++) {
        for (int dx = 0; dx < AI_INPUT_WIDTH; dx++) {
            int sx = (int)(dx * scale_x);
            int sy = (int)(dy * scale_y);
            sx = (sx >= w) ? w - 1 : sx;
            sy = (sy >= h) ? h - 1 : sy;

            int idx = (dy * AI_INPUT_WIDTH + dx) * 3;
            int s_idx = (sy * w + sx) * 3;

            // INT8 量化: pixel/255 * 255 → 直接赋值
            ai_input[idx + 0] = src[s_idx + 0];  // R
            ai_input[idx + 1] = src[s_idx + 1];  // G
            ai_input[idx + 2] = src[s_idx + 2];  // B
        }
    }
}
```

#### 6.5.4 AI 推理函数

```c
/**
 * @brief  执行一次人脸检测推理
 * @retval 检测到的人脸数量
 */
static int ai_detect_faces(uint8_t* image, int w, int h)
{
    // 1. 预处理
    preprocess_image(image, w, h);

    // 2. 配置输入
    ai_buffer in_buf;
    in_buf = ai_network_inputs_get(network_handle, NULL);
    memcpy(in_buf.data, ai_input, AI_INPUT_SIZE);

    // 3. 配置输出
    ai_buffer out_buf;
    out_buf = ai_network_outputs_get(network_handle, NULL);

    // 4. 执行推理
    ai_error err = ai_network_run(network_handle, &in_buf, &out_buf);
    if (err.type != AI_ERROR_NONE) {
        printf("[AI] Inference error: %d\r\n", err.code);
        return 0;
    }

    // 5. 后处理：解析输出 + NMS
    int num_faces = post_process_ssd(
        (float*)out_buf.data,
        ai_output_boxes,
        ai_output_scores,
        MAX_DETECTIONS
    );

    return num_faces;
}
```

#### 6.5.5 SSD 后处理（NMS）

```c
/**
 * @brief  SSD 输出 → 边界框 + NMS 后处理
 *
 * SSD MobileNetV1 0.25 输出格式:
 *   输出张量 [N_anchors × (4 + num_classes + 1)]
 *   或两个独立输出: boxes [N_anchors, 4] + scores [N_anchors]
 *
 * 详细格式取决于具体模型的输出层配置，需要从 Cube.AI
 * 生成的 network_config.h 确认。
 *
 * @param raw_output:   模型原始输出
 * @param out_boxes:    输出框 [N, 4] (y1, x1, y2, x2)
 * @param out_scores:   输出分数 [N]
 * @param max_detections: 最大检测数
 * @return 实际检测数
 */
static int post_process_ssd(float* raw_output, float* out_boxes,
                            float* out_scores, int max_detections)
{
    // 此函数需要根据实际模型输出格式调整
    // 以下是伪代码框架：

    // 1. 解析输出 → 解码锚框 → 获取候选框
    // 2. 筛选 score > SCORE_THRESHOLD 的框
    // 3. 按 score 降序排列
    // 4. NMS (IoU > NMS_THRESHOLD → 删除)
    // 5. 返回前 max_detections 个

    // 详细实现参考:
    //   - 当前 face_detection/models/detector.py 中的 decode_boxes + NMS
    //   - FP-AI-VISION1 中的后处理代码 (utm2611)

    return 0; // 占位
}
```

#### 6.5.6 绘制检测框到帧缓冲

```c
/**
 * @brief  在 SDRAM 帧缓冲上绘制人脸检测框
 *
 * 帧缓冲格式: ARGB8888 (4 bytes/pixel)
 * 屏幕尺寸: 800×480
 *
 * @param fps:   帧缓冲首地址 (0xD0000000)
 * @param boxes: 边界框 [y1, x1, y2, x2] — 模型输入坐标
 * @param scores: 置信度
 * @param num:   检测数量
 * @param color: 框颜色 (ARGB)
 * @param scale_x: 模型坐标 → 屏幕坐标 X 比例
 * @param scale_y: 模型坐标 → 屏幕坐标 Y 比例
 */
static void draw_boxes_on_fb(uint32_t* fb, float* boxes, float* scores,
                             int num, uint32_t color,
                             float scale_x, float scale_y)
{
    for (int i = 0; i < num; i++) {
        int y1 = (int)(boxes[i*4 + 0] * scale_y);
        int x1 = (int)(boxes[i*4 + 1] * scale_x);
        int y2 = (int)(boxes[i*4 + 2] * scale_y);
        int x2 = (int)(boxes[i*4 + 3] * scale_x);

        // 裁剪到屏幕范围
        if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
        if (x2 >= 800) x2 = 799;
        if (y2 >= 480) y2 = 479;

        // 画水平线 (上、下)
        for (int x = x1; x <= x2; x++) {
            fb[y1 * 800 + x] = color;
            fb[y2 * 800 + x] = color;
        }
        // 画垂直线 (左、右)
        for (int y = y1; y <= y2; y++) {
            fb[y * 800 + x1] = color;
            fb[y * 800 + x2] = color;
        }
    }
}
```

#### 6.5.7 修改主循环

```c
int main(void)
{
    // ... 原有初始化 (MPU, Cache, HAL, SystemClock, HSEM, GPIO, USART, LED, SDRAM, LCD) ...

    // ★ 新增：初始化 AI 模型
    if (ai_init() != 0) {
        printf("[Error] AI init failed!\r\n");
        while (1);  // 根据策略决定是否继续
    }

    // ★ 新增：初始化输入源（选择一种）
    // 选项 A：SD 卡（推荐演示用）
    // BSP_SD_Init();
    // FATFS_Init();
    //
    // 选项 B：USART 串口（推荐调试用）
    // (已有 MX_USART1_UART_Init())
    //
    // 选项 C：DCMI 摄像头（推荐实时检测用）
    // BSP_CAMERA_Init();

    // 加载默认图片到 SDRAM 输入缓冲
    // (可以是 Flash 中的测试图片或 SD 卡中的图片)

    while (1)
    {
        if (pending_buffer < 0)  // 等待上一帧刷新完毕
        {
            // === 1. 准备输入图片 ===
            // 从源（SD卡/USART/摄像头）获取图片数据到 AI_INPUT_BUFFER
            // get_image_from_source(AI_INPUT_BUFFER, ...);

            // === 2. AI 推理 ===
            int num_faces = ai_detect_faces(AI_INPUT_BUFFER,
                                            source_width, source_height);

            // === 3. 将来源图片放大显示到 LCD ===
            // 使用 DMA2D CopyBuffer 将源图拷贝到帧缓冲
            CopyBuffer(source_image, LCD_FRAME_BUFFER,
                       offset_x, offset_y, source_width, source_height);

            // === 4. 在帧缓冲上绘制人脸检测框 ===
            if (num_faces > 0) {
                float scale_x = (float)LCD_WIDTH / AI_INPUT_WIDTH;
                float scale_y = (float)LCD_HEIGHT / AI_INPUT_HEIGHT;
                draw_boxes_on_fb((uint32_t*)LCD_FRAME_BUFFER,
                                 ai_output_boxes, ai_output_scores,
                                 num_faces,
                                 0xFF00FF00,  // ARGB Green
                                 scale_x, scale_y);
            }

            // === 5. 刷新 LCD ===
            HAL_DSI_Refresh(&hlcd_dsi);
        }
        HAL_Delay(100);  // ~10 fps
    }
}
```

---

## 七、MCU 端后处理（NMS）详解

### 7.1 后处理实现指南

SSD MobileNet 的后处理在 MCU 端需要手动实现（Cube.AI 不支持 NMS 算子），参考以下资源：

| 资源 | 位置 |
|------|------|
| **当前仓库的 Python NMS 实现** | `face_detection/models/detector.py` → `decode_boxes()` + `batched_nms()` |
| **FP-AI-VISION1 后处理** | `Projects/STM32H747I-DISCO/Applications/FP-AI-VISION1/` |
| **WIDER Face 评估 IoU** | `face_detection/utils/metrics.py` → `compute_map()` |

### 7.2 简化 NMS 伪代码（用于 MCU 移植）

```c
// 简化版: 只保留 top-K 的 NMS
// 假设模型直接输出解码后的框 + 分数

typedef struct {
    float y1, x1, y2, x2;
    float score;
    int valid;
} Detection;

static float iou(Detection* a, Detection* b) {
    float inter_x1 = fmaxf(a->x1, b->x1);
    float inter_y1 = fmaxf(a->y1, b->y1);
    float inter_x2 = fminf(a->x2, b->x2);
    float inter_y2 = fminf(a->y2, b->y2);
    if (inter_x1 >= inter_x2 || inter_y1 >= inter_y2) return 0.0f;

    float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
    float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);
    return inter_area / (area_a + area_b - inter_area + 1e-10f);
}

int nms(Detection* dets, int n, float threshold, int max_output) {
    // 1. 按 score 降序排列 (bubble sort for small n)
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (dets[j].score > dets[i].score) {
                Detection tmp = dets[i];
                dets[i] = dets[j];
                dets[j] = tmp;
            }
        }
    }

    // 2. NMS
    int keep_count = 0;
    for (int i = 0; i < n && keep_count < max_output; i++) {
        if (!dets[i].valid) continue;
        dets[keep_count++] = dets[i];
        for (int j = i + 1; j < n; j++) {
            if (!dets[j].valid) continue;
            if (iou(&dets[i], &dets[j]) > threshold) {
                dets[j].valid = 0;
            }
        }
    }
    return keep_count;
}
```

---

## 八、PC 端测试与验证

### 8.1 测试量化模型（部署前验证）

```bash
cd workspace

# 用 Python + TFLite Runtime 测试量化模型
python << 'EOF'
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
import json

# 加载量化模型
interpreter = tf.lite.Interpreter(
    model_path="../output/quantized_models/face_detector_int8.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input: {input_details[0]['shape']}, {input_details[0]['dtype']}")
print(f"Output: {len(output_details)} tensors")

# 测试单张图片
image = cv2.imread("test_face.jpg")
image = cv2.resize(image, (192, 192))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 设置输入
input_data = np.expand_dims(image_rgb, axis=0).astype(np.uint8)
interpreter.set_tensor(input_details[0]['index'], input_data)

# 推理
import time
t0 = time.time()
interpreter.invoke()
t_elapsed = time.time() - t0
print(f"TFLite inference: {t_elapsed*1000:.1f} ms")

# 获取输出
for out_detail in output_details:
    out_data = interpreter.get_tensor(out_detail['index'])
    print(f"{out_detail['name']}: shape={out_data.shape}, "
          f"range=[{out_data.min():.3f}, {out_data.max():.3f}]")
EOF
```

### 8.2 精度验证

```bash
# 在 WIDER Face 验证集上评估 mAP
python ../model_zoo/stm32ai-modelzoo-services/face_detection/evaluation/evaluate.py \
    --model_path ../output/quantized_models/face_detector_int8.tflite \
    --dataset_path ../data/coco_format/wider_face_val.json \
    --score_threshold 0.5 \
    --nms_threshold 0.45
```

---

## 九、硬件部署与调试

### 9.1 部署前检查清单

| 检查项 | 状态 |
|--------|------|
| Keil 工程编译通过（含 Cube.AI 生成文件） | ☐ |
| `stm32h7xx_hal_conf.h` 启用 `HAL_CRC_MODULE_ENABLED` | ☐ |
| 激活缓冲地址在 `stm32h747xx_flash_CM7.sct` 中可用 | ☐ |
| Flash 总大小 < 1MB | ☐ |
| 输入缓冲地址不与帧缓冲重叠 | ☐ |
| USART1 可用（115200，输出调试日志） | ☐ |
| LCD 显示正常（基线验证） | ☐ |

### 9.2 烧录与验证

```bash
# 1. 在 Keil 中编译 CM7 Target → 生成 .hex 文件
# 2. 使用 STM32CubeProgrammer 烧录
STM32_Programmer_CLI -c port=SWD -w build/LCD_DSI_CMD_mode_Single_Buffer_CM7.hex -v

# 3. 打开串口监视（PuTTY / Tera Term），波特率 115200
# 4. 观察 UART 输出：
#    [AI] Model: xxx weights, xxx activations
#    [AI] Flash: xxx B, RAM: xxx B
#    [AI] Inference: xxx ms, xxx faces detected
```

### 9.3 性能目标

| 指标 | 目标值 | 备注 |
|------|--------|------|
| 模型 Flash | < 512KB | SSD MNetV1 0.25 ≈ 438KB |
| 模型 RAM | < 256KB | ≈ 196KB @ 192×192 |
| 推理时间 | < 500ms | M7 @ 400MHz, ~50M MACC |
| 检测 mAP@0.5 | > 0.70 | WIDER Face Easy |
| LCD 刷新率 | > 5 fps | 含推理 + 绘制 |

---

## 十、风险与备选方案

| 风险 | 概率 | 应对 |
|------|------|------|
| Model Zoo 人脸检测模型不支持 Cube.AI 某些算子 | 中 | 1. 使用 operator checking 工具提前验证 2. 回退到 SSDLite MobileNetV2（已知兼容） |
| 量化后精度下降 > 5% | 低 | 1. 使用 QAT 替代 PTQ 2. 减小输入尺寸（160×160） 3. 使用 FP16 量化 |
| Flash/RAM 超限 | 中 | 1. 使用 ST Yolo LC v1（更小模型） 2. 输入尺寸降到 160×160 3. 减少锚框数量 |
| ST Yolo LC 预训练权重获取困难 | 高 | 优先使用 SSD MobileNet（公开权重），Yolo LC 作为备选 |
| Cube.AI 不支持 INT8 输入 | 低 | 改为 FP32 输入 + INT8 权重（混合精度） |

---

## 十一、完整时间估算

| 阶段 | 预计耗时 | 依赖 |
|------|---------|------|
| 1. 环境与仓库准备 | 2-4 小时 | 网络下载速度 |
| 2. 数据下载 | 2-6 小时 | 网络下载速度 |
| 3. 数据格式转换 | 0.5 小时 | 阶段 2 |
| 4. 模型微调（50 epochs, GPU） | 4-6 小时 | 阶段 1+3 |
| 5. 量化 + 导出 | 1 小时 | 阶段 4 |
| 6. C 代码生成 | 0.5 小时 | 阶段 5 |
| 7. MCU 集成（main.c 修改） | 4-8 小时 | 阶段 6 |
| 8. 编译 + 烧录 + 调试 | 4-8 小时 | 阶段 7 |
| **总计** | **约 18-34 小时** | — |

---

## 十二、文件对照表（新工作空间 ↔ 原 Keil 工程）

| 新工作空间文件 | 目的地（Keil 工程） | 说明 |
|---------------|-------------------|------|
| `output/quantized_models/face_detector_int8.tflite` | Cube.AI 输入 | 量化模型 |
| `mcu_deploy/generated/network.c` | `CM7/Core/Src/` | 模型包装 |
| `mcu_deploy/generated/network.h` | `CM7/Core/Inc/` | 模型头文件 |
| `mcu_deploy/generated/network_data.c` | `CM7/Core/Src/` | 权重数据 |
| `mcu_deploy/generated/network_data.h` | `CM7/Core/Inc/` | 权重头文件 |
| `mcu_deploy/ai_postprocess.c` | `CM7/Core/Src/` | NMS 后处理 |
| `mcu_deploy/ai_postprocess.h` | `CM7/Core/Inc/` | 后处理头文件 |
| — | `CM7/Core/Inc/stm32h7xx_hal_conf.h` | 启用 CRC |
| — | `CM7/Core/Src/main.c` | 集成 AI 推理 |
| — | `MDK-ARM/*.uvprojx` | 添加源文件 + include path |

---

## 附录：有用的命令速查

```bash
# 检查 Git LFS 文件是否正确下载
cd model_zoo/stm32ai-modelzoo
git lfs ls-files

# 列出人脸检测模型文件
find face_detection/models/ -name "*.h5" -o -name "*.tflite" -o -name "*.onnx"

# 检查 ONNX 模型算子
python -c "
import onnx
m = onnx.load('../output/face_detector.onnx')
ops = set(n.op_type for n in m.graph.node)
print('Operators:', sorted(ops))
"

# 查看 TFLite 模型信息
python -c "
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path='../output/quantized_models/face_detector_int8.tflite')
interpreter.allocate_tensors()
print('Input:', interpreter.get_input_details())
print('Output:', interpreter.get_output_details())
"
```
