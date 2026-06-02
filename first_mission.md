``
你需要完成以下任务：

## 任务：训练一个轻量级人脸检测模型，用于部署到 STM32H747I-DISCO 嵌入式平台

### 目标平台约束
- MCU: STM32H747XI, Cortex-M7 @ 400MHz
- 可用 RAM: 约 512KB (DTCM 128KB + D1 AXI SRAM 512KB + D2 SRAM 288KB)
- 可用 Flash: 1MB (CM7 Bank1)
- 外部 SDRAM: 32MB (可用于存储中间结果/输入图像)
- 输入图像尺寸: 建议 160×120 或 320×240 (RGB888 或灰度)
- 推理引擎: STM32 Cube.AI (支持 Keras/TFLite/ONNX)
- 模型大小限制: Flash 占用 < 512KB, RAM 占用 < 256KB

### 模型选择建议 (按优先级)
1. **SSD MobileNetV1/V2** — FPN 结构, 轻量, Cube.AI 官方示例使用
2. **Ultra-Light-Fast-Generic-Face-Detector** — 专为嵌入式设计, < 1MB
3. **YOLO-Fastest** — 极致裁剪的 YOLO, 适合 MCU
4. **BlazeFace** — Google 的轻量人脸检测, 但可能需要 MobileNetV2 的 DepthwiseConv 支持

### 训练数据集
- 推荐: WIDER Face (http://shuoyang1213.me/WIDERFACE/)
- 或: FDDB (Face Detection Data Set and Benchmark)
- 标注格式: 转换为 YOLO 或 COCO 格式的边界框 (x, y, w, h)

### 训练步骤要求
1. **数据准备**
   - 下载 WIDER Face 数据集
   - 将标注转换为所选模型的格式
   - 数据增强: 随机翻转、裁剪、色彩抖动、缩放
   - 划分训练集/验证集 (80/20)

2. **模型定义**
   - 选择合适的轻量级骨干网络 (MobileNetV1/V2 或 ShuffleNetV2)
   - 添加检测头 (SSD head 或 YOLO head)
   - 输入尺寸建议 160×160 或 320×320

3. **训练配置**
   - 优化器: Adam 或 SGD with momentum
   - 初始学习率: 1e-3 (Adam) 或 1e-2 (SGD)
   - 学习率调度: CosineAnnealing 或 StepLR
   - Batch size: 32-64 (根据 GPU 内存调整)
   - Epochs: 100-300
   - 损失函数: 分类损失 (CrossEntropy/Focal Loss) + 回归损失 (SmoothL1/IoU Loss)

4. **模型量化感知训练 (QAT)**
   - 使用 PyTorch 的量化工具 (torch.quantization)
   - 目标: INT8 量化
   - QAT 训练额外 10-20 epochs
   - 验证量化后精度损失 < 2% mAP

5. **模型导出**
   - 导出为 ONNX 格式 (opset version 11 或 12)
   - 使用 ONNX Runtime 验证导出的模型
   - 可选: 转换为 TensorFlow Lite (通过 onnx2tf 或直接使用 TF 训练)
   - 确保算子集兼容 Cube.AI

6. **测试评估**
   - 在 WIDER Face 验证集上评估 mAP
   - 测试推理时间 (在 PC 上模拟 INT8 推理)
   - 内存占用分析 (参数量 + 中间激活值)

### 输出物
- 训练好的 PyTorch 模型 (.pth)
- 导出的 ONNX 模型 (.onnx)
- TFLite 模型 (.tflite) [推荐, Cube.AI 原生支持更好]
- 训练日志和评估报告
- 一个 Python 推理脚本 (test_camera.py), 可读取单张图片或摄像头进行测试
```