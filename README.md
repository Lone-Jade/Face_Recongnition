# 基于STM32H747I-DISCO的人脸检测系统

**Face Detection System on STM32H747I-DISCO**

> **课程作业声明 / Coursework Disclaimer**  
> 本项目为同济大学《嵌入式系统》课程大作业。  
> This project is a coursework assignment for the *Embedded Systems* course at **Tongji University**.

[中文](#chinese) | [English](#english)

---

<span id="chinese"></span>
## 中文

### 项目简介

本项目在**STM32H747I-DISCO**开发板上实现了基于深度学习的人脸检测系统。系统通过USART串口接收PC端发送的图像，利用STM32 Cube.AI运行时库在Cortex-M7上运行**YuNet-320**轻量级人脸检测模型，并在OTM8009A LCD面板上实时显示带检测框标注的结果。

**核心功能：**
- MIPI DSI Command Mode LCD显示（800×480, DMA2D硬件加速）
- Cube.AI部署YuNet-320人脸检测模型（Float32, ~140M MACCs）
- USART二进制协议接收320×240图像（支持单张和多张批处理）
- 检测后处理（盒子解码 + NMS去重）+ 绿色边框绘制
- 双核架构：CM7主控+AI推理，CM4待机辅助

**项目结构：**

```
Face_Detection/
├── CM7/Core/              # CM7 主核源码
│   ├── Src/
│   │   ├── main.c         # 主循环：LCD显示 + AI推理 + USART接收
│   │   ├── ai_detection.c # 预处理/后处理/绘制
│   │   └── usart.c        # USART1配置
│   └── Inc/
│       ├── ai_detection.h # 检测参数、SDRAM布局、接口声明
│       └── stm32h7xx_hal_conf.h
├── CM4/Core/              # CM4 辅助核源码（待机）
├── AI/App/                # Cube.AI 运行时（app_x-cube-ai.c/h）
├── ai_generated_network/  # Cube.AI 生成的网络代码（network.c/h + data）
├── PC_side/
│   └── send_image.py      # PC端图像发送脚本
├── MDK-ARM/               # Keil MDK 工程文件
├── model_stm32/           # STM32 Model Zoo 配置与文档
├── txt_or_md/
│   ├── report.md          # 实验报告
│   └── debug_code_reference.c  # 调试代码参考
└── .ioc                   # STM32CubeMX 配置文件
```

### 软硬件平台

| 项目 | 规格 |
|------|------|
| 开发板 | STM32H747I-DISCO (Discovery Kit) |
| MCU | STM32H747XI (Cortex-M7 @ 480MHz + Cortex-M4 @ 240MHz) |
| 显示 | OTM8009A (800×480 RGB888, MIPI DSI Command Mode) |
| 外存 | IS42S32800J SDRAM 32MB @ 0xD0000000 |
| 通信 | USART1 (PA9/PA10, 115200bps) |
| IDE | Keil MDK-ARM uVision 5, ARMCC V5.06 |
| AI工具链 | STM32 Cube.AI / X-CUBE-AI (ST.AI Embedded Client API) |
| PC环境 | Python 3.8+ / pyserial / Pillow |
| 设备包 | Keil.STM32H7xx_DFP.2.6.0 |

### 模型来源

本项目的YuNet-320人脸检测模型来自 **STM32 Model Zoo**（ST官方模型动物园）：

- **仓库地址：** [stm32ai-modelzoo/face_detection](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/master/face_detection/)
- **模型名称：** `yunetn_320` (YuNet Nano, 输入 320×320)
- **模型信息：**
  - 输入：1×3×320×320, Float32, NCHW, BGR
  - 输出：12个张量（cls×3, obj×3, bbox×4×3, kps×10×3），3个stride（8, 16, 32）
  - 参数量：~295KB（权重）
  - 计算量：~140M MACCs
  - 激活内存：~2.46MB（位于SDRAM 0xD0800000）

模型通过**STM32 Cube.AI Studio**（或X-CUBE-AI CLI）编译为嵌入式C代码，生成在`ai_generated_network/`和`AI/App/`目录下。如需重新生成：

```bash
# STM32 Cube.AI CLI 方式
stm32ai generate -m yunetn_320.onnx -o ai_generated_network/ --target stm32h7
```

### 数据集获取方式

本项目检测的人脸图像来自 **WIDER Face** 数据集（使用其子集用于测试）：

- **WIDER Face 官网：** [http://shuoyang1213.me/WIDERFACE/](http://shuoyang1213.me/WIDERFACE/)
- **下载方式：**
  1. 访问官网下载 WIDER_train.zip / WIDER_val.zip / WIDER_test.zip
  2. 解压后在 `WIDER_test/images/` 中选择图像文件夹（如 `0--Parade`, `1--Handshaking` 等）
  3. 使用 `PC_side/send_image.py` 将图像发送到STM32

**注意：** WIDER Face的ground truth标注文件用于PC端模型训练/评估。在STM32端只需原始图像即可，检测由模型自动完成。

### 使用方法

#### 1. 编译固件

在Keil MDK-ARM中打开 `MDK-ARM/LCD_DSI_CMD_mode_Single_Buffer.uvprojx`，选择目标 `LCD_DSI_CMD_mode_Single_Buffer_CM7`，编译下载到开发板。

> **注意：** `.gitignore` 排除了 `Drivers/` 和 `Middlewares/` 目录。克隆后需通过STM32CubeMX打开 `.ioc` 文件重新生成HAL驱动代码。

#### 2. PC端准备

```bash
pip install pyserial pillow
```

#### 3. 发送图像

```bash
# 单张图像（开发板自动识别并循环显示检测结果）
python PC_side/send_image.py path/to/face.jpg

# 整个文件夹（最大10张，开发板循环显示）
python PC_side/send_image.py WIDERFace/WIDER_test/images/0--Parade

# 指定串口和波特率
python PC_side/send_image.py face.jpg COM4 921600
```

#### 4. 观察结果

- 编译固件中包含2张测试图像，开机自动循环显示并进行人脸检测
- 通过USART发送的图像会覆盖内置图像，在LCD上显示检测结果
- LCD底部显示已检测人脸数量（"Faces detected: N"）
- 检测到的人脸用绿色边框标注

#### 5. 内置/外发两种模式

- **内置模式**（默认）：使用Flash中烧写的 `face_img_0` 和 `face_img_1` 循环检测
- **USART模式**：PC端发送图像后自动切换，循环显示所有接收到的图像

### 通信协议

**单张图像：**
```
[4字节 "FACE" (0x45434146)] [4字节 数据长度 LE] [原始 ARGB8888 像素数据]
图像固定尺寸: 320 × 240 × 4 = 307200 字节
```

**多图结束标志：**
```
[4字节 "MULT" (0x544C554D)] [4字节 图像数量 LE] [4字节 0]
```

### SDRAM 内存布局

| 地址范围 | 用途 | 大小 |
|----------|------|------|
| 0xD0000000 | LCD帧缓冲 (800×480×4) | ~1.5MB |
| 0xD0400000 | USART接收图像缓冲 (最多10张) | ~3MB |
| 0xD0600000 | 相机缓冲（预留） | - |
| 0xD0800000 | AI推理激活缓冲 | ~2.46MB |

### 注意事项

1. **D-Cache一致性**：预处理后必须调用 `SCB_CleanDCache_by_Addr()` 将数据从D-Cache刷入SDRAM，否则AI推理会读到脏数据
2. **Keil项目依赖**：克隆后需要STM32CubeMX重新生成 `Drivers/` 和 `Middlewares/`
3. **Cube.AI版本**：推荐使用STM32 Cube.AI Studio v1.0+ 或 X-CUBE-AI v9.0.0+
4. **串口连接**：使用板载ST-Link/V3-1虚拟串口，Windows下通常识别为 `COM3` 或 `COM4`
5. **图像格式**：PC端只发送320×240 ARGB8888原始像素数据，预处理（缩放/通道转换）在PC端完成
6. **检测阈值**：当前阈值（DET_THRESHOLD=0.40, NMS_THRESHOLD=0.40, MIN_BOX_SIZE=10.0）经过Float32模型调优，INT8量化模型可能需要调整
7. **电源**：建议使用开发板附带的5V/3A电源适配器，USB供电可能不够稳定

---

<span id="english"></span>
## English

### Project Overview

This project implements a **deep learning-based face detection system** on the **STM32H747I-DISCO** discovery kit. The system receives images from a PC via USART serial, runs the **YuNet-320** lightweight face detection model using the STM32 Cube.AI runtime on the Cortex-M7 core, and displays the results with bounding box annotations on the OTM8009A LCD panel in real time.

**Core Features:**
- MIPI DSI Command Mode LCD display (800×480, DMA2D hardware acceleration)
- Cube.AI-deployed YuNet-320 face detection model (Float32, ~140M MACCs)
- USART binary protocol for receiving 320×240 images (single or batch)
- Detection post-processing (box decoding + NMS) with green bounding box drawing
- Dual-core architecture: CM7 handles main control + AI inference, CM4 in standby

**Project Structure:**

```
Face_Detection/
├── CM7/Core/              # CM7 core source code
│   ├── Src/
│   │   ├── main.c         # Main loop: LCD display + AI inference + USART receive
│   │   ├── ai_detection.c # Preprocessing / postprocessing / drawing
│   │   └── usart.c        # USART1 configuration
│   └── Inc/
│       ├── ai_detection.h # Detection params, SDRAM layout, API declarations
│       └── stm32h7xx_hal_conf.h
├── CM4/Core/              # CM4 auxiliary core (standby)
├── AI/App/                # Cube.AI runtime (app_x-cube-ai.c/h)
├── ai_generated_network/  # Cube.AI generated network code (network.c/h + data)
├── PC_side/
│   └── send_image.py      # PC-side image sender script
├── MDK-ARM/               # Keil MDK project files
├── model_stm32/           # STM32 Model Zoo configs and docs
├── txt_or_md/
│   ├── report.md          # Experiment report (Chinese)
│   └── debug_code_reference.c  # Debug code reference
└── .ioc                   # STM32CubeMX project file
```

### Hardware & Software Platform

| Item | Specification |
|------|---------------|
| Board | STM32H747I-DISCO (Discovery Kit) |
| MCU | STM32H747XI (Cortex-M7 @ 480MHz + Cortex-M4 @ 240MHz) |
| Display | OTM8009A (800×480 RGB888, MIPI DSI Command Mode) |
| External RAM | IS42S32800J SDRAM 32MB @ 0xD0000000 |
| Communication | USART1 (PA9/PA10, 115200bps) |
| IDE | Keil MDK-ARM uVision 5, ARMCC V5.06 |
| AI Toolchain | STM32 Cube.AI / X-CUBE-AI (ST.AI Embedded Client API) |
| PC environment | Python 3.8+ / pyserial / Pillow |
| Device Pack | Keil.STM32H7xx_DFP.2.6.0 |

### Model Source

The YuNet-320 face detection model used in this project comes from the **STM32 Model Zoo**:

- **Repository:** [stm32ai-modelzoo/face_detection](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/master/face_detection/)
- **Model Name:** `yunetn_320` (YuNet Nano, 320×320 input)
- **Model Info:**
  - Input: 1×3×320×320, Float32, NCHW, BGR
  - Output: 12 tensors (cls×3, obj×3, bbox×4×3, kps×10×3), 3 strides (8, 16, 32)
  - Parameters: ~295KB (weights)
  - MACCs: ~140M
  - Activation memory: ~2.46MB (at SDRAM 0xD0800000)

The model is compiled into embedded C code using **STM32 Cube.AI Studio** (or X-CUBE-AI CLI), producing the generated files in `ai_generated_network/` and `AI/App/`. To regenerate:

```bash
# Using STM32 Cube.AI CLI
stm32ai generate -m yunetn_320.onnx -o ai_generated_network/ --target stm32h7
```

### Dataset

Face images used for testing come from the **WIDER Face** dataset:

- **WIDER Face website:** [http://shuoyang1213.me/WIDERFACE/](http://shuoyang1213.me/WIDERFACE/)
- **Download steps:**
  1. Download WIDER_train.zip / WIDER_val.zip / WIDER_test.zip from the official site
  2. Extract and navigate to `WIDER_test/images/` to find scene folders (e.g., `0--Parade`, `1--Handshaking`)
  3. Use `PC_side/send_image.py` to send images to the STM32 board

**Note:** WIDER Face ground-truth annotation files are used for PC-side model training/evaluation only. The STM32 side only needs raw images — detection is performed automatically by the model.

### Usage

#### 1. Build Firmware

Open `MDK-ARM/LCD_DSI_CMD_mode_Single_Buffer.uvprojx` in Keil MDK-ARM, select target `LCD_DSI_CMD_mode_Single_Buffer_CM7`, build and download to the board.

> **Note:** The `.gitignore` excludes `Drivers/` and `Middlewares/` directories. After cloning, open the `.ioc` file with STM32CubeMX and regenerate the HAL driver code.

#### 2. PC-side Setup

```bash
pip install pyserial pillow
```

#### 3. Send Images

```bash
# Single image (board auto-detects and loops detection results)
python PC_side/send_image.py path/to/face.jpg

# Entire folder (max 10 images, board cycles through them)
python PC_side/send_image.py WIDERFace/WIDER_test/images/0--Parade

# Specify COM port and baudrate
python PC_side/send_image.py face.jpg COM4 921600
```

#### 4. Observe Results

- The firmware includes 2 built-in test images that cycle on boot with face detection
- USART-sent images override built-in images; detection results appear on LCD
- The LCD bottom line shows detected face count ("Faces detected: N")
- Detected faces are marked with green bounding boxes

#### 5. Two Operating Modes

- **Built-in mode** (default): cycles through `face_img_0` and `face_img_1` from Flash
- **USART mode**: auto-switches when PC sends images; cycles through all received images

### Communication Protocol

**Single image:**
```
[4B "FACE" (0x45434146)] [4B data length LE] [raw ARGB8888 pixel data]
Fixed image size: 320 × 240 × 4 = 307200 bytes
```

**End-of-batch marker:**
```
[4B "MULT" (0x544C554D)] [4B image count LE] [4B 0]
```

### SDRAM Memory Layout

| Address Range | Usage | Size |
|---------------|-------|------|
| 0xD0000000 | LCD Frame Buffer (800×480×4) | ~1.5MB |
| 0xD0400000 | USART Image Buffer (max 10 images) | ~3MB |
| 0xD0600000 | Camera Buffer (reserved) | - |
| 0xD0800000 | AI Activation Buffer | ~2.46MB |

### Important Notes

1. **D-Cache Coherency**: Always call `SCB_CleanDCache_by_Addr()` after preprocessing to flush D-Cache to SDRAM; otherwise the AI engine reads stale data
2. **Keil Project Dependencies**: After cloning, regenerate `Drivers/` and `Middlewares/` using STM32CubeMX with the `.ioc` file
3. **Cube.AI Version**: STM32 Cube.AI Studio v1.0+ or X-CUBE-AI v9.0.0+ recommended
4. **Serial Connection**: Use the onboard ST-Link/V3-1 Virtual COM Port; typically appears as `COM3` or `COM4` on Windows
5. **Image Format**: PC side sends 320×240 ARGB8888 raw pixel data; resizing and format conversion are handled PC-side
6. **Detection Thresholds**: Current values (DET_THRESHOLD=0.40, NMS_THRESHOLD=0.40, MIN_BOX_SIZE=10.0) are tuned for Float32; adjust for INT8 quantized models if needed
7. **Power Supply**: Use the included 5V/3A power adapter; USB power may be insufficient
