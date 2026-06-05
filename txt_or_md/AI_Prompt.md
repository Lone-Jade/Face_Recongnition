# AI_Prompt.md — STM32H747I-DISCO 人脸检测系统部署完整提示词

---

## 一、项目概述

本项目目标：在 **STM32H747I-DISCO** 开发板上部署**人脸检测系统**。输入包含人脸的图片（通过串口/SD卡/USB/以太网/摄像头），在自带 DSI 触摸屏上显示图片并框出人脸位置。

整个开发流程分为三大阶段：

| 阶段 | 内容 | 工具 |
|------|------|------|
| **阶段A** | LCD DSI 基础显示工程（已完成，作为本项目基线） | STM32CubeMX + Keil5 |
| **阶段B** | 人脸检测模型训练与导出 | Python + PyTorch |
| **阶段C** | 模型部署到 STM32H747I-DISCO，集成 LCD 显示 | Cube.AI / FP-AI-VISION + Keil5 |

本文档提供：阶段A 的完整总结 + 阶段B/C 的详细实施提示词，供下一个 AI 直接执行。

---

## 二、阶段A 基线项目：LCD DSI Command Mode 单缓冲显示

### 2.1 硬件平台

| 项目 | 详情 |
|------|------|
| 开发板 | STM32H747I-DISCO (MB1248D) |
| 主控 | STM32H747XIHx (Cortex-M7 @ 400MHz + Cortex-M4 @ 200MHz) |
| 外部晶振 | HSE = 25 MHz |
| LCD 面板 | 800×480, OTM8009A 控制器, DSI 接口 (2 data lanes) |
| SDRAM | IS42S32800J, 32MB, 挂载于 FMC Bank2, 起始地址 `0xD0000000` |
| 触摸屏 | FT5336 (I2C) |

### 2.2 软件架构概览

```
Flash中的Image[] (ARGB8888, 320×240)
       │
       ▼  DMA2D M2M (CopyBuffer, 位置 x=240,y=160 居中)
       │
SDRAM 帧缓冲 (0xD0000000, 800×480 ARGB8888)
       │
       ▼  LTDC Layer 0 (从SDRAM读取, 输出DPI信号到DSI Wrapper)
       │
DSI Host (Adapted Command Mode, Tearing Effect同步)
       │
       ▼  DSI Link (2 lanes)
       │
OTM8009A 面板 (800×480, RGB888)
```

### 2.3 完整目录结构

```
LCD_DSI_CMD_mode_Single_Buffer/
├── CM7/Core/
│   ├── Inc/
│   │   ├── main.h                  # 定义 LED1 引脚
│   │   ├── stm32h7xx_hal_conf.h    # HAL 模块裁剪
│   │   ├── stm32h7xx_it.h
│   │   ├── gpio.h
│   │   └── usart.h
│   └── Src/
│       ├── main.c                  # ★ 核心应用 (CM7)
│       ├── gpio.c
│       ├── usart.c
│       ├── stm32h7xx_hal_msp.c
│       └── stm32h7xx_it.c
├── CM4/Core/
│   ├── Inc/
│   │   ├── main.h
│   │   ├── stm32h7xx_hal_conf.h
│   │   └── stm32h7xx_it.h
│   └── Src/
│       ├── main.c                  # CM4: 进入STOP模式,等待HSEM唤醒后空转
│       ├── stm32h7xx_hal_msp.c
│       └── stm32h7xx_it.c
├── Common/Src/
│   └── system_stm32h7xx_dualcore_boot_cm4_cm7.c  # 双核启动代码
├── Drivers/
│   ├── CMSIS/                      # CMSIS Core + STM32H7 设备头文件
│   └── STM32H7xx_HAL_Driver/       # HAL 库 (Inc + Src)
├── BSP/
│   ├── Components/
│   │   ├── Common/                 # 抽象接口 (lcd.h, ts.h, camera.h, audio.h)
│   │   ├── otm8009a/               # OTM8009A 驱动 (ID=0x40, RGB888, landscape)
│   │   ├── adv7533/                # HDMI 发射器 (未使用)
│   │   ├── nt35510/                # 备用 LCD 控制器 (未使用)
│   │   ├── is42s32800j/            # SDRAM 组件
│   │   ├── ft5336/                 # 触摸屏控制器
│   │   └── ...                     # 其他外设驱动
│   ├── STM32H747I-DISCO/           # 板级 BSP
│   │   ├── stm32h747i_discovery_lcd.c/.h  # LCD BSP 驱动
│   │   ├── stm32h747i_discovery_sdram.c/.h # SDRAM BSP 驱动
│   │   ├── stm32h747i_discovery_bus.c/.h   # I2C4 总线
│   │   └── ...
│   ├── Inc/
│   │   ├── stm32h747i_discovery_conf.h     # ★ 板级配置
│   │   ├── image_320x240_argb8888.h        # 图片1 (320×240 ARGB8888, 307KB)
│   │   └── life_augmented_argb8888.h       # 图片2 (320×240 ARGB8888, 307KB)
│   └── Utilities/
│       ├── lcd/
│       │   ├── stm32_lcd.c/.h      # UTIL_LCD 高级绘图 + 文字
│       │   └── Fonts/              # 字体 (8/12/16/20/24px)
│       └── CPU/
└── MDK-ARM/
    ├── LCD_DSI_CMD_mode_Single_Buffer.uvprojx  # Keil 工程文件
    ├── stm32h747xx_flash_CM7.sct   # CM7 链接脚本: Flash=0x08000000, RAM=0x20000000
    ├── stm32h747xx_flash_CM4.sct   # CM4 链接脚本: Flash=0x08100000, RAM=0x10000000
    └── startup_stm32h747xx_CM7.s / _CM4.s  # 启动汇编
```

### 2.4 双核启动流程

```
1. 上电 → CM7+CM4 同时启动
2. CM4: 使能 HSEM 中断通知(信号量0) → 进入 D2 STOP 模式 (WFE)
3. CM7: MPU_Config() → 使能 I-Cache/D-Cache → 等待 CM4 进入 STOP
4. CM7: HAL_Init() → SystemClock_Config() (HSE→PLL1→400MHz)
5. CM7: 释放 HSEM 信号量0 → 唤醒 CM4
6. CM4 醒来: 清除 HSEM 标志 → HAL_Init() → while(1) 空转
7. CM7 继续: GPIO/USART 初始化 → SDRAM 初始化 → LCD 初始化
8. CM7 主循环: 拷贝图片 → DSI 刷新 → 等待 TE 回调 → 循环
```

### 2.5 核心配置参数

#### 时钟树
```
HSE (25MHz) → PLL1 (/2, ×64, /2) → SYSCLK = 400MHz (CM7)
                                  → HCLK = SYSCLK/2 = 200MHz
HSE (25MHz) → PLL3 (/5, ×160, /19) → LTDC pixel clock = 42MHz
HSE (25MHz) → DSI PLL (IDF=/5, NDIV=100, ODF=/1)
            → VCO = 5×100 = 500MHz → Byte lane clock ≈ 62.5MHz
```

#### LTDC 时序 (800×480)
```
HSYNC=1, HBP=1, HFP=1, HACT=800 → 总宽度=803
VSYNC=1, VBP=1, VFP=1, VACT=480 → 总高度=483
极性: HS=AL, VS=AL, DE=AL, PC=IPC
```

#### DSI 命令模式配置
```
AdaptedCommandMode:
  - ColorCoding = DSI_RGB888
  - CommandSize = 800 (HACT)
  - AutomaticRefresh = DISABLE (手动调用HAL_DSI_Refresh)
  - TearingEffectSource = DSI_TE_DSILINK
  - TEAcknowledgeRequest = ENABLE
  - 2 Data Lanes, TXEscapeCkdiv = 4
```

#### SDRAM
```
设备: IS42S32800J (32MB)
地址: 0xD0000000 (SDRAM_DEVICE_ADDR)
帧缓冲: LCD_FRAME_BUFFER = 0xD0000000 → Layer0
备用层: LCD_LAYER_1_ADDRESS = 0xD0200000
相机缓冲: CAMERA_FRAME_BUFFER = 0xD0600000
```

#### MPU 配置
```
Region 0: 0x00000000, 4GB, 无访问权限, 子区域禁用掩码=0x87
Region 1: 0xD0000000, 32MB, 完全访问, 可缓存, 不可共享
```

### 2.6 HAL 模块裁剪清单

**已启用的 HAL 模块** (`stm32h7xx_hal_conf.h`):
```
HAL_MODULE_ENABLED          HAL_GPIO_MODULE_ENABLED
HAL_DMA_MODULE_ENABLED      HAL_MDMA_MODULE_ENABLED
HAL_RCC_MODULE_ENABLED      HAL_FLASH_MODULE_ENABLED
HAL_EXTI_MODULE_ENABLED     HAL_PWR_MODULE_ENABLED
HAL_I2C_MODULE_ENABLED      HAL_CORTEX_MODULE_ENABLED
HAL_HSEM_MODULE_ENABLED     HAL_UART_MODULE_ENABLED
HAL_DSI_MODULE_ENABLED      HAL_LTDC_MODULE_ENABLED
HAL_SDRAM_MODULE_ENABLED    HAL_DMA2D_MODULE_ENABLED
HAL_ADC_MODULE_ENABLED
```

### 2.7 Keil5 工程配置要点

**uvprojx 工程设置:**
- 设备包: `Keil.STM32H7xx_DFP.2.6.0`
- 编译器: ARMCC V5.06 update 6
- 两个 Target: `_CM7` 和 `_CM4`
- 优化级别: `-O4` (尺寸优化)
- 生成 HEX: 是
- CM7 链接脚本: `stm32h747xx_flash_CM7.sct` (Flash=0x08000000 1MB, RAM=0x20000000 128KB)
- CM4 链接脚本: `stm32h747xx_sram2_CM4.sct` (SRAM 运行)

**CM7 编译器预定义符号:**
`CORE_CM7, USE_HAL_DRIVER, STM32H747xx`

**CM7 Include Paths:**
```
../CM7/Core/Inc
../Drivers/STM32H7xx_HAL_Driver/Inc
../Drivers/STM32H7xx_HAL_Driver/Inc/Legacy
../Drivers/CMSIS/Device/ST/STM32H7xx/Include
../Drivers/CMSIS/Include
../BSP/Components/Common
../BSP/STM32H747I-DISCO
../BSP/Inc
../BSP/Utilities/CPU
../BSP/Utilities/Fonts
../BSP/Utilities/lcd
../BSP/Components/adv7533
../BSP/Components/is42s32800j
../BSP/Components/otm8009a
../BSP/Components/nt35510
```

**参与编译的 HAL 源文件列表** (27个):
```
stm32h7xx_hal.c          stm32h7xx_hal_cortex.c    stm32h7xx_hal_tim.c
stm32h7xx_hal_tim_ex.c   stm32h7xx_hal_uart.c      stm32h7xx_hal_uart_ex.c
stm32h7xx_hal_rcc.c      stm32h7xx_hal_rcc_ex.c    stm32h7xx_hal_flash.c
stm32h7xx_hal_flash_ex.c stm32h7xx_hal_gpio.c      stm32h7xx_hal_hsem.c
stm32h7xx_hal_dma.c      stm32h7xx_hal_dma_ex.c    stm32h7xx_hal_mdma.c
stm32h7xx_hal_pwr.c      stm32h7xx_hal_pwr_ex.c    stm32h7xx_hal_i2c.c
stm32h7xx_hal_i2c_ex.c   stm32h7xx_hal_exti.c      stm32h7xx_hal_adc.c
stm32h7xx_hal_adc_ex.c   stm32h7xx_hal_dma2d.c     stm32h7xx_hal_dsi.c
stm32h7xx_hal_ltdc.c     stm32h7xx_hal_ltdc_ex.c   stm32h7xx_hal_sdram.c
stm32h7xx_ll_fmc.c
```

**BSP 源文件:**
```
stm32h747i_discovery.c          stm32h747i_discovery_lcd.c
stm32h747i_discovery_sdram.c    stm32h747i_discovery_bus.c
otm8009a.c                      otm8009a_reg.c
adv7533_reg.c                   stm32_lcd.c
system_stm32h7xx_dualcore_boot_cm4_cm7.c
```

### 2.8 main.c 关键代码流程 (CM7)

```c
main() {
    MPU_Config();           // SDRAM区域设为可缓存
    SCB_EnableICache();
    SCB_EnableDCache();
    // 等待CM4进入STOP → HAL_Init() → SystemClock_Config()
    // 释放HSEM唤醒CM4
    MX_GPIO_Init();         // LED1 (PI12)
    MX_USART1_UART_Init();  // USART1, 115200
    BSP_LED_Init(LED3);     // PI14
    BSP_SDRAM_Init(0);      // 初始化IS42S32800J
    LCD_Init();             // DSI PLL → DSI Cmd Mode → LTDC → OTM8009A
    // 配置 LCD Context (800×480, ARGB8888, 4Bpp)
    // 禁用 DSI Wrapper → 初始化 LTDC Layer0 → 使能 DSI Wrapper
    LCD_BriefDisplay();     // 蓝色标题栏 + 白色背景
    CopyBuffer(Images[0], LCD_FRAME_BUFFER, 240, 160, 320, 240);
    HAL_DSI_Refresh(&hlcd_dsi);

    while(1) {
        if(pending_buffer < 0) {       // 等待上一帧刷新完毕
            CopyBuffer(Images[i], ...);  // DMA2D 拷贝到SDRAM
            HAL_DSI_Refresh(&hlcd_dsi);  // 触发DSI刷新
        }
        HAL_Delay(2000);
    }
}
```

**关键函数说明:**

| 函数 | 功能 |
|------|------|
| `LCD_Init()` | 完整的 LCD 初始化: 硬件复位 → MSP 时钟+NVIC → PLL3 (LTDC时钟) → DSI PLL → DSI 命令模式配置 → LTDC → DSI PHY 时序 → OTM8009A 驱动初始化 |
| `CopyBuffer()` | DMA2D M2M 模式拷贝图片到帧缓冲指定位置 (OutputOffset = 800-xsize) |
| `HAL_DSI_EndOfRefreshCallback()` | Tearing Effect 回调, 设置 `pending_buffer = -1` 表示可以写入新帧 |
| `LCD_BriefDisplay()` | 使用 UTIL_LCD 绘制标题栏和说明文字 |

### 2.9 OTM8009A 面板初始化要点

- ID 读取: `0x40`
- 颜色格式: `OTM8009A_COLMOD_RGB888` (24bpp)
- 方向: `LCD_ORIENTATION_LANDSCAPE` (800×480)
- I/O 接口: 通过 DSI 命令 (`DSI_IO_Write` / `DSI_IO_Read`) 访问 DCS 寄存器
- 初始化时所有 LP (Low Power) 命令使能; 完成后全部禁用仅保留高速模式
- 使能 BTA (Bus Turnaround) 流控; RX 强制低功耗

---

## 三、阶段B：人脸检测模型训练 (Python + PyTorch)

### 3.1 训练提示词

```
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

### 3.2 训练代码框架参考

```python
# 建议的项目结构
face_detection_training/
├── data/
│   ├── wider_face/           # 数据集
│   └── data_loader.py        # 数据加载器
├── models/
│   ├── detector.py           # 模型定义
│   └── backbone.py           # 骨干网络
├── train.py                  # 训练脚本
├── train_qat.py              # 量化感知训练脚本
├── export_onnx.py            # 导出 ONNX
├── export_tflite.py          # 导出 TFLite
├── test_image.py             # 单张图片测试
├── test_camera.py            # 摄像头实时测试
├── utils/
│   ├── loss.py               # 损失函数
│   ├── metrics.py            # mAP 评估
│   └── augment.py            # 数据增强
└── config.py                 # 配置文件
```

### 3.3 输出模型要求

| 指标 | 目标值 |
|------|--------|
| 输入尺寸 | 160×160×3 或 320×320×3 |
| 检测类别 | 1 (face) |
| 模型大小 (INT8) | < 500KB |
| RAM 占用 | < 256KB |
| 推理时间 (M7@400MHz) | < 500ms |
| mAP@0.5 (WIDER Face Easy) | > 0.70 |

---

## 四、阶段C：模型部署到 STM32H747I-DISCO

### 4.1 Cube.AI 集成提示词

```
## 任务：将训练好的人脸检测模型部署到 STM32H747I-DISCO

### 4.1.1 Cube.AI 配置

在 STM32CubeMX 中操作：

1. **打开 Cube.AI 插件**
   - 进入 "Software Packs" → "Select Components"
   - 启用 "STMicroelectronics.X-CUBE-AI" (最新版本, 如 9.0.0)
   - 在 Pinout 界面会出现 "Additional Software" → "X-CUBE-AI"

2. **加载模型**
   - 点击 X-CUBE-AI → "Add Network"
   - 选择导出的 ONNX 或 TFLite 模型文件
   - Cube.AI 会分析模型并显示:
     * Flash 占用 (权重 + 代码)
     * RAM 占用 (运行时缓冲)
     * 推理时间估算 (M7 @ 400MHz)
     * 算子兼容性报告
   - 如果某些算子不支持, Cube.AI 会报错, 需要回到训练阶段修改模型

3. **生成代码**
   - Cube.AI 会生成以下文件:
     * `network.c` / `network.h`        — 模型包装函数
     * `network_data.c` / `network_data.h` — 权重数据 (存入 Flash)
     * `app_x-cube-ai.c` / `.h`          — AI 应用模板

4. **CubeMX 需要的额外配置**
   - 启用 CRC 外设 (Cube.AI 验证用)
   - 如果使用摄像头: 启用 DCMI + DMA
   - 如果使用 SD 卡: 启用 SDMMC + FATFS
   - 增大 USART 波特率 (921600 或更高, 方便传输图片)
```

### 4.1.2 需要新增的 HAL 模块

在原项目基础上, `stm32h7xx_hal_conf.h` 需要额外启用:
```c
#define HAL_TIM_MODULE_ENABLED      // 定时器, 用于帧率统计
// 根据输入方式选择:
#define HAL_SD_MODULE_ENABLED       // SD卡读取图片
#define HAL_DCMI_MODULE_ENABLED     // DCMI 摄像头接口
#define HAL_CRC_MODULE_ENABLED      // Cube.AI 需要
```

### 4.1.3 需要新增的 BSP 模块

```
BSP/Components/
├── ov5640/                   # 如果使用摄像头 (已存在于项目中)
│   ├── ov5640.c/.h
│   └── ov5640_reg.c/.h

BSP/STM32H747I-DISCO/
└── stm32h747i_discovery_camera.c/.h  # 相机 BSP (已存在)

Middlewares/ST/
└── AI/                       # Cube.AI 生成的 Runtime 库
    ├── Inc/
    │   ├── ai_platform.h
    │   ├── ai_network.h
    │   └── ...
    └── Lib/
        └── libNetworkRuntime.a
```

### 4.1.4 修改 main.c 的架构

```c
// === 新增 include ===
#include "app_x-cube-ai.h"       // Cube.AI 应用层
#include "ai_platform.h"
#include "network.h"             // 模型网络结构

// === 新增全局变量 ===
#define INPUT_WIDTH   160    // 模型输入宽
#define INPUT_HEIGHT  120    // 模型输入高
#define INPUT_CHANNELS 3     // RGB

static ai_handle network;
static ai_float input_data[INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS];
static ai_float output_data[MAX_DETECTIONS * 6];  // [x1,y1,x2,y2,score,class] each

// === 修改后的 main 函数流程 ===
main() {
    // ... 原有初始化 (MPU, Cache, HAL, SystemClock, HSEM, GPIO, USART, LED, SDRAM, LCD) ...

    // ★ 新增: 初始化 AI 模型
    ai_init();
    ai_network_create(&network, ...);

    // ★ 新增: 初始化输入源 (选择以下之一):
    // 方式A: 从 SD 卡读取图片文件
    BSP_SD_Init();
    FATFS_Init();
    // 方式B: 使用摄像头
    BSP_CAMERA_Init();
    // 方式C: 通过 USART 接收图片数据
    // (已有 MX_USART1_UART_Init(), 在中断中接收)

    while(1) {
        // 1. 获取输入图片
        load_image_from_source(input_data);  // 读取并缩放到 INPUT_WIDTH×INPUT_HEIGHT

        // 2. AI 推理
        ai_network_run(network, input_data, output_data);

        // 3. 后处理 (NMS 非极大值抑制)
        int num_faces = post_process(output_data, detections);

        // 4. 准备 LCD 显示
        // 4a. 将输入图片放大并拷贝到帧缓冲
        CopyBuffer(scaled_image, LCD_FRAME_BUFFER, ...);

        // 4b. 在帧缓冲上绘制人脸检测框
        for (int i = 0; i < num_faces; i++) {
            draw_rect_on_buffer(LCD_FRAME_BUFFER,
                detections[i].x1 * SCALE_X,
                detections[i].y1 * SCALE_Y,
                detections[i].x2 * SCALE_X,
                detections[i].y2 * SCALE_Y,
                COLOR_GREEN);
        }

        // 5. 刷新显示
        HAL_DSI_Refresh(&hlcd_dsi);
        while(pending_buffer >= 0);  // 等待 TE 回调

        HAL_Delay(100);  // 约 10fps
    }
}
```

### 4.1.5 人脸检测后处理伪代码

```c
typedef struct {
    float x1, y1, x2, y2;
    float score;
} Detection;

// NMS 阈值
#define NMS_THRESHOLD  0.45f
#define SCORE_THRESHOLD 0.5f
#define MAX_DETECTIONS 20

int post_process(float* raw_output, Detection* detections) {
    // 1. 解析模型输出 (根据你的模型输出格式调整)
    //    SSD 输出: [num_anchors × (4 + num_classes + 1)]
    //    YOLO 输出: [grid_h × grid_w × (5 + num_classes)]

    // 2. 坐标解码 (anchor box → 绝对坐标)

    // 3. 筛选 score > SCORE_THRESHOLD 的框

    // 4. NMS 去重

    // 5. 返回检测到的人脸数量
}
```

### 4.1.6 图片输入源实现选项

**选项A: USART 串口接收 (推荐用于调试)**
```c
// 使用 YModem 或自定义协议从 PC 传输图片
// PC端: Python 脚本读取图片 → 串口发送
// MCU端: USART IDLE + DMA 接收, 存入 SDRAM 缓冲区
```

**选项B: SD 卡读取 (推荐用于演示)**
```c
// 使用 FATFS 读取 BMP/JPEG 文件
// 使用 STM32H7 的 JPEG 硬件解码器 (HAL_JPEG_MODULE_ENABLED)
// 缩小/裁剪到模型输入尺寸
```

**选项C: DCMI 摄像头 (推荐用于实时检测)**
```c
// 配置 OV5640 → DCMI → DMA → SDRAM 缓冲区
// 从 640×480 (或更低) 缩放到模型输入尺寸
// 使用 DMA2D 进行缩放和颜色转换
```

### 4.1.7 Keil5 工程需要添加的文件组

在 uVision 工程中新增:

**Group: Application/User/AI:**
```
app_x-cube-ai.c
network.c
network_data.c
```

**Group: Middlewares/ST/AI:**
```
(由 Cube.AI 生成的 Runtime 源文件)
```

**新增 Include Paths:**
```
../Middlewares/ST/AI/Inc
../CM7/Core/Inc/AI
```

### 4.1.8 内存规划

```
SDRAM 0xD0000000:
  ├── 0xD0000000  LCD Frame Buffer (800×480×4 = 1.5MB)
  ├── 0xD0200000  Layer1 / 备用 (可选)
  ├── 0xD0400000  AI Input Buffer (320×240×4 = 300KB)
  ├── 0xD0500000  Camera Snapshot (640×480×2 = 600KB RGB565)
  └── 0xD0600000  Camera Frame Buffer (CAMERA_FRAME_BUFFER)

Internal SRAM:
  ├── DTCM (0x20000000): 关键变量, 栈
  ├── AXI SRAM (0x24000000): AI 运行时缓冲 (activation buffer), DMA 描述符
  └── D2 SRAM (0x30000000): 外设数据, FATFS 缓冲区
```

### 4.1.9 Cube.AI 生成的 API 使用模板

```c
#include "ai_platform.h"
#include "network.h"
#include "network_data.h"

// 全局句柄和缓冲
static ai_handle network_handle = AI_HANDLE_NULL;
static ai_network_report report;

// 初始化
int ai_init(void) {
    ai_error err;

    // 创建网络实例
    err = ai_network_create(&network_handle, AI_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE) return -1;

    // 初始化网络 (分配运行时缓冲)
    const ai_network_params params = {
        AI_NETWORK_DATA_WEIGHTS(ai_network_data_weights_get()),
        AI_NETWORK_DATA_ACTIVATIONS(activations_buffer)  // 需预分配
    };
    err = ai_network_init(network_handle, &params);
    if (err.type != AI_ERROR_NONE) return -1;

    return 0;
}

// 推理一次
int ai_run(float* input, float* output) {
    ai_iarray in_array, out_array;
    ai_buffer in_buf, out_buf;

    // 配置输入
    in_buf = ai_network_inputs_get(network_handle, NULL);
    // 填充 input_data 到 in_buf.data ...

    // 配置输出
    out_buf = ai_network_outputs_get(network_handle, NULL);

    // 执行推理
    ai_network_run(network_handle, &in_buf, &out_buf);

    // 从 out_buf.data 复制到 output
    return 0;
}
```

### 4.1.10 DMA2D 用于图像预处理

可以利用 DMA2D (Chrom-ART) 加速图像预处理:

```c
// 颜色空间转换: RGB565 → ARGB8888
void convert_rgb565_to_argb8888(uint16_t* src, uint32_t* dst, int w, int h) {
    hdma2d.Init.Mode         = DMA2D_M2M_PFC;  // 带像素格式转换
    hdma2d.Init.ColorMode    = DMA2D_OUTPUT_ARGB8888;
    hdma2d.Init.OutputOffset = LCD_WIDTH - w;
    hdma2d.LayerCfg[1].InputColorMode = DMA2D_INPUT_RGB565;
    // ... 启动转换
}

// 缩放: 640×480 → 160×120 (4倍缩小)
void downscale_image(uint32_t* src, uint32_t* dst, int src_w, int src_h,
                     int dst_w, int dst_h) {
    // DMA2D 没有硬件缩放, 可使用简单的双线性插值
    // 或使用 MDMA 批量搬运
}
```

---

## 五、FP-AI-VISION 替代方案

如果 Cube.AI 直接部署遇到困难, 可以考虑使用 ST 官方的 **FP-AI-VISION** 功能包:

### FP-AI-VISION 简介
- ST 官方 AI 视觉功能包, 专为 STM32H747I-DISCO 设计
- 包含完整的人脸检测/识别/人体姿态估计示例
- 基于 TensorFlow Lite Micro 运行时
- 集成相机 (OV5640) + LCD 显示 + TouchGFX

### 使用方法
1. 从 ST 官网下载 `FP-AI-VISION` 软件包
2. 导入到 STM32CubeMX (或直接使用预构建的工程)
3. 替换预置的模型文件为你训练的人脸检测模型
4. 调整输入/输出张量格式和预处理逻辑
5. 编译烧录

---

## 六、PC 端测试工具 (Python)

### 6.1 串口图片传输工具

```python
# pc_send_image.py
import serial
import struct
from PIL import Image

def send_image_via_uart(port, image_path, baudrate=921600):
    """通过串口发送图片到 STM32"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((320, 240))  # 或 160×120 取决于模型输入

    ser = serial.Serial(port, baudrate, timeout=1)

    # 发送图片尺寸
    w, h = img.size
    ser.write(struct.pack('<HH', w, h))

    # 发送像素数据 (RGB888)
    data = img.tobytes()
    ser.write(data)

    ser.close()
```

### 6.2 SD 卡测试图片准备

```python
# prepare_test_images.py
from PIL import Image
import os

# 将 WIDER Face 测试图片转换为 BMP 格式存入 SD 卡
def prepare_sd_card_images(input_dir, output_dir, size=(320, 240)):
    for fname in os.listdir(input_dir):
        img = Image.open(os.path.join(input_dir, fname))
        img = img.resize(size)
        # 保存为 BMP (24-bit)
        img.save(os.path.join(output_dir, fname.replace('.jpg', '.bmp')))
```

---

## 七、总结：向新 AI 交付的文件包

如果你想将此任务委托给另一个 AI, 请交付以下内容:

### 7.1 从本项目复制的文件 (阶段A基线)

```
LCD_DSI_CMD_mode_Single_Buffer/    (整个文件夹)
```

### 7.2 CubeMX 项目文件 (.ioc)

在新项目中用 CubeMX 生成初始代码后, 手动修改的内容 (详见第二章节):
- `main.c` 中的 LCD_Init / LTDC_Init / CopyBuffer / LCD_MspInit
- `stm32h7xx_hal_conf.h` 中的 HAL 模块启用列表
- `stm32h747i_discovery_conf.h` 中的板级配置
- Keil 工程中的 Include 路径和源文件分组

### 7.3 已训练好的模型文件

```
face_detection_model.onnx       (或 .tflite)
face_detection_model.h5         (Keras 格式, 可选)
weights_quantized_int8.pth      (PyTorch 量化权重, 可选)
```

### 7.4 训练的图片数据(可选, 太大可省略)

```
test_images/                    (10-20张测试图片)
coco_annotations.json           (标注信息, 可选)
```

### 7.5 完整的提示词文档

```
AI_Prompt.md                    (本文档)
```

---

## 八、给执行 AI 的总提示词

```
你是嵌入式AI部署专家。你要在STM32H747I-DISCO开发板上完成人脸检测系统。

## 已提供的基线项目
项目目录包含一个完整的LCD DSI显示基线工程，基于STM32H747I-DISCO平台。
该工程已实现:
- 双核启动 (CM7 + CM4)
- SDRAM 初始化 (32MB @ 0xD0000000)
- LTDC (800×480 ARGB8888) + DSI Command Mode + OTM8009A LCD 驱动
- DMA2D 图像拷贝
- UTIL_LCD 文字和图形绘制
- USART1 串口通信
- 定时切换显示两张预设图片

## 你要完成的任务
1. 阅读理解基线项目的代码结构和关键函数
2. 使用 STM32CubeMX + Cube.AI 插件加载人脸检测模型 (.tflite/.onnx)
3. 在 Keil5 中集成 Cube.AI 生成的 Runtime 和网络代码
4. 修改 main.c: 加入 AI 推理流程, 输入图片人脸检测, 结果绘制到 LCD
5. 实现USART串口输入图片
6. 在帧缓冲上绘制人脸边界框 (绿色矩形框)
7. 编译通过, 生成可烧录的 .hex 文件

## 输入要求
- 模型文件放在项目 Middlewares/ST/AI/ 目录下
- 网络权重自动加载到 Flash (由 Cube.AI 生成的 network_data.c)
- 运行时激活缓冲放在 AXI SRAM (0x24000000)
- 输入图片缓冲放在 SDRAM (0xD0400000 起)
- 继续使用现有的 DSI Command Mode 和 TE 同步机制

## 输出要求
- 可以在 Keil5 中直接编译的完整工程
- main.c 包含完整的 AI 推理 + 结果显示流程
- 添加详细注释说明每个新增步骤
- 提供 PC 端测试脚本 (Python) 用于串口发送测试图片

## 约束
- 不要修改现有的 BSP 驱动层 (LCD/SDRAM/BUS)
- 保持双核启动架构不变
- 不要使用 RTOS (裸机)
- 模型必须部署在 CM7 核心
- 帧缓冲继续使用 SDRAM 单一缓冲 (0xD0000000), TE 同步不变
```

---

## 附录A：关键文件映射表

| 文件 | 功能 | 是否需要修改 |
|------|------|------------|
| `CM7/Core/Src/main.c` | CM7 主程序 | **是 — 加入AI推理 + 检测框绘制** |
| `CM7/Core/Inc/main.h` | CM7 头文件 | **是 — 新增AI相关声明** |
| `CM7/Core/Inc/stm32h7xx_hal_conf.h` | HAL 模块配置 | **是 — 启用 CRC/JPEG/DCMI/SD等** |
| `CM4/Core/Src/main.c` | CM4 固件 | 否 |
| `BSP/Inc/stm32h747i_discovery_conf.h` | 板级配置 | 可能 (调整缓冲地址) |
| `MDK-ARM/xxx.uvprojx` | Keil 工程 | **是 — 添加新文件组和Include路径** |
| `MDK-ARM/stm32h747xx_flash_CM7.sct` | CM7 链接脚本 | 可能 (如果Flash/RAM不够) |
| `Middlewares/ST/AI/*` | Cube.AI Runtime | **新增** |
| `CM7/Core/Src/network*.c` | 模型权重+包装 | **新增 (Cube.AI生成)** |

## 附录B：参考资料

- STM32H747I-DISCO 用户手册: UM2198
- STM32H747xx 参考手册: RM0399
- OTM8009A 数据手册: DS12180
- X-CUBE-AI 用户手册: UM2526
- FP-AI-VISION 用户手册: UM2611
- STM32 Cube.AI 官方文档: https://www.st.com/en/embedded-software/x-cube-ai.html
- WIDER Face 数据集: http://shuoyang1213.me/WIDERFACE/
