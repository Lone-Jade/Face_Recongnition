# 概述
本项目目标：在 **STM32H747I-DISCO** 开发板上部署**人脸检测系统**。输入包含人脸的图片（通过串口），在自带 DSI 触摸屏上显示图片并框出人脸位置。

# 开发板
| 项目     | 详情                                                       |
| -------- | ---------------------------------------------------------- |
| 开发板   | STM32H747I-DISCO (MB1248D)                                 |
| 主控     | STM32H747XIHx (Cortex-M7 @ 400MHz + Cortex-M4 @ 200MHz)    |
| 外部晶振 | HSE = 25 MHz                                               |
| LCD 面板 | 800×480, OTM8009A 控制器, DSI 接口 (2 data lanes)          |
| SDRAM    | IS42S32800J, 32MB, 挂载于 FMC Bank2, 起始地址 `0xD0000000` |
| 触摸屏   | FT5336 (I2C)                                               |

# 要求
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
2. 阅读新添加的模型和由该模型生成的开发板配套代码
3. 在 Keil5 中集成 Cube.AI 生成的 Runtime 和网络代码
4. 修改 main.c: 加入 AI 推理流程, 输入图片人脸检测, 结果绘制到 LCD
5. 实现USART串口输入图片
6. 在帧缓冲上绘制人脸边界框 (绿色矩形框)
7. 编译通过, 生成可烧录的 .hex/.axf 文件

## 输入要求
- 模型原始文件放在项目 model_stm32 文件夹下；由这个模型在STM32Cube Ai Studio生成的模型文件放在项目 AI, ai_generated_work文件夹下
- 网络权重自动加载到 Flash (由 Ai Studio 生成的 network_data.c)
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
- 所有代码必须写在/* USER CODE BEGIN xxx */和/* USER CODE END xxx */之间
- 严禁修改时钟配置
- 严禁修改 BSP, HAL库等文件

## 其他
### 本机已安装：
STM32CubeMX: D:\EmbeddedSystem\STM32CubeMX\STM32CubeMX.exe
STM32CubeProgrammer:E:\XIWEI++\STM43CubeProgrammer\bin\STM32CubeProgrammer.exe
STM32CubeIDE:D:\EmbeddedSystem\STM32CubeIDE\STM32CubeIDE_2.1.1\STM32CubeIDE\stm32cubeide.exe (非必要不使用)
STM32CubeAIStudio:E:\EmbeddedSystem\stm32cubeaistudio\STM32Cube-AI-Studio\STM32Cube-AI-Studio.exe 
Keil5:D:\EmbeddedSystem\Keil_v5\UV4\UV4.exe
### python环境要求
D:\PythonConda\envs\pytorch_face_detection\python.exe
```