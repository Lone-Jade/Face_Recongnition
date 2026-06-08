# 实验报告：基于STM32H747I-DISCO的人脸检测系统

---

## 一、实验目的

1. 掌握STM32H747I-DISCO开发板双核（CM7+CM4）启动流程及外设驱动开发方法
2. 掌握DSI Command Mode + LTDC + DMA2D的LCD显示管线配置，实现图像帧缓冲显示
3. 掌握STM32 Cube.AI工具链的使用，完成深度学习模型从ONNX到嵌入式C代码的转换与部署
4. 实现完整的"PC端发送图像→USART接收→AI推理→LCD显示检测框"嵌入式视觉应用流水线
5. 深入理解嵌入式AI推理中的内存布局（SDRAM分配、NCHW张量格式）、量化精度损失及后处理优化策略

---

## 二、实验平台

### 硬件

| 项目 | 规格 |
|------|------|
| 开发板 | STM32H747I-DISCO (Discovery Kit) |
| MCU | STM32H747XI (双核：Cortex-M7 @ 480 MHz + Cortex-M4 @ 240 MHz) |
| 显示面板 | OTM8009A (800×480 RGB888, MIPI DSI Command Mode) |
| SDRAM | IS42S32800J, 32MB @ 0xD0000000 |
| 串口 | USART1 (PA9-TX, PA10-RX), 115200 baud |
| 连接 | ST-Link/V3 板载调试器 + 虚拟串口 (VCP) |

### 软件

| 工具 | 版本/说明 |
|------|-----------|
| IDE | Keil MDK-ARM uVision 5, ARMCC V5.06 update 6 |
| 设备包 | Keil.STM32H7xx_DFP.2.6.0 |
| Cube.AI | STM32 Cube.AI Studio / X-CUBE-AI, ST.AI Embedded Client API |
| PC端脚本 | Python 3 + pyserial + Pillow |
| 模型 | YuNet-320 (轻量级人脸检测), 输入 1×3×320×320, 输出12个张量 |
| 优化等级 | -O4 |

---

## 三、实验步骤

### 3.1 屏幕显示部分

#### 3.1.1 DSI + LTDC 显示管线初始化

STM32H747I-DISCO采用MIPI DSI Adapted Command Mode驱动OTM8009A面板。显示数据流如下：

```
Images[] (Flash, ARGB8888, 320×240)
  → DMA2D M2M (CopyBuffer, 偏移 x=240,y=160 实现居中)
  → SDRAM Frame Buffer (0xD0000000, 800×480 ARGB8888)
  → LTDC Layer 0 (读取SDRAM, 输出DPI到DSI Wrapper)
  → DSI Host (Adapted Command Mode, Tearing Effect同步)
  → OTM8009A 面板 (800×480, RGB888, 横屏)
```

**关键时钟配置代码：**

```c
// PLL3 配置: HSE 25MHz / 5 * 160 / 19 = 42MHz (LTDC像素时钟)
PeriphClkInitStruct.PLL3.PLL3M = 5;
PeriphClkInitStruct.PLL3.PLL3N = 160;
PeriphClkInitStruct.PLL3.PLL3R = 19;

// DSI PLL: HSE/5 = 5MHz, VCO = 5*100 = 500MHz, 字节通道时钟 ≈ 62.5MHz
dsiPllInit.PLLNDIV = 100;
dsiPllInit.PLLIDF = DSI_PLL_IN_DIV5;
dsiPllInit.PLLODF = DSI_PLL_OUT_DIV1;
```

**LTDC时序配置（800×480）：**

```c
#define VSYNC   1
#define VBP     1
#define VFP     1
#define VACT    480
#define HSYNC   1
#define HBP     1
#define HFP     1
#define HACT    800

hlcd_ltdc.Init.HorizontalSync = HSYNC;
hlcd_ltdc.Init.VerticalSync = VSYNC;
hlcd_ltdc.Init.AccumulatedHBP = HSYNC + HBP;
hlcd_ltdc.Init.AccumulatedVBP = VSYNC + VBP;
hlcd_ltdc.Init.AccumulatedActiveH = VSYNC + VBP + VACT;
hlcd_ltdc.Init.AccumulatedActiveW = HSYNC + HBP + HACT;
hlcd_ltdc.Init.TotalHeigh = VSYNC + VBP + VACT + VFP;
hlcd_ltdc.Init.TotalWidth = HSYNC + HBP + HACT + HFP;
hlcd_ltdc.Init.HSPolarity = LTDC_HSPOLARITY_AL;   // 所有极性: Active Low
hlcd_ltdc.Init.VSPolarity = LTDC_VSPOLARITY_AL;
hlcd_ltdc.Init.DEPolarity = LTDC_DEPOLARITY_AL;
hlcd_ltdc.Init.PCPolarity = LTDC_PCPOLARITY_IPC;   // 像素时钟反相
```

**DSI Command Mode配置：**

```c
CmdCfg.VirtualChannelID = 0;
CmdCfg.ColorCoding = DSI_RGB888;
CmdCfg.CommandSize = HACT;                       // 800
CmdCfg.TearingEffectSource = DSI_TE_DSILINK;     // TE信号来自DSI链路
CmdCfg.TearingEffectPolarity = DSI_TE_RISING_EDGE;
CmdCfg.AutomaticRefresh = DSI_AR_DISABLE;        // 手动刷新模式
CmdCfg.TEAcknowledgeRequest = DSI_TE_ACKNOWLEDGE_ENABLE;
HAL_DSI_ConfigAdaptedCommandMode(&hlcd_dsi, &CmdCfg);
```

#### 3.1.2 帧缓冲复制（DMA2D M2M）

320×240的原始图像居中放置到800×480的帧缓冲中（水平偏移240，垂直偏移160）：

```c
static void CopyBuffer(uint32_t *pSrc, uint32_t *pDst,
                       uint16_t x, uint16_t y,
                       uint16_t xsize, uint16_t ysize)
{
  uint32_t destination = (uint32_t)pDst + (y * 800 + x) * 4;
  uint32_t source = (uint32_t)pSrc;

  hdma2d.Init.Mode          = DMA2D_M2M;
  hdma2d.Init.ColorMode     = DMA2D_OUTPUT_ARGB8888;
  hdma2d.Init.OutputOffset  = 800 - xsize;  // 行尾跳转到下一行
  hdma2d.Init.AlphaInverted = DMA2D_REGULAR_ALPHA;
  hdma2d.Init.RedBlueSwap   = DMA2D_RB_REGULAR;

  hdma2d.LayerCfg[1].AlphaMode      = DMA2D_NO_MODIF_ALPHA;
  hdma2d.LayerCfg[1].InputAlpha     = 0xFF;
  hdma2d.LayerCfg[1].InputColorMode = DMA2D_INPUT_ARGB8888;
  hdma2d.LayerCfg[1].InputOffset    = 0;

  if (HAL_DMA2D_Init(&hdma2d) == HAL_OK) {
    if (HAL_DMA2D_ConfigLayer(&hdma2d, 1) == HAL_OK) {
      if (HAL_DMA2D_Start(&hdma2d, source, destination, xsize, ysize) == HAL_OK) {
        HAL_DMA2D_PollForTransfer(&hdma2d, 100);
      }
    }
  }
}
```

#### 3.1.3 Tearing Effect 同步机制

DSI Command Mode下依赖TE（Tearing Effect）信号来同步帧更新，避免画面撕裂：

```c
static int32_t pending_buffer = -1;  // -1: 空闲, 0: 刷新中

// DSI刷新完成回调 — 由TE信号触发
void HAL_DSI_EndOfRefreshCallback(DSI_HandleTypeDef *hdsi)
{
  if (pending_buffer >= 0) {
    pending_buffer = -1;  // 释放帧缓冲，允许下一帧写入
  }
}

// 主循环中的同步逻辑
if (pending_buffer < 0) {
    CopyBuffer(src_img, (uint32_t *)LCD_FRAME_BUFFER, 240, 160, 320, 240);
    pending_buffer = 0;
    HAL_DSI_Refresh(&hlcd_dsi);  // 触发刷新，等待TE信号
}
```

### 3.2 模型部署部分

#### 3.2.1 模型信息

| 属性 | 值 |
|------|-----|
| 原始模型 | YuNetN-320 (轻量级人脸检测) |
| 输入 | 1×3×320×320, Float32, NCHW, BGR |
| 输出 | 12个张量 (cls×3, obj×3, bbox×3, kps×3) |
| 参数量 | 295KB (权重) |
| MACCs | ~140M |
| 激活内存 | 约2.46MB (包括buf2的大张量) |
| 量化 | Float32 (未做INT8量化) |

#### 3.2.2 Cube.AI 初始化流程

通过ST.AI Embedded Client API初始化模型：

```c
// app_x-cube-ai.c 中的初始化
int aiInit(void) {
  stai_return_code ret_code;

  // 1: 初始化运行时库
  ret_code = stai_runtime_init();

  // 2: 初始化网络模型上下文
  ret_code = user_stai_network_init(network_context);

  // 3: 设置网络激活缓冲区 (DTCM + SDRAM)
  stai_ptr data_activations[] = { DTCMRAM, (ai_handle)0xd0800000 };
  ret_code = stai_network_set_activations(network_context, data_activations,
                                          STAI_NETWORK_ACTIVATIONS_NUM);

  // 4: 获取输入/输出张量指针
  ret_code = stai_network_get_inputs(network_context, stai_input, &in_length);
  ret_code = stai_network_get_outputs(network_context, stai_output, &out_length);

  return 0;
}
```

#### 3.2.3 SDRAM 缓冲区规划

32MB SDRAM (0xD0000000) 的多功能分区：

```
#define LCD_FRAME_BUFFER_ADDR   0xD0000000  // 帧缓冲:   800×480×4 ≈ 1.5MB
#define AI_IMG_BUF_ADDR         0xD0400000  // USART图像: 最大10张 ≈ 3MB
#define AI_CAM_BUF_ADDR         0xD0600000  // 相机缓冲 (预留)
#define AI_ACTIVATION_2_ADDR    0xD0800000  // AI激活buf2: 2.46MB
```

#### 3.2.4 图像预处理

将320×240 ARGB8888图像转为模型所需的1×3×320×320 Float32 NCHW BGR张量：

```c
void ai_preprocess(uint32_t *src, int src_w, int src_h,
                   float *dst, int dst_w, int dst_h)
{
  int x_ratio = ((src_w << 16) / dst_w) + 1;
  int y_ratio = ((src_h << 16) / dst_h) + 1;
  int plane_size = dst_h * dst_w;

  for (int c = 0; c < 3; c++) {
    float *dst_c = dst + c * plane_size;  // NCHW: 通道优先
    for (int y = 0; y < dst_h; y++) {
      int sy = (y * y_ratio) >> 16;
      if (sy >= src_h) sy = src_h - 1;
      for (int x = 0; x < dst_w; x++) {
        int sx = (x * x_ratio) >> 16;
        if (sx >= src_w) sx = src_w - 1;

        uint32_t pixel = src[sy * src_w + sx];
        uint8_t b = pixel & 0xFF;
        uint8_t g = (pixel >> 8) & 0xFF;
        uint8_t r = (pixel >> 16) & 0xFF;

        float val;
        if (c == 0)      val = (float)b;   // B
        else if (c == 1) val = (float)g;   // G
        else             val = (float)r;   // R

        dst_c[y * dst_w + x] = val;
      }
    }
  }
}
```

#### 3.2.5 推理执行

主循环中的推理流水线（预处理→缓存清洗→推理→后处理→绘制）：

```c
if (pending_buffer < 0) {
    // Step 1: DMA2D 复制图像到帧缓冲
    CopyBuffer(src_img, (uint32_t *)LCD_FRAME_BUFFER, 240, 160, 320, 240);

    // Step 2: 预处理 ARGB8888 → NCHW Float32 BGR
    ai_preprocess(src_img, 320, 240, ai_input, 320, 320);
    // 关键: D-Cache中新建的数据必须在AI推理前刷入SDRAM
    SCB_CleanDCache_by_Addr((uint32_t *)ai_input,
                            3 * 320 * 320 * sizeof(float));

    // Step 3: 执行AI推理 (同步模式)
    stai_return_code ret = aiRun();

    // Step 4: 后处理 — 解码YuNet输出 + NMS
    num_detections = ai_postprocess(stai_output, detections, MAX_DETECTIONS,
                                     DET_THRESHOLD, NMS_THRESHOLD, MIN_BOX_SIZE);

    // Step 5: 在帧缓冲上绘制检测框
    ai_draw_detections(detections, num_detections,
                       (uint32_t *)LCD_FRAME_BUFFER,
                       800, 240, 160, 320, 240);

    pending_buffer = 0;
    HAL_DSI_Refresh(&hlcd_dsi);
}
```

#### 3.2.6 后处理 — YuNet输出解码与NMS

YuNet输出12个张量，按3种stride（8, 16, 32）组织：

```
outputs[0..2]  = cls_8, cls_16, cls_32     (1×1×gridsize, 已过sigmoid)
outputs[3..5]  = obj_8, obj_16, obj_32     (1×1×gridsize, 已过sigmoid)
outputs[6..8]  = bbox_8, bbox_16, bbox_32  (1×4×gridsize)
outputs[9..11] = kps_8, kps_16, kps_32     (1×10×gridsize, 本实验未用)
```

解码核心逻辑（使用Interleaved索引，非CHW）：

```c
for (int i = 0; i < gs; i++) {
  for (int j = 0; j < gs; j++) {
    int loc = i * gs + j;
    float score = cls[loc] * obj[loc];  // 联合置信度
    if (score < threshold) continue;

    // 注意: ST AI输出的bbox为Interleaved格式 (N×4), 非CHW (4×N)
    float dx = bbox[loc * 4 + 0];
    float dy = bbox[loc * 4 + 1];
    float dw = bbox[loc * 4 + 2];
    float dh = bbox[loc * 4 + 3];

    // 锚点位于网格左上角: (cx, cy) = (j+dx, i+dy) * stride
    float cx = ((float)j + dx) * stride;
    float cy = ((float)i + dy) * stride;
    float bw = expf(dw) * stride;
    float bh = expf(dh) * stride;

    // 转换为 (x1, y1, x2, y2)
    float x1 = cx - bw * 0.5f;
    float y1 = cy - bh * 0.5f;
    float x2 = cx + bw * 0.5f;
    float y2 = cy + bh * 0.5f;
    // ... 裁剪到 [0, 320] 范围 ...
  }
}
```

NMS采用IoU + 包含率双重过滤：

```c
// IoU + 包含率过滤: 若小框75%以上面积被大框覆盖，直接抑制
float smaller_area = (area_a < area_b) ? area_a : area_b;
if (iou > nms_threshold || inter > 0.75f * smaller_area) {
    dets[b].score = -1.0f;  // 标记为抑制
}
```

#### 3.2.7 USART 图像接收协议

PC端通过二进制协议发送图像和批处理指令：

```
单张图像: [4B "FACE"] [4B size LE] [raw ARGB8888 bytes]
批次结束: [4B "MULT"] [4B count LE] [4B 0]
```

STM32端采用中断+状态机逐字节接收：

```c
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  switch (rx_state) {
    case RX_STATE_IDLE:
      // 累积头部，检测 "FACE" 或 "MULT" 魔法字
      break;
    case RX_STATE_GOT_MAGIC:
      // 解析size字段 (FACE) 或count字段 (MULT)
      break;
    case RX_STATE_RECEIVING:
      // 逐字节存入SDRAM (DMA不可行，因为每次RX长度=1)
      ((uint8_t *)usart_get_img(usart_img_count))[rx_received] = byte;
      break;
  }
  USART_StartRxByte();  // 重新启动接收
}
```

PC端发送脚本（Python）：

```python
def send_single_image(ser, raw_data, img_index, total):
    header = b"FACE" + struct.pack("<I", len(raw_data))
    payload = header + raw_data
    CHUNK = 4096
    for i in range(0, len(payload), CHUNK):
        ser.write(payload[i:i + CHUNK])

# 多图发送后，发送MULT结束包
if total > 1:
    header = b"MULT" + struct.pack("<I", total) + struct.pack("<I", 0)
    ser.write(header)
```

---

## 四、踩坑记录与注意事项

### 4.1 D-Cache一致性问题（最隐蔽的坑）

**现象：** AI推理结果始终为0或全噪声，检测不到任何人脸。

**原因：** STM32H7的CM7内核有D-Cache。当CPU将预处理后的数据写入`ai_input`（位于SDRAM的激活缓冲区）时，数据可能只停留在D-Cache中，未被写入物理SDRAM。Cube.AI的推理引擎通过DMA或其他总线主设备读取SDRAM时，读到的却是旧数据。

**解决：** 在调用`aiRun()`之前，必须显式清洗D-Cache：

```c
SCB_CleanDCache_by_Addr((uint32_t *)ai_input,
                        3 * 320 * 320 * sizeof(float));
```

这是嵌入式AI部署中非常容易忽略但致命的问题。任何CPU写入+硬件外设读取的场景都需要确保Cache一致性。

### 4.2 bbox输出格式混淆（CHW vs Interleaved）

**现象：** 检测框位置和大小完全错误（如23×24像素而非74×140像素），与PC端ONNX推理结果不一致。

**原因：** YuNet原始模型的bbox输出是CHW格式 `[4, H, W]`（即 `bbox[channel][y][x]`），但STM32 Cube.AI编译后的网络实际输出为Interleaved格式 `[H*W, 4]`（即 `bbox[loc*4+ch]`）。两种索引方式产生的解码坐标完全不同。

**解决：** 在USART调试输出中同时打印两种索引方式的结果，与PC端已知正确的ONNX输出比对，确认Cube.AI编译器实际使用Interleaved格式。最终采用 `bbox[loc*4+ch]` 索引方式。

### 4.3 模型输出格式需要插入sigmoid

**现象：** 初次部署时检测完全没有响应，所有score接近0或1的极端值。

**原因：** PC端训练/导出ONNX时，最后一层sigmoid是内嵌在模型计算图中的。但通过Cube.AI编译后的模型可能将sigmoid层从计算图中分离或丢弃（取决于优化配置和模型导出方式）。

**解决：** 在PC端重新导出ONNX，确保最后一层包含sigmoid。同时在`ai_postprocess`中直接使用cls和obj的乘积作为置信度（不再单独计算sigmoid）。对应Git提交：`b963a27 before add sigmod` → `977d790 before add sigmod to ai_detection.c` → `36cfd4a first run model on board ok`。

### 4.4 OTM8009A初始化需要二次调用`LCD_BriefDisplay()`

**现象：** LCD首次初始化后显示异常（花屏或颜色偏移）。

**原因：** OTM8009A驱动器有时在首次初始化序列后未完全就绪，需要重复一次配置才能稳定工作。

**解决：** 在主初始化流程中连续调用两次`LCD_BriefDisplay()`。

### 4.5 USART逐字节接收无法使用DMA

**现象：** 使用DMA批量接收时频繁丢字节或数据错位。

**原因：** 协议中每帧长度可变（取决于图像大小字段），且"MULT"和"FACE"两类数据包长度不同。DMA的固定长度传输不适合这种变长协议。

**解决：** 采用中断方式逐字节接收+状态机解析，每次`HAL_UART_RxCpltCallback`触发后立即重新注册下一字节的接收。

### 4.6 DMA2D硬件只支持特定像素格式的PFC

**注意：** DMA2D的PFC（Pixel Format Converter）只支持特定的输入格式转换（如RGB565→ARGB8888）。对于ARGB8888→ARGB8888的纯拷贝，不能通过PFC，必须使用M2M（Memory-to-Memory）模式，并正确配置InputOffset。

---

## 五、实验收获与心得

1. **嵌入式AI部署的完整流水线认知：** 从PC端模型训练到Cube.AI编译，再到MCU上预处理→推理→后处理的完整闭环，实际踩过了Cache一致性、输出格式、量化精度等多个坑，对嵌入式AI的工程落地的理解从理论走向了实践。

2. **Cache一致性是嵌入式AI的最大陷阱：** D-Cache对于MCU性能至关重要，但在CPU写入/硬件读取的场景下会引入极其隐蔽的数据不一致bug。`SCB_CleanDCache_by_Addr`这一行代码是整个系统能跑通的关键，也让我深刻理解了ARMv7-M架构中Cache机制对AI推理的影响。

3. **Cube.AI工具链的特性与局限：** Cube.AI能自动将ONNX编译为C代码并在STM32上运行，但编译器可能改变张量的内存布局（如CHW→Interleaved），需要在实际部署中通过调试输出来验证输出格式。不能盲目假设编译器产出的内存布局与原始模型框架一致。

4. **SDRAM作为AI激活缓冲区的可行性：** 本项目的YuNet-320需要约2.46MB的激活内存，远超CM7的DTCM（128KB），必须使用外部SDRAM。虽然SDRAM延迟较高，但对于140MMACs的模型，推理时延仍在可接受范围（通过MPU配置为cacheable来缓解瓶颈）。

5. **DSI Command Mode + TE同步是嵌入式显示的正确范式：** 相比Video Mode的固定刷新率，Command Mode允许MCU在推理完成后才触发刷新，天然适配"检测→绘制→显示"的异步工作流。TE同步信号确保了帧缓冲在显示期间不被修改，彻底消除画面撕裂。

6. **YOLO类Anchor-Free检测头的后处理复杂度可控：** 3个stride下的bbox解码+NMS排序，在CM7上纯C实现完全可行。关键优化在于尽早过滤低置信度候选框（`if (score < threshold) continue`），大幅减少后续NMS的计算量。
