# STM32H747I-DISCO YuNet-320 人脸检测部署与调试日志

## 一、初始状态

### 1.1 项目背景
- **硬件平台**：STM32H747I-DISCO 开发板（CM7 + CM4 双核）
- **编译环境**：Keil MDK-ARM (uVision 5)，ARMCC V5.06 update 6，-O4 优化
- **AI 框架**：STM32Cube.AI Studio / ST Edge AI Core v4.0.1，ST-AI (stai) 嵌入式客户端 API
- **模型**：YuNet-320 人脸检测，1×3×320×320 float32 NCHW BGR 输入，12 个输出张量（cls / obj / bbox / kps，各 3 个 stride：8 / 16 / 32）
- **模型编译格式**：float32（非 INT8），权重 295KB Flash，激活 2.5MB（DTCM + SDRAM）

### 1.2 SDRAM 内存布局
| 地址 | 用途 | 大小 |
|------|------|------|
| `0xD0000000` | LCD 帧缓冲（800×480 ARGB8888） | ~1.5MB |
| `0xD0400000` | USART 接收图像缓冲 | ~300KB |
| `0xD0600000` | 相机帧缓冲（预留） | — |
| `0xD0800000` | AI 激活缓冲区 2 | 2.34MB |

### 1.3 初始问题
开发板运行后人脸检测完全错误：
- **显示**：一张人脸显示 81 个绿框，小框密集覆盖在鼻子、眼睛、脸颊等面部特征上
- **预期**：PC 端 ONNX Runtime 推理能正确框出人脸（位置、数目、框大小均正确）
- **现象**：人脸越大，显示的框越多

---

## 二、调试流程

### 2.1 第一阶段：添加串口调试输出

**目标**：获取开发板实际运行时的检测数据（分数、位置、数量），与 PC 端对比。

**实现**：
1. 利用 ST AI 测试工具库（`aiTestUtility.c`）已有的 `fputc` 重定向（输出到 USART1），避免重复定义链接错误
2. 在 `ai_detection.c` 的 `ai_postprocess()` 中添加详细调试输出（宏 `AI_DBG`，后因与 ST AI 库的 `AI_DEBUG` 宏冲突改名为 `AI_DBG`）：
   - 每个 stride 的 cls / obj 前 5 个采样值
   - bbox[0,0] 值
   - 每个 stride 的原始候选框数量
   - Top-5 分数
   - NMS 前后检测框详情
3. 在 `main.c` 中添加：
   - 源图像像素采样（前 5 个 ARGB8888 像素值）
   - 预处理后 B/G/R 通道采样
   - D-Cache Clean（`SCB_CleanDCache_by_Addr`）
   - AI 推理返回码

**串口输出示例**（关键发现）：
```
Stride 16 (gs=20): cls[0..4]=0.6299 0.6207 0.5949 ...  obj[0..4]=0.0002 0.0000 ...
  Stride 16 raw candidates: 8 (threshold=0.700)
  Top-5 scores: 0.8563 0.8521 0.8499 0.8476 0.8404
Total raw candidates (all strides): 8
NMS: suppressed 0, remaining 8
=== Final detections: 8 ===
  #0: score=0.8563 box=(184.7,102.8)-(208.1,126.9) w=23 h=24
  #1: score=0.8521 box=(163.4,102.5)-(182.2,139.9) w=19 h=37
  ...
```

**初步分析**：
- cls 值 0.5-0.6（sigmoid 近 0 输出，背景区域正常）
- obj 值全面接近 0.0000（模型对背景的强烈否定，正常）
- NMS 抑制 0 个框 —— 所有 8 个框都存活
- 框尺寸 19-66px（在 320×320 输入空间中太小，实际人脸应 ~150px）

---

### 2.2 第二阶段：NMS 抑制失效分析

**问题**：为什么 NMS 没有抑制重叠框？

**分析数据**（Frame 1，1 张人脸，8 个检测）：
```
#1: score=0.8499 box=(155.6,99.9,221.2,160.9) size=66x61   ← 最大框
#0: score=0.8563 box=(184.7,102.8)-(208.1,126.9) w=23 h=24  ← 小框
```

**IoU 计算**：
- 大框面积：66 × 61 = 4026
- 小框面积：23 × 24 = 552
- 交集：小框几乎完全被大框包含 → inter = 552
- **IoU = 552 / (4026 + 552 - 552) = 552 / 4026 = 0.137**
- **0.137 < 0.45（NMS 阈值）→ 小框不被抑制！**

**根因**：小框被大框完全包住时，IoU = 小框面积 / 大框面积，比例太低。标准 IoU-based NMS 对小框包大框的情况无能为力。

**修复**：在 NMS 中新增**包含率过滤器**：
```c
float smaller_area = (area_a < area_b) ? area_a : area_b;
if (iou > nms_threshold || inter > 0.75f * smaller_area) {
    dets[b].score = -1.0f;  // 抑制
}
```
- 如果小框 75% 以上面积被大框覆盖 → 直接抑制，不管 IoU
- 效果：NMS 抑制数从 0 提升到 4（Frame 1，8→4）

**其他修复**：
- LCD 文本残留问题：显示前设置白色背景色（`UTIL_LCD_SetBackColor(UTIL_LCD_COLOR_WHITE)`）
- 参数调整：`DET_THRESHOLD` 0.60→0.70，`NMS_THRESHOLD` 0.45→0.30

---

### 2.3 第三阶段：PC 端对比测试——定位根本差异

**关键问题**：包含率过滤器改善了结果（8→4），但仍有 4 个检测（应为 1 个），且所有框仍然很小（19-53px），从未出现正确的大框。

**方法**：编写 PC 端 Python 测试，**完全复制 STM32 的预处理和后处理逻辑**，与 ONNX Runtime 推理结果对比。

#### 测试脚本 1：`compare_preprocessing.py`
- 从 C 头文件 `face_img_0.h` / `face_img_1.h` 解析 ARGB8888 像素数据
- 用与 STM32 完全相同的整数比最近邻采样预处理
- 用 ONNX Runtime 推理
- 用与 STM32 C 代码完全相同的后处理（包含率过滤器 + NMS）

**结果**：
```
[STM32 nearest-neighbor] inference: 82.3ms, detections: 1
  score=0.8563 box=(145,48)-(219,188) w=74 h=140    ← 正确的大框！
```

**PC 产生 1 个 74×140 的正确检测，STM32 产生 4 个 19-53px 的错误检测！**

#### 测试脚本 2：`deep_compare.py`
- 逐值比较模型输出（cls / obj / bbox）
- STM32 串口输出 vs Python ONNX 输出

**结果**：
```
cls_16 max|diff| USART vs Python: 0.0000    ← 完全相同！
Match. Model outputs are identical.
```

**关键发现**：
- cls / obj 值在 STM32 和 PC 之间**完全相同**（max|diff| = 0.0000）
- bbox[0,0]（loc=0）值也完全相同
- 模型编译正确（float32，cos_sim=1.0），预处理正确（输入值一致），**但最终检测结果不同**

---

### 2.4 第四阶段：定位根因——bbox 内存格式

**思路**：模型输出相同，后处理逻辑相同，但结果不同 → **bbox 数据的读取方式必然不同**。

**假设**：ST AI 编译后的模型输出的 bbox 内存格式，与 `network.h` 声明的 CHW 格式不一致。

`network.h` 声明：
```c
#define STAI_NETWORK_OUT_7_SHAPE  {1,4,40,40}          // CHW: channel × height × width
#define STAI_NETWORK_OUT_7_FLAGS  (STAI_FLAG_CHANNEL_FIRST | ...)
```

C 代码按 CHW 格式读取：
```c
float dx = bbox[0 * gs * gs + loc];  // 通道 0, 位置 loc
float dy = bbox[1 * gs * gs + loc];  // 通道 1, 位置 loc
float dw = bbox[2 * gs * gs + loc];  // 通道 2, 位置 loc
float dh = bbox[3 * gs * gs + loc];  // 通道 3, 位置 loc
```

ONNX 模型原生输出是 **interleaved**（交错）格式 `(1, N, 4)`——即 N 个网格位置，每个位置连续存储 4 个值。

**验证方法**：在 `ai_detection.c` 中添加对比调试，在最佳候选位置同时用两种索引方案读取 bbox 并解码框：

```c
// CHW 索引
printf("bbox CHW[loc=%d]: dx=%.4f dy=%.4f dw=%.4f dh=%.4f\r\n",
       best_loc,
       bbox[0*gs*gs+best_loc], bbox[1*gs*gs+best_loc],
       bbox[2*gs*gs+best_loc], bbox[3*gs*gs+best_loc]);

// Interleaved 索引
printf("bbox IL [loc=%d]: dx=%.4f dy=%.4f dw=%.4f dh=%.4f\r\n",
       best_loc,
       bbox[best_loc*4+0], bbox[best_loc*4+1],
       bbox[best_loc*4+2], bbox[best_loc*4+3]);
```

**决定性输出**：
```
Best candidate grid=(7,12) loc=152 score=0.8563
bbox CHW[loc=152]: dx=0.2758 dy=0.1787 dw=0.3773 dh=0.4091
bbox IL [loc=152]: dx=-0.6490 dy=0.4033 dw=1.5310 dh=2.1694
-> CHW box: (185,103)-(208,127) 23x24    ← 错误的 23×24 小框
-> IL  box: (145,48)-(219,188) 74x140    ← 正确的 74×140 大框，匹配PC！
```

**同时验证 loc=0 的对比**：
```
bbox[0,0] CHW: dx=0.7080 dy=0.5722 dw=0.4949 dh=0.5376
bbox[0,0] IL:  dx=0.7080 dy=0.8980 dw=0.5351 dh=0.8584
```
- dx 相同（loc=0 时两种索引碰巧读到同一位置），但 dy / dw / dh 不同！

**根因确认**：ST AI 编译器输出的 bbox 实际是 **interleaved 格式 `(N, 4)`**，而非 `network.h` 声明的 CHW 格式 `(4, H, W)`。C 代码按 CHW 索引读取，读到了错误的数据。

---

### 2.5 第五阶段：根本修复

**修改 `ai_detection.c` 中 bbox 读取方式**：
```c
// 旧代码（CHW 索引 —— 错误）
float dx = bbox[0 * gs * gs + loc];
float dy = bbox[1 * gs * gs + loc];
float dw = bbox[2 * gs * gs + loc];
float dh = bbox[3 * gs * gs + loc];

// 新代码（Interleaved 索引 —— 正确）
float dx = bbox[loc * 4 + 0];
float dy = bbox[loc * 4 + 1];
float dw = bbox[loc * 4 + 2];
float dh = bbox[loc * 4 + 3];
```

**效果**：
- 框大小从 19-53px 变为正确的 74×140（匹配 PC ONNX 结果）
- 大框通过包含率过滤器抑制所有内部小框
- 检测数从 4 降至 1（正确）

---

### 2.6 第六阶段：参数调优

**最终参数**：
| 参数 | 最终值 | 说明 |
|------|--------|------|
| `DET_THRESHOLD` | 0.50 | 较低的置信度阈值，捕获较小/模糊的人脸 |
| `NMS_THRESHOLD` | 0.40 | IoU 抑制阈值 |
| `MIN_BOX_SIZE` | 12.0 | 最小框尺寸（320×320 输入空间） |
| `MAX_DETECTIONS` | 50 | 最大检测数 |
| 包含率过滤 | `inter > 0.75 × smaller_area` | 小框被大框包住 75% 以上 → 抑制 |

---

## 三、问题与修复总结

| # | 问题 | 症状 | 根因 | 修复 |
|---|------|------|------|------|
| 1 | 人脸被检测 81 次 | 大量小绿框密集覆盖面部特征 | NMS 无法抑制被大框包住的小框（IoU 太低） | 添加包含率过滤器 |
| 2 | 检测数从 81→8→4 但仍不对 | 所有框都太小（19-53px），从无正确大框 | bbox 按 CHW 格式索引，**但 ST AI 实际输出 interleaved 格式** | 改为交错索引 `bbox[loc*4+c]` |
| 3 | LCD 显示文本重叠 | "8" + "1" = "81" | 显示前未清背景 | 设置白色背景色后再显示 |
| 4 | 宏名冲突 | `AI_DEBUG` 与 ST AI 库重复定义 | ST AI 库也定义了同名宏 | 重命名为 `AI_DBG` |
| 5 | fputc 重定义 | 链接错误 | ST AI 测试工具已有 fputc | 删除自定义 fputc |

---

## 四、关键调试方法

### 4.1 串口 printf 调试
- 利用 ST AI 库现有的 `fputc` → USART1 重定向
- 输出模型输入/输出采样值、检测候选数、框坐标
- **关键输出**：两种 bbox 索引方案的框解码对比

### 4.2 PC 端精确复现
- 用 Python 完全复制 STM32 C 代码的预处理和后处理逻辑
- 从 C 头文件解析 ARGB8888 数据，模拟整数比最近邻采样
- 对比 ONNX 推理结果，**排除模型和预处理问题**

### 4.3 二分定位法
- 先确认模型输出是否相同（cls/obj 值对比）→ 相同 ✓
- 再确认预处理是否相同（输入值对比）→ 相同 ✓
- 最后聚焦后处理差异 → bbox 索引方式不同 ✗

### 4.4 A/B 对比测试
- 在**同一位置**用两种索引方案读取 bbox，解码并对比框大小
- CHW 方案 → 23×24（错误） vs IL 方案 → 74×140（正确，匹配 PC）
- **一目了然地定位了根因**

---

## 五、关键文件修改清单

| 文件 | 修改内容 |
|------|----------|
| `CM7/Core/Src/ai_detection.c` | **核心修复**：bbox 交错索引；包含率过滤器；调试输出 |
| `CM7/Core/Inc/ai_detection.h` | 检测参数（DET_THRESHOLD / NMS_THRESHOLD / MIN_BOX_SIZE） |
| `CM7/Core/Src/main.c` | 预处理调试输出；D-Cache Clean；LCD 背景色设置 |
| `PC_test/compare_preprocessing.py` | PC 端对比测试：STM32 预处理 + ONNX 推理 |
| `PC_test/deep_compare.py` | PC 端深度对比：逐值比较模型输出 |
| `PC_test/onnx_infer.py` | PC 端标准推理（bilinear resize） |
| `PC_test/compare_validation.py` | ONNX vs ST AI NPZ 验证对比 |

---

## 六、经验教训

1. **不要完全信任自动生成的元数据**：ST AI 的 `network.h` 声明 bbox 为 CHW 格式 `{1,4,40,40}`，但实际运行时输出是 interleaved `(N,4)`。编译器可能在优化过程中改变了内存布局。

2. **PC 端复现是调试嵌入式 AI 最有效的手段**：在 PC 上用 Python 精确复制嵌入式预处理和后处理逻辑，可以快速排除模型和预处理问题，将排查范围缩小到嵌入式特有的内存/格式差异。

3. **A/B 对比是最直观的定位方法**：在同一位置用两种方案读取数据并对比结果，比逐值分析更高效。

4. **IoU-based NMS 的局限性**：小框被大框包住时 IoU 极低，必须用包含率过滤等补充策略。

5. **保留调试基础设施**：`AI_DBG` 宏可以在测试时开启详细输出，发布时关闭，非常适合嵌入式调试。
