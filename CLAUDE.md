# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STM32H747I-DISCO bare-metal firmware — **Phase A baseline**: LCD DSI Command Mode display with single frame buffer. The larger goal (documented in [AI_Prompt.md](AI_Prompt.md)) is deploying a face detection model on this board using Cube.AI. This baseline project handles display only; face detection (Phases B/C) has not been implemented yet.

## Build System

Built with **Keil MDK-ARM (uVision 5)** on Windows. There is no CLI build.

- Project file: `MDK-ARM/LCD_DSI_CMD_mode_Single_Buffer.uvprojx`
- Two targets: `LCD_DSI_CMD_mode_Single_Buffer_CM7` (main) and `_CM4` (stub)
- Device pack: `Keil.STM32H7xx_DFP.2.6.0`
- Compiler: ARMCC V5.06 update 6, optimization `-O4`
- CM7 preprocessor defines: `CORE_CM7, USE_HAL_DRIVER, STM32H747xx`

> **Important**: The `.gitignore` excludes `Drivers/` and `Middlewares/`. After a fresh clone, you must regenerate these via STM32CubeMX (open the `.ioc` file, generate code) or copy from an STM32Cube firmware package. The HAL and CMSIS files you see on disk are NOT in git.

## Architecture: Dual-Core Boot Sequence

1. Both CM7 and CM4 boot. **CM4** enables HSEM notification on semaphore 0, then enters D2 STOP mode (`HAL_PWREx_EnterSTOPMode`), waiting for CM7.
2. **CM7** configures MPU, enables I-Cache/D-Cache, calls `HAL_Init()` + `SystemClock_Config()`, then releases HSEM semaphore 0 to wake CM4.
3. **CM4** wakes, calls `HAL_Init()`, and enters an empty `while(1)` loop.
4. **CM7** initializes GPIO, USART1, LED, SDRAM, LCD (DSI + LTDC + OTM8009A), DMA2D.
5. **CM7 main loop**: copies an image into the SDRAM frame buffer via DMA2D M2M → calls `HAL_DSI_Refresh()` → waits for `HAL_DSI_EndOfRefreshCallback` (Tearing Effect signal from panel) → copies the next image → repeats.

The dual-core boot entry point is `Common/Src/system_stm32h7xx_dualcore_boot_cm4_cm7.c`.

## Display Pipeline

```
Images[] (Flash, ARGB8888, 320×240)
    → DMA2D M2M (CopyBuffer, positions at x=240,y=160 for centering)
    → SDRAM frame buffer (0xD0000000, 800×480 ARGB8888)
    → LTDC Layer 0 (reads SDRAM, outputs DPI to DSI wrapper)
    → DSI Host (Adapted Command Mode, Tearing Effect sync)
    → OTM8009A panel (800×480, RGB888, Landscape)
```

After `HAL_DSI_Refresh()`, the DSI host waits for a Tearing Effect signal. When refresh completes, `HAL_DSI_EndOfRefreshCallback` sets `pending_buffer = -1`, signaling the main loop can copy the next frame.

## Key Addresses & Peripherals

| Resource | Address/Value |
|----------|---------------|
| Frame buffer (`LCD_FRAME_BUFFER`) | `0xD0000000` (SDRAM Bank 2) |
| LCD Layer 1 buffer | `0xD0200000` |
| Camera frame buffer | `0xD0600000` |
| SDRAM device address | `SDRAM_DEVICE_ADDR` = `0xD0000000` |
| SDRAM size | 32MB (IS42S32800J) |
| CM7 Flash | `0x08000000` (1MB) |
| CM7 RAM (DTCM) | `0x20000000` (128KB) |
| CM4 Flash | `0x08100000` |
| CM4 RAM | `0x10000000` |

## Critical Clock & Timing Config

- **HSE**: 25 MHz (confirmed in `stm32h7xx_hal_conf.h`)
- **SYSCLK**: HSE → PLL1 (/2, ×64, /2) = 400 MHz (CM7)
- **HCLK**: SYSCLK/2 = 200 MHz
- **LTDC pixel clock**: PLL3 (HSE/5×160/19) ≈ 42 MHz
- **DSI PLL**: HSE/5=5MHz → VCO=5×100=500MHz → byte lane clock ≈ 62.5 MHz (NDIV=100, IDF=/5, ODF=/1)
- **LTDC timing**: 800×480, HSYNC=HBP=HFP=1, VSYNC=VBP=VFP=1, all polarities active-low, pixel clock inverted
- **DSI**: 2 data lanes, Adapted Command Mode, Tearing Effect on DSI link, TE rising edge, automatic refresh disabled

## MPU Configuration

- **Region 0**: 0x00000000, 4GB, no access, sub-region disable mask=0x87 (background deny-all)
- **Region 1**: SDRAM_DEVICE_ADDR, 32MB, full access, cacheable, non-shareable (overrides Region 0 for SDRAM range)

## Enabled HAL Modules

Verified from `CM7/Core/Inc/stm32h7xx_hal_conf.h`: `HAL_MODULE_ENABLED`, `GPIO`, `DMA`, `MDMA`, `RCC`, `FLASH`, `EXTI`, `PWR`, `I2C`, `CORTEX`, `HSEM`, `UART`, `DSI`, `LTDC`, `SDRAM`, `DMA2D`, `ADC`.

All others (CRC, DCMI, JPEG, SD, TIM, etc.) are commented out but can be enabled when needed for face detection.

## The `pending_buffer` TE Sync Mechanism

The variable `pending_buffer` (declared in `main.c`) gates the main loop:

- `pending_buffer = -1` → frame buffer is free, main loop can write next image
- `pending_buffer = 0` → a refresh is in progress, do not touch the buffer
- `HAL_DSI_EndOfRefreshCallback` sets `pending_buffer = -1` when the TE signal arrives
- `CopyBuffer` then sets it back to `0` after copying

The main loop polls `pending_buffer < 0` with a `HAL_Delay(2000)` between iterations.

## What To Modify for Face Detection (Phase C)

See [AI_Prompt.md](AI_Prompt.md) for the full plan. Key files to modify:

| File | Change |
|------|--------|
| `CM7/Core/Src/main.c` | Add AI inference + face detection box drawing |
| `CM7/Core/Inc/stm32h7xx_hal_conf.h` | Enable CRC, DCMI/SD/JPEG as needed |
| `MDK-ARM/*.uvprojx` | Add Cube.AI generated files + include paths |
| `Middlewares/ST/AI/` | **New** — Cube.AI runtime + model files |

Do NOT modify: CM4 code, BSP drivers (LCD/SDRAM/BUS), dual-core boot sequence, DSI command mode configuration, TE sync mechanism.

## Project Context

- The `AI_Prompt.md` file contains the complete three-phase project spec: Phase A (LCD baseline — done), Phase B (model training in Python/PyTorch), Phase C (Cube.AI deployment on STM32H7)
- This repo is Phase A only. Phase B (model training) happens separately in Python, producing `.tflite`/`.onnx` model files
- The `.uvprojx` Keil project may reference files with paths relative to the project directory
