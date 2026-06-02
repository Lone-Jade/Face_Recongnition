# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STM32H747I-DISCO bare-metal firmware demonstrating **LCD DSI Command Mode with single frame buffer**. Uses DMA2D to copy images into SDRAM (frame buffer at `0xD0000000`), then drives a DSI-connected OTM8009A LCD panel via LTDC + DSI Host in command mode.

## Build System

Built with **Keil MDK-ARM (uVision 5)**. The `.uvprojx` project file is at `MDK-ARM/LCD_DSI_CMD_mode_Single_Buffer.uvprojx`. It has two targets:
- `LCD_DSI_CMD_mode_Single_Buffer_CM7` — Cortex-M7 (main application)
- `LCD_DSI_CMD_mode_Single_Buffer_CM4` — Cortex-M4 stub

Both targets generate `.hex` files. Device pack: `Keil.STM32H7xx_DFP.2.6.0`.

There is no CLI build. Compilation is done through the uVision IDE on Windows.

## Architecture: Dual-Core Boot Sequence

1. Both cores boot. **CM4** immediately enables HSEM notification on semaphore 0, then enters STOP mode (`PWR_D2_DOMAIN`), waiting for CM7.
2. **CM7** configures MPU, caches, system clock, GPIO, USART. Then initializes SDRAM → LCD (DSI + LTDC + OTM8009A) → DMA2D. After init, CM7 releases HSEM semaphore 0, waking CM4.
3. **CM4** wakes, calls `HAL_Init()`, and spins in an empty `while(1)` loop.
4. **CM7** main loop: copies an image into the SDRAM frame buffer via DMA2D M2M, calls `HAL_DSI_Refresh()`, waits for `HAL_DSI_EndOfRefreshCallback` (Tearing Effect from panel), then copies the next image.

The dual-core boot entry point is `Common/Src/system_stm32h7xx_dualcore_boot_cm4_cm7.c`.

## Source Map

| Directory | Role |
|-----------|------|
| `CM7/Core/` | Cortex-M7 application code (`main.c`, `gpio.c`, `usart.c`, HAL MSP, IT handlers) |
| `CM4/Core/` | Cortex-M4 stub (`main.c` — just waits for HSEM then idles) |
| `Common/Src/` | Dual-core system startup (`system_stm32h7xx_dualcore_boot_cm4_cm7.c`) |
| `Drivers/STM32H7xx_HAL_Driver/` | STM32H7 HAL library |
| `Drivers/CMSIS/` | CMSIS core + device headers for STM32H7 |
| `BSP/STM32H747I-DISCO/` | Board BSP (LCD init, SDRAM, bus, audio, camera, touch screen, QSPI) |
| `BSP/Components/otm8009a/` | OTM8009A LCD panel driver IC |
| `BSP/Components/adv7533/` | ADV7533 HDMI transmitter driver |
| `BSP/Components/Common/` | Abstract interfaces (lcd.h, ts.h, audio.h, etc.) |
| `BSP/Utilities/lcd/` | LCD utility layer (`stm32_lcd.c` — context, font rendering, drawing) |
| `BSP/Utilities/Fonts/` | Bitmap fonts (8/12/16/20/24px) |
| `MDK-ARM/` | uVision project + startup assembly + linker scatter files |

## Critical: Manual HAL Module Enabling

CubeMX did **not** configure LCD-related peripherals. The root `stm32h7xx_hal_conf.h` contains the reference enabled modules needed for LCD usage (DSI, LTDC, SDRAM, DMA2D, ADC). The CM7 build's `CM7/Core/Inc/stm32h7xx_hal_conf.h` must have these modules enabled (check against the root copy). Currently the CM7 copy enables: `GPIO`, `DMA`, `MDMA`, `RCC`, `FLASH`, `EXTI`, `PWR`, `I2C`, `CORTEX`, `HSEM`, `UART`, `DSI`, `LTDC`, `SDRAM`, `DMA2D`, `ADC`.

## Key Peripherals & Addresses

- **Frame buffer**: `0xD0000000` (SDRAM Bank 2, 32MB at `SDRAM_DEVICE_ADDR`)
- **LTDC**: 800×480, ARGB8888, configured in command mode (no autonomous refresh)
- **DSI**: 2 data lanes, DSI PLL input = HSE/5 = 5 MHz → VCO = 5×100 = 500 MHz → byte lane clock ~62.5 MHz (PLLNDIV=100, IDF=/5, ODF=/1)
- **LCD controller**: OTM8009A, RGB888 color mode, landscape orientation
- **DMA2D**: Memory-to-memory (M2M) with ARGB8888 input/output, used to blit images into the frame buffer

## Data Flow

```
Images[] (Flash, ARGB8888, 320×240)
    → DMA2D M2M (CopyBuffer, positions at x=240,y=160)
    → SDRAM frame buffer (0xD0000000, 800×480 ARGB8888)
    → LTDC layer 0 (reads SDRAM, outputs to DSI wrapper)
    → DSI Host (command mode, DPI→DSI packets, Tearing Effect sync)
    → OTM8009A panel
```

After `HAL_DSI_Refresh()`, the DSI host waits for a Tearing Effect signal from the panel. When refresh completes, `HAL_DSI_EndOfRefreshCallback` sets `pending_buffer = -1`, allowing the main loop to copy the next image.
