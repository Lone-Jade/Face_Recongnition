/**
  ******************************************************************************
  * @file           : LCD_Display.h
  * @brief          : LCD Display module header for STM32H747I-DISCO
  *
  *                   Provides LCD initialization, layer configuration, DMA2D
  *                   copy, DSI interrupt handling, and frame buffer sync for
  *                   the OTM8009A panel (800x480, DSI Adapted Command Mode).
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __LCD_DISPLAY_H
#define __LCD_DISPLAY_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32h7xx_hal.h"
#include "stm32h747i_discovery_lcd.h"
#include "stm32_lcd.h"

/* Exported constants --------------------------------------------------------*/

/** @defgroup LCD_Display_Exported_Constants Exported Constants
  * @{
  */
/** @brief LCD frame buffer base address (SDRAM Bank 2) */
#define LCD_FRAME_BUFFER                0xD0000000U

/** @brief LTDC timing parameters for 800x480 panel (pixel clock ~42 MHz) */
#define LCD_VSYNC                       1U
#define LCD_VBP                         1U
#define LCD_VFP                         1U
#define LCD_VACT                        480U
#define LCD_HSYNC                       1U
#define LCD_HBP                         1U
#define LCD_HFP                         1U
#define LCD_HACT                        800U

/**
  * @}
  */

/* Exported types ------------------------------------------------------------*/

/* Exported variables --------------------------------------------------------*/

/** @brief Frame buffer busy flag.
  *        -1 = free (tearing effect received), 0 = busy (refresh in progress) */
extern int32_t pending_buffer;

/* Exported functions prototypes ---------------------------------------------*/

/** @addtogroup LCD_Display_Exported_Functions
  * @{
  */

/**
  * @brief  Initialize the DSI LCD (PLL3, DSI host, LTDC, OTM8009A panel).
  * @retval BSP_ERROR_NONE on success, otherwise a BSP error code.
  */
uint8_t LCD_Init(void);

/**
  * @brief  Initialize an LTDC layer with the given frame buffer address.
  * @param  LayerIndex: LTDC layer index (0 or 1).
  * @param  Address:    frame buffer start address in SDRAM.
  * @retval None
  */
void LCD_LayertInit(uint16_t LayerIndex, uint32_t Address);

/**
  * @brief  Display project title and description text on LCD.
  * @retval None
  */
void LCD_BriefDisplay(void);

/**
  * @brief  DMA2D memory-to-memory copy: transfers an image to the LCD frame buffer.
  * @param  pSrc:  pointer to source image data (ARGB8888).
  * @param  pDst:  pointer to destination frame buffer.
  * @param  x:     horizontal offset on LCD (pixels).
  * @param  y:     vertical offset on LCD (pixels).
  * @param  xsize: source image width (pixels).
  * @param  ysize: source image height (pixels).
  * @retval None
  */
void CopyBuffer(uint32_t *pSrc, uint32_t *pDst,
                uint16_t x, uint16_t y,
                uint16_t xsize, uint16_t ysize);

/**
  * @brief  Get LCD X resolution (width).
  * @param  Instance: LCD instance index.
  * @param  XSize:    pointer to store the width value.
  * @retval BSP_ERROR_NONE on success.
  */
int32_t LCD_GetXSize(uint32_t Instance, uint32_t *XSize);

/**
  * @brief  Get LCD Y resolution (height).
  * @param  Instance: LCD instance index.
  * @param  YSize:    pointer to store the height value.
  * @retval BSP_ERROR_NONE on success.
  */
int32_t LCD_GetYSize(uint32_t Instance, uint32_t *YSize);

/**
  * @brief  DSI global interrupt handler.
  *         Calls HAL_DSI_IRQHandler() to process DSI events.
  * @retval None
  */
void DSI_IRQHandler(void);

/**
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* __LCD_DISPLAY_H */
