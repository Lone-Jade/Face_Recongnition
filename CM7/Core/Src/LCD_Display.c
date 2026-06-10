/**
  ******************************************************************************
  * @file           : LCD_Display.c
  * @brief          : LCD Display module for STM32H747I-DISCO
  *
  *                   Manages DSI host, LTDC controller, DMA2D frame-buffer copy,
  *                   OTM8009A panel initialization, and Tearing Effect (TE) sync.
  *                   The LCD is 800x480 ARGB8888, driven in DSI Adapted Command
  *                   Mode with 2 data lanes.
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

/* Includes ------------------------------------------------------------------*/
#include "LCD_Display.h"
#include "stm32h747i_discovery_bus.h"

/* Private typedef -----------------------------------------------------------*/

/* Private define ------------------------------------------------------------*/

/* Private macro -------------------------------------------------------------*/

/* Private variables ---------------------------------------------------------*/

/** @brief DMA2D handle for memory-to-memory frame-buffer copy operations */
static DMA2D_HandleTypeDef hdma2d;

/** @brief DSI Adapted Command Mode configuration */
static DSI_CmdCfgTypeDef CmdCfg;

/** @brief DSI low-power command configuration */
static DSI_LPCmdTypeDef LPCmd;

/** @brief DSI PLL initialization structure */
static DSI_PLLInitTypeDef dsiPllInit;

/** @brief Peripheral clock configuration for PLL3 (LTDC clock) */
static RCC_PeriphCLKInitTypeDef PeriphClkInitStruct;

/** @brief Reference to BSP LCD handle (exposed via stm32h747i_discovery_lcd.h) */
extern LTDC_HandleTypeDef  hlcd_ltdc;
extern DSI_HandleTypeDef   hlcd_dsi;
extern BSP_LCD_Ctx_t       Lcd_Ctx[];

/* Exported variables ---------------------------------------------------------*/

/**
  * @brief Frame buffer synchronization flag.
  *        -1 : frame buffer is free (TE signal received, ready for next frame).
  *         0 : refresh in progress, do not touch the frame buffer.
  */
int32_t pending_buffer = -1;

/* Private function prototypes -----------------------------------------------*/

/**
  * @brief  Initialize the LTDC controller with panel timing parameters.
  * @retval None
  */
static void LTDC_Init(void);

/**
  * @brief  Write a value to an LCD panel register via DSI command.
  * @param  ChannelNbr: DSI virtual channel ID.
  * @param  Reg:        register address.
  * @param  pData:      pointer to data byte(s).
  * @param  Size:       number of data bytes (0 or 1 for short, >1 for long).
  * @retval BSP_ERROR_NONE on success, BSP_ERROR_BUS_FAILURE on error.
  */
static int32_t DSI_IO_Write(uint16_t ChannelNbr, uint16_t Reg,
                            uint8_t *pData, uint16_t Size);

/**
  * @brief  Read a value from an LCD panel register via DSI command.
  * @param  ChannelNbr: DSI virtual channel ID.
  * @param  Reg:        register address.
  * @param  pData:      pointer to store read data.
  * @param  Size:       number of bytes to read.
  * @retval BSP_ERROR_NONE on success, BSP_ERROR_BUS_FAILURE on error.
  */
static int32_t DSI_IO_Read(uint16_t ChannelNbr, uint16_t Reg,
                           uint8_t *pData, uint16_t Size);

/**
  * @brief  MSP initialization for LTDC, DSI, and DMA2D peripherals.
  *         Enables clocks, releases resets, configures NVIC priorities.
  * @retval None
  */
static void LCD_MspInit(void);

/* Exported functions ---------------------------------------------------------*/

/**
  * @brief  End of Refresh DSI callback (Tearing Effect signal received).
  *         Signals the main loop that the frame buffer is free for next frame.
  * @param  hdsi: pointer to DSI handle.
  * @retval None
  */
void HAL_DSI_EndOfRefreshCallback(DSI_HandleTypeDef *hdsi)
{
  if (pending_buffer >= 0)
  {
    pending_buffer = -1;
  }
}

/**
  * @brief  Get LCD X resolution (width).
  * @param  Instance: LCD instance index.
  * @param  XSize:    pointer to store the width value.
  * @retval BSP_ERROR_NONE on success.
  */
int32_t LCD_GetXSize(uint32_t Instance, uint32_t *XSize)
{
  *XSize = Lcd_Ctx[0].XSize;
  return BSP_ERROR_NONE;
}

/**
  * @brief  Get LCD Y resolution (height).
  * @param  Instance: LCD instance index.
  * @param  YSize:    pointer to store the height value.
  * @retval BSP_ERROR_NONE on success.
  */
int32_t LCD_GetYSize(uint32_t Instance, uint32_t *YSize)
{
  *YSize = Lcd_Ctx[0].YSize;
  return BSP_ERROR_NONE;
}

/**
  * @brief  Initialize the DSI LCD (PLL3, DSI host, LTDC, OTM8009A panel driver).
  *         Configures the display pipeline for 800x480 ARGB8888 Adapted Command
  *         Mode with Tearing Effect synchronization.
  * @retval BSP_ERROR_NONE on success.
  */
uint8_t LCD_Init(void)
{
  DSI_PHY_TimerTypeDef PhyTimings;
  OTM8009A_IO_t IOCtx;
  static OTM8009A_Object_t OTM8009AObj;
  static void *Lcd_CompObj = NULL;

  /* Hardware reset of the DSI LCD (XRES signal, active low) */
  BSP_LCD_Reset(0);

  /* MSP init: clocks, resets, NVIC for LTDC/DSI/DMA2D */
  LCD_MspInit();

  /* PLL3 for LTDC clock: HSE / 5 * 160 / 19 ≈ 42 MHz */
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_LTDC;
  PeriphClkInitStruct.PLL3.PLL3M = 5;
  PeriphClkInitStruct.PLL3.PLL3N = 160;
  PeriphClkInitStruct.PLL3.PLL3FRACN = 0;
  PeriphClkInitStruct.PLL3.PLL3P = 2;
  PeriphClkInitStruct.PLL3.PLL3Q = 2;
  PeriphClkInitStruct.PLL3.PLL3R = 19;
  PeriphClkInitStruct.PLL3.PLL3VCOSEL = RCC_PLL3VCOWIDE;
  PeriphClkInitStruct.PLL3.PLL3RGE = RCC_PLL3VCIRANGE_2;
  HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct);

  /* DSI init: 2 data lanes, PLL VCO = HSE/5 * 100 = 500 MHz */
  hlcd_dsi.Instance = DSI;
  HAL_DSI_DeInit(&(hlcd_dsi));
  dsiPllInit.PLLNDIV = 100;
  dsiPllInit.PLLIDF = DSI_PLL_IN_DIV5;
  dsiPllInit.PLLODF = DSI_PLL_OUT_DIV1;
  hlcd_dsi.Init.NumberOfLanes = DSI_TWO_DATA_LANES;
  hlcd_dsi.Init.TXEscapeCkdiv = 0x4;
  HAL_DSI_Init(&(hlcd_dsi), &(dsiPllInit));

  /* DSI Adapted Command Mode with Tearing Effect on DSI link */
  CmdCfg.VirtualChannelID = 0;
  CmdCfg.HSPolarity = DSI_HSYNC_ACTIVE_HIGH;
  CmdCfg.VSPolarity = DSI_VSYNC_ACTIVE_HIGH;
  CmdCfg.DEPolarity = DSI_DATA_ENABLE_ACTIVE_HIGH;
  CmdCfg.ColorCoding = DSI_RGB888;
  CmdCfg.CommandSize = LCD_HACT;
  CmdCfg.TearingEffectSource = DSI_TE_DSILINK;
  CmdCfg.TearingEffectPolarity = DSI_TE_RISING_EDGE;
  CmdCfg.VSyncPol = DSI_VSYNC_FALLING;
  CmdCfg.AutomaticRefresh = DSI_AR_DISABLE;
  CmdCfg.TEAcknowledgeRequest = DSI_TE_ACKNOWLEDGE_ENABLE;
  HAL_DSI_ConfigAdaptedCommandMode(&hlcd_dsi, &CmdCfg);

  /* Low-power command configuration (enabled during init) */
  LPCmd.LPGenShortWriteNoP = DSI_LP_GSW0P_ENABLE;
  LPCmd.LPGenShortWriteOneP = DSI_LP_GSW1P_ENABLE;
  LPCmd.LPGenShortWriteTwoP = DSI_LP_GSW2P_ENABLE;
  LPCmd.LPGenShortReadNoP = DSI_LP_GSR0P_ENABLE;
  LPCmd.LPGenShortReadOneP = DSI_LP_GSR1P_ENABLE;
  LPCmd.LPGenShortReadTwoP = DSI_LP_GSR2P_ENABLE;
  LPCmd.LPGenLongWrite = DSI_LP_GLW_ENABLE;
  LPCmd.LPDcsShortWriteNoP = DSI_LP_DSW0P_ENABLE;
  LPCmd.LPDcsShortWriteOneP = DSI_LP_DSW1P_ENABLE;
  LPCmd.LPDcsShortReadNoP = DSI_LP_DSR0P_ENABLE;
  LPCmd.LPDcsLongWrite = DSI_LP_DLW_ENABLE;
  HAL_DSI_ConfigCommand(&hlcd_dsi, &LPCmd);

  LTDC_Init();
  HAL_DSI_Start(&(hlcd_dsi));

  /* DSI PHY timings */
  PhyTimings.ClockLaneHS2LPTime = 35;
  PhyTimings.ClockLaneLP2HSTime = 35;
  PhyTimings.DataLaneHS2LPTime = 35;
  PhyTimings.DataLaneLP2HSTime = 35;
  PhyTimings.DataLaneMaxReadTime = 0;
  PhyTimings.StopWaitTime = 10;
  HAL_DSI_ConfigPhyTimer(&hlcd_dsi, &PhyTimings);

  /* OTM8009A panel driver initialization */
  IOCtx.Address = 0;
  IOCtx.GetTick = BSP_GetTick;
  IOCtx.WriteReg = DSI_IO_Write;
  IOCtx.ReadReg = DSI_IO_Read;
  OTM8009A_RegisterBusIO(&OTM8009AObj, &IOCtx);
  Lcd_CompObj = (&OTM8009AObj);
  OTM8009A_Init(Lcd_CompObj, OTM8009A_COLMOD_RGB888, LCD_ORIENTATION_LANDSCAPE);

  /* Disable low-power commands after init for normal operation */
  LPCmd.LPGenShortWriteNoP = DSI_LP_GSW0P_DISABLE;
  LPCmd.LPGenShortWriteOneP = DSI_LP_GSW1P_DISABLE;
  LPCmd.LPGenShortWriteTwoP = DSI_LP_GSW2P_DISABLE;
  LPCmd.LPGenShortReadNoP = DSI_LP_GSR0P_DISABLE;
  LPCmd.LPGenShortReadOneP = DSI_LP_GSR1P_DISABLE;
  LPCmd.LPGenShortReadTwoP = DSI_LP_GSR2P_DISABLE;
  LPCmd.LPGenLongWrite = DSI_LP_GLW_DISABLE;
  LPCmd.LPDcsShortWriteNoP = DSI_LP_DSW0P_DISABLE;
  LPCmd.LPDcsShortWriteOneP = DSI_LP_DSW1P_DISABLE;
  LPCmd.LPDcsShortReadNoP = DSI_LP_DSR0P_DISABLE;
  LPCmd.LPDcsLongWrite = DSI_LP_DLW_DISABLE;
  HAL_DSI_ConfigCommand(&hlcd_dsi, &LPCmd);

  HAL_DSI_ConfigFlowControl(&hlcd_dsi, DSI_FLOW_CONTROL_BTA);
  HAL_DSI_ForceRXLowPower(&hlcd_dsi, ENABLE);

  return BSP_ERROR_NONE;
}

/**
  * @brief  Initialize an LTDC layer with the given frame buffer address.
  *         Configures layer window, pixel format, alpha blending, and
  *         color keying for ARGB8888 output.
  * @param  LayerIndex: LTDC layer index (0 or 1).
  * @param  Address:    frame buffer start address in SDRAM.
  * @retval None
  */
void LCD_LayertInit(uint16_t LayerIndex, uint32_t Address)
{
  LTDC_LayerCfgTypeDef layercfg;

  layercfg.WindowX0 = 0;
  layercfg.WindowX1 = Lcd_Ctx[0].XSize;
  layercfg.WindowY0 = 0;
  layercfg.WindowY1 = Lcd_Ctx[0].YSize;
  layercfg.PixelFormat = LTDC_PIXEL_FORMAT_ARGB8888;
  layercfg.FBStartAdress = Address;
  layercfg.Alpha = 255;
  layercfg.Alpha0 = 0;
  layercfg.Backcolor.Blue = 0;
  layercfg.Backcolor.Green = 0;
  layercfg.Backcolor.Red = 0;
  layercfg.BlendingFactor1 = LTDC_BLENDING_FACTOR1_PAxCA;
  layercfg.BlendingFactor2 = LTDC_BLENDING_FACTOR2_PAxCA;
  layercfg.ImageWidth = Lcd_Ctx[0].XSize;
  layercfg.ImageHeight = Lcd_Ctx[0].YSize;

  HAL_LTDC_ConfigLayer(&hlcd_ltdc, &layercfg, LayerIndex);
}

/**
  * @brief  Display title and info on LCD.
  *         Draws a blue header bar (0-112 px) with project title,
  *         and a white area (112-480 px) with description text.
  * @retval None
  */
void LCD_BriefDisplay(void)
{
  /* Blue header background: 6 lines, 24+24+16+16+16+16 = 112 px */
  UTIL_LCD_SetFont(&Font24);
  UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_BLUE);
  UTIL_LCD_FillRect(0, 0, 800, 112, UTIL_LCD_COLOR_BLUE);
  UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_WHITE);
  UTIL_LCD_SetBackColor(UTIL_LCD_COLOR_BLUE);
  UTIL_LCD_DisplayStringAtLine(1, (uint8_t *)"            Final Project Face_Detection");

  /* White area: 112 + 368 = 480 px total */
  UTIL_LCD_FillRect(0, 112, 800, 368, UTIL_LCD_COLOR_WHITE);
  UTIL_LCD_SetFont(&Font16);
  UTIL_LCD_DisplayStringAtLine(4, (uint8_t *)"The image input through the serial port is used to recognize");
  UTIL_LCD_DisplayStringAtLine(5, (uint8_t *)"faces on the development board and display the results.");
}

/**
  * @brief  DMA2D memory-to-memory copy: transfers an image to the LCD frame buffer.
  *         The source image is placed at (x, y) on the 800x480 LCD with the
  *         specified dimensions. Uses polling mode with 100 ms timeout.
  * @param  pSrc:  pointer to source image data (ARGB8888 format).
  * @param  pDst:  pointer to destination frame buffer base.
  * @param  x:     horizontal offset on LCD (pixels).
  * @param  y:     vertical offset on LCD (pixels).
  * @param  xsize: source image width (pixels).
  * @param  ysize: source image height (pixels).
  * @retval None
  */
void CopyBuffer(uint32_t *pSrc, uint32_t *pDst,
                uint16_t x, uint16_t y,
                uint16_t xsize, uint16_t ysize)
{
  uint32_t destination = (uint32_t)pDst + (y * 800 + x) * 4;
  uint32_t source = (uint32_t)pSrc;

  hdma2d.Init.Mode = DMA2D_M2M;
  hdma2d.Init.ColorMode = DMA2D_OUTPUT_ARGB8888;
  hdma2d.Init.OutputOffset = 800 - xsize;
  hdma2d.Init.AlphaInverted = DMA2D_REGULAR_ALPHA;
  hdma2d.Init.RedBlueSwap = DMA2D_RB_REGULAR;
  hdma2d.XferCpltCallback = NULL;

  hdma2d.LayerCfg[1].AlphaMode = DMA2D_NO_MODIF_ALPHA;
  hdma2d.LayerCfg[1].InputAlpha = 0xFF;
  hdma2d.LayerCfg[1].InputColorMode = DMA2D_INPUT_ARGB8888;
  hdma2d.LayerCfg[1].InputOffset = 0;
  hdma2d.LayerCfg[1].RedBlueSwap = DMA2D_RB_REGULAR;
  hdma2d.LayerCfg[1].AlphaInverted = DMA2D_REGULAR_ALPHA;

  hdma2d.Instance = DMA2D;

  if (HAL_DMA2D_Init(&hdma2d) == HAL_OK)
  {
    if (HAL_DMA2D_ConfigLayer(&hdma2d, 1) == HAL_OK)
    {
      if (HAL_DMA2D_Start(&hdma2d, source, destination, xsize, ysize) == HAL_OK)
      {
        HAL_DMA2D_PollForTransfer(&hdma2d, 100);
      }
    }
  }
}

/**
  * @brief  DSI global interrupt handler.
  *         Dispatches to HAL_DSI_IRQHandler for all DSI interrupt sources
  *         including Tearing Effect (TE) and end-of-refresh events.
  * @retval None
  */
void DSI_IRQHandler(void)
{
  HAL_DSI_IRQHandler(&hlcd_dsi);
}

/* Private functions ----------------------------------------------------------*/

/**
  * @brief  Initialize the LTDC controller with panel timing parameters.
  *         Configures 800x480 resolution with all sync pulse widths set to 1,
  *         active-low polarities, and inverted pixel clock.
  * @retval None
  */
static void LTDC_Init(void)
{
  hlcd_ltdc.Instance = LTDC;
  HAL_LTDC_DeInit(&hlcd_ltdc);

  hlcd_ltdc.Init.HorizontalSync = LCD_HSYNC;
  hlcd_ltdc.Init.VerticalSync = LCD_VSYNC;
  hlcd_ltdc.Init.AccumulatedHBP = LCD_HSYNC + LCD_HBP;
  hlcd_ltdc.Init.AccumulatedVBP = LCD_VSYNC + LCD_VBP;
  hlcd_ltdc.Init.AccumulatedActiveH = LCD_VSYNC + LCD_VBP + LCD_VACT;
  hlcd_ltdc.Init.AccumulatedActiveW = LCD_HSYNC + LCD_HBP + LCD_HACT;
  hlcd_ltdc.Init.TotalHeigh = LCD_VSYNC + LCD_VBP + LCD_VACT + LCD_VFP;
  hlcd_ltdc.Init.TotalWidth = LCD_HSYNC + LCD_HBP + LCD_HACT + LCD_HFP;
  hlcd_ltdc.Init.Backcolor.Blue = 0;
  hlcd_ltdc.Init.Backcolor.Green = 0;
  hlcd_ltdc.Init.Backcolor.Red = 0;
  hlcd_ltdc.Init.HSPolarity = LTDC_HSPOLARITY_AL;
  hlcd_ltdc.Init.VSPolarity = LTDC_VSPOLARITY_AL;
  hlcd_ltdc.Init.DEPolarity = LTDC_DEPOLARITY_AL;
  hlcd_ltdc.Init.PCPolarity = LTDC_PCPOLARITY_IPC;
  hlcd_ltdc.Instance = LTDC;

  HAL_LTDC_Init(&hlcd_ltdc);
}

/**
  * @brief  Write a value to an LCD panel register via DSI command.
  *         Uses short write for 1-byte payloads and long write for multi-byte.
  * @param  ChannelNbr: DSI virtual channel ID.
  * @param  Reg:        register address.
  * @param  pData:      pointer to data byte(s).
  * @param  Size:       number of data bytes.
  * @retval BSP_ERROR_NONE on success, BSP_ERROR_BUS_FAILURE on error.
  */
static int32_t DSI_IO_Write(uint16_t ChannelNbr, uint16_t Reg,
                            uint8_t *pData, uint16_t Size)
{
  int32_t ret = BSP_ERROR_NONE;
  if (Size <= 1U)
  {
    if (HAL_DSI_ShortWrite(&hlcd_dsi, ChannelNbr, DSI_DCS_SHORT_PKT_WRITE_P1,
                           Reg, (uint32_t)pData[Size]) != HAL_OK)
    {
      ret = BSP_ERROR_BUS_FAILURE;
    }
  }
  else
  {
    if (HAL_DSI_LongWrite(&hlcd_dsi, ChannelNbr, DSI_DCS_LONG_PKT_WRITE,
                          Size, (uint32_t)Reg, pData) != HAL_OK)
    {
      ret = BSP_ERROR_BUS_FAILURE;
    }
  }
  return ret;
}

/**
  * @brief  Read a value from an LCD panel register via DSI command.
  * @param  ChannelNbr: DSI virtual channel ID.
  * @param  Reg:        register address.
  * @param  pData:      pointer to store read data bytes.
  * @param  Size:       number of bytes to read.
  * @retval BSP_ERROR_NONE on success, BSP_ERROR_BUS_FAILURE on error.
  */
static int32_t DSI_IO_Read(uint16_t ChannelNbr, uint16_t Reg,
                           uint8_t *pData, uint16_t Size)
{
  int32_t ret = BSP_ERROR_NONE;
  if (HAL_DSI_Read(&hlcd_dsi, ChannelNbr, pData, Size,
                   DSI_DCS_SHORT_PKT_READ, Reg, pData) != HAL_OK)
  {
    ret = BSP_ERROR_BUS_FAILURE;
  }
  return ret;
}

/**
  * @brief  MSP initialization for LTDC, DSI, and DMA2D peripherals.
  *         Enables peripheral clocks, releases resets, and configures
  *         NVIC interrupt priorities (group priority 9, sub-priority 0xF).
  * @retval None
  */
static void LCD_MspInit(void)
{
  __HAL_RCC_LTDC_CLK_ENABLE();
  __HAL_RCC_LTDC_FORCE_RESET();
  __HAL_RCC_LTDC_RELEASE_RESET();

  __HAL_RCC_DMA2D_CLK_ENABLE();
  __HAL_RCC_DMA2D_FORCE_RESET();
  __HAL_RCC_DMA2D_RELEASE_RESET();

  __HAL_RCC_DSI_CLK_ENABLE();
  __HAL_RCC_DSI_FORCE_RESET();
  __HAL_RCC_DSI_RELEASE_RESET();

  HAL_NVIC_SetPriority(LTDC_IRQn, 9, 0xf);
  HAL_NVIC_EnableIRQ(LTDC_IRQn);

  HAL_NVIC_SetPriority(DMA2D_IRQn, 9, 0xf);
  HAL_NVIC_EnableIRQ(DMA2D_IRQn);

  HAL_NVIC_SetPriority(DSI_IRQn, 9, 0xf);
  HAL_NVIC_EnableIRQ(DSI_IRQn);
}
