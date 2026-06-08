/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body - Face Detection on STM32H747I-DISCO
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
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stm32h747i_discovery.h"
#include "stm32h747i_discovery_lcd.h"
#include "stm32h747i_discovery_sdram.h"
#include "stm32h747i_discovery_bus.h"
#include "stm32_lcd.h"

#include "face_img_0.h"
#include "face_img_1.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "app_x-cube-ai.h"
#include "bsp_ai.h"
#include "network.h"
#include "network_data.h"
#include "ai_detection.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USART RX state machine states */
typedef enum
{
	RX_STATE_IDLE = 0,	/* Waiting for "FACE" or "MULT" magic header (4 bytes) */
	RX_STATE_GOT_MAGIC, /* Magic received, now reading size/count field (4 bytes) */
	RX_STATE_GOT_SIZE,	/* Size parsed, now receiving raw image pixel data */
	RX_STATE_RECEIVING	/* (Alias) actively receiving image bytes into SDRAM */
} rx_state_t;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define LCD_FRAME_BUFFER 0xD0000000

#ifndef HSEM_ID_0
#define HSEM_ID_0 (0U) /* HW semaphore 0 */
#endif

/* ---- USART Image Receive Protocol ----
   Single image:  [4B "FACE"] [4B size LE] [raw ARGB8888 bytes]
   End of batch:  [4B "MULT"] [4B count LE] [4B 0]
   Image spec: 320x240 ARGB8888, 307200 bytes each.
   Storage: SDRAM from 0xD0400000, max 10 images. */
#define USART_RX_MAGIC_FACE 0x45434146 /* "FACE" little-endian */
#define USART_RX_MAGIC_MULT 0x544C554D /* "MULT" little-endian */
#define USART_RX_IMG_SIZE (320 * 240 * 4)
#define USART_IMG_MAX 10 /* Max images in SDRAM (4MB / 300KB) */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
#define VSYNC 1
#define VBP 1
#define VFP 1
#define VACT 480
#define HSYNC 1
#define HBP 1
#define HFP 1
#define HACT 800

#define LAYER0_ADDRESS (LCD_FB_START_ADDRESS)
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* ---- LCD / DMA2D / DSI handles ---- */
extern LTDC_HandleTypeDef hlcd_ltdc;
static DMA2D_HandleTypeDef hdma2d;
extern DSI_HandleTypeDef hlcd_dsi;
DSI_VidCfgTypeDef hdsivideo_handle;
DSI_CmdCfgTypeDef CmdCfg;
DSI_LPCmdTypeDef LPCmd;
DSI_PLLInitTypeDef dsiPllInit;
static RCC_PeriphCLKInitTypeDef PeriphClkInitStruct;
OTM8009A_Object_t *pObj;

/* ---- Frame buffer synchronization ---- */
static int32_t pending_buffer = -1;

/* ---- Built-in demo images ---- */
static int ImageIndex = 0;
static const uint32_t *Images[] = {
	face_img_0, /* WIDER Face: Parade */
	face_img_1, /* WIDER Face: Students */
};
static int num_builtin = 2;
static int use_builtin = 1; /* 1 = demo mode, 0 = USART mode */

uint8_t My_String[80]; /* General-purpose string buffer for LCD display */

/* ---- USART RX state machine variables ---- */
static rx_state_t rx_state = RX_STATE_IDLE;
static uint32_t rx_expected_size = 0;
static uint32_t rx_received = 0;
static uint8_t rx_header_buf[8]; /* 4 magic + 4 size */
static uint8_t rx_header_idx = 0;
static uint32_t rx_magic_type = 0; /* FACE or MULT */
static volatile uint8_t rx_img_ready = 0;
static uint8_t rx_byte; /* Single-byte RX buffer */

/* ---- Multi-image mode ---- */
static int usart_img_count = 0;				   /* Number of USART images received */
static int usart_img_index = 0;				   /* Current display index */
static volatile uint8_t usart_multi_ready = 0; /* MULT end-of-batch received */

/* ---- AI inference variables ---- */
static float *ai_input;						 /* Model input tensor pointer */
static Detection detections[MAX_DETECTIONS]; /* Face detection results */
static int num_detections = 0;

/* Forward declarations for functions used by LCD_UTIL_Driver below */
int32_t LCD_GetXSize(uint32_t Instance, uint32_t *XSize);
int32_t LCD_GetYSize(uint32_t Instance, uint32_t *YSize);

/* ---- LCD utility driver ---- */
const LCD_UTILS_Drv_t LCD_UTIL_Driver = {
	BSP_LCD_DrawBitmap,
	BSP_LCD_FillRGBRect,
	BSP_LCD_DrawHLine,
	BSP_LCD_DrawVLine,
	BSP_LCD_FillRect,
	BSP_LCD_ReadPixel,
	BSP_LCD_WritePixel,
	LCD_GetXSize,
	LCD_GetYSize,
	BSP_LCD_SetActiveLayer,
	BSP_LCD_GetPixelFormat};
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MPU_Config(void);
/* USER CODE BEGIN PFP */
static void USART_StartRxByte(void);
static uint32_t *usart_get_img(int n);

static void CopyBuffer(uint32_t *pSrc, uint32_t *pDst,
					   uint16_t x, uint16_t y,
					   uint16_t xsize, uint16_t ysize);
static uint8_t LCD_Init(void);
void LTDC_Init(void);
static void LCD_LayertInit(uint16_t LayerIndex, uint32_t Address);
static int32_t DSI_IO_Write(uint16_t ChannelNbr, uint16_t Reg,
							uint8_t *pData, uint16_t Size);
static int32_t DSI_IO_Read(uint16_t ChannelNbr, uint16_t Reg,
						   uint8_t *pData, uint16_t Size);
int32_t LCD_GetXSize(uint32_t Instance, uint32_t *XSize);
int32_t LCD_GetYSize(uint32_t Instance, uint32_t *YSize);
void LCD_MspInit(void);
static void LCD_BriefDisplay(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* External reference to AI network context (defined in app_x-cube-ai.c) */
extern uint8_t network_context[STAI_NETWORK_CONTEXT_SIZE];
extern stai_ptr stai_input[STAI_NETWORK_IN_NUM];
extern stai_ptr stai_output[STAI_NETWORK_OUT_NUM];

/* ---- USART helper functions ---- */

/* Start single-byte interrupt reception on USART1 */
static void USART_StartRxByte(void)
{
	HAL_UART_Receive_IT(&huart1, &rx_byte, 1);
}

/* Get pointer to n-th received image in SDRAM (each image is 307200 bytes) */
static uint32_t *usart_get_img(int n)
{
	return (uint32_t *)(AI_IMG_BUF_ADDR + n * USART_RX_IMG_SIZE);
}
/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void)
{
	/* USER CODE BEGIN 1 */

	/* USER CODE END 1 */
	/* USER CODE BEGIN Boot_Mode_Sequence_0 */
	int32_t timeout;
	/* USER CODE END Boot_Mode_Sequence_0 */

	/* MPU Configuration--------------------------------------------------------*/
	MPU_Config();

	/* Enable the CPU Cache */
	/* Enable I-Cache---------------------------------------------------------*/
	SCB_EnableICache();
	/* Enable D-Cache---------------------------------------------------------*/
	SCB_EnableDCache();

	/* USER CODE BEGIN Boot_Mode_Sequence_1 */
	/* Wait until CPU2 boots and enters stop mode or timeout */
	timeout = 0xFFFF;
	while ((__HAL_RCC_GET_FLAG(RCC_FLAG_D2CKRDY) != RESET) && (timeout-- > 0))
		;
	if (timeout < 0)
	{
		Error_Handler();
	}
	/* USER CODE END Boot_Mode_Sequence_1 */
	/* MCU Configuration--------------------------------------------------------*/

	/* Reset of all peripherals, Initializes the Flash interface and the Systick. */
	HAL_Init();

	/* USER CODE BEGIN Init */

	/* USER CODE END Init */

	/* Configure the system clock */
	SystemClock_Config();
	/* USER CODE BEGIN Boot_Mode_Sequence_2 */
	/* Release CM4 from stop mode via HSEM semaphore */
	__HAL_RCC_HSEM_CLK_ENABLE();
	HAL_HSEM_FastTake(HSEM_ID_0);
	HAL_HSEM_Release(HSEM_ID_0, 0);
	timeout = 0xFFFF;
	while ((__HAL_RCC_GET_FLAG(RCC_FLAG_D2CKRDY) == RESET) && (timeout-- > 0))
		;
	if (timeout < 0)
	{
		Error_Handler();
	}
	/* USER CODE END Boot_Mode_Sequence_2 */

	/* USER CODE BEGIN SysInit */

	/* USER CODE END SysInit */

	/* Initialize all configured peripherals */
	MX_GPIO_Init();
	MX_USART1_UART_Init();
	/* USER CODE BEGIN 2 */
	/* Initialize LED */
	BSP_LED_Init(LED3);

	/* Initialize SDRAM */
	BSP_SDRAM_Init(0);

	/* Initialize LCD */
	if (LCD_Init() != BSP_ERROR_NONE)
	{
		Error_Handler();
	}

	/* Configure LCD context */
	Lcd_Ctx[0].ActiveLayer = 0;
	Lcd_Ctx[0].PixelFormat = LCD_PIXEL_FORMAT_ARGB8888;
	Lcd_Ctx[0].BppFactor = 4;
	Lcd_Ctx[0].XSize = 800;
	Lcd_Ctx[0].YSize = 480;

	/* Initialize LTDC layer 0 */
	__HAL_DSI_WRAPPER_DISABLE(&hlcd_dsi);
	LCD_LayertInit(0, LCD_FRAME_BUFFER);
	UTIL_LCD_SetFuncDriver(&LCD_UTIL_Driver);
	__HAL_DSI_WRAPPER_ENABLE(&hlcd_dsi);

	/* Display title and info */
	LCD_BriefDisplay();
	LCD_BriefDisplay();

	/* Draw first built-in image (centered at x=240, y=160) */
	CopyBuffer((uint32_t *)Images[ImageIndex++],
			   (uint32_t *)LCD_FRAME_BUFFER, 240, 160, 320, 240);
	pending_buffer = 0;
	HAL_DSI_Refresh(&hlcd_dsi);

	/* Initialize Cube.AI runtime and model */
	STM32CubeAI_Studio_AI_Init();
	ai_input = (float *)stai_input[0]; /* Cache input tensor: 1x3x320x320 float32 */

	/* Start USART1 RX interrupt for image reception */
	USART_StartRxByte();
	/* USER CODE END 2 */

	/* Infinite loop */
	/* USER CODE BEGIN WHILE */
	while (1)
	{
		/* ---- Check MULT end-of-batch packet (enter multi-image cycle mode) ---- */
		if (usart_multi_ready)
		{
			usart_multi_ready = 0;
			use_builtin = 0;
			sprintf((char *)My_String, "USART: %d images", usart_img_count);
			UTIL_LCD_DisplayStringAtLine(7, (uint8_t *)My_String);
		}

		/* ---- Check USART-received image ---- */
		if (rx_img_ready)
		{
			rx_img_ready = 0;
			use_builtin = 0;
			sprintf((char *)My_String, "USART: recv %d image(s)", usart_img_count);
			UTIL_LCD_DisplayStringAtLine(7, (uint8_t *)My_String);
		}

		/* ---- Frame processing (TE-synchronized) ---- */
		if (pending_buffer < 0)
		{
			uint32_t *src_img;

			/* Select image source: built-in demo or USART */
			if (use_builtin)
			{
				src_img = (uint32_t *)Images[ImageIndex];
			}
			else
			{
				src_img = usart_get_img(usart_img_index);
			}

			/* Step 1: Copy image to LCD frame buffer via DMA2D */
			CopyBuffer(src_img, (uint32_t *)LCD_FRAME_BUFFER, 240, 160, 320, 240);

			/* Step 2: Preprocess - ARGB8888 to NCHW BGR float32 */
			ai_preprocess(src_img, 320, 240, ai_input, 320, 320);
			/* Clean D-Cache so ST AI runtime reads correct input from SDRAM */
			SCB_CleanDCache_by_Addr((uint32_t *)ai_input,
									(int32_t)(3 * 320 * 320 * sizeof(float)));

			/* Step 3: Run AI inference */
			stai_return_code ret = aiRun();
			if (ret != STAI_SUCCESS)
			{
				sprintf((char *)My_String, "AI err: %d", (int)ret);
				UTIL_LCD_DisplayStringAtLine(26, (uint8_t *)My_String);
			}
			else
			{
				/* Step 4: Post-process - decode detections + NMS + containment filter */
				num_detections = ai_postprocess(stai_output, detections, MAX_DETECTIONS,
												DET_THRESHOLD, NMS_THRESHOLD,
												MIN_BOX_SIZE);

				/* Step 5: Draw green bounding boxes on LCD frame buffer */
				ai_draw_detections(detections, num_detections,
								   (uint32_t *)LCD_FRAME_BUFFER,
								   800, 240, 160, 320, 240);

				/* Display detection count below the image (Line 26, Font16, y~416) */
				UTIL_LCD_SetBackColor(UTIL_LCD_COLOR_WHITE);
				UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_BLACK);
				sprintf((char *)My_String, "Faces detected: %d", num_detections);
				UTIL_LCD_DisplayStringAtLine(26, (uint8_t *)My_String);
			}

			/* Trigger DSI refresh (waits for Tearing Effect signal) */
			pending_buffer = 0;
			HAL_DSI_Refresh(&hlcd_dsi);

			/* Advance to next image */
			if (use_builtin)
			{
				ImageIndex = (ImageIndex + 1) % num_builtin;
			}
			else if (usart_img_count > 1)
			{
				usart_img_index = (usart_img_index + 1) % usart_img_count;
			}
		}
		HAL_Delay(2000);
		/* USER CODE END WHILE */
		/* USER CODE BEGIN 3 */
	}
	/* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void)
{
	RCC_OscInitTypeDef RCC_OscInitStruct = {0};
	RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

	/** Supply configuration update enable */
	HAL_PWREx_ConfigSupply(PWR_SMPS_2V5_SUPPLIES_EXT);

	/** Configure the main internal regulator output voltage */
	__HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);
	while (!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY))
	{
	}

	/** Initializes the RCC Oscillators */
	RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
	RCC_OscInitStruct.HSEState = RCC_HSE_ON;
	RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
	RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
	RCC_OscInitStruct.PLL.PLLM = 5;
	RCC_OscInitStruct.PLL.PLLN = 192;
	RCC_OscInitStruct.PLL.PLLP = 2;
	RCC_OscInitStruct.PLL.PLLQ = 2;
	RCC_OscInitStruct.PLL.PLLR = 2;
	RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_2;
	RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
	RCC_OscInitStruct.PLL.PLLFRACN = 0;
	if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
	{
		Error_Handler();
	}

	/** Initializes the CPU, AHB and APB buses clocks */
	RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2 | RCC_CLOCKTYPE_D3PCLK1 | RCC_CLOCKTYPE_D1PCLK1;
	RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
	RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
	RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
	RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
	RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
	RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
	RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

	if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
	{
		Error_Handler();
	}
}

/* USER CODE BEGIN 4 */

/* ---- LCD Utility Functions ---- */

int32_t LCD_GetXSize(uint32_t Instance, uint32_t *XSize)
{
	*XSize = Lcd_Ctx[0].XSize;
	return BSP_ERROR_NONE;
}

int32_t LCD_GetYSize(uint32_t Instance, uint32_t *YSize)
{
	*YSize = Lcd_Ctx[0].YSize;
	return BSP_ERROR_NONE;
}

/* ---- DSI / LTDC / DMA2D Callbacks and Helpers ---- */

/**
 * @brief  End of Refresh DSI callback (Tearing Effect signal received).
 *         Signals the main loop that the frame buffer is free for next frame.
 */
void HAL_DSI_EndOfRefreshCallback(DSI_HandleTypeDef *hdsi)
{
	if (pending_buffer >= 0)
	{
		pending_buffer = -1;
	}
}

/**
 * @brief  Initialize the DSI LCD (PLL, DSI, LTDC, OTM8009A driver).
 */
static uint8_t LCD_Init(void)
{
	DSI_PHY_TimerTypeDef PhyTimings;
	OTM8009A_IO_t IOCtx;
	static OTM8009A_Object_t OTM8009AObj;
	static void *Lcd_CompObj = NULL;

	/* Hardware reset of the DSI LCD (XRES signal, active low) */
	BSP_LCD_Reset(0);

	/* MSP init: clocks, resets, NVIC for LTDC/DSI/DMA2D */
	LCD_MspInit();

	/* PLL3 for LTDC clock: HSE/5*160/19 = ~42 MHz */
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

	/* DSI init: 2 data lanes, PLL = HSE/5 * 100 = 500 MHz VCO */
	hlcd_dsi.Instance = DSI;
	HAL_DSI_DeInit(&(hlcd_dsi));
	dsiPllInit.PLLNDIV = 100;
	dsiPllInit.PLLIDF = DSI_PLL_IN_DIV5;
	dsiPllInit.PLLODF = DSI_PLL_OUT_DIV1;
	hlcd_dsi.Init.NumberOfLanes = DSI_TWO_DATA_LANES;
	hlcd_dsi.Init.TXEscapeCkdiv = 0x4;
	HAL_DSI_Init(&(hlcd_dsi), &(dsiPllInit));

	/* DSI Adapted Command Mode with Tearing Effect */
	CmdCfg.VirtualChannelID = 0;
	CmdCfg.HSPolarity = DSI_HSYNC_ACTIVE_HIGH;
	CmdCfg.VSPolarity = DSI_VSYNC_ACTIVE_HIGH;
	CmdCfg.DEPolarity = DSI_DATA_ENABLE_ACTIVE_HIGH;
	CmdCfg.ColorCoding = DSI_RGB888;
	CmdCfg.CommandSize = HACT;
	CmdCfg.TearingEffectSource = DSI_TE_DSILINK;
	CmdCfg.TearingEffectPolarity = DSI_TE_RISING_EDGE;
	CmdCfg.VSyncPol = DSI_VSYNC_FALLING;
	CmdCfg.AutomaticRefresh = DSI_AR_DISABLE;
	CmdCfg.TEAcknowledgeRequest = DSI_TE_ACKNOWLEDGE_ENABLE;
	HAL_DSI_ConfigAdaptedCommandMode(&hlcd_dsi, &CmdCfg);

	/* Low-power command configuration */
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

	/* OTM8009A panel driver init */
	IOCtx.Address = 0;
	IOCtx.GetTick = BSP_GetTick;
	IOCtx.WriteReg = DSI_IO_Write;
	IOCtx.ReadReg = DSI_IO_Read;
	OTM8009A_RegisterBusIO(&OTM8009AObj, &IOCtx);
	Lcd_CompObj = (&OTM8009AObj);
	OTM8009A_Init(Lcd_CompObj, OTM8009A_COLMOD_RGB888, LCD_ORIENTATION_LANDSCAPE);

	/* Disable low-power commands after init */
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

void LTDC_Init(void)
{
	hlcd_ltdc.Instance = LTDC;
	HAL_LTDC_DeInit(&hlcd_ltdc);

	hlcd_ltdc.Init.HorizontalSync = HSYNC;
	hlcd_ltdc.Init.VerticalSync = VSYNC;
	hlcd_ltdc.Init.AccumulatedHBP = HSYNC + HBP;
	hlcd_ltdc.Init.AccumulatedVBP = VSYNC + VBP;
	hlcd_ltdc.Init.AccumulatedActiveH = VSYNC + VBP + VACT;
	hlcd_ltdc.Init.AccumulatedActiveW = HSYNC + HBP + HACT;
	hlcd_ltdc.Init.TotalHeigh = VSYNC + VBP + VACT + VFP;
	hlcd_ltdc.Init.TotalWidth = HSYNC + HBP + HACT + HFP;
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

static void LCD_LayertInit(uint16_t LayerIndex, uint32_t Address)
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

void LCD_MspInit(void)
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

/**
 * @brief  Display title and info on LCD.
 *         Blue header (y=0-112): project title
 *         White area (y=112-480): description text and image area
 */
static void LCD_BriefDisplay(void)
{
	/* Blue header background: 6 lines, 24+24+16+16+16+16 = 112px */
	UTIL_LCD_SetFont(&Font24);
	UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_BLUE);
	UTIL_LCD_FillRect(0, 0, 800, 112, UTIL_LCD_COLOR_BLUE);
	UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_WHITE);
	UTIL_LCD_SetBackColor(UTIL_LCD_COLOR_BLUE);
	UTIL_LCD_DisplayStringAtLine(1, (uint8_t *)"            Final Project Face_Detection");

	/* White area: 112+368 = 480px total */
	UTIL_LCD_FillRect(0, 112, 800, 368, UTIL_LCD_COLOR_WHITE);
	UTIL_LCD_SetFont(&Font16);
	UTIL_LCD_DisplayStringAtLine(4, (uint8_t *)"The image input through the serial port is used to recognize");
	UTIL_LCD_DisplayStringAtLine(5, (uint8_t *)"faces on the development board and display the results.");
}

/**
 * @brief  DMA2D memory-to-memory copy: transfers image to LCD frame buffer.
 *         Image is placed at (x,y) with size (xsize, ysize) on 800x480 LCD.
 */
static void CopyBuffer(uint32_t *pSrc, uint32_t *pDst,
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

/******************************************************************************/
/*                 STM32H7xx Peripherals Interrupt Handlers                   */
/******************************************************************************/

void DSI_IRQHandler(void)
{
	HAL_DSI_IRQHandler(&hlcd_dsi);
}

/**
 * @brief  Dummy ExitRun0Mode - not used. CM4 wakeup via HSEM semaphore.
 */
void ExitRun0Mode(void)
{
	/* Empty: CM4 is woken via HSEM, not via RUN0 mode exit */
}

/**
 * @brief  USART1 RX Complete callback.
 *         State machine for "FACE"/"MULT" image receive protocol.
 *         Receives one byte at a time via interrupt, reassembles packets.
 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
	if (huart->Instance != USART1)
		return;

	uint8_t byte = rx_byte;

	switch (rx_state)
	{
	case RX_STATE_IDLE: /* Waiting for "FACE" or "MULT" magic */
		rx_header_buf[rx_header_idx++] = byte;
		if (rx_header_idx >= 4)
		{
			uint32_t magic = *(uint32_t *)rx_header_buf;
			if (magic == USART_RX_MAGIC_FACE || magic == USART_RX_MAGIC_MULT)
			{
				rx_magic_type = magic;
				rx_state = RX_STATE_GOT_MAGIC;
				rx_header_idx = 0;
			}
			else
			{
				/* Shift buffer to resync on next byte */
				rx_header_buf[0] = rx_header_buf[1];
				rx_header_buf[1] = rx_header_buf[2];
				rx_header_buf[2] = rx_header_buf[3];
				rx_header_idx = 3;
			}
		}
		break;

	case RX_STATE_GOT_MAGIC: /* Reading size/count field */
		rx_header_buf[rx_header_idx++] = byte;
		if (rx_header_idx >= 4)
		{
			if (rx_magic_type == USART_RX_MAGIC_MULT)
			{
				/* MULT: count = number of images in batch, enter cycle mode */
				uint32_t count = *(uint32_t *)(rx_header_buf);
				if (count > 0 && count <= USART_IMG_MAX)
				{
					usart_img_count = (int)count;
					usart_img_index = 0;
					usart_multi_ready = 1;
				}
				rx_state = RX_STATE_IDLE;
				rx_header_idx = 0;
			}
			else
			{
				/* FACE: size = raw ARGB8888 bytes to follow */
				rx_expected_size = *(uint32_t *)(rx_header_buf);
				if (rx_expected_size > 0 && rx_expected_size <= USART_RX_IMG_SIZE * 4)
				{
					rx_received = 0;
					rx_state = RX_STATE_RECEIVING;
				}
				else
				{
					rx_state = RX_STATE_IDLE;
					rx_header_idx = 0;
				}
			}
		}
		break;

	case RX_STATE_RECEIVING: /* Receiving raw image bytes into SDRAM */
		((uint8_t *)usart_get_img(usart_img_count))[rx_received] = byte;
		rx_received++;
		if (rx_received >= rx_expected_size)
		{
			usart_img_count++;
			rx_img_ready = 1;
			rx_state = RX_STATE_IDLE;
			rx_header_idx = 0;
		}
		break;

	default:
		rx_state = RX_STATE_IDLE;
		rx_header_idx = 0;
		break;
	}

	USART_StartRxByte();
}

/* USER CODE END 4 */

/* MPU Configuration */

void MPU_Config(void)
{
	MPU_Region_InitTypeDef MPU_InitStruct = {0};

	HAL_MPU_Disable();

	/** Region 0: 4GB background region - no access by default */
	MPU_InitStruct.Enable = MPU_REGION_ENABLE;
	MPU_InitStruct.Number = MPU_REGION_NUMBER0;
	MPU_InitStruct.BaseAddress = 0x0;
	MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
	MPU_InitStruct.SubRegionDisable = 0x87;
	MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
	MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
	MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
	MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
	MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
	MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;
	HAL_MPU_ConfigRegion(&MPU_InitStruct);

	/** Region 1: SDRAM 32MB - full access, cacheable */
	MPU_InitStruct.Number = MPU_REGION_NUMBER1;
	MPU_InitStruct.BaseAddress = SDRAM_DEVICE_ADDR;
	MPU_InitStruct.Size = MPU_REGION_SIZE_32MB;
	MPU_InitStruct.SubRegionDisable = 0x0;
	MPU_InitStruct.AccessPermission = MPU_REGION_FULL_ACCESS;
	MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_ENABLE;
	MPU_InitStruct.IsShareable = MPU_ACCESS_NOT_SHAREABLE;
	MPU_InitStruct.IsCacheable = MPU_ACCESS_CACHEABLE;
	HAL_MPU_ConfigRegion(&MPU_InitStruct);

	HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

/**
 * @brief  This function is executed in case of error occurrence.
 */
void Error_Handler(void)
{
	/* USER CODE BEGIN Error_Handler_Debug */
	__disable_irq();
	while (1)
	{
	}
	/* USER CODE END Error_Handler_Debug */
}

#ifdef USE_FULL_ASSERT
/**
 * @brief  Reports the name of the source file and the source line number
 *         where the assert_param error has occurred.
 */
void assert_failed(uint8_t *file, uint32_t line)
{
	/* USER CODE BEGIN 6 */
	/* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
