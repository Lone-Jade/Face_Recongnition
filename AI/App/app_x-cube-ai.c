
/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

  /**
    * Description
    * v1.0: Minimum template to show how to use the Embedded Client API ST-AI 
    *
        */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

#if defined ( __ICCARM__ )
#define AI_DTCMRAM   _Pragma("location=\"AI_DTCMRAM\"")
#define AI_ITCMRAM   _Pragma("location=\"AI_ITCMRAM\"")
#define AI_RAM_D1   _Pragma("location=\"AI_RAM_D1\"")
#define AI_FMC   _Pragma("location=\"AI_FMC\"")
#elif defined ( __CC_ARM ) || ( __GNUC__ )
#define AI_DTCMRAM   __attribute__((section(".AI_DTCMRAM")))
#define AI_ITCMRAM   __attribute__((section(".AI_ITCMRAM")))
#define AI_RAM_D1   __attribute__((section(".AI_RAM_D1")))
#define AI_FMC   __attribute__((section(".AI_FMC")))
#endif

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include "app_x-cube-ai.h"
#include "bsp_ai.h"
#include "stai.h"
/**
STAI_ALIGNED(32) static uint8_t data_in_1[STAI_NETWORK_IN_1_SIZE_BYTES];

// Array to store the data of the input tensor
stai_ptr data_ins[] = {
  data_in_1
}; 
*/

/* Output defs ----------------------------------------------------------------*/

/**
STAI_ALIGNED(32) 
static uint8_t data_out_1[STAI_NETWORK_OUT_1_SIZE_BYTES];
STAI_ALIGNED(32) 
static uint8_t data_out_2[STAI_NETWORK_OUT_2_SIZE_BYTES];
STAI_ALIGNED(32) 
static uint8_t data_out_3[STAI_NETWORK_OUT_3_SIZE_BYTES];
STAI_ALIGNED(32) 
static uint8_t data_out_4[STAI_NETWORK_OUT_4_SIZE_BYTES];
STAI_ALIGNED(32) 
static uint8_t data_out_5[STAI_NETWORK_OUT_5_SIZE_BYTES];
STAI_ALIGNED(32) 
static uint8_t data_out_6[STAI_NETWORK_OUT_6_SIZE_BYTES];
STAI_ALIGNED(32) 
static uint8_t data_out_7[STAI_NETWORK_OUT_7_SIZE_BYTES];
STAI_ALIGNED(32) 
static uint8_t data_out_8[STAI_NETWORK_OUT_8_SIZE_BYTES];
STAI_ALIGNED(32) 
static uint8_t data_out_9[STAI_NETWORK_OUT_9_SIZE_BYTES];
STAI_ALIGNED(32) 
static uint8_t data_out_10[STAI_NETWORK_OUT_10_SIZE_BYTES];
STAI_ALIGNED(32) 
static uint8_t data_out_11[STAI_NETWORK_OUT_11_SIZE_BYTES];
STAI_ALIGNED(32) 
static uint8_t data_out_12[STAI_NETWORK_OUT_12_SIZE_BYTES];

// c-array to store the data of the output tensor
stai_ptr data_outs[] = {
  data_out_1,
  data_out_2,
  data_out_3,
  data_out_4,
  data_out_5,
  data_out_6,
  data_out_7,
  data_out_8,
  data_out_9,
  data_out_10,
  data_out_11,
  data_out_12
}; 
*/




/* Global byte buffer to save instantiated C-model network context */
STAI_NETWORK_CONTEXT_DECLARE(network_context, STAI_NETWORK_CONTEXT_SIZE)

/* Activations buffers -------------------------------------------------------*/
STAI_ALIGNED(32) 
AI_DTCMRAM 
static uint8_t DTCMRAM[STAI_NETWORK_ACTIVATION_1_SIZE_BYTES];


/* Global c-array to handle the activations buffer */
/* Activation buf1: DTCM (59KB).  Activation buf2: SDRAM @ 0xD0800000 (2.34MB).
   SDRAM layout: LCD_FB@0xD0000000, IMG@0xD0400000, CAM@0xD0600000, AI_ACT2@0xD0800000.
   See CM7/Core/Inc/ai_detection.h for full map. */
stai_ptr data_activations[] = { DTCMRAM, (ai_handle)0xd0800000 };

STAI_ALIGNED(32) static uint8_t states_1[4];
stai_ptr data_states[] = {
    states_1
};




/* Entry points --------------------------------------------------------------*/

/* Array of pointer to manage the model's input/output tensors.
   NOT static — main.c references them via extern. */
stai_size in_length, out_length;
stai_ptr stai_input[STAI_NETWORK_IN_NUM];
stai_ptr stai_output[STAI_NETWORK_OUT_NUM];


/* 
 * Bootstrap
 */
int aiInit(void) {
  stai_return_code ret_code;

  /* 1: Initialize runtime library */
  ret_code = stai_runtime_init();
  if (ret_code != STAI_SUCCESS) return (int)ret_code;

  /* 2: Initialize network model context */
  ret_code = user_stai_network_init(network_context);
  if (ret_code != STAI_SUCCESS) return (int)ret_code;

  /* 3: Set network activations buffers */
  ret_code = stai_network_set_activations(network_context, data_activations, STAI_NETWORK_ACTIVATIONS_NUM);
  if (ret_code != STAI_SUCCESS) return (int)ret_code;

  /* 4: Get AI input/output buffer pointers (allocated internally or set above) */
  ret_code = stai_network_get_inputs(network_context, stai_input, &in_length);
  if (ret_code != STAI_SUCCESS) return (int)ret_code;

  ret_code = stai_network_get_outputs(network_context, stai_output, &out_length);
  if (ret_code != STAI_SUCCESS) return (int)ret_code;

  return 0;
}

int aiDeinit(void) {
  stai_return_code ret_code;

  /* 1: Deinitialize network model context */
  ret_code = stai_network_deinit(network_context);
  if (ret_code != STAI_SUCCESS) return (int)ret_code;

  /* 2: Deinitialize runtime library */
  ret_code = stai_runtime_deinit();
  if (ret_code != STAI_SUCCESS) return (int)ret_code;

  return 0;
}

/*
 * Run inference (synchronous, blocking). Caller must fill stai_input[0]
 * before calling, and read stai_output[] after this returns.
 */
stai_return_code aiRun(void) {
  stai_return_code ret_code;

  /* Perform the inference */
  ret_code = stai_network_run(network_context, STAI_MODE_SYNC);
  if (ret_code != STAI_SUCCESS) {
      ret_code = stai_network_get_error(network_context);
  }

  return ret_code;
}


int acquire_and_process_data()
{
  /* USER CODE BEGIN acquire_and_process_data */
  /* fill the inputs of the c-model 
  for (int idx=0; idx < STAI_NETWORK_IN_NUM; idx++ )
  {
      stai_input[idx] = ....
  }

  */
  return 0;
  /* USER CODE END acquire_and_process_data */
}

int post_process()
{
  /* USER CODE BEGIN post_process */
  /* process the predictions
  for (int idx=0; idx < STAI_NETWORK_OUT_NUM; idx++ )
  {
      stai_output[idx] = ....
  }

  */
  return 0;
  /* USER CODE END post_process */
}



/* 
 * Example of main loop function
 */
void main_loop() {
  /* USER CODE BEGIN main_loop */
  while (1) {
    /* 1 - Acquire, pre-process and fill the input buffers */
    acquire_and_process_data();

    /* 2 - Call inference engine */
    aiRun();

    /* 3 - Post-process the predictions */
    post_process();
  }
  /* USER CODE END main_loop */
}


/* Entry points --------------------------------------------------------------*/


void STM32CubeAI_Studio_AI_Init(void)
{
    aiInit();
    /* USER CODE BEGIN init */
    /* USER CODE END init */
}

void STM32CubeAI_Studio_AI_Process(void)
{
    main_loop();
} 

void STM32CubeAI_Studio_AI_Deinit(void)
{
    aiDeinit();
} 


#ifdef __cplusplus
}
#endif
