/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-06-05T15:48:46+0800
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "ai_lite_inspect.h"
#include "ai_platform_interface.h"
#include "layers.h"
#include "core_convert.h"
#include "network.h"
#include "network_details.h"
#include "network_data.h"
#include "stai_events.h"

#include "lite_operators.h"

#include "ai_lite_inspect.h"
/*****************************************************************************/
#define STAI_INTERNAL_API_MAJOR               (1)
#define STAI_INTERNAL_API_MINOR               (0)
#define STAI_INTERNAL_API_MICRO               (0)

#define STAI_MAGIC                            (0xB1C00100)

/*****************************************************************************/
#define _STAI_CONCAT_ARG(a, b)     a ## b
#define STAI_CONCAT(a, b)         _STAI_CONCAT_ARG(a, b)

/*!  STAI_CAST SECTION                       *********************************/
#define STAI_CAST(type, expr) \
  ((type)(expr))


/*****************************************************************************/
#define STAI_SIZE(_size) \
  ((stai_size)(_size))

/*****************************************************************************/
#define STAI_INIT_BUFFER(_flags, _size, _address) \
  { \
    .size = (_size), \
    .address = (uintptr_t)(_address), \
    .flags = (_flags), \
  }

#define STAI_INIT_TENSOR(_name, _flags, _fmt, _size_bytes, _shape, _scale, _zeropoint) \
  { \
    .size_bytes = (_size_bytes), \
    .flags = (_flags), \
    .format = (stai_format)(_fmt), \
    .shape = STAI_PACK(_shape), \
    .scale = STAI_PACK(_scale), \
    .zeropoint = STAI_PACK(_zeropoint), \
    .name = (_name) \
  }

#define STAI_INIT_ARRAY(_size, _ptr) \
  { .size = STAI_SIZE(_size), .data = STAI_PACK(_ptr) }


#define STAI_CAST_ARRAY(_type, _size, _ptr) \
  { .size = STAI_SIZE(_size), .data = (_type)STAI_PACK(_ptr) }


#define STAI_DECLARE_ARRAY(_type, _size, ...) \
  { .size = STAI_SIZE(_size), .data = (_type[_size]) { STAI_PACK(__VA_ARGS__) } }


#define STAI_EMPTY_ARRAY() \
  { .size = 0, .data = NULL }


#define STAI_INIT_VERSION(_major, _minor, _micro) \
  { .major = (_major), .minor = (_minor), .micro = (_micro), .reserved = 0x0 }

/*****************************************************************************/
/**  Getters and setters  **/

#define STAI_GET_ARRAY_SIZE(nd_array) \
  (nd_array.size)


#define STAI_GET_ARRAY_ELEM(nd_array, pos) \
  (nd_array.data[(pos)])

#define _STAI_SET_ERROR(net_ctx, cond, value, exit) { \
  if (!(net_ctx)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE; } \
  if (((uintptr_t)net_ctx) & (_STAI_CONTEXT_ALIGNMENT-1)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_ALIGNMENT; } \
  if (((value) >= STAI_ERROR_GENERIC) && (cond)) { \
    if ((net_ctx)->_return_code == STAI_SUCCESS) { \
      (net_ctx)->_return_code = (value); \
    } \
    return (exit); \
  } \
}

/*****************************************************************************/
/* TODO REMOVE THESE TWO MACROS */
#define STAI_EVENT_NODE_START_CB
#define STAI_EVENT_NODE_STOP_CB

#ifdef STAI_EVENT_NODE_START_CB
#ifndef _STAI_NETWORK_EVENT_NODE_START_CB
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _start_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(const stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_START, (const void*)&_start_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_START_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_START_CB */

#ifdef STAI_EVENT_NODE_STOP_CB
#ifndef _STAI_NETWORK_EVENT_NODE_STOP_CB
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _stop_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_STOP, (const void*)&_stop_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_STOP_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_STOP_CB */


/*****************************************************************************/
#define _STAI_NETWORK_MODEL_SIGNATURE     "0x28cbcb2715420c9ce9d9e3b61204a6be"
#define _STAI_NETWORK_DATETIME            "2026-06-05T15:48:46+0800"
#define _STAI_NETWORK_COMPILE_DATETIME    __DATE__ " " __TIME__

#define _STAI_CONTEXT_ALIGNMENT        STAI_NETWORK_CONTEXT_ALIGNMENT

/*****************************************************************************/
#define g_network_activations_1     (NULL)
#define g_network_activations_2     (NULL)




#if defined(HAVE_NETWORK_INFO)
/*****************************************************************************/
static const stai_network_info g_network_info = {
  .model_signature = _STAI_NETWORK_MODEL_SIGNATURE,
  .c_compile_datetime = _STAI_NETWORK_COMPILE_DATETIME,
  .c_model_name = STAI_NETWORK_MODEL_NAME,
  .c_model_datetime = _STAI_NETWORK_DATETIME,
  .c_model_signature = 0x0,
  .runtime_version = STAI_INIT_VERSION(12, 0, 1),
  .tool_version = STAI_INIT_VERSION(4, 0, 1),
  .api_version = STAI_INIT_VERSION(1, 0, 0),
  .n_macc = STAI_NETWORK_MACC_NUM,
  .n_nodes = STAI_NETWORK_NODES_NUM,
  .flags = STAI_NETWORK_FLAGS,
  .n_inputs = STAI_NETWORK_IN_NUM,
  .n_outputs = STAI_NETWORK_OUT_NUM,
  .n_activations = STAI_NETWORK_ACTIVATIONS_NUM,
  .n_weights = STAI_NETWORK_WEIGHTS_NUM,
  .n_states = STAI_NETWORK_STATES_NUM,
  .inputs = (stai_tensor[STAI_NETWORK_IN_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_IN_1_NAME,
      STAI_NETWORK_IN_1_FLAGS,
      STAI_NETWORK_IN_1_FORMAT,
      STAI_NETWORK_IN_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 3, 320, 320),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 3, 1, 1600, 1),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_2_NAME,
      STAI_NETWORK_OUT_2_FLAGS,
      STAI_NETWORK_OUT_2_FORMAT,
      STAI_NETWORK_OUT_2_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 3, 1, 400, 1),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_3_NAME,
      STAI_NETWORK_OUT_3_FLAGS,
      STAI_NETWORK_OUT_3_FORMAT,
      STAI_NETWORK_OUT_3_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 3, 1, 100, 1),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_4_NAME,
      STAI_NETWORK_OUT_4_FLAGS,
      STAI_NETWORK_OUT_4_FORMAT,
      STAI_NETWORK_OUT_4_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 3, 1, 1600, 1),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_5_NAME,
      STAI_NETWORK_OUT_5_FLAGS,
      STAI_NETWORK_OUT_5_FORMAT,
      STAI_NETWORK_OUT_5_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 3, 1, 400, 1),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_6_NAME,
      STAI_NETWORK_OUT_6_FLAGS,
      STAI_NETWORK_OUT_6_FORMAT,
      STAI_NETWORK_OUT_6_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 3, 1, 100, 1),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_7_NAME,
      STAI_NETWORK_OUT_7_FLAGS,
      STAI_NETWORK_OUT_7_FORMAT,
      STAI_NETWORK_OUT_7_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 4, 40, 40),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_8_NAME,
      STAI_NETWORK_OUT_8_FLAGS,
      STAI_NETWORK_OUT_8_FORMAT,
      STAI_NETWORK_OUT_8_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 4, 20, 20),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_9_NAME,
      STAI_NETWORK_OUT_9_FLAGS,
      STAI_NETWORK_OUT_9_FORMAT,
      STAI_NETWORK_OUT_9_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 4, 10, 10),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_10_NAME,
      STAI_NETWORK_OUT_10_FLAGS,
      STAI_NETWORK_OUT_10_FORMAT,
      STAI_NETWORK_OUT_10_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 10, 40, 40),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_11_NAME,
      STAI_NETWORK_OUT_11_FLAGS,
      STAI_NETWORK_OUT_11_FORMAT,
      STAI_NETWORK_OUT_11_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 10, 20, 20),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_12_NAME,
      STAI_NETWORK_OUT_12_FLAGS,
      STAI_NETWORK_OUT_12_FORMAT,
      STAI_NETWORK_OUT_12_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 10, 10, 10),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .activations = (stai_tensor[STAI_NETWORK_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 59564),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_2_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_2_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 2457600),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_NETWORK_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 295360),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },

  .states = NULL
};
#endif

#define _STAI_CONTEXT_ACQUIRE(_net_ctx, _net_handle) \
  _stai_network_context* _net_ctx = (_stai_network_context*)(_net_handle); \
  STAI_ASSERT(_net_ctx != NULL) \
  _STAI_SET_ERROR(_net_ctx, _net_ctx->_magic != STAI_MAGIC, \
                  STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE, _net_ctx->_return_code)


/*****************************************************************************/
static
void _stai_network_check(_stai_network_context* net_ctx)
{
  stai_size idx;

// Check activations status
  for (idx=0; idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    if (net_ctx->_activations[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_ACTIVATIONS_NUM) ? STAI_FLAG_ACTIVATIONS : STAI_FLAG_NONE;
// Check inputs status
  for (idx=0; idx<STAI_NETWORK_IN_NUM; idx++) {
    if (net_ctx->_inputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_IN_NUM) ? STAI_FLAG_INPUTS : STAI_FLAG_NONE;

  // Check outputs status
  for (idx=0; idx<STAI_NETWORK_OUT_NUM; idx++) {
    if (net_ctx->_outputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_OUT_NUM) ? STAI_FLAG_OUTPUTS : STAI_FLAG_NONE;

// Check weights status
  for (idx=0; idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    if (net_ctx->_weights[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_WEIGHTS_NUM) ? STAI_FLAG_WEIGHTS : STAI_FLAG_NONE;
STAI_PRINT("  [_stai_network_check] flags: 0x%08x\n", net_ctx->_flags)
}


/*****************************************************************************/
STAI_API_ENTRY
stai_return_code stai_network_init(
  stai_network* network)
{
  /* Memory where to store internal context is provided by applications as a raw byte buffer */
  _stai_network_context* net_ctx = (_stai_network_context*)(network);
  net_ctx->_return_code = STAI_SUCCESS;
  STAI_PRINT("[Entering Network Init] network(%p) context_size(%d)\n", net_ctx, (int32_t)sizeof(_stai_network_context))

  _STAI_SET_ERROR(net_ctx, STAI_NETWORK_CONTEXT_SIZE != sizeof(_stai_network_context),
                 STAI_ERROR_NETWORK_INVALID_CONTEXT_SIZE, net_ctx->_return_code)

  {
    const _stai_network_context _network_context = {
      ._magic = STAI_MAGIC,
      ._signature = STAI_NETWORK_MODEL_SIGNATURE,
      ._flags = STAI_NETWORK_FLAGS,
      ._return_code = STAI_SUCCESS,
      ._callback = NULL,
      ._callback_cookie = NULL,
      ._activations = {
      (stai_ptr)g_network_activations_1,(stai_ptr)g_network_activations_2
      },
      ._weights = {
      (stai_ptr)g_network_weights_array
      },
      ._inputs = {
    NULL},
      ._outputs = {
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL},
    };

    // Deep copy of internal context to opaque buffer provided by app
    *net_ctx = _network_context;

    _stai_network_check(net_ctx);
  }

  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_deinit(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /*  Reset flags to initial state  */
  net_ctx->_flags = STAI_NETWORK_FLAGS;
  return net_ctx->_return_code;
}

/*****************************************************************************/





/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 307200, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  input_Transpose_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 307200, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  node_215_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 409600, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  node_216_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  node_232_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 409600, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  node_233_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  node_241_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  node_242_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  node_250_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  node_251_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6400, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  node_263_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6400, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  node_268_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  node_269_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  node_273_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  node_278_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  node_279_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 102400, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  input_Transpose_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 320, 320), AI_STRIDE_INIT(4, 4, 4, 12, 3840),
  1, &input_Transpose_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  input_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 320, 320, 3), AI_STRIDE_INIT(4, 4, 4, 1280, 409600),
  1, &input_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  node_215_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 160, 160), AI_STRIDE_INIT(4, 4, 4, 64, 10240),
  1, &node_215_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  node_216_output, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 80, 80), AI_STRIDE_INIT(4, 4, 4, 64, 5120),
  1, &node_216_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  node_232_output, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 80, 80), AI_STRIDE_INIT(4, 4, 4, 256, 20480),
  1, &node_232_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  node_233_output, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &node_233_output_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  node_241_output, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &node_241_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  node_242_output, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &node_242_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  node_250_output, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &node_250_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  node_251_output, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 4, 4, 256, 2560),
  1, &node_251_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  node_263_output, AI_STATIC,
  69, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 4, 4, 256, 2560),
  1, &node_263_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  node_268_output, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &node_268_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  node_269_output, AI_STATIC,
  71, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &node_269_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  node_273_output, AI_STATIC,
  76, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 4, 4, 256, 5120),
  1, &node_273_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  node_278_output, AI_STATIC,
  77, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &node_278_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  node_279_output, AI_STATIC,
  78, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 40, 40), AI_STRIDE_INIT(4, 4, 4, 256, 10240),
  1, &node_279_output_array, NULL)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_Transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_Transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_Transpose_layer, 2,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &input_Transpose_chain,
  NULL, &input_Transpose_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  node_216_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &node_215_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &node_216_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  node_216_layer, 6,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &node_216_chain,
  NULL, &node_216_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  node_233_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &node_232_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &node_233_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  node_233_layer, 19,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &node_233_chain,
  NULL, &node_233_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  node_242_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &node_241_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &node_242_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  node_242_layer, 26,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &node_242_chain,
  NULL, &node_242_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  node_251_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &node_250_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &node_251_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  node_251_layer, 33,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &node_251_chain,
  NULL, &node_251_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_float node_268_scales_data[] = { 2.0, 2.0, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    node_268_scales, AI_ARRAY_FORMAT_FLOAT,
    node_268_scales_data, node_268_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  node_268_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &node_263_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &node_268_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  node_268_layer, 43,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_nearest,
  &node_268_chain,
  NULL, &node_268_layer, AI_STATIC, 
  .scales = &node_268_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_NEAREST, 
  .nearest_mode = AI_ROUND_FLOOR, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  node_269_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &node_250_output, &node_268_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &node_269_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  node_269_layer, 44,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &node_269_chain,
  NULL, &node_269_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)


AI_STATIC_CONST ai_float node_278_scales_data[] = { 2.0, 2.0, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    node_278_scales, AI_ARRAY_FORMAT_FLOAT,
    node_278_scales_data, node_278_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  node_278_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &node_273_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &node_278_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  node_278_layer, 48,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_nearest,
  &node_278_chain,
  NULL, &node_278_layer, AI_STATIC, 
  .scales = &node_278_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_NEAREST, 
  .nearest_mode = AI_ROUND_FLOOR, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  node_279_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &node_241_output, &node_278_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &node_279_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  node_279_layer, 49,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &node_279_chain,
  NULL, &node_279_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite_transpose_input_Transpose(_stai_network_context* net_ctx)
{
  input_output_array.data = AI_PTR(net_ctx->_inputs[0] + 0);
  input_output_array.data_start = AI_PTR(net_ctx->_inputs[0] + 0);
  input_Transpose_output_array.data = AI_PTR(net_ctx->_activations[1] + 1228800);
  input_Transpose_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 1228800);
  _STAI_NETWORK_EVENT_NODE_START_CB(2, 1, { input_output.data->data});
  forward_transpose(&input_Transpose_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(2, 1, { input_Transpose_output.data->data});
}
void forward_lite_mp_node_216(_stai_network_context* net_ctx)
{
  node_215_output_array.data = AI_PTR(net_ctx->_activations[1] + 771520);
  node_215_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 771520);
  node_216_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  node_216_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(6, 1, { node_215_output.data->data});
  forward_mp(&node_216_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(6, 1, { node_216_output.data->data});
}
void forward_lite_mp_node_233(_stai_network_context* net_ctx)
{
  node_232_output_array.data = AI_PTR(net_ctx->_activations[1] + 653056);
  node_232_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 653056);
  node_233_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  node_233_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(19, 1, { node_232_output.data->data});
  forward_mp(&node_233_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(19, 1, { node_233_output.data->data});
}
void forward_lite_mp_node_242(_stai_network_context* net_ctx)
{
  node_241_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  node_241_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  node_242_output_array.data = AI_PTR(net_ctx->_activations[1] + 409600);
  node_242_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 409600);
  _STAI_NETWORK_EVENT_NODE_START_CB(26, 1, { node_241_output.data->data});
  forward_mp(&node_242_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(26, 1, { node_242_output.data->data});
}
void forward_lite_mp_node_251(_stai_network_context* net_ctx)
{
  node_250_output_array.data = AI_PTR(net_ctx->_activations[1] + 409600);
  node_250_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 409600);
  node_251_output_array.data = AI_PTR(net_ctx->_activations[0] + 25708);
  node_251_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 25708);
  _STAI_NETWORK_EVENT_NODE_START_CB(33, 1, { node_250_output.data->data});
  forward_mp(&node_251_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(33, 1, { node_251_output.data->data});
}
void forward_lite_upsample_nearest_node_268(_stai_network_context* net_ctx)
{
  node_263_output_array.data = AI_PTR(net_ctx->_activations[0] + 364);
  node_263_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 364);
  node_268_output_array.data = AI_PTR(net_ctx->_activations[1] + 2355200);
  node_268_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 2355200);
  _STAI_NETWORK_EVENT_NODE_START_CB(43, 1, { node_263_output.data->data});
  forward_upsample_nearest(&node_268_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(43, 1, { node_268_output.data->data});
}
void forward_lite_eltwise_node_269(_stai_network_context* net_ctx)
{
  node_250_output_array.data = AI_PTR(net_ctx->_activations[1] + 409600);
  node_250_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 409600);
  node_268_output_array.data = AI_PTR(net_ctx->_activations[1] + 2355200);
  node_268_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 2355200);
  node_269_output_array.data = AI_PTR(net_ctx->_activations[1] + 2252800);
  node_269_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 2252800);
  _STAI_NETWORK_EVENT_NODE_START_CB(44, 2, { node_250_output.data->data,node_268_output.data->data});
  forward_eltwise(&node_269_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(44, 1, { node_269_output.data->data});
}
void forward_lite_upsample_nearest_node_278(_stai_network_context* net_ctx)
{
  node_273_output_array.data = AI_PTR(net_ctx->_activations[1] + 2247680);
  node_273_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 2247680);
  node_278_output_array.data = AI_PTR(net_ctx->_activations[1] + 1838080);
  node_278_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 1838080);
  _STAI_NETWORK_EVENT_NODE_START_CB(48, 1, { node_273_output.data->data});
  forward_upsample_nearest(&node_278_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(48, 1, { node_278_output.data->data});
}
void forward_lite_eltwise_node_279(_stai_network_context* net_ctx)
{
  node_241_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  node_241_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  node_278_output_array.data = AI_PTR(net_ctx->_activations[1] + 1838080);
  node_278_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 1838080);
  node_279_output_array.data = AI_PTR(net_ctx->_activations[1] + 1428480);
  node_279_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 1428480);
  _STAI_NETWORK_EVENT_NODE_START_CB(49, 2, { node_241_output.data->data,node_278_output.data->data});
  forward_eltwise(&node_279_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(49, 1, { node_279_output.data->data});
}

/*****************************************************************************/



static const ai_u32 node_458_t_in_0_shape_ch_const_u32 = 3;
static const ai_u32 node_458_t_out_0_shape_ch_const_u32 = 16;
static const ai_u32 node_458_t_in_0_shape_w_const_u32 = 320;
static const ai_u32 node_458_t_in_0_shape_h_const_u32 = 320;
static const ai_u32 node_458_t_out_0_shape_w_const_u32 = 160;
static const ai_u32 node_458_t_out_0_shape_h_const_u32 = 160;
static const ai_u32 node_458_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_458_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_458_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_458_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_458_l_stride_1_const_u16 = 2;
static const ai_u16 node_458_l_stride_0_const_u16 = 2;
static const ai_u16 node_458_l_dilation_H_const_u16 = 1;
static const ai_u16 node_458_l_dilation_W_const_u16 = 1;

static const ai_i32 node_211_t_in_0_shape_ch_h_w_prod_const_s32 = 409600;

static const ai_u32 node_212_t_in_0_shape_ch_const_u32 = 16;
static const ai_u32 node_212_t_out_0_shape_ch_const_u32 = 16;
static const ai_u32 node_212_t_in_0_shape_w_const_u32 = 160;
static const ai_u32 node_212_t_in_0_shape_h_const_u32 = 160;
static const ai_u32 node_212_t_out_0_shape_w_const_u32 = 160;
static const ai_u32 node_212_t_out_0_shape_h_const_u32 = 160;
static const ai_u32 node_212_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_212_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_212_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_212_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_212_l_stride_1_const_u16 = 1;
static const ai_u16 node_212_l_stride_0_const_u16 = 1;
static const ai_u16 node_212_l_dilation_H_const_u16 = 1;
static const ai_u16 node_212_l_dilation_W_const_u16 = 1;

static const ai_u32 node_461_t_in_0_shape_ch_const_u32 = 16;
static const ai_u32 node_461_t_out_0_shape_ch_const_u32 = 16;
static const ai_u32 node_461_t_in_0_shape_w_const_u32 = 160;
static const ai_u32 node_461_t_in_0_shape_h_const_u32 = 160;
static const ai_u32 node_461_t_out_0_shape_w_const_u32 = 160;
static const ai_u32 node_461_t_out_0_shape_h_const_u32 = 160;
static const ai_u32 node_461_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_461_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_461_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_461_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_461_l_stride_1_const_u16 = 1;
static const ai_u16 node_461_l_stride_0_const_u16 = 1;
static const ai_u16 node_461_l_dilation_H_const_u16 = 1;
static const ai_u16 node_461_l_dilation_W_const_u16 = 1;

static const ai_i32 node_215_t_in_0_shape_ch_h_w_prod_const_s32 = 409600;


static const ai_u32 node_217_t_in_0_shape_ch_const_u32 = 16;
static const ai_u32 node_217_t_out_0_shape_ch_const_u32 = 16;
static const ai_u32 node_217_t_in_0_shape_w_const_u32 = 80;
static const ai_u32 node_217_t_in_0_shape_h_const_u32 = 80;
static const ai_u32 node_217_t_out_0_shape_w_const_u32 = 80;
static const ai_u32 node_217_t_out_0_shape_h_const_u32 = 80;
static const ai_u32 node_217_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_217_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_217_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_217_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_217_l_stride_1_const_u16 = 1;
static const ai_u16 node_217_l_stride_0_const_u16 = 1;
static const ai_u16 node_217_l_dilation_H_const_u16 = 1;
static const ai_u16 node_217_l_dilation_W_const_u16 = 1;

static const ai_u32 node_464_t_in_0_shape_ch_const_u32 = 16;
static const ai_u32 node_464_t_out_0_shape_ch_const_u32 = 16;
static const ai_u32 node_464_t_in_0_shape_w_const_u32 = 80;
static const ai_u32 node_464_t_in_0_shape_h_const_u32 = 80;
static const ai_u32 node_464_t_out_0_shape_w_const_u32 = 80;
static const ai_u32 node_464_t_out_0_shape_h_const_u32 = 80;
static const ai_u32 node_464_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_464_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_464_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_464_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_464_l_stride_1_const_u16 = 1;
static const ai_u16 node_464_l_stride_0_const_u16 = 1;
static const ai_u16 node_464_l_dilation_H_const_u16 = 1;
static const ai_u16 node_464_l_dilation_W_const_u16 = 1;

static const ai_i32 node_220_t_in_0_shape_ch_h_w_prod_const_s32 = 102400;

static const ai_u32 node_221_t_in_0_shape_ch_const_u32 = 16;
static const ai_u32 node_221_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_221_t_in_0_shape_w_const_u32 = 80;
static const ai_u32 node_221_t_in_0_shape_h_const_u32 = 80;
static const ai_u32 node_221_t_out_0_shape_w_const_u32 = 80;
static const ai_u32 node_221_t_out_0_shape_h_const_u32 = 80;
static const ai_u32 node_221_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_221_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_221_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_221_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_221_l_stride_1_const_u16 = 1;
static const ai_u16 node_221_l_stride_0_const_u16 = 1;
static const ai_u16 node_221_l_dilation_H_const_u16 = 1;
static const ai_u16 node_221_l_dilation_W_const_u16 = 1;

static const ai_u32 node_467_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_467_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_467_t_in_0_shape_w_const_u32 = 80;
static const ai_u32 node_467_t_in_0_shape_h_const_u32 = 80;
static const ai_u32 node_467_t_out_0_shape_w_const_u32 = 80;
static const ai_u32 node_467_t_out_0_shape_h_const_u32 = 80;
static const ai_u32 node_467_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_467_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_467_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_467_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_467_l_stride_1_const_u16 = 1;
static const ai_u16 node_467_l_stride_0_const_u16 = 1;
static const ai_u16 node_467_l_dilation_H_const_u16 = 1;
static const ai_u16 node_467_l_dilation_W_const_u16 = 1;

static const ai_i32 node_224_t_in_0_shape_ch_h_w_prod_const_s32 = 409600;

static const ai_u32 node_225_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_225_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_225_t_in_0_shape_w_const_u32 = 80;
static const ai_u32 node_225_t_in_0_shape_h_const_u32 = 80;
static const ai_u32 node_225_t_out_0_shape_w_const_u32 = 80;
static const ai_u32 node_225_t_out_0_shape_h_const_u32 = 80;
static const ai_u32 node_225_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_225_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_225_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_225_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_225_l_stride_1_const_u16 = 1;
static const ai_u16 node_225_l_stride_0_const_u16 = 1;
static const ai_u16 node_225_l_dilation_H_const_u16 = 1;
static const ai_u16 node_225_l_dilation_W_const_u16 = 1;

static const ai_u32 node_470_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_470_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_470_t_in_0_shape_w_const_u32 = 80;
static const ai_u32 node_470_t_in_0_shape_h_const_u32 = 80;
static const ai_u32 node_470_t_out_0_shape_w_const_u32 = 80;
static const ai_u32 node_470_t_out_0_shape_h_const_u32 = 80;
static const ai_u32 node_470_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_470_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_470_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_470_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_470_l_stride_1_const_u16 = 1;
static const ai_u16 node_470_l_stride_0_const_u16 = 1;
static const ai_u16 node_470_l_dilation_H_const_u16 = 1;
static const ai_u16 node_470_l_dilation_W_const_u16 = 1;

static const ai_i32 node_228_t_in_0_shape_ch_h_w_prod_const_s32 = 409600;

static const ai_u32 node_229_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_229_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_229_t_in_0_shape_w_const_u32 = 80;
static const ai_u32 node_229_t_in_0_shape_h_const_u32 = 80;
static const ai_u32 node_229_t_out_0_shape_w_const_u32 = 80;
static const ai_u32 node_229_t_out_0_shape_h_const_u32 = 80;
static const ai_u32 node_229_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_229_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_229_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_229_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_229_l_stride_1_const_u16 = 1;
static const ai_u16 node_229_l_stride_0_const_u16 = 1;
static const ai_u16 node_229_l_dilation_H_const_u16 = 1;
static const ai_u16 node_229_l_dilation_W_const_u16 = 1;

static const ai_u32 node_473_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_473_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_473_t_in_0_shape_w_const_u32 = 80;
static const ai_u32 node_473_t_in_0_shape_h_const_u32 = 80;
static const ai_u32 node_473_t_out_0_shape_w_const_u32 = 80;
static const ai_u32 node_473_t_out_0_shape_h_const_u32 = 80;
static const ai_u32 node_473_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_473_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_473_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_473_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_473_l_stride_1_const_u16 = 1;
static const ai_u16 node_473_l_stride_0_const_u16 = 1;
static const ai_u16 node_473_l_dilation_H_const_u16 = 1;
static const ai_u16 node_473_l_dilation_W_const_u16 = 1;

static const ai_i32 node_232_t_in_0_shape_ch_h_w_prod_const_s32 = 409600;


static const ai_u32 node_234_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_234_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_234_t_in_0_shape_w_const_u32 = 40;
static const ai_u32 node_234_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 node_234_t_out_0_shape_w_const_u32 = 40;
static const ai_u32 node_234_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 node_234_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_234_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_234_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_234_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_234_l_stride_1_const_u16 = 1;
static const ai_u16 node_234_l_stride_0_const_u16 = 1;
static const ai_u16 node_234_l_dilation_H_const_u16 = 1;
static const ai_u16 node_234_l_dilation_W_const_u16 = 1;

static const ai_u32 node_476_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_476_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_476_t_in_0_shape_w_const_u32 = 40;
static const ai_u32 node_476_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 node_476_t_out_0_shape_w_const_u32 = 40;
static const ai_u32 node_476_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 node_476_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_476_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_476_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_476_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_476_l_stride_1_const_u16 = 1;
static const ai_u16 node_476_l_stride_0_const_u16 = 1;
static const ai_u16 node_476_l_dilation_H_const_u16 = 1;
static const ai_u16 node_476_l_dilation_W_const_u16 = 1;

static const ai_i32 node_237_t_in_0_shape_ch_h_w_prod_const_s32 = 102400;

static const ai_u32 node_238_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_238_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_238_t_in_0_shape_w_const_u32 = 40;
static const ai_u32 node_238_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 node_238_t_out_0_shape_w_const_u32 = 40;
static const ai_u32 node_238_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 node_238_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_238_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_238_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_238_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_238_l_stride_1_const_u16 = 1;
static const ai_u16 node_238_l_stride_0_const_u16 = 1;
static const ai_u16 node_238_l_dilation_H_const_u16 = 1;
static const ai_u16 node_238_l_dilation_W_const_u16 = 1;

static const ai_u32 node_479_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_479_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_479_t_in_0_shape_w_const_u32 = 40;
static const ai_u32 node_479_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 node_479_t_out_0_shape_w_const_u32 = 40;
static const ai_u32 node_479_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 node_479_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_479_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_479_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_479_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_479_l_stride_1_const_u16 = 1;
static const ai_u16 node_479_l_stride_0_const_u16 = 1;
static const ai_u16 node_479_l_dilation_H_const_u16 = 1;
static const ai_u16 node_479_l_dilation_W_const_u16 = 1;

static const ai_i32 node_241_t_in_0_shape_ch_h_w_prod_const_s32 = 102400;


static const ai_u32 node_243_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_243_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_243_t_in_0_shape_w_const_u32 = 20;
static const ai_u32 node_243_t_in_0_shape_h_const_u32 = 20;
static const ai_u32 node_243_t_out_0_shape_w_const_u32 = 20;
static const ai_u32 node_243_t_out_0_shape_h_const_u32 = 20;
static const ai_u32 node_243_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_243_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_243_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_243_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_243_l_stride_1_const_u16 = 1;
static const ai_u16 node_243_l_stride_0_const_u16 = 1;
static const ai_u16 node_243_l_dilation_H_const_u16 = 1;
static const ai_u16 node_243_l_dilation_W_const_u16 = 1;

static const ai_u32 node_482_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_482_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_482_t_in_0_shape_w_const_u32 = 20;
static const ai_u32 node_482_t_in_0_shape_h_const_u32 = 20;
static const ai_u32 node_482_t_out_0_shape_w_const_u32 = 20;
static const ai_u32 node_482_t_out_0_shape_h_const_u32 = 20;
static const ai_u32 node_482_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_482_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_482_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_482_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_482_l_stride_1_const_u16 = 1;
static const ai_u16 node_482_l_stride_0_const_u16 = 1;
static const ai_u16 node_482_l_dilation_H_const_u16 = 1;
static const ai_u16 node_482_l_dilation_W_const_u16 = 1;

static const ai_i32 node_246_t_in_0_shape_ch_h_w_prod_const_s32 = 25600;

static const ai_u32 node_247_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_247_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_247_t_in_0_shape_w_const_u32 = 20;
static const ai_u32 node_247_t_in_0_shape_h_const_u32 = 20;
static const ai_u32 node_247_t_out_0_shape_w_const_u32 = 20;
static const ai_u32 node_247_t_out_0_shape_h_const_u32 = 20;
static const ai_u32 node_247_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_247_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_247_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_247_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_247_l_stride_1_const_u16 = 1;
static const ai_u16 node_247_l_stride_0_const_u16 = 1;
static const ai_u16 node_247_l_dilation_H_const_u16 = 1;
static const ai_u16 node_247_l_dilation_W_const_u16 = 1;

static const ai_u32 node_485_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_485_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_485_t_in_0_shape_w_const_u32 = 20;
static const ai_u32 node_485_t_in_0_shape_h_const_u32 = 20;
static const ai_u32 node_485_t_out_0_shape_w_const_u32 = 20;
static const ai_u32 node_485_t_out_0_shape_h_const_u32 = 20;
static const ai_u32 node_485_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_485_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_485_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_485_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_485_l_stride_1_const_u16 = 1;
static const ai_u16 node_485_l_stride_0_const_u16 = 1;
static const ai_u16 node_485_l_dilation_H_const_u16 = 1;
static const ai_u16 node_485_l_dilation_W_const_u16 = 1;

static const ai_i32 node_250_t_in_0_shape_ch_h_w_prod_const_s32 = 25600;


static const ai_u32 node_252_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_252_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_252_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 node_252_t_in_0_shape_h_const_u32 = 10;
static const ai_u32 node_252_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 node_252_t_out_0_shape_h_const_u32 = 10;
static const ai_u32 node_252_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_252_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_252_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_252_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_252_l_stride_1_const_u16 = 1;
static const ai_u16 node_252_l_stride_0_const_u16 = 1;
static const ai_u16 node_252_l_dilation_H_const_u16 = 1;
static const ai_u16 node_252_l_dilation_W_const_u16 = 1;

static const ai_u32 node_488_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_488_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_488_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 node_488_t_in_0_shape_h_const_u32 = 10;
static const ai_u32 node_488_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 node_488_t_out_0_shape_h_const_u32 = 10;
static const ai_u32 node_488_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_488_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_488_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_488_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_488_l_stride_1_const_u16 = 1;
static const ai_u16 node_488_l_stride_0_const_u16 = 1;
static const ai_u16 node_488_l_dilation_H_const_u16 = 1;
static const ai_u16 node_488_l_dilation_W_const_u16 = 1;

static const ai_i32 node_255_t_in_0_shape_ch_h_w_prod_const_s32 = 6400;

static const ai_u32 node_256_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_256_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_256_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 node_256_t_in_0_shape_h_const_u32 = 10;
static const ai_u32 node_256_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 node_256_t_out_0_shape_h_const_u32 = 10;
static const ai_u32 node_256_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_256_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_256_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_256_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_256_l_stride_1_const_u16 = 1;
static const ai_u16 node_256_l_stride_0_const_u16 = 1;
static const ai_u16 node_256_l_dilation_H_const_u16 = 1;
static const ai_u16 node_256_l_dilation_W_const_u16 = 1;

static const ai_u32 node_491_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_491_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_491_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 node_491_t_in_0_shape_h_const_u32 = 10;
static const ai_u32 node_491_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 node_491_t_out_0_shape_h_const_u32 = 10;
static const ai_u32 node_491_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_491_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_491_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_491_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_491_l_stride_1_const_u16 = 1;
static const ai_u16 node_491_l_stride_0_const_u16 = 1;
static const ai_u16 node_491_l_dilation_H_const_u16 = 1;
static const ai_u16 node_491_l_dilation_W_const_u16 = 1;

static const ai_i32 node_259_t_in_0_shape_ch_h_w_prod_const_s32 = 6400;

static const ai_u32 node_260_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_260_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_260_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 node_260_t_in_0_shape_h_const_u32 = 10;
static const ai_u32 node_260_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 node_260_t_out_0_shape_h_const_u32 = 10;
static const ai_u32 node_260_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_260_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_260_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_260_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_260_l_stride_1_const_u16 = 1;
static const ai_u16 node_260_l_stride_0_const_u16 = 1;
static const ai_u16 node_260_l_dilation_H_const_u16 = 1;
static const ai_u16 node_260_l_dilation_W_const_u16 = 1;

static const ai_u32 node_494_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_494_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_494_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 node_494_t_in_0_shape_h_const_u32 = 10;
static const ai_u32 node_494_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 node_494_t_out_0_shape_h_const_u32 = 10;
static const ai_u32 node_494_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_494_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_494_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_494_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_494_l_stride_1_const_u16 = 1;
static const ai_u16 node_494_l_stride_0_const_u16 = 1;
static const ai_u16 node_494_l_dilation_H_const_u16 = 1;
static const ai_u16 node_494_l_dilation_W_const_u16 = 1;

static const ai_i32 node_263_t_in_0_shape_ch_h_w_prod_const_s32 = 6400;

static const ai_u32 node_292_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_292_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_292_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 node_292_t_in_0_shape_h_const_u32 = 10;
static const ai_u32 node_292_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 node_292_t_out_0_shape_h_const_u32 = 10;
static const ai_u32 node_292_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_292_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_292_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_292_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_292_l_stride_1_const_u16 = 1;
static const ai_u16 node_292_l_stride_0_const_u16 = 1;
static const ai_u16 node_292_l_dilation_H_const_u16 = 1;
static const ai_u16 node_292_l_dilation_W_const_u16 = 1;

static const ai_u32 node_509_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_509_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_509_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 node_509_t_in_0_shape_h_const_u32 = 10;
static const ai_u32 node_509_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 node_509_t_out_0_shape_h_const_u32 = 10;
static const ai_u32 node_509_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_509_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_509_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_509_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_509_l_stride_1_const_u16 = 1;
static const ai_u16 node_509_l_stride_0_const_u16 = 1;
static const ai_u16 node_509_l_dilation_H_const_u16 = 1;
static const ai_u16 node_509_l_dilation_W_const_u16 = 1;

static const ai_i32 node_295_t_in_0_shape_ch_h_w_prod_const_s32 = 6400;

static const ai_u32 node_318_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_318_t_out_0_shape_ch_const_u32 = 10;
static const ai_u32 node_318_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 node_318_t_in_0_shape_h_const_u32 = 10;
static const ai_u32 node_318_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 node_318_t_out_0_shape_h_const_u32 = 10;
static const ai_u32 node_318_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_318_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_318_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_318_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_318_l_stride_1_const_u16 = 1;
static const ai_u16 node_318_l_stride_0_const_u16 = 1;
static const ai_u16 node_318_l_dilation_H_const_u16 = 1;
static const ai_u16 node_318_l_dilation_W_const_u16 = 1;

static const ai_u32 node_319_t_in_0_shape_ch_const_u32 = 10;
static const ai_u32 node_319_t_out_0_shape_ch_const_u32 = 10;
static const ai_u32 node_319_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 node_319_t_in_0_shape_h_const_u32 = 10;
static const ai_u32 node_319_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 node_319_t_out_0_shape_h_const_u32 = 10;
static const ai_u32 node_319_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_319_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_319_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_319_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_319_l_stride_1_const_u16 = 1;
static const ai_u16 node_319_l_stride_0_const_u16 = 1;
static const ai_u16 node_319_l_dilation_H_const_u16 = 1;
static const ai_u16 node_319_l_dilation_W_const_u16 = 1;

static const ai_u32 node_312_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_312_t_out_0_shape_ch_const_u32 = 1;
static const ai_u32 node_312_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 node_312_t_in_0_shape_h_const_u32 = 10;
static const ai_u32 node_312_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 node_312_t_out_0_shape_h_const_u32 = 10;
static const ai_u32 node_312_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_312_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_312_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_312_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_312_l_stride_1_const_u16 = 1;
static const ai_u16 node_312_l_stride_0_const_u16 = 1;
static const ai_u16 node_312_l_dilation_H_const_u16 = 1;
static const ai_u16 node_312_l_dilation_W_const_u16 = 1;

static const ai_u32 node_313_t_in_0_shape_ch_const_u32 = 1;
static const ai_u32 node_313_t_out_0_shape_ch_const_u32 = 1;
static const ai_u32 node_313_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 node_313_t_in_0_shape_h_const_u32 = 10;
static const ai_u32 node_313_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 node_313_t_out_0_shape_h_const_u32 = 10;
static const ai_u32 node_313_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_313_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_313_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_313_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_313_l_stride_1_const_u16 = 1;
static const ai_u16 node_313_l_stride_0_const_u16 = 1;
static const ai_u16 node_313_l_dilation_H_const_u16 = 1;
static const ai_u16 node_313_l_dilation_W_const_u16 = 1;

static const ai_i32 obj_32_t_in_0_shape_ch_h_prod_const_s32 = 100;

static const ai_u32 node_306_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_306_t_out_0_shape_ch_const_u32 = 4;
static const ai_u32 node_306_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 node_306_t_in_0_shape_h_const_u32 = 10;
static const ai_u32 node_306_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 node_306_t_out_0_shape_h_const_u32 = 10;
static const ai_u32 node_306_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_306_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_306_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_306_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_306_l_stride_1_const_u16 = 1;
static const ai_u16 node_306_l_stride_0_const_u16 = 1;
static const ai_u16 node_306_l_dilation_H_const_u16 = 1;
static const ai_u16 node_306_l_dilation_W_const_u16 = 1;

static const ai_u32 node_307_t_in_0_shape_ch_const_u32 = 4;
static const ai_u32 node_307_t_out_0_shape_ch_const_u32 = 4;
static const ai_u32 node_307_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 node_307_t_in_0_shape_h_const_u32 = 10;
static const ai_u32 node_307_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 node_307_t_out_0_shape_h_const_u32 = 10;
static const ai_u32 node_307_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_307_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_307_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_307_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_307_l_stride_1_const_u16 = 1;
static const ai_u16 node_307_l_stride_0_const_u16 = 1;
static const ai_u16 node_307_l_dilation_H_const_u16 = 1;
static const ai_u16 node_307_l_dilation_W_const_u16 = 1;

static const ai_u32 node_300_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_300_t_out_0_shape_ch_const_u32 = 1;
static const ai_u32 node_300_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 node_300_t_in_0_shape_h_const_u32 = 10;
static const ai_u32 node_300_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 node_300_t_out_0_shape_h_const_u32 = 10;
static const ai_u32 node_300_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_300_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_300_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_300_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_300_l_stride_1_const_u16 = 1;
static const ai_u16 node_300_l_stride_0_const_u16 = 1;
static const ai_u16 node_300_l_dilation_H_const_u16 = 1;
static const ai_u16 node_300_l_dilation_W_const_u16 = 1;

static const ai_u32 node_301_t_in_0_shape_ch_const_u32 = 1;
static const ai_u32 node_301_t_out_0_shape_ch_const_u32 = 1;
static const ai_u32 node_301_t_in_0_shape_w_const_u32 = 10;
static const ai_u32 node_301_t_in_0_shape_h_const_u32 = 10;
static const ai_u32 node_301_t_out_0_shape_w_const_u32 = 10;
static const ai_u32 node_301_t_out_0_shape_h_const_u32 = 10;
static const ai_u32 node_301_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_301_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_301_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_301_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_301_l_stride_1_const_u16 = 1;
static const ai_u16 node_301_l_stride_0_const_u16 = 1;
static const ai_u16 node_301_l_dilation_H_const_u16 = 1;
static const ai_u16 node_301_l_dilation_W_const_u16 = 1;

static const ai_i32 cls_32_t_in_0_shape_ch_h_prod_const_s32 = 100;



static const ai_u32 node_270_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_270_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_270_t_in_0_shape_w_const_u32 = 20;
static const ai_u32 node_270_t_in_0_shape_h_const_u32 = 20;
static const ai_u32 node_270_t_out_0_shape_w_const_u32 = 20;
static const ai_u32 node_270_t_out_0_shape_h_const_u32 = 20;
static const ai_u32 node_270_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_270_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_270_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_270_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_270_l_stride_1_const_u16 = 1;
static const ai_u16 node_270_l_stride_0_const_u16 = 1;
static const ai_u16 node_270_l_dilation_H_const_u16 = 1;
static const ai_u16 node_270_l_dilation_W_const_u16 = 1;

static const ai_u32 node_497_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_497_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_497_t_in_0_shape_w_const_u32 = 20;
static const ai_u32 node_497_t_in_0_shape_h_const_u32 = 20;
static const ai_u32 node_497_t_out_0_shape_w_const_u32 = 20;
static const ai_u32 node_497_t_out_0_shape_h_const_u32 = 20;
static const ai_u32 node_497_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_497_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_497_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_497_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_497_l_stride_1_const_u16 = 1;
static const ai_u16 node_497_l_stride_0_const_u16 = 1;
static const ai_u16 node_497_l_dilation_H_const_u16 = 1;
static const ai_u16 node_497_l_dilation_W_const_u16 = 1;

static const ai_i32 node_273_t_in_0_shape_ch_h_w_prod_const_s32 = 25600;

static const ai_u32 node_288_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_288_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_288_t_in_0_shape_w_const_u32 = 20;
static const ai_u32 node_288_t_in_0_shape_h_const_u32 = 20;
static const ai_u32 node_288_t_out_0_shape_w_const_u32 = 20;
static const ai_u32 node_288_t_out_0_shape_h_const_u32 = 20;
static const ai_u32 node_288_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_288_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_288_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_288_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_288_l_stride_1_const_u16 = 1;
static const ai_u16 node_288_l_stride_0_const_u16 = 1;
static const ai_u16 node_288_l_dilation_H_const_u16 = 1;
static const ai_u16 node_288_l_dilation_W_const_u16 = 1;

static const ai_u32 node_506_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_506_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_506_t_in_0_shape_w_const_u32 = 20;
static const ai_u32 node_506_t_in_0_shape_h_const_u32 = 20;
static const ai_u32 node_506_t_out_0_shape_w_const_u32 = 20;
static const ai_u32 node_506_t_out_0_shape_h_const_u32 = 20;
static const ai_u32 node_506_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_506_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_506_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_506_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_506_l_stride_1_const_u16 = 1;
static const ai_u16 node_506_l_stride_0_const_u16 = 1;
static const ai_u16 node_506_l_dilation_H_const_u16 = 1;
static const ai_u16 node_506_l_dilation_W_const_u16 = 1;

static const ai_i32 node_291_t_in_0_shape_ch_h_w_prod_const_s32 = 25600;

static const ai_u32 node_316_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_316_t_out_0_shape_ch_const_u32 = 10;
static const ai_u32 node_316_t_in_0_shape_w_const_u32 = 20;
static const ai_u32 node_316_t_in_0_shape_h_const_u32 = 20;
static const ai_u32 node_316_t_out_0_shape_w_const_u32 = 20;
static const ai_u32 node_316_t_out_0_shape_h_const_u32 = 20;
static const ai_u32 node_316_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_316_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_316_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_316_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_316_l_stride_1_const_u16 = 1;
static const ai_u16 node_316_l_stride_0_const_u16 = 1;
static const ai_u16 node_316_l_dilation_H_const_u16 = 1;
static const ai_u16 node_316_l_dilation_W_const_u16 = 1;

static const ai_u32 node_317_t_in_0_shape_ch_const_u32 = 10;
static const ai_u32 node_317_t_out_0_shape_ch_const_u32 = 10;
static const ai_u32 node_317_t_in_0_shape_w_const_u32 = 20;
static const ai_u32 node_317_t_in_0_shape_h_const_u32 = 20;
static const ai_u32 node_317_t_out_0_shape_w_const_u32 = 20;
static const ai_u32 node_317_t_out_0_shape_h_const_u32 = 20;
static const ai_u32 node_317_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_317_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_317_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_317_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_317_l_stride_1_const_u16 = 1;
static const ai_u16 node_317_l_stride_0_const_u16 = 1;
static const ai_u16 node_317_l_dilation_H_const_u16 = 1;
static const ai_u16 node_317_l_dilation_W_const_u16 = 1;

static const ai_u32 node_310_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_310_t_out_0_shape_ch_const_u32 = 1;
static const ai_u32 node_310_t_in_0_shape_w_const_u32 = 20;
static const ai_u32 node_310_t_in_0_shape_h_const_u32 = 20;
static const ai_u32 node_310_t_out_0_shape_w_const_u32 = 20;
static const ai_u32 node_310_t_out_0_shape_h_const_u32 = 20;
static const ai_u32 node_310_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_310_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_310_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_310_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_310_l_stride_1_const_u16 = 1;
static const ai_u16 node_310_l_stride_0_const_u16 = 1;
static const ai_u16 node_310_l_dilation_H_const_u16 = 1;
static const ai_u16 node_310_l_dilation_W_const_u16 = 1;

static const ai_u32 node_311_t_in_0_shape_ch_const_u32 = 1;
static const ai_u32 node_311_t_out_0_shape_ch_const_u32 = 1;
static const ai_u32 node_311_t_in_0_shape_w_const_u32 = 20;
static const ai_u32 node_311_t_in_0_shape_h_const_u32 = 20;
static const ai_u32 node_311_t_out_0_shape_w_const_u32 = 20;
static const ai_u32 node_311_t_out_0_shape_h_const_u32 = 20;
static const ai_u32 node_311_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_311_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_311_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_311_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_311_l_stride_1_const_u16 = 1;
static const ai_u16 node_311_l_stride_0_const_u16 = 1;
static const ai_u16 node_311_l_dilation_H_const_u16 = 1;
static const ai_u16 node_311_l_dilation_W_const_u16 = 1;

static const ai_i32 obj_16_t_in_0_shape_ch_h_prod_const_s32 = 400;

static const ai_u32 node_304_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_304_t_out_0_shape_ch_const_u32 = 4;
static const ai_u32 node_304_t_in_0_shape_w_const_u32 = 20;
static const ai_u32 node_304_t_in_0_shape_h_const_u32 = 20;
static const ai_u32 node_304_t_out_0_shape_w_const_u32 = 20;
static const ai_u32 node_304_t_out_0_shape_h_const_u32 = 20;
static const ai_u32 node_304_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_304_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_304_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_304_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_304_l_stride_1_const_u16 = 1;
static const ai_u16 node_304_l_stride_0_const_u16 = 1;
static const ai_u16 node_304_l_dilation_H_const_u16 = 1;
static const ai_u16 node_304_l_dilation_W_const_u16 = 1;

static const ai_u32 node_305_t_in_0_shape_ch_const_u32 = 4;
static const ai_u32 node_305_t_out_0_shape_ch_const_u32 = 4;
static const ai_u32 node_305_t_in_0_shape_w_const_u32 = 20;
static const ai_u32 node_305_t_in_0_shape_h_const_u32 = 20;
static const ai_u32 node_305_t_out_0_shape_w_const_u32 = 20;
static const ai_u32 node_305_t_out_0_shape_h_const_u32 = 20;
static const ai_u32 node_305_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_305_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_305_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_305_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_305_l_stride_1_const_u16 = 1;
static const ai_u16 node_305_l_stride_0_const_u16 = 1;
static const ai_u16 node_305_l_dilation_H_const_u16 = 1;
static const ai_u16 node_305_l_dilation_W_const_u16 = 1;

static const ai_u32 node_298_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_298_t_out_0_shape_ch_const_u32 = 1;
static const ai_u32 node_298_t_in_0_shape_w_const_u32 = 20;
static const ai_u32 node_298_t_in_0_shape_h_const_u32 = 20;
static const ai_u32 node_298_t_out_0_shape_w_const_u32 = 20;
static const ai_u32 node_298_t_out_0_shape_h_const_u32 = 20;
static const ai_u32 node_298_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_298_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_298_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_298_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_298_l_stride_1_const_u16 = 1;
static const ai_u16 node_298_l_stride_0_const_u16 = 1;
static const ai_u16 node_298_l_dilation_H_const_u16 = 1;
static const ai_u16 node_298_l_dilation_W_const_u16 = 1;

static const ai_u32 node_299_t_in_0_shape_ch_const_u32 = 1;
static const ai_u32 node_299_t_out_0_shape_ch_const_u32 = 1;
static const ai_u32 node_299_t_in_0_shape_w_const_u32 = 20;
static const ai_u32 node_299_t_in_0_shape_h_const_u32 = 20;
static const ai_u32 node_299_t_out_0_shape_w_const_u32 = 20;
static const ai_u32 node_299_t_out_0_shape_h_const_u32 = 20;
static const ai_u32 node_299_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_299_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_299_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_299_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_299_l_stride_1_const_u16 = 1;
static const ai_u16 node_299_l_stride_0_const_u16 = 1;
static const ai_u16 node_299_l_dilation_H_const_u16 = 1;
static const ai_u16 node_299_l_dilation_W_const_u16 = 1;

static const ai_i32 cls_16_t_in_0_shape_ch_h_prod_const_s32 = 400;



static const ai_u32 node_280_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_280_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_280_t_in_0_shape_w_const_u32 = 40;
static const ai_u32 node_280_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 node_280_t_out_0_shape_w_const_u32 = 40;
static const ai_u32 node_280_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 node_280_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_280_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_280_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_280_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_280_l_stride_1_const_u16 = 1;
static const ai_u16 node_280_l_stride_0_const_u16 = 1;
static const ai_u16 node_280_l_dilation_H_const_u16 = 1;
static const ai_u16 node_280_l_dilation_W_const_u16 = 1;

static const ai_u32 node_500_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_500_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_500_t_in_0_shape_w_const_u32 = 40;
static const ai_u32 node_500_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 node_500_t_out_0_shape_w_const_u32 = 40;
static const ai_u32 node_500_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 node_500_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_500_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_500_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_500_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_500_l_stride_1_const_u16 = 1;
static const ai_u16 node_500_l_stride_0_const_u16 = 1;
static const ai_u16 node_500_l_dilation_H_const_u16 = 1;
static const ai_u16 node_500_l_dilation_W_const_u16 = 1;

static const ai_i32 node_283_t_in_0_shape_ch_h_w_prod_const_s32 = 102400;

static const ai_u32 node_284_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_284_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_284_t_in_0_shape_w_const_u32 = 40;
static const ai_u32 node_284_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 node_284_t_out_0_shape_w_const_u32 = 40;
static const ai_u32 node_284_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 node_284_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_284_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_284_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_284_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_284_l_stride_1_const_u16 = 1;
static const ai_u16 node_284_l_stride_0_const_u16 = 1;
static const ai_u16 node_284_l_dilation_H_const_u16 = 1;
static const ai_u16 node_284_l_dilation_W_const_u16 = 1;

static const ai_u32 node_503_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_503_t_out_0_shape_ch_const_u32 = 64;
static const ai_u32 node_503_t_in_0_shape_w_const_u32 = 40;
static const ai_u32 node_503_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 node_503_t_out_0_shape_w_const_u32 = 40;
static const ai_u32 node_503_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 node_503_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_503_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_503_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_503_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_503_l_stride_1_const_u16 = 1;
static const ai_u16 node_503_l_stride_0_const_u16 = 1;
static const ai_u16 node_503_l_dilation_H_const_u16 = 1;
static const ai_u16 node_503_l_dilation_W_const_u16 = 1;

static const ai_i32 node_287_t_in_0_shape_ch_h_w_prod_const_s32 = 102400;

static const ai_u32 node_314_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_314_t_out_0_shape_ch_const_u32 = 10;
static const ai_u32 node_314_t_in_0_shape_w_const_u32 = 40;
static const ai_u32 node_314_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 node_314_t_out_0_shape_w_const_u32 = 40;
static const ai_u32 node_314_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 node_314_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_314_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_314_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_314_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_314_l_stride_1_const_u16 = 1;
static const ai_u16 node_314_l_stride_0_const_u16 = 1;
static const ai_u16 node_314_l_dilation_H_const_u16 = 1;
static const ai_u16 node_314_l_dilation_W_const_u16 = 1;

static const ai_u32 node_315_t_in_0_shape_ch_const_u32 = 10;
static const ai_u32 node_315_t_out_0_shape_ch_const_u32 = 10;
static const ai_u32 node_315_t_in_0_shape_w_const_u32 = 40;
static const ai_u32 node_315_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 node_315_t_out_0_shape_w_const_u32 = 40;
static const ai_u32 node_315_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 node_315_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_315_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_315_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_315_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_315_l_stride_1_const_u16 = 1;
static const ai_u16 node_315_l_stride_0_const_u16 = 1;
static const ai_u16 node_315_l_dilation_H_const_u16 = 1;
static const ai_u16 node_315_l_dilation_W_const_u16 = 1;

static const ai_u32 node_308_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_308_t_out_0_shape_ch_const_u32 = 1;
static const ai_u32 node_308_t_in_0_shape_w_const_u32 = 40;
static const ai_u32 node_308_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 node_308_t_out_0_shape_w_const_u32 = 40;
static const ai_u32 node_308_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 node_308_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_308_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_308_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_308_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_308_l_stride_1_const_u16 = 1;
static const ai_u16 node_308_l_stride_0_const_u16 = 1;
static const ai_u16 node_308_l_dilation_H_const_u16 = 1;
static const ai_u16 node_308_l_dilation_W_const_u16 = 1;

static const ai_u32 node_309_t_in_0_shape_ch_const_u32 = 1;
static const ai_u32 node_309_t_out_0_shape_ch_const_u32 = 1;
static const ai_u32 node_309_t_in_0_shape_w_const_u32 = 40;
static const ai_u32 node_309_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 node_309_t_out_0_shape_w_const_u32 = 40;
static const ai_u32 node_309_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 node_309_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_309_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_309_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_309_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_309_l_stride_1_const_u16 = 1;
static const ai_u16 node_309_l_stride_0_const_u16 = 1;
static const ai_u16 node_309_l_dilation_H_const_u16 = 1;
static const ai_u16 node_309_l_dilation_W_const_u16 = 1;

static const ai_i32 obj_8_t_in_0_shape_ch_h_prod_const_s32 = 1600;

static const ai_u32 node_302_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_302_t_out_0_shape_ch_const_u32 = 4;
static const ai_u32 node_302_t_in_0_shape_w_const_u32 = 40;
static const ai_u32 node_302_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 node_302_t_out_0_shape_w_const_u32 = 40;
static const ai_u32 node_302_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 node_302_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_302_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_302_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_302_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_302_l_stride_1_const_u16 = 1;
static const ai_u16 node_302_l_stride_0_const_u16 = 1;
static const ai_u16 node_302_l_dilation_H_const_u16 = 1;
static const ai_u16 node_302_l_dilation_W_const_u16 = 1;

static const ai_u32 node_303_t_in_0_shape_ch_const_u32 = 4;
static const ai_u32 node_303_t_out_0_shape_ch_const_u32 = 4;
static const ai_u32 node_303_t_in_0_shape_w_const_u32 = 40;
static const ai_u32 node_303_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 node_303_t_out_0_shape_w_const_u32 = 40;
static const ai_u32 node_303_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 node_303_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_303_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_303_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_303_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_303_l_stride_1_const_u16 = 1;
static const ai_u16 node_303_l_stride_0_const_u16 = 1;
static const ai_u16 node_303_l_dilation_H_const_u16 = 1;
static const ai_u16 node_303_l_dilation_W_const_u16 = 1;

static const ai_u32 node_296_t_in_0_shape_ch_const_u32 = 64;
static const ai_u32 node_296_t_out_0_shape_ch_const_u32 = 1;
static const ai_u32 node_296_t_in_0_shape_w_const_u32 = 40;
static const ai_u32 node_296_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 node_296_t_out_0_shape_w_const_u32 = 40;
static const ai_u32 node_296_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 node_296_t_weight_0_shape_w_const_u32 = 1;
static const ai_u32 node_296_t_weight_0_shape_h_const_u32 = 1;
static const ai_i32 node_296_l_pad_W_0_const_s32 = 0;
static const ai_i32 node_296_l_pad_H_0_const_s32 = 0;
static const ai_u16 node_296_l_stride_1_const_u16 = 1;
static const ai_u16 node_296_l_stride_0_const_u16 = 1;
static const ai_u16 node_296_l_dilation_H_const_u16 = 1;
static const ai_u16 node_296_l_dilation_W_const_u16 = 1;

static const ai_u32 node_297_t_in_0_shape_ch_const_u32 = 1;
static const ai_u32 node_297_t_out_0_shape_ch_const_u32 = 1;
static const ai_u32 node_297_t_in_0_shape_w_const_u32 = 40;
static const ai_u32 node_297_t_in_0_shape_h_const_u32 = 40;
static const ai_u32 node_297_t_out_0_shape_w_const_u32 = 40;
static const ai_u32 node_297_t_out_0_shape_h_const_u32 = 40;
static const ai_u32 node_297_t_weight_0_shape_w_const_u32 = 3;
static const ai_u32 node_297_t_weight_0_shape_h_const_u32 = 3;
static const ai_i32 node_297_l_pad_W_0_const_s32 = 1;
static const ai_i32 node_297_l_pad_H_0_const_s32 = 1;
static const ai_u16 node_297_l_stride_1_const_u16 = 1;
static const ai_u16 node_297_l_stride_0_const_u16 = 1;
static const ai_u16 node_297_l_dilation_H_const_u16 = 1;
static const ai_u16 node_297_l_dilation_W_const_u16 = 1;

static const ai_i32 cls_8_t_in_0_shape_ch_h_prod_const_s32 = 1600;
STAI_API_ENTRY
stai_return_code stai_network_run(
  stai_network* network,
  const stai_run_mode mode)
{
   STAI_UNUSED(mode)
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_ACTIVATIONS) != STAI_FLAG_ACTIVATIONS,
        STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_INPUTS) != STAI_FLAG_INPUTS,
                  STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_OUTPUTS) != STAI_FLAG_OUTPUTS,
                  STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_WEIGHTS) != STAI_FLAG_WEIGHTS,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)


  /* LITE_KERNEL_SECTION BEGIN input_Transpose */
  {
    
  forward_lite_transpose_input_Transpose(net_ctx);
  }
  /* LITE_KERNEL_SECTION END input_Transpose */
  /* LITE_KERNEL_SECTION BEGIN node_458 */
  {
      const ai_float* node_458_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 1228800);
    ai_float* node_458_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 802432);
    const ai_u8* node_458_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 0);
    const ai_u8* node_458_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 1728);
    ai_float* node_458_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 256);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(1, 1, {(stai_ptr) node_458_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_458_t_in_0_ptr_const_f32, node_458_t_out_0_ptr_f32, node_458_t_weight_0_ptr_const_u8, node_458_t_weight_1_ptr_const_u8, node_458_t_scratch_0_ptr_f32, node_458_t_in_0_shape_ch_const_u32, node_458_t_out_0_shape_ch_const_u32, node_458_t_in_0_shape_w_const_u32, node_458_t_in_0_shape_h_const_u32, node_458_t_out_0_shape_w_const_u32, node_458_t_out_0_shape_h_const_u32, node_458_t_weight_0_shape_w_const_u32, node_458_t_weight_0_shape_h_const_u32, node_458_l_pad_W_0_const_s32, node_458_l_pad_H_0_const_s32, node_458_l_stride_1_const_u16, node_458_l_stride_0_const_u16, 3, 3, node_458_l_dilation_H_const_u16, node_458_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(1, 1, {(stai_ptr) node_458_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_458 */
  /* LITE_KERNEL_SECTION BEGIN node_211 */
  {
      ai_handle node_211_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[1] + 802432);
    const ai_handle node_211_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[1] + 802432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(2, 1, {(stai_ptr) node_211_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_211_t_out_0_ptr_handle, node_211_t_in_0_ptr_const_handle, node_211_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(2, 1, {(stai_ptr) node_211_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_211 */
  /* LITE_KERNEL_SECTION BEGIN node_212 */
  {
      const ai_float* node_212_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 802432);
    ai_float* node_212_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 792192);
    const ai_u8* node_212_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 1792);
    const ai_u8* node_212_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 2816);
    ai_float* node_212_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 256);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(3, 1, {(stai_ptr) node_212_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_212_t_in_0_ptr_const_f32, node_212_t_out_0_ptr_f32, node_212_t_weight_0_ptr_const_u8, node_212_t_weight_1_ptr_const_u8, node_212_t_scratch_0_ptr_f32, node_212_t_in_0_shape_ch_const_u32, node_212_t_out_0_shape_ch_const_u32, node_212_t_in_0_shape_w_const_u32, node_212_t_in_0_shape_h_const_u32, node_212_t_out_0_shape_w_const_u32, node_212_t_out_0_shape_h_const_u32, node_212_t_weight_0_shape_w_const_u32, node_212_t_weight_0_shape_h_const_u32, node_212_l_pad_W_0_const_s32, node_212_l_pad_H_0_const_s32, node_212_l_stride_1_const_u16, node_212_l_stride_0_const_u16, 1, 1, node_212_l_dilation_H_const_u16, node_212_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(3, 1, {(stai_ptr) node_212_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_212 */
  /* LITE_KERNEL_SECTION BEGIN node_461 */
  {
      const ai_float* node_461_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 792192);
    ai_float* node_461_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 771520);
    const ai_u8* node_461_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 2880);
    const ai_u8* node_461_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 3456);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(4, 1, {(stai_ptr) node_461_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_461_t_in_0_ptr_const_f32, node_461_t_out_0_ptr_f32, node_461_t_weight_0_ptr_const_u8, node_461_t_weight_1_ptr_const_u8, node_461_t_in_0_shape_ch_const_u32, node_461_t_out_0_shape_ch_const_u32, node_461_t_in_0_shape_w_const_u32, node_461_t_in_0_shape_h_const_u32, node_461_t_out_0_shape_w_const_u32, node_461_t_out_0_shape_h_const_u32, node_461_t_weight_0_shape_w_const_u32, node_461_t_weight_0_shape_h_const_u32, node_461_l_pad_W_0_const_s32, node_461_l_pad_H_0_const_s32, node_461_l_stride_1_const_u16, node_461_l_stride_0_const_u16, 3, 3, node_461_l_dilation_H_const_u16, node_461_l_dilation_W_const_u16, (ai_size)(16));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(4, 1, {(stai_ptr) node_461_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_461 */
  /* LITE_KERNEL_SECTION BEGIN node_215 */
  {
      ai_handle node_215_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[1] + 771520);
    const ai_handle node_215_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[1] + 771520);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(5, 1, {(stai_ptr) node_215_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_215_t_out_0_ptr_handle, node_215_t_in_0_ptr_const_handle, node_215_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(5, 1, {(stai_ptr) node_215_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_215 */
  /* LITE_KERNEL_SECTION BEGIN node_216 */
  {
    
  forward_lite_mp_node_216(net_ctx);
  }
  /* LITE_KERNEL_SECTION END node_216 */
  /* LITE_KERNEL_SECTION BEGIN node_217 */
  {
      const ai_float* node_217_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 0);
    ai_float* node_217_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 409600);
    const ai_u8* node_217_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 3520);
    const ai_u8* node_217_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 4544);
    ai_float* node_217_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 256);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(7, 1, {(stai_ptr) node_217_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_217_t_in_0_ptr_const_f32, node_217_t_out_0_ptr_f32, node_217_t_weight_0_ptr_const_u8, node_217_t_weight_1_ptr_const_u8, node_217_t_scratch_0_ptr_f32, node_217_t_in_0_shape_ch_const_u32, node_217_t_out_0_shape_ch_const_u32, node_217_t_in_0_shape_w_const_u32, node_217_t_in_0_shape_h_const_u32, node_217_t_out_0_shape_w_const_u32, node_217_t_out_0_shape_h_const_u32, node_217_t_weight_0_shape_w_const_u32, node_217_t_weight_0_shape_h_const_u32, node_217_l_pad_W_0_const_s32, node_217_l_pad_H_0_const_s32, node_217_l_stride_1_const_u16, node_217_l_stride_0_const_u16, 1, 1, node_217_l_dilation_H_const_u16, node_217_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(7, 1, {(stai_ptr) node_217_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_217 */
  /* LITE_KERNEL_SECTION BEGIN node_464 */
  {
      const ai_float* node_464_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 409600);
    ai_float* node_464_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 0);
    const ai_u8* node_464_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 4608);
    const ai_u8* node_464_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 5184);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(8, 1, {(stai_ptr) node_464_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_464_t_in_0_ptr_const_f32, node_464_t_out_0_ptr_f32, node_464_t_weight_0_ptr_const_u8, node_464_t_weight_1_ptr_const_u8, node_464_t_in_0_shape_ch_const_u32, node_464_t_out_0_shape_ch_const_u32, node_464_t_in_0_shape_w_const_u32, node_464_t_in_0_shape_h_const_u32, node_464_t_out_0_shape_w_const_u32, node_464_t_out_0_shape_h_const_u32, node_464_t_weight_0_shape_w_const_u32, node_464_t_weight_0_shape_h_const_u32, node_464_l_pad_W_0_const_s32, node_464_l_pad_H_0_const_s32, node_464_l_stride_1_const_u16, node_464_l_stride_0_const_u16, 3, 3, node_464_l_dilation_H_const_u16, node_464_l_dilation_W_const_u16, (ai_size)(16));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(8, 1, {(stai_ptr) node_464_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_464 */
  /* LITE_KERNEL_SECTION BEGIN node_220 */
  {
      ai_handle node_220_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[1] + 409600);
    const ai_handle node_220_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[1] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(9, 1, {(stai_ptr) node_220_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_220_t_out_0_ptr_handle, node_220_t_in_0_ptr_const_handle, node_220_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(9, 1, {(stai_ptr) node_220_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_220 */
  /* LITE_KERNEL_SECTION BEGIN node_221 */
  {
      const ai_float* node_221_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 409600);
    ai_float* node_221_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 819200);
    const ai_u8* node_221_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 5248);
    const ai_u8* node_221_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 9344);
    ai_float* node_221_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 300);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(10, 1, {(stai_ptr) node_221_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_221_t_in_0_ptr_const_f32, node_221_t_out_0_ptr_f32, node_221_t_weight_0_ptr_const_u8, node_221_t_weight_1_ptr_const_u8, node_221_t_scratch_0_ptr_f32, node_221_t_in_0_shape_ch_const_u32, node_221_t_out_0_shape_ch_const_u32, node_221_t_in_0_shape_w_const_u32, node_221_t_in_0_shape_h_const_u32, node_221_t_out_0_shape_w_const_u32, node_221_t_out_0_shape_h_const_u32, node_221_t_weight_0_shape_w_const_u32, node_221_t_weight_0_shape_h_const_u32, node_221_l_pad_W_0_const_s32, node_221_l_pad_H_0_const_s32, node_221_l_stride_1_const_u16, node_221_l_stride_0_const_u16, 1, 1, node_221_l_dilation_H_const_u16, node_221_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(10, 1, {(stai_ptr) node_221_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_221 */
  /* LITE_KERNEL_SECTION BEGIN node_467 */
  {
      const ai_float* node_467_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 819200);
    ai_float* node_467_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 777472);
    const ai_u8* node_467_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 9600);
    const ai_u8* node_467_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 11904);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(11, 1, {(stai_ptr) node_467_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_467_t_in_0_ptr_const_f32, node_467_t_out_0_ptr_f32, node_467_t_weight_0_ptr_const_u8, node_467_t_weight_1_ptr_const_u8, node_467_t_in_0_shape_ch_const_u32, node_467_t_out_0_shape_ch_const_u32, node_467_t_in_0_shape_w_const_u32, node_467_t_in_0_shape_h_const_u32, node_467_t_out_0_shape_w_const_u32, node_467_t_out_0_shape_h_const_u32, node_467_t_weight_0_shape_w_const_u32, node_467_t_weight_0_shape_h_const_u32, node_467_l_pad_W_0_const_s32, node_467_l_pad_H_0_const_s32, node_467_l_stride_1_const_u16, node_467_l_stride_0_const_u16, 3, 3, node_467_l_dilation_H_const_u16, node_467_l_dilation_W_const_u16, (ai_size)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(11, 1, {(stai_ptr) node_467_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_467 */
  /* LITE_KERNEL_SECTION BEGIN node_224 */
  {
      ai_handle node_224_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[1] + 777472);
    const ai_handle node_224_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[1] + 777472);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(12, 1, {(stai_ptr) node_224_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_224_t_out_0_ptr_handle, node_224_t_in_0_ptr_const_handle, node_224_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(12, 1, {(stai_ptr) node_224_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_224 */
  /* LITE_KERNEL_SECTION BEGIN node_225 */
  {
      const ai_float* node_225_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 777472);
    ai_float* node_225_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 756992);
    const ai_u8* node_225_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 12160);
    const ai_u8* node_225_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 28544);
    ai_float* node_225_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(13, 1, {(stai_ptr) node_225_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_225_t_in_0_ptr_const_f32, node_225_t_out_0_ptr_f32, node_225_t_weight_0_ptr_const_u8, node_225_t_weight_1_ptr_const_u8, node_225_t_scratch_0_ptr_f32, node_225_t_in_0_shape_ch_const_u32, node_225_t_out_0_shape_ch_const_u32, node_225_t_in_0_shape_w_const_u32, node_225_t_in_0_shape_h_const_u32, node_225_t_out_0_shape_w_const_u32, node_225_t_out_0_shape_h_const_u32, node_225_t_weight_0_shape_w_const_u32, node_225_t_weight_0_shape_h_const_u32, node_225_l_pad_W_0_const_s32, node_225_l_pad_H_0_const_s32, node_225_l_stride_1_const_u16, node_225_l_stride_0_const_u16, 1, 1, node_225_l_dilation_H_const_u16, node_225_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(13, 1, {(stai_ptr) node_225_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_225 */
  /* LITE_KERNEL_SECTION BEGIN node_470 */
  {
      const ai_float* node_470_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 756992);
    ai_float* node_470_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 715264);
    const ai_u8* node_470_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 28800);
    const ai_u8* node_470_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 31104);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(14, 1, {(stai_ptr) node_470_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_470_t_in_0_ptr_const_f32, node_470_t_out_0_ptr_f32, node_470_t_weight_0_ptr_const_u8, node_470_t_weight_1_ptr_const_u8, node_470_t_in_0_shape_ch_const_u32, node_470_t_out_0_shape_ch_const_u32, node_470_t_in_0_shape_w_const_u32, node_470_t_in_0_shape_h_const_u32, node_470_t_out_0_shape_w_const_u32, node_470_t_out_0_shape_h_const_u32, node_470_t_weight_0_shape_w_const_u32, node_470_t_weight_0_shape_h_const_u32, node_470_l_pad_W_0_const_s32, node_470_l_pad_H_0_const_s32, node_470_l_stride_1_const_u16, node_470_l_stride_0_const_u16, 3, 3, node_470_l_dilation_H_const_u16, node_470_l_dilation_W_const_u16, (ai_size)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(14, 1, {(stai_ptr) node_470_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_470 */
  /* LITE_KERNEL_SECTION BEGIN node_228 */
  {
      ai_handle node_228_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[1] + 715264);
    const ai_handle node_228_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[1] + 715264);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(15, 1, {(stai_ptr) node_228_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_228_t_out_0_ptr_handle, node_228_t_in_0_ptr_const_handle, node_228_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(15, 1, {(stai_ptr) node_228_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_228 */
  /* LITE_KERNEL_SECTION BEGIN node_229 */
  {
      const ai_float* node_229_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 715264);
    ai_float* node_229_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 694784);
    const ai_u8* node_229_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 31360);
    const ai_u8* node_229_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 47744);
    ai_float* node_229_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(16, 1, {(stai_ptr) node_229_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_229_t_in_0_ptr_const_f32, node_229_t_out_0_ptr_f32, node_229_t_weight_0_ptr_const_u8, node_229_t_weight_1_ptr_const_u8, node_229_t_scratch_0_ptr_f32, node_229_t_in_0_shape_ch_const_u32, node_229_t_out_0_shape_ch_const_u32, node_229_t_in_0_shape_w_const_u32, node_229_t_in_0_shape_h_const_u32, node_229_t_out_0_shape_w_const_u32, node_229_t_out_0_shape_h_const_u32, node_229_t_weight_0_shape_w_const_u32, node_229_t_weight_0_shape_h_const_u32, node_229_l_pad_W_0_const_s32, node_229_l_pad_H_0_const_s32, node_229_l_stride_1_const_u16, node_229_l_stride_0_const_u16, 1, 1, node_229_l_dilation_H_const_u16, node_229_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(16, 1, {(stai_ptr) node_229_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_229 */
  /* LITE_KERNEL_SECTION BEGIN node_473 */
  {
      const ai_float* node_473_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 694784);
    ai_float* node_473_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 653056);
    const ai_u8* node_473_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 48000);
    const ai_u8* node_473_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 50304);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(17, 1, {(stai_ptr) node_473_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_473_t_in_0_ptr_const_f32, node_473_t_out_0_ptr_f32, node_473_t_weight_0_ptr_const_u8, node_473_t_weight_1_ptr_const_u8, node_473_t_in_0_shape_ch_const_u32, node_473_t_out_0_shape_ch_const_u32, node_473_t_in_0_shape_w_const_u32, node_473_t_in_0_shape_h_const_u32, node_473_t_out_0_shape_w_const_u32, node_473_t_out_0_shape_h_const_u32, node_473_t_weight_0_shape_w_const_u32, node_473_t_weight_0_shape_h_const_u32, node_473_l_pad_W_0_const_s32, node_473_l_pad_H_0_const_s32, node_473_l_stride_1_const_u16, node_473_l_stride_0_const_u16, 3, 3, node_473_l_dilation_H_const_u16, node_473_l_dilation_W_const_u16, (ai_size)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(17, 1, {(stai_ptr) node_473_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_473 */
  /* LITE_KERNEL_SECTION BEGIN node_232 */
  {
      ai_handle node_232_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[1] + 653056);
    const ai_handle node_232_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[1] + 653056);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(18, 1, {(stai_ptr) node_232_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_232_t_out_0_ptr_handle, node_232_t_in_0_ptr_const_handle, node_232_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(18, 1, {(stai_ptr) node_232_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_232 */
  /* LITE_KERNEL_SECTION BEGIN node_233 */
  {
    
  forward_lite_mp_node_233(net_ctx);
  }
  /* LITE_KERNEL_SECTION END node_233 */
  /* LITE_KERNEL_SECTION BEGIN node_234 */
  {
      const ai_float* node_234_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 0);
    ai_float* node_234_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 409600);
    const ai_u8* node_234_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 50560);
    const ai_u8* node_234_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 66944);
    ai_float* node_234_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(20, 1, {(stai_ptr) node_234_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_234_t_in_0_ptr_const_f32, node_234_t_out_0_ptr_f32, node_234_t_weight_0_ptr_const_u8, node_234_t_weight_1_ptr_const_u8, node_234_t_scratch_0_ptr_f32, node_234_t_in_0_shape_ch_const_u32, node_234_t_out_0_shape_ch_const_u32, node_234_t_in_0_shape_w_const_u32, node_234_t_in_0_shape_h_const_u32, node_234_t_out_0_shape_w_const_u32, node_234_t_out_0_shape_h_const_u32, node_234_t_weight_0_shape_w_const_u32, node_234_t_weight_0_shape_h_const_u32, node_234_l_pad_W_0_const_s32, node_234_l_pad_H_0_const_s32, node_234_l_stride_1_const_u16, node_234_l_stride_0_const_u16, 1, 1, node_234_l_dilation_H_const_u16, node_234_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(20, 1, {(stai_ptr) node_234_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_234 */
  /* LITE_KERNEL_SECTION BEGIN node_476 */
  {
      const ai_float* node_476_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 409600);
    ai_float* node_476_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 0);
    const ai_u8* node_476_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 67200);
    const ai_u8* node_476_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 69504);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(21, 1, {(stai_ptr) node_476_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_476_t_in_0_ptr_const_f32, node_476_t_out_0_ptr_f32, node_476_t_weight_0_ptr_const_u8, node_476_t_weight_1_ptr_const_u8, node_476_t_in_0_shape_ch_const_u32, node_476_t_out_0_shape_ch_const_u32, node_476_t_in_0_shape_w_const_u32, node_476_t_in_0_shape_h_const_u32, node_476_t_out_0_shape_w_const_u32, node_476_t_out_0_shape_h_const_u32, node_476_t_weight_0_shape_w_const_u32, node_476_t_weight_0_shape_h_const_u32, node_476_l_pad_W_0_const_s32, node_476_l_pad_H_0_const_s32, node_476_l_stride_1_const_u16, node_476_l_stride_0_const_u16, 3, 3, node_476_l_dilation_H_const_u16, node_476_l_dilation_W_const_u16, (ai_size)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(21, 1, {(stai_ptr) node_476_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_476 */
  /* LITE_KERNEL_SECTION BEGIN node_237 */
  {
      ai_handle node_237_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[1] + 409600);
    const ai_handle node_237_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[1] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(22, 1, {(stai_ptr) node_237_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_237_t_out_0_ptr_handle, node_237_t_in_0_ptr_const_handle, node_237_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(22, 1, {(stai_ptr) node_237_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_237 */
  /* LITE_KERNEL_SECTION BEGIN node_238 */
  {
      const ai_float* node_238_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 409600);
    ai_float* node_238_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 2048000);
    const ai_u8* node_238_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 69760);
    const ai_u8* node_238_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 86144);
    ai_float* node_238_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(23, 1, {(stai_ptr) node_238_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_238_t_in_0_ptr_const_f32, node_238_t_out_0_ptr_f32, node_238_t_weight_0_ptr_const_u8, node_238_t_weight_1_ptr_const_u8, node_238_t_scratch_0_ptr_f32, node_238_t_in_0_shape_ch_const_u32, node_238_t_out_0_shape_ch_const_u32, node_238_t_in_0_shape_w_const_u32, node_238_t_in_0_shape_h_const_u32, node_238_t_out_0_shape_w_const_u32, node_238_t_out_0_shape_h_const_u32, node_238_t_weight_0_shape_w_const_u32, node_238_t_weight_0_shape_h_const_u32, node_238_l_pad_W_0_const_s32, node_238_l_pad_H_0_const_s32, node_238_l_stride_1_const_u16, node_238_l_stride_0_const_u16, 1, 1, node_238_l_dilation_H_const_u16, node_238_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(23, 1, {(stai_ptr) node_238_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_238 */
  /* LITE_KERNEL_SECTION BEGIN node_479 */
  {
      const ai_float* node_479_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 2048000);
    ai_float* node_479_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 2026752);
    const ai_u8* node_479_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 86400);
    const ai_u8* node_479_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 88704);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(24, 1, {(stai_ptr) node_479_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_479_t_in_0_ptr_const_f32, node_479_t_out_0_ptr_f32, node_479_t_weight_0_ptr_const_u8, node_479_t_weight_1_ptr_const_u8, node_479_t_in_0_shape_ch_const_u32, node_479_t_out_0_shape_ch_const_u32, node_479_t_in_0_shape_w_const_u32, node_479_t_in_0_shape_h_const_u32, node_479_t_out_0_shape_w_const_u32, node_479_t_out_0_shape_h_const_u32, node_479_t_weight_0_shape_w_const_u32, node_479_t_weight_0_shape_h_const_u32, node_479_l_pad_W_0_const_s32, node_479_l_pad_H_0_const_s32, node_479_l_stride_1_const_u16, node_479_l_stride_0_const_u16, 3, 3, node_479_l_dilation_H_const_u16, node_479_l_dilation_W_const_u16, (ai_size)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(24, 1, {(stai_ptr) node_479_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_479 */
  /* LITE_KERNEL_SECTION BEGIN node_241 */
  {
      ai_handle node_241_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[1] + 0);
    const ai_handle node_241_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[1] + 2026752);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(25, 1, {(stai_ptr) node_241_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_241_t_out_0_ptr_handle, node_241_t_in_0_ptr_const_handle, node_241_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(25, 1, {(stai_ptr) node_241_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_241 */
  /* LITE_KERNEL_SECTION BEGIN node_242 */
  {
    
  forward_lite_mp_node_242(net_ctx);
  }
  /* LITE_KERNEL_SECTION END node_242 */
  /* LITE_KERNEL_SECTION BEGIN node_243 */
  {
      const ai_float* node_243_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 409600);
    ai_float* node_243_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 512000);
    const ai_u8* node_243_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 88960);
    const ai_u8* node_243_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 105344);
    ai_float* node_243_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(27, 1, {(stai_ptr) node_243_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_243_t_in_0_ptr_const_f32, node_243_t_out_0_ptr_f32, node_243_t_weight_0_ptr_const_u8, node_243_t_weight_1_ptr_const_u8, node_243_t_scratch_0_ptr_f32, node_243_t_in_0_shape_ch_const_u32, node_243_t_out_0_shape_ch_const_u32, node_243_t_in_0_shape_w_const_u32, node_243_t_in_0_shape_h_const_u32, node_243_t_out_0_shape_w_const_u32, node_243_t_out_0_shape_h_const_u32, node_243_t_weight_0_shape_w_const_u32, node_243_t_weight_0_shape_h_const_u32, node_243_l_pad_W_0_const_s32, node_243_l_pad_H_0_const_s32, node_243_l_stride_1_const_u16, node_243_l_stride_0_const_u16, 1, 1, node_243_l_dilation_H_const_u16, node_243_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(27, 1, {(stai_ptr) node_243_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_243 */
  /* LITE_KERNEL_SECTION BEGIN node_482 */
  {
      const ai_float* node_482_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 512000);
    ai_float* node_482_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 409600);
    const ai_u8* node_482_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 105600);
    const ai_u8* node_482_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 107904);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(28, 1, {(stai_ptr) node_482_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_482_t_in_0_ptr_const_f32, node_482_t_out_0_ptr_f32, node_482_t_weight_0_ptr_const_u8, node_482_t_weight_1_ptr_const_u8, node_482_t_in_0_shape_ch_const_u32, node_482_t_out_0_shape_ch_const_u32, node_482_t_in_0_shape_w_const_u32, node_482_t_in_0_shape_h_const_u32, node_482_t_out_0_shape_w_const_u32, node_482_t_out_0_shape_h_const_u32, node_482_t_weight_0_shape_w_const_u32, node_482_t_weight_0_shape_h_const_u32, node_482_l_pad_W_0_const_s32, node_482_l_pad_H_0_const_s32, node_482_l_stride_1_const_u16, node_482_l_stride_0_const_u16, 3, 3, node_482_l_dilation_H_const_u16, node_482_l_dilation_W_const_u16, (ai_size)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(28, 1, {(stai_ptr) node_482_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_482 */
  /* LITE_KERNEL_SECTION BEGIN node_246 */
  {
      ai_handle node_246_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[1] + 512000);
    const ai_handle node_246_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[1] + 409600);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(29, 1, {(stai_ptr) node_246_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_246_t_out_0_ptr_handle, node_246_t_in_0_ptr_const_handle, node_246_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(29, 1, {(stai_ptr) node_246_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_246 */
  /* LITE_KERNEL_SECTION BEGIN node_247 */
  {
      const ai_float* node_247_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 512000);
    ai_float* node_247_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 614400);
    const ai_u8* node_247_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 108160);
    const ai_u8* node_247_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 124544);
    ai_float* node_247_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(30, 1, {(stai_ptr) node_247_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_247_t_in_0_ptr_const_f32, node_247_t_out_0_ptr_f32, node_247_t_weight_0_ptr_const_u8, node_247_t_weight_1_ptr_const_u8, node_247_t_scratch_0_ptr_f32, node_247_t_in_0_shape_ch_const_u32, node_247_t_out_0_shape_ch_const_u32, node_247_t_in_0_shape_w_const_u32, node_247_t_in_0_shape_h_const_u32, node_247_t_out_0_shape_w_const_u32, node_247_t_out_0_shape_h_const_u32, node_247_t_weight_0_shape_w_const_u32, node_247_t_weight_0_shape_h_const_u32, node_247_l_pad_W_0_const_s32, node_247_l_pad_H_0_const_s32, node_247_l_stride_1_const_u16, node_247_l_stride_0_const_u16, 1, 1, node_247_l_dilation_H_const_u16, node_247_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(30, 1, {(stai_ptr) node_247_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_247 */
  /* LITE_KERNEL_SECTION BEGIN node_485 */
  {
      const ai_float* node_485_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 614400);
    ai_float* node_485_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 512000);
    const ai_u8* node_485_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 124800);
    const ai_u8* node_485_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 127104);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(31, 1, {(stai_ptr) node_485_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_485_t_in_0_ptr_const_f32, node_485_t_out_0_ptr_f32, node_485_t_weight_0_ptr_const_u8, node_485_t_weight_1_ptr_const_u8, node_485_t_in_0_shape_ch_const_u32, node_485_t_out_0_shape_ch_const_u32, node_485_t_in_0_shape_w_const_u32, node_485_t_in_0_shape_h_const_u32, node_485_t_out_0_shape_w_const_u32, node_485_t_out_0_shape_h_const_u32, node_485_t_weight_0_shape_w_const_u32, node_485_t_weight_0_shape_h_const_u32, node_485_l_pad_W_0_const_s32, node_485_l_pad_H_0_const_s32, node_485_l_stride_1_const_u16, node_485_l_stride_0_const_u16, 3, 3, node_485_l_dilation_H_const_u16, node_485_l_dilation_W_const_u16, (ai_size)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(31, 1, {(stai_ptr) node_485_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_485 */
  /* LITE_KERNEL_SECTION BEGIN node_250 */
  {
      ai_handle node_250_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[1] + 409600);
    const ai_handle node_250_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[1] + 512000);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(32, 1, {(stai_ptr) node_250_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_250_t_out_0_ptr_handle, node_250_t_in_0_ptr_const_handle, node_250_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(32, 1, {(stai_ptr) node_250_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_250 */
  /* LITE_KERNEL_SECTION BEGIN node_251 */
  {
    
  forward_lite_mp_node_251(net_ctx);
  }
  /* LITE_KERNEL_SECTION END node_251 */
  /* LITE_KERNEL_SECTION BEGIN node_252 */
  {
      const ai_float* node_252_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 25708);
    ai_float* node_252_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 23148);
    const ai_u8* node_252_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 127360);
    const ai_u8* node_252_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 143744);
    ai_float* node_252_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(34, 1, {(stai_ptr) node_252_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_252_t_in_0_ptr_const_f32, node_252_t_out_0_ptr_f32, node_252_t_weight_0_ptr_const_u8, node_252_t_weight_1_ptr_const_u8, node_252_t_scratch_0_ptr_f32, node_252_t_in_0_shape_ch_const_u32, node_252_t_out_0_shape_ch_const_u32, node_252_t_in_0_shape_w_const_u32, node_252_t_in_0_shape_h_const_u32, node_252_t_out_0_shape_w_const_u32, node_252_t_out_0_shape_h_const_u32, node_252_t_weight_0_shape_w_const_u32, node_252_t_weight_0_shape_h_const_u32, node_252_l_pad_W_0_const_s32, node_252_l_pad_H_0_const_s32, node_252_l_stride_1_const_u16, node_252_l_stride_0_const_u16, 1, 1, node_252_l_dilation_H_const_u16, node_252_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(34, 1, {(stai_ptr) node_252_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_252 */
  /* LITE_KERNEL_SECTION BEGIN node_488 */
  {
      const ai_float* node_488_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 23148);
    ai_float* node_488_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 17260);
    const ai_u8* node_488_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 144000);
    const ai_u8* node_488_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 146304);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(35, 1, {(stai_ptr) node_488_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_488_t_in_0_ptr_const_f32, node_488_t_out_0_ptr_f32, node_488_t_weight_0_ptr_const_u8, node_488_t_weight_1_ptr_const_u8, node_488_t_in_0_shape_ch_const_u32, node_488_t_out_0_shape_ch_const_u32, node_488_t_in_0_shape_w_const_u32, node_488_t_in_0_shape_h_const_u32, node_488_t_out_0_shape_w_const_u32, node_488_t_out_0_shape_h_const_u32, node_488_t_weight_0_shape_w_const_u32, node_488_t_weight_0_shape_h_const_u32, node_488_l_pad_W_0_const_s32, node_488_l_pad_H_0_const_s32, node_488_l_stride_1_const_u16, node_488_l_stride_0_const_u16, 3, 3, node_488_l_dilation_H_const_u16, node_488_l_dilation_W_const_u16, (ai_size)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(35, 1, {(stai_ptr) node_488_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_488 */
  /* LITE_KERNEL_SECTION BEGIN node_255 */
  {
      ai_handle node_255_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 17260);
    const ai_handle node_255_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 17260);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(36, 1, {(stai_ptr) node_255_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_255_t_out_0_ptr_handle, node_255_t_in_0_ptr_const_handle, node_255_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(36, 1, {(stai_ptr) node_255_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_255 */
  /* LITE_KERNEL_SECTION BEGIN node_256 */
  {
      const ai_float* node_256_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 17260);
    ai_float* node_256_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 14700);
    const ai_u8* node_256_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 146560);
    const ai_u8* node_256_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 162944);
    ai_float* node_256_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(37, 1, {(stai_ptr) node_256_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_256_t_in_0_ptr_const_f32, node_256_t_out_0_ptr_f32, node_256_t_weight_0_ptr_const_u8, node_256_t_weight_1_ptr_const_u8, node_256_t_scratch_0_ptr_f32, node_256_t_in_0_shape_ch_const_u32, node_256_t_out_0_shape_ch_const_u32, node_256_t_in_0_shape_w_const_u32, node_256_t_in_0_shape_h_const_u32, node_256_t_out_0_shape_w_const_u32, node_256_t_out_0_shape_h_const_u32, node_256_t_weight_0_shape_w_const_u32, node_256_t_weight_0_shape_h_const_u32, node_256_l_pad_W_0_const_s32, node_256_l_pad_H_0_const_s32, node_256_l_stride_1_const_u16, node_256_l_stride_0_const_u16, 1, 1, node_256_l_dilation_H_const_u16, node_256_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(37, 1, {(stai_ptr) node_256_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_256 */
  /* LITE_KERNEL_SECTION BEGIN node_491 */
  {
      const ai_float* node_491_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 14700);
    ai_float* node_491_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 8812);
    const ai_u8* node_491_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 163200);
    const ai_u8* node_491_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 165504);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(38, 1, {(stai_ptr) node_491_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_491_t_in_0_ptr_const_f32, node_491_t_out_0_ptr_f32, node_491_t_weight_0_ptr_const_u8, node_491_t_weight_1_ptr_const_u8, node_491_t_in_0_shape_ch_const_u32, node_491_t_out_0_shape_ch_const_u32, node_491_t_in_0_shape_w_const_u32, node_491_t_in_0_shape_h_const_u32, node_491_t_out_0_shape_w_const_u32, node_491_t_out_0_shape_h_const_u32, node_491_t_weight_0_shape_w_const_u32, node_491_t_weight_0_shape_h_const_u32, node_491_l_pad_W_0_const_s32, node_491_l_pad_H_0_const_s32, node_491_l_stride_1_const_u16, node_491_l_stride_0_const_u16, 3, 3, node_491_l_dilation_H_const_u16, node_491_l_dilation_W_const_u16, (ai_size)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(38, 1, {(stai_ptr) node_491_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_491 */
  /* LITE_KERNEL_SECTION BEGIN node_259 */
  {
      ai_handle node_259_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 8812);
    const ai_handle node_259_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 8812);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(39, 1, {(stai_ptr) node_259_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_259_t_out_0_ptr_handle, node_259_t_in_0_ptr_const_handle, node_259_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(39, 1, {(stai_ptr) node_259_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_259 */
  /* LITE_KERNEL_SECTION BEGIN node_260 */
  {
      const ai_float* node_260_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 8812);
    ai_float* node_260_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 6252);
    const ai_u8* node_260_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 165760);
    const ai_u8* node_260_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 182144);
    ai_float* node_260_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(40, 1, {(stai_ptr) node_260_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_260_t_in_0_ptr_const_f32, node_260_t_out_0_ptr_f32, node_260_t_weight_0_ptr_const_u8, node_260_t_weight_1_ptr_const_u8, node_260_t_scratch_0_ptr_f32, node_260_t_in_0_shape_ch_const_u32, node_260_t_out_0_shape_ch_const_u32, node_260_t_in_0_shape_w_const_u32, node_260_t_in_0_shape_h_const_u32, node_260_t_out_0_shape_w_const_u32, node_260_t_out_0_shape_h_const_u32, node_260_t_weight_0_shape_w_const_u32, node_260_t_weight_0_shape_h_const_u32, node_260_l_pad_W_0_const_s32, node_260_l_pad_H_0_const_s32, node_260_l_stride_1_const_u16, node_260_l_stride_0_const_u16, 1, 1, node_260_l_dilation_H_const_u16, node_260_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(40, 1, {(stai_ptr) node_260_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_260 */
  /* LITE_KERNEL_SECTION BEGIN node_494 */
  {
      const ai_float* node_494_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 6252);
    ai_float* node_494_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 364);
    const ai_u8* node_494_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 182400);
    const ai_u8* node_494_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 184704);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(41, 1, {(stai_ptr) node_494_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_494_t_in_0_ptr_const_f32, node_494_t_out_0_ptr_f32, node_494_t_weight_0_ptr_const_u8, node_494_t_weight_1_ptr_const_u8, node_494_t_in_0_shape_ch_const_u32, node_494_t_out_0_shape_ch_const_u32, node_494_t_in_0_shape_w_const_u32, node_494_t_in_0_shape_h_const_u32, node_494_t_out_0_shape_w_const_u32, node_494_t_out_0_shape_h_const_u32, node_494_t_weight_0_shape_w_const_u32, node_494_t_weight_0_shape_h_const_u32, node_494_l_pad_W_0_const_s32, node_494_l_pad_H_0_const_s32, node_494_l_stride_1_const_u16, node_494_l_stride_0_const_u16, 3, 3, node_494_l_dilation_H_const_u16, node_494_l_dilation_W_const_u16, (ai_size)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(41, 1, {(stai_ptr) node_494_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_494 */
  /* LITE_KERNEL_SECTION BEGIN node_263 */
  {
      ai_handle node_263_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 364);
    const ai_handle node_263_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 364);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(42, 1, {(stai_ptr) node_263_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_263_t_out_0_ptr_handle, node_263_t_in_0_ptr_const_handle, node_263_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(42, 1, {(stai_ptr) node_263_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_263 */
  /* LITE_KERNEL_SECTION BEGIN node_292 */
  {
      const ai_float* node_292_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 364);
    ai_float* node_292_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 25964);
    const ai_u8* node_292_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 184960);
    const ai_u8* node_292_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 201344);
    ai_float* node_292_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(59, 1, {(stai_ptr) node_292_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_292_t_in_0_ptr_const_f32, node_292_t_out_0_ptr_f32, node_292_t_weight_0_ptr_const_u8, node_292_t_weight_1_ptr_const_u8, node_292_t_scratch_0_ptr_f32, node_292_t_in_0_shape_ch_const_u32, node_292_t_out_0_shape_ch_const_u32, node_292_t_in_0_shape_w_const_u32, node_292_t_in_0_shape_h_const_u32, node_292_t_out_0_shape_w_const_u32, node_292_t_out_0_shape_h_const_u32, node_292_t_weight_0_shape_w_const_u32, node_292_t_weight_0_shape_h_const_u32, node_292_l_pad_W_0_const_s32, node_292_l_pad_H_0_const_s32, node_292_l_stride_1_const_u16, node_292_l_stride_0_const_u16, 1, 1, node_292_l_dilation_H_const_u16, node_292_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(59, 1, {(stai_ptr) node_292_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_292 */
  /* LITE_KERNEL_SECTION BEGIN node_509 */
  {
      const ai_float* node_509_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 25964);
    ai_float* node_509_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 512000);
    const ai_u8* node_509_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 201600);
    const ai_u8* node_509_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 203904);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(60, 1, {(stai_ptr) node_509_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_509_t_in_0_ptr_const_f32, node_509_t_out_0_ptr_f32, node_509_t_weight_0_ptr_const_u8, node_509_t_weight_1_ptr_const_u8, node_509_t_in_0_shape_ch_const_u32, node_509_t_out_0_shape_ch_const_u32, node_509_t_in_0_shape_w_const_u32, node_509_t_in_0_shape_h_const_u32, node_509_t_out_0_shape_w_const_u32, node_509_t_out_0_shape_h_const_u32, node_509_t_weight_0_shape_w_const_u32, node_509_t_weight_0_shape_h_const_u32, node_509_l_pad_W_0_const_s32, node_509_l_pad_H_0_const_s32, node_509_l_stride_1_const_u16, node_509_l_stride_0_const_u16, 3, 3, node_509_l_dilation_H_const_u16, node_509_l_dilation_W_const_u16, (ai_size)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(60, 1, {(stai_ptr) node_509_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_509 */
  /* LITE_KERNEL_SECTION BEGIN node_295 */
  {
      ai_handle node_295_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[0] + 25964);
    const ai_handle node_295_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[1] + 512000);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(61, 1, {(stai_ptr) node_295_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_295_t_out_0_ptr_handle, node_295_t_in_0_ptr_const_handle, node_295_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(61, 1, {(stai_ptr) node_295_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_295 */
  /* LITE_KERNEL_SECTION BEGIN node_318 */
  {
      const ai_float* node_318_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 25964);
    ai_float* node_318_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 51564);
    const ai_u8* node_318_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 204160);
    const ai_u8* node_318_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 206720);
    ai_float* node_318_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(84, 1, {(stai_ptr) node_318_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_318_t_in_0_ptr_const_f32, node_318_t_out_0_ptr_f32, node_318_t_weight_0_ptr_const_u8, node_318_t_weight_1_ptr_const_u8, node_318_t_scratch_0_ptr_f32, node_318_t_in_0_shape_ch_const_u32, node_318_t_out_0_shape_ch_const_u32, node_318_t_in_0_shape_w_const_u32, node_318_t_in_0_shape_h_const_u32, node_318_t_out_0_shape_w_const_u32, node_318_t_out_0_shape_h_const_u32, node_318_t_weight_0_shape_w_const_u32, node_318_t_weight_0_shape_h_const_u32, node_318_l_pad_W_0_const_s32, node_318_l_pad_H_0_const_s32, node_318_l_stride_1_const_u16, node_318_l_stride_0_const_u16, 1, 1, node_318_l_dilation_H_const_u16, node_318_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(84, 1, {(stai_ptr) node_318_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_318 */
  /* LITE_KERNEL_SECTION BEGIN node_319 */
  {
      const ai_float* node_319_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 51564);
    ai_float* node_319_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_outputs[11] + 0);
    const ai_u8* node_319_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 206760);
    const ai_u8* node_319_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 207120);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(85, 1, {(stai_ptr) node_319_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_319_t_in_0_ptr_const_f32, node_319_t_out_0_ptr_f32, node_319_t_weight_0_ptr_const_u8, node_319_t_weight_1_ptr_const_u8, node_319_t_in_0_shape_ch_const_u32, node_319_t_out_0_shape_ch_const_u32, node_319_t_in_0_shape_w_const_u32, node_319_t_in_0_shape_h_const_u32, node_319_t_out_0_shape_w_const_u32, node_319_t_out_0_shape_h_const_u32, node_319_t_weight_0_shape_w_const_u32, node_319_t_weight_0_shape_h_const_u32, node_319_l_pad_W_0_const_s32, node_319_l_pad_H_0_const_s32, node_319_l_stride_1_const_u16, node_319_l_stride_0_const_u16, 3, 3, node_319_l_dilation_H_const_u16, node_319_l_dilation_W_const_u16, (ai_size)(10));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(85, 1, {(stai_ptr) node_319_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_319 */
  /* LITE_KERNEL_SECTION BEGIN node_312 */
  {
      const ai_float* node_312_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 25964);
    ai_float* node_312_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 51564);
    const ai_u8* node_312_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 207160);
    const ai_u8* node_312_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 207416);
    ai_float* node_312_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(78, 1, {(stai_ptr) node_312_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_312_t_in_0_ptr_const_f32, node_312_t_out_0_ptr_f32, node_312_t_weight_0_ptr_const_u8, node_312_t_weight_1_ptr_const_u8, node_312_t_scratch_0_ptr_f32, node_312_t_in_0_shape_ch_const_u32, node_312_t_out_0_shape_ch_const_u32, node_312_t_in_0_shape_w_const_u32, node_312_t_in_0_shape_h_const_u32, node_312_t_out_0_shape_w_const_u32, node_312_t_out_0_shape_h_const_u32, node_312_t_weight_0_shape_w_const_u32, node_312_t_weight_0_shape_h_const_u32, node_312_l_pad_W_0_const_s32, node_312_l_pad_H_0_const_s32, node_312_l_stride_1_const_u16, node_312_l_stride_0_const_u16, 1, 1, node_312_l_dilation_H_const_u16, node_312_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(78, 1, {(stai_ptr) node_312_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_312 */
  /* LITE_KERNEL_SECTION BEGIN node_313 */
  {
      const ai_float* node_313_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 51564);
    ai_float* node_313_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 51964);
    const ai_u8* node_313_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 207420);
    const ai_u8* node_313_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 207456);
    ai_float* node_313_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(79, 1, {(stai_ptr) node_313_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_313_t_in_0_ptr_const_f32, node_313_t_out_0_ptr_f32, node_313_t_weight_0_ptr_const_u8, node_313_t_weight_1_ptr_const_u8, node_313_t_scratch_0_ptr_f32, node_313_t_in_0_shape_ch_const_u32, node_313_t_out_0_shape_ch_const_u32, node_313_t_in_0_shape_w_const_u32, node_313_t_in_0_shape_h_const_u32, node_313_t_out_0_shape_w_const_u32, node_313_t_out_0_shape_h_const_u32, node_313_t_weight_0_shape_w_const_u32, node_313_t_weight_0_shape_h_const_u32, node_313_l_pad_W_0_const_s32, node_313_l_pad_H_0_const_s32, node_313_l_stride_1_const_u16, node_313_l_stride_0_const_u16, 3, 3, node_313_l_dilation_H_const_u16, node_313_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(79, 1, {(stai_ptr) node_313_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_313 */
  /* LITE_KERNEL_SECTION BEGIN obj_32 */
  {
      ai_handle obj_32_t_out_0_ptr_handle = (ai_handle)(net_ctx->_outputs[5] + 0);
    const ai_handle obj_32_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 51964);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(103, 1, {(stai_ptr) obj_32_t_in_0_ptr_const_handle});
    
  forward_lite_nl_sigmoid_if32of32(obj_32_t_out_0_ptr_handle, obj_32_t_in_0_ptr_const_handle, obj_32_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(103, 1, {(stai_ptr) obj_32_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END obj_32 */
  /* LITE_KERNEL_SECTION BEGIN node_306 */
  {
      const ai_float* node_306_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 25964);
    ai_float* node_306_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 51964);
    const ai_u8* node_306_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 207460);
    const ai_u8* node_306_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 208484);
    ai_float* node_306_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(72, 1, {(stai_ptr) node_306_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_306_t_in_0_ptr_const_f32, node_306_t_out_0_ptr_f32, node_306_t_weight_0_ptr_const_u8, node_306_t_weight_1_ptr_const_u8, node_306_t_scratch_0_ptr_f32, node_306_t_in_0_shape_ch_const_u32, node_306_t_out_0_shape_ch_const_u32, node_306_t_in_0_shape_w_const_u32, node_306_t_in_0_shape_h_const_u32, node_306_t_out_0_shape_w_const_u32, node_306_t_out_0_shape_h_const_u32, node_306_t_weight_0_shape_w_const_u32, node_306_t_weight_0_shape_h_const_u32, node_306_l_pad_W_0_const_s32, node_306_l_pad_H_0_const_s32, node_306_l_stride_1_const_u16, node_306_l_stride_0_const_u16, 1, 1, node_306_l_dilation_H_const_u16, node_306_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(72, 1, {(stai_ptr) node_306_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_306 */
  /* LITE_KERNEL_SECTION BEGIN node_307 */
  {
      const ai_float* node_307_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 51964);
    ai_float* node_307_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_outputs[8] + 0);
    const ai_u8* node_307_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 208500);
    const ai_u8* node_307_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 208644);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(73, 1, {(stai_ptr) node_307_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_307_t_in_0_ptr_const_f32, node_307_t_out_0_ptr_f32, node_307_t_weight_0_ptr_const_u8, node_307_t_weight_1_ptr_const_u8, node_307_t_in_0_shape_ch_const_u32, node_307_t_out_0_shape_ch_const_u32, node_307_t_in_0_shape_w_const_u32, node_307_t_in_0_shape_h_const_u32, node_307_t_out_0_shape_w_const_u32, node_307_t_out_0_shape_h_const_u32, node_307_t_weight_0_shape_w_const_u32, node_307_t_weight_0_shape_h_const_u32, node_307_l_pad_W_0_const_s32, node_307_l_pad_H_0_const_s32, node_307_l_stride_1_const_u16, node_307_l_stride_0_const_u16, 3, 3, node_307_l_dilation_H_const_u16, node_307_l_dilation_W_const_u16, (ai_size)(4));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(73, 1, {(stai_ptr) node_307_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_307 */
  /* LITE_KERNEL_SECTION BEGIN node_300 */
  {
      const ai_float* node_300_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 25964);
    ai_float* node_300_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 51964);
    const ai_u8* node_300_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 208660);
    const ai_u8* node_300_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 208916);
    ai_float* node_300_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(66, 1, {(stai_ptr) node_300_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_300_t_in_0_ptr_const_f32, node_300_t_out_0_ptr_f32, node_300_t_weight_0_ptr_const_u8, node_300_t_weight_1_ptr_const_u8, node_300_t_scratch_0_ptr_f32, node_300_t_in_0_shape_ch_const_u32, node_300_t_out_0_shape_ch_const_u32, node_300_t_in_0_shape_w_const_u32, node_300_t_in_0_shape_h_const_u32, node_300_t_out_0_shape_w_const_u32, node_300_t_out_0_shape_h_const_u32, node_300_t_weight_0_shape_w_const_u32, node_300_t_weight_0_shape_h_const_u32, node_300_l_pad_W_0_const_s32, node_300_l_pad_H_0_const_s32, node_300_l_stride_1_const_u16, node_300_l_stride_0_const_u16, 1, 1, node_300_l_dilation_H_const_u16, node_300_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(66, 1, {(stai_ptr) node_300_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_300 */
  /* LITE_KERNEL_SECTION BEGIN node_301 */
  {
      const ai_float* node_301_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 51964);
    ai_float* node_301_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 53164);
    const ai_u8* node_301_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 208920);
    const ai_u8* node_301_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 208956);
    ai_float* node_301_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(67, 1, {(stai_ptr) node_301_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_301_t_in_0_ptr_const_f32, node_301_t_out_0_ptr_f32, node_301_t_weight_0_ptr_const_u8, node_301_t_weight_1_ptr_const_u8, node_301_t_scratch_0_ptr_f32, node_301_t_in_0_shape_ch_const_u32, node_301_t_out_0_shape_ch_const_u32, node_301_t_in_0_shape_w_const_u32, node_301_t_in_0_shape_h_const_u32, node_301_t_out_0_shape_w_const_u32, node_301_t_out_0_shape_h_const_u32, node_301_t_weight_0_shape_w_const_u32, node_301_t_weight_0_shape_h_const_u32, node_301_l_pad_W_0_const_s32, node_301_l_pad_H_0_const_s32, node_301_l_stride_1_const_u16, node_301_l_stride_0_const_u16, 3, 3, node_301_l_dilation_H_const_u16, node_301_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(67, 1, {(stai_ptr) node_301_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_301 */
  /* LITE_KERNEL_SECTION BEGIN cls_32 */
  {
      ai_handle cls_32_t_out_0_ptr_handle = (ai_handle)(net_ctx->_outputs[2] + 0);
    const ai_handle cls_32_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 53164);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(94, 1, {(stai_ptr) cls_32_t_in_0_ptr_const_handle});
    
  forward_lite_nl_sigmoid_if32of32(cls_32_t_out_0_ptr_handle, cls_32_t_in_0_ptr_const_handle, cls_32_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(94, 1, {(stai_ptr) cls_32_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END cls_32 */
  /* LITE_KERNEL_SECTION BEGIN node_268 */
  {
    
  forward_lite_upsample_nearest_node_268(net_ctx);
  }
  /* LITE_KERNEL_SECTION END node_268 */
  /* LITE_KERNEL_SECTION BEGIN node_269 */
  {
    
  forward_lite_eltwise_node_269(net_ctx);
  }
  /* LITE_KERNEL_SECTION END node_269 */
  /* LITE_KERNEL_SECTION BEGIN node_270 */
  {
      const ai_float* node_270_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 2252800);
    ai_float* node_270_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 2247680);
    const ai_u8* node_270_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 208960);
    const ai_u8* node_270_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 225344);
    ai_float* node_270_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(45, 1, {(stai_ptr) node_270_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_270_t_in_0_ptr_const_f32, node_270_t_out_0_ptr_f32, node_270_t_weight_0_ptr_const_u8, node_270_t_weight_1_ptr_const_u8, node_270_t_scratch_0_ptr_f32, node_270_t_in_0_shape_ch_const_u32, node_270_t_out_0_shape_ch_const_u32, node_270_t_in_0_shape_w_const_u32, node_270_t_in_0_shape_h_const_u32, node_270_t_out_0_shape_w_const_u32, node_270_t_out_0_shape_h_const_u32, node_270_t_weight_0_shape_w_const_u32, node_270_t_weight_0_shape_h_const_u32, node_270_l_pad_W_0_const_s32, node_270_l_pad_H_0_const_s32, node_270_l_stride_1_const_u16, node_270_l_stride_0_const_u16, 1, 1, node_270_l_dilation_H_const_u16, node_270_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(45, 1, {(stai_ptr) node_270_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_270 */
  /* LITE_KERNEL_SECTION BEGIN node_497 */
  {
      const ai_float* node_497_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 2247680);
    ai_float* node_497_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 2145280);
    const ai_u8* node_497_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 225600);
    const ai_u8* node_497_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 227904);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(46, 1, {(stai_ptr) node_497_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_497_t_in_0_ptr_const_f32, node_497_t_out_0_ptr_f32, node_497_t_weight_0_ptr_const_u8, node_497_t_weight_1_ptr_const_u8, node_497_t_in_0_shape_ch_const_u32, node_497_t_out_0_shape_ch_const_u32, node_497_t_in_0_shape_w_const_u32, node_497_t_in_0_shape_h_const_u32, node_497_t_out_0_shape_w_const_u32, node_497_t_out_0_shape_h_const_u32, node_497_t_weight_0_shape_w_const_u32, node_497_t_weight_0_shape_h_const_u32, node_497_l_pad_W_0_const_s32, node_497_l_pad_H_0_const_s32, node_497_l_stride_1_const_u16, node_497_l_stride_0_const_u16, 3, 3, node_497_l_dilation_H_const_u16, node_497_l_dilation_W_const_u16, (ai_size)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(46, 1, {(stai_ptr) node_497_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_497 */
  /* LITE_KERNEL_SECTION BEGIN node_273 */
  {
      ai_handle node_273_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[1] + 2247680);
    const ai_handle node_273_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[1] + 2145280);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(47, 1, {(stai_ptr) node_273_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_273_t_out_0_ptr_handle, node_273_t_in_0_ptr_const_handle, node_273_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(47, 1, {(stai_ptr) node_273_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_273 */
  /* LITE_KERNEL_SECTION BEGIN node_288 */
  {
      const ai_float* node_288_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 2247680);
    ai_float* node_288_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 2355200);
    const ai_u8* node_288_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 228160);
    const ai_u8* node_288_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 244544);
    ai_float* node_288_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(56, 1, {(stai_ptr) node_288_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_288_t_in_0_ptr_const_f32, node_288_t_out_0_ptr_f32, node_288_t_weight_0_ptr_const_u8, node_288_t_weight_1_ptr_const_u8, node_288_t_scratch_0_ptr_f32, node_288_t_in_0_shape_ch_const_u32, node_288_t_out_0_shape_ch_const_u32, node_288_t_in_0_shape_w_const_u32, node_288_t_in_0_shape_h_const_u32, node_288_t_out_0_shape_w_const_u32, node_288_t_out_0_shape_h_const_u32, node_288_t_weight_0_shape_w_const_u32, node_288_t_weight_0_shape_h_const_u32, node_288_l_pad_W_0_const_s32, node_288_l_pad_H_0_const_s32, node_288_l_stride_1_const_u16, node_288_l_stride_0_const_u16, 1, 1, node_288_l_dilation_H_const_u16, node_288_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(56, 1, {(stai_ptr) node_288_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_288 */
  /* LITE_KERNEL_SECTION BEGIN node_506 */
  {
      const ai_float* node_506_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 2355200);
    ai_float* node_506_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 2145280);
    const ai_u8* node_506_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 244800);
    const ai_u8* node_506_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 247104);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(57, 1, {(stai_ptr) node_506_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_506_t_in_0_ptr_const_f32, node_506_t_out_0_ptr_f32, node_506_t_weight_0_ptr_const_u8, node_506_t_weight_1_ptr_const_u8, node_506_t_in_0_shape_ch_const_u32, node_506_t_out_0_shape_ch_const_u32, node_506_t_in_0_shape_w_const_u32, node_506_t_in_0_shape_h_const_u32, node_506_t_out_0_shape_w_const_u32, node_506_t_out_0_shape_h_const_u32, node_506_t_weight_0_shape_w_const_u32, node_506_t_weight_0_shape_h_const_u32, node_506_l_pad_W_0_const_s32, node_506_l_pad_H_0_const_s32, node_506_l_stride_1_const_u16, node_506_l_stride_0_const_u16, 3, 3, node_506_l_dilation_H_const_u16, node_506_l_dilation_W_const_u16, (ai_size)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(57, 1, {(stai_ptr) node_506_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_506 */
  /* LITE_KERNEL_SECTION BEGIN node_291 */
  {
      ai_handle node_291_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[1] + 409600);
    const ai_handle node_291_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[1] + 2145280);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(58, 1, {(stai_ptr) node_291_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_291_t_out_0_ptr_handle, node_291_t_in_0_ptr_const_handle, node_291_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(58, 1, {(stai_ptr) node_291_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_291 */
  /* LITE_KERNEL_SECTION BEGIN node_316 */
  {
      const ai_float* node_316_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 409600);
    ai_float* node_316_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 364);
    const ai_u8* node_316_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 247360);
    const ai_u8* node_316_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 249920);
    ai_float* node_316_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(82, 1, {(stai_ptr) node_316_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_316_t_in_0_ptr_const_f32, node_316_t_out_0_ptr_f32, node_316_t_weight_0_ptr_const_u8, node_316_t_weight_1_ptr_const_u8, node_316_t_scratch_0_ptr_f32, node_316_t_in_0_shape_ch_const_u32, node_316_t_out_0_shape_ch_const_u32, node_316_t_in_0_shape_w_const_u32, node_316_t_in_0_shape_h_const_u32, node_316_t_out_0_shape_w_const_u32, node_316_t_out_0_shape_h_const_u32, node_316_t_weight_0_shape_w_const_u32, node_316_t_weight_0_shape_h_const_u32, node_316_l_pad_W_0_const_s32, node_316_l_pad_H_0_const_s32, node_316_l_stride_1_const_u16, node_316_l_stride_0_const_u16, 1, 1, node_316_l_dilation_H_const_u16, node_316_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(82, 1, {(stai_ptr) node_316_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_316 */
  /* LITE_KERNEL_SECTION BEGIN node_317 */
  {
      const ai_float* node_317_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 364);
    ai_float* node_317_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_outputs[10] + 0);
    const ai_u8* node_317_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 249960);
    const ai_u8* node_317_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 250320);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(83, 1, {(stai_ptr) node_317_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_317_t_in_0_ptr_const_f32, node_317_t_out_0_ptr_f32, node_317_t_weight_0_ptr_const_u8, node_317_t_weight_1_ptr_const_u8, node_317_t_in_0_shape_ch_const_u32, node_317_t_out_0_shape_ch_const_u32, node_317_t_in_0_shape_w_const_u32, node_317_t_in_0_shape_h_const_u32, node_317_t_out_0_shape_w_const_u32, node_317_t_out_0_shape_h_const_u32, node_317_t_weight_0_shape_w_const_u32, node_317_t_weight_0_shape_h_const_u32, node_317_l_pad_W_0_const_s32, node_317_l_pad_H_0_const_s32, node_317_l_stride_1_const_u16, node_317_l_stride_0_const_u16, 3, 3, node_317_l_dilation_H_const_u16, node_317_l_dilation_W_const_u16, (ai_size)(10));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(83, 1, {(stai_ptr) node_317_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_317 */
  /* LITE_KERNEL_SECTION BEGIN node_310 */
  {
      const ai_float* node_310_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 409600);
    ai_float* node_310_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 364);
    const ai_u8* node_310_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 250360);
    const ai_u8* node_310_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 250616);
    ai_float* node_310_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(76, 1, {(stai_ptr) node_310_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_310_t_in_0_ptr_const_f32, node_310_t_out_0_ptr_f32, node_310_t_weight_0_ptr_const_u8, node_310_t_weight_1_ptr_const_u8, node_310_t_scratch_0_ptr_f32, node_310_t_in_0_shape_ch_const_u32, node_310_t_out_0_shape_ch_const_u32, node_310_t_in_0_shape_w_const_u32, node_310_t_in_0_shape_h_const_u32, node_310_t_out_0_shape_w_const_u32, node_310_t_out_0_shape_h_const_u32, node_310_t_weight_0_shape_w_const_u32, node_310_t_weight_0_shape_h_const_u32, node_310_l_pad_W_0_const_s32, node_310_l_pad_H_0_const_s32, node_310_l_stride_1_const_u16, node_310_l_stride_0_const_u16, 1, 1, node_310_l_dilation_H_const_u16, node_310_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(76, 1, {(stai_ptr) node_310_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_310 */
  /* LITE_KERNEL_SECTION BEGIN node_311 */
  {
      const ai_float* node_311_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 364);
    ai_float* node_311_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 1964);
    const ai_u8* node_311_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 250620);
    const ai_u8* node_311_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 250656);
    ai_float* node_311_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(77, 1, {(stai_ptr) node_311_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_311_t_in_0_ptr_const_f32, node_311_t_out_0_ptr_f32, node_311_t_weight_0_ptr_const_u8, node_311_t_weight_1_ptr_const_u8, node_311_t_scratch_0_ptr_f32, node_311_t_in_0_shape_ch_const_u32, node_311_t_out_0_shape_ch_const_u32, node_311_t_in_0_shape_w_const_u32, node_311_t_in_0_shape_h_const_u32, node_311_t_out_0_shape_w_const_u32, node_311_t_out_0_shape_h_const_u32, node_311_t_weight_0_shape_w_const_u32, node_311_t_weight_0_shape_h_const_u32, node_311_l_pad_W_0_const_s32, node_311_l_pad_H_0_const_s32, node_311_l_stride_1_const_u16, node_311_l_stride_0_const_u16, 3, 3, node_311_l_dilation_H_const_u16, node_311_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(77, 1, {(stai_ptr) node_311_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_311 */
  /* LITE_KERNEL_SECTION BEGIN obj_16 */
  {
      ai_handle obj_16_t_out_0_ptr_handle = (ai_handle)(net_ctx->_outputs[4] + 0);
    const ai_handle obj_16_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 1964);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(100, 1, {(stai_ptr) obj_16_t_in_0_ptr_const_handle});
    
  forward_lite_nl_sigmoid_if32of32(obj_16_t_out_0_ptr_handle, obj_16_t_in_0_ptr_const_handle, obj_16_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(100, 1, {(stai_ptr) obj_16_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END obj_16 */
  /* LITE_KERNEL_SECTION BEGIN node_304 */
  {
      const ai_float* node_304_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 409600);
    ai_float* node_304_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 1964);
    const ai_u8* node_304_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 250660);
    const ai_u8* node_304_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 251684);
    ai_float* node_304_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(70, 1, {(stai_ptr) node_304_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_304_t_in_0_ptr_const_f32, node_304_t_out_0_ptr_f32, node_304_t_weight_0_ptr_const_u8, node_304_t_weight_1_ptr_const_u8, node_304_t_scratch_0_ptr_f32, node_304_t_in_0_shape_ch_const_u32, node_304_t_out_0_shape_ch_const_u32, node_304_t_in_0_shape_w_const_u32, node_304_t_in_0_shape_h_const_u32, node_304_t_out_0_shape_w_const_u32, node_304_t_out_0_shape_h_const_u32, node_304_t_weight_0_shape_w_const_u32, node_304_t_weight_0_shape_h_const_u32, node_304_l_pad_W_0_const_s32, node_304_l_pad_H_0_const_s32, node_304_l_stride_1_const_u16, node_304_l_stride_0_const_u16, 1, 1, node_304_l_dilation_H_const_u16, node_304_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(70, 1, {(stai_ptr) node_304_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_304 */
  /* LITE_KERNEL_SECTION BEGIN node_305 */
  {
      const ai_float* node_305_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 1964);
    ai_float* node_305_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_outputs[7] + 0);
    const ai_u8* node_305_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 251700);
    const ai_u8* node_305_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 251844);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(71, 1, {(stai_ptr) node_305_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_305_t_in_0_ptr_const_f32, node_305_t_out_0_ptr_f32, node_305_t_weight_0_ptr_const_u8, node_305_t_weight_1_ptr_const_u8, node_305_t_in_0_shape_ch_const_u32, node_305_t_out_0_shape_ch_const_u32, node_305_t_in_0_shape_w_const_u32, node_305_t_in_0_shape_h_const_u32, node_305_t_out_0_shape_w_const_u32, node_305_t_out_0_shape_h_const_u32, node_305_t_weight_0_shape_w_const_u32, node_305_t_weight_0_shape_h_const_u32, node_305_l_pad_W_0_const_s32, node_305_l_pad_H_0_const_s32, node_305_l_stride_1_const_u16, node_305_l_stride_0_const_u16, 3, 3, node_305_l_dilation_H_const_u16, node_305_l_dilation_W_const_u16, (ai_size)(4));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(71, 1, {(stai_ptr) node_305_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_305 */
  /* LITE_KERNEL_SECTION BEGIN node_298 */
  {
      const ai_float* node_298_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 409600);
    ai_float* node_298_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 1964);
    const ai_u8* node_298_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 251860);
    const ai_u8* node_298_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 252116);
    ai_float* node_298_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(64, 1, {(stai_ptr) node_298_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_298_t_in_0_ptr_const_f32, node_298_t_out_0_ptr_f32, node_298_t_weight_0_ptr_const_u8, node_298_t_weight_1_ptr_const_u8, node_298_t_scratch_0_ptr_f32, node_298_t_in_0_shape_ch_const_u32, node_298_t_out_0_shape_ch_const_u32, node_298_t_in_0_shape_w_const_u32, node_298_t_in_0_shape_h_const_u32, node_298_t_out_0_shape_w_const_u32, node_298_t_out_0_shape_h_const_u32, node_298_t_weight_0_shape_w_const_u32, node_298_t_weight_0_shape_h_const_u32, node_298_l_pad_W_0_const_s32, node_298_l_pad_H_0_const_s32, node_298_l_stride_1_const_u16, node_298_l_stride_0_const_u16, 1, 1, node_298_l_dilation_H_const_u16, node_298_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(64, 1, {(stai_ptr) node_298_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_298 */
  /* LITE_KERNEL_SECTION BEGIN node_299 */
  {
      const ai_float* node_299_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 1964);
    ai_float* node_299_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 49964);
    const ai_u8* node_299_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 252120);
    const ai_u8* node_299_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 252156);
    ai_float* node_299_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(65, 1, {(stai_ptr) node_299_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_299_t_in_0_ptr_const_f32, node_299_t_out_0_ptr_f32, node_299_t_weight_0_ptr_const_u8, node_299_t_weight_1_ptr_const_u8, node_299_t_scratch_0_ptr_f32, node_299_t_in_0_shape_ch_const_u32, node_299_t_out_0_shape_ch_const_u32, node_299_t_in_0_shape_w_const_u32, node_299_t_in_0_shape_h_const_u32, node_299_t_out_0_shape_w_const_u32, node_299_t_out_0_shape_h_const_u32, node_299_t_weight_0_shape_w_const_u32, node_299_t_weight_0_shape_h_const_u32, node_299_l_pad_W_0_const_s32, node_299_l_pad_H_0_const_s32, node_299_l_stride_1_const_u16, node_299_l_stride_0_const_u16, 3, 3, node_299_l_dilation_H_const_u16, node_299_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(65, 1, {(stai_ptr) node_299_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_299 */
  /* LITE_KERNEL_SECTION BEGIN cls_16 */
  {
      ai_handle cls_16_t_out_0_ptr_handle = (ai_handle)(net_ctx->_outputs[1] + 0);
    const ai_handle cls_16_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 49964);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(91, 1, {(stai_ptr) cls_16_t_in_0_ptr_const_handle});
    
  forward_lite_nl_sigmoid_if32of32(cls_16_t_out_0_ptr_handle, cls_16_t_in_0_ptr_const_handle, cls_16_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(91, 1, {(stai_ptr) cls_16_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END cls_16 */
  /* LITE_KERNEL_SECTION BEGIN node_278 */
  {
    
  forward_lite_upsample_nearest_node_278(net_ctx);
  }
  /* LITE_KERNEL_SECTION END node_278 */
  /* LITE_KERNEL_SECTION BEGIN node_279 */
  {
    
  forward_lite_eltwise_node_279(net_ctx);
  }
  /* LITE_KERNEL_SECTION END node_279 */
  /* LITE_KERNEL_SECTION BEGIN node_280 */
  {
      const ai_float* node_280_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 1428480);
    ai_float* node_280_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 1838080);
    const ai_u8* node_280_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 252160);
    const ai_u8* node_280_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 268544);
    ai_float* node_280_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(50, 1, {(stai_ptr) node_280_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_280_t_in_0_ptr_const_f32, node_280_t_out_0_ptr_f32, node_280_t_weight_0_ptr_const_u8, node_280_t_weight_1_ptr_const_u8, node_280_t_scratch_0_ptr_f32, node_280_t_in_0_shape_ch_const_u32, node_280_t_out_0_shape_ch_const_u32, node_280_t_in_0_shape_w_const_u32, node_280_t_in_0_shape_h_const_u32, node_280_t_out_0_shape_w_const_u32, node_280_t_out_0_shape_h_const_u32, node_280_t_weight_0_shape_w_const_u32, node_280_t_weight_0_shape_h_const_u32, node_280_l_pad_W_0_const_s32, node_280_l_pad_H_0_const_s32, node_280_l_stride_1_const_u16, node_280_l_stride_0_const_u16, 1, 1, node_280_l_dilation_H_const_u16, node_280_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(50, 1, {(stai_ptr) node_280_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_280 */
  /* LITE_KERNEL_SECTION BEGIN node_500 */
  {
      const ai_float* node_500_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 1838080);
    ai_float* node_500_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 1428480);
    const ai_u8* node_500_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 268800);
    const ai_u8* node_500_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 271104);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(51, 1, {(stai_ptr) node_500_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_500_t_in_0_ptr_const_f32, node_500_t_out_0_ptr_f32, node_500_t_weight_0_ptr_const_u8, node_500_t_weight_1_ptr_const_u8, node_500_t_in_0_shape_ch_const_u32, node_500_t_out_0_shape_ch_const_u32, node_500_t_in_0_shape_w_const_u32, node_500_t_in_0_shape_h_const_u32, node_500_t_out_0_shape_w_const_u32, node_500_t_out_0_shape_h_const_u32, node_500_t_weight_0_shape_w_const_u32, node_500_t_weight_0_shape_h_const_u32, node_500_l_pad_W_0_const_s32, node_500_l_pad_H_0_const_s32, node_500_l_stride_1_const_u16, node_500_l_stride_0_const_u16, 3, 3, node_500_l_dilation_H_const_u16, node_500_l_dilation_W_const_u16, (ai_size)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(51, 1, {(stai_ptr) node_500_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_500 */
  /* LITE_KERNEL_SECTION BEGIN node_283 */
  {
      ai_handle node_283_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[1] + 1018880);
    const ai_handle node_283_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[1] + 1428480);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(52, 1, {(stai_ptr) node_283_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_283_t_out_0_ptr_handle, node_283_t_in_0_ptr_const_handle, node_283_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(52, 1, {(stai_ptr) node_283_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_283 */
  /* LITE_KERNEL_SECTION BEGIN node_284 */
  {
      const ai_float* node_284_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 1018880);
    ai_float* node_284_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 0);
    const ai_u8* node_284_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 271360);
    const ai_u8* node_284_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 287744);
    ai_float* node_284_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 108);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(53, 1, {(stai_ptr) node_284_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_284_t_in_0_ptr_const_f32, node_284_t_out_0_ptr_f32, node_284_t_weight_0_ptr_const_u8, node_284_t_weight_1_ptr_const_u8, node_284_t_scratch_0_ptr_f32, node_284_t_in_0_shape_ch_const_u32, node_284_t_out_0_shape_ch_const_u32, node_284_t_in_0_shape_w_const_u32, node_284_t_in_0_shape_h_const_u32, node_284_t_out_0_shape_w_const_u32, node_284_t_out_0_shape_h_const_u32, node_284_t_weight_0_shape_w_const_u32, node_284_t_weight_0_shape_h_const_u32, node_284_l_pad_W_0_const_s32, node_284_l_pad_H_0_const_s32, node_284_l_stride_1_const_u16, node_284_l_stride_0_const_u16, 1, 1, node_284_l_dilation_H_const_u16, node_284_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(53, 1, {(stai_ptr) node_284_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_284 */
  /* LITE_KERNEL_SECTION BEGIN node_503 */
  {
      const ai_float* node_503_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 0);
    ai_float* node_503_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 2048000);
    const ai_u8* node_503_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 288000);
    const ai_u8* node_503_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 290304);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(54, 1, {(stai_ptr) node_503_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_503_t_in_0_ptr_const_f32, node_503_t_out_0_ptr_f32, node_503_t_weight_0_ptr_const_u8, node_503_t_weight_1_ptr_const_u8, node_503_t_in_0_shape_ch_const_u32, node_503_t_out_0_shape_ch_const_u32, node_503_t_in_0_shape_w_const_u32, node_503_t_in_0_shape_h_const_u32, node_503_t_out_0_shape_w_const_u32, node_503_t_out_0_shape_h_const_u32, node_503_t_weight_0_shape_w_const_u32, node_503_t_weight_0_shape_h_const_u32, node_503_l_pad_W_0_const_s32, node_503_l_pad_H_0_const_s32, node_503_l_stride_1_const_u16, node_503_l_stride_0_const_u16, 3, 3, node_503_l_dilation_H_const_u16, node_503_l_dilation_W_const_u16, (ai_size)(64));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(54, 1, {(stai_ptr) node_503_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_503 */
  /* LITE_KERNEL_SECTION BEGIN node_287 */
  {
      ai_handle node_287_t_out_0_ptr_handle = (ai_handle)(net_ctx->_activations[1] + 1638400);
    const ai_handle node_287_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[1] + 2048000);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(55, 1, {(stai_ptr) node_287_t_in_0_ptr_const_handle});
    
  forward_lite_nl_relu_if32of32(node_287_t_out_0_ptr_handle, node_287_t_in_0_ptr_const_handle, node_287_t_in_0_shape_ch_h_w_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(55, 1, {(stai_ptr) node_287_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END node_287 */
  /* LITE_KERNEL_SECTION BEGIN node_314 */
  {
      const ai_float* node_314_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 1638400);
    ai_float* node_314_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 2048000);
    const ai_u8* node_314_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 290560);
    const ai_u8* node_314_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 293120);
    ai_float* node_314_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(80, 1, {(stai_ptr) node_314_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_314_t_in_0_ptr_const_f32, node_314_t_out_0_ptr_f32, node_314_t_weight_0_ptr_const_u8, node_314_t_weight_1_ptr_const_u8, node_314_t_scratch_0_ptr_f32, node_314_t_in_0_shape_ch_const_u32, node_314_t_out_0_shape_ch_const_u32, node_314_t_in_0_shape_w_const_u32, node_314_t_in_0_shape_h_const_u32, node_314_t_out_0_shape_w_const_u32, node_314_t_out_0_shape_h_const_u32, node_314_t_weight_0_shape_w_const_u32, node_314_t_weight_0_shape_h_const_u32, node_314_l_pad_W_0_const_s32, node_314_l_pad_H_0_const_s32, node_314_l_stride_1_const_u16, node_314_l_stride_0_const_u16, 1, 1, node_314_l_dilation_H_const_u16, node_314_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(80, 1, {(stai_ptr) node_314_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_314 */
  /* LITE_KERNEL_SECTION BEGIN node_315 */
  {
      const ai_float* node_315_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 2048000);
    ai_float* node_315_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_outputs[9] + 0);
    const ai_u8* node_315_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 293160);
    const ai_u8* node_315_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 293520);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(81, 1, {(stai_ptr) node_315_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_315_t_in_0_ptr_const_f32, node_315_t_out_0_ptr_f32, node_315_t_weight_0_ptr_const_u8, node_315_t_weight_1_ptr_const_u8, node_315_t_in_0_shape_ch_const_u32, node_315_t_out_0_shape_ch_const_u32, node_315_t_in_0_shape_w_const_u32, node_315_t_in_0_shape_h_const_u32, node_315_t_out_0_shape_w_const_u32, node_315_t_out_0_shape_h_const_u32, node_315_t_weight_0_shape_w_const_u32, node_315_t_weight_0_shape_h_const_u32, node_315_l_pad_W_0_const_s32, node_315_l_pad_H_0_const_s32, node_315_l_stride_1_const_u16, node_315_l_stride_0_const_u16, 3, 3, node_315_l_dilation_H_const_u16, node_315_l_dilation_W_const_u16, (ai_size)(10));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(81, 1, {(stai_ptr) node_315_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_315 */
  /* LITE_KERNEL_SECTION BEGIN node_308 */
  {
      const ai_float* node_308_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 1638400);
    ai_float* node_308_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 32364);
    const ai_u8* node_308_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 293560);
    const ai_u8* node_308_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 293816);
    ai_float* node_308_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(74, 1, {(stai_ptr) node_308_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_308_t_in_0_ptr_const_f32, node_308_t_out_0_ptr_f32, node_308_t_weight_0_ptr_const_u8, node_308_t_weight_1_ptr_const_u8, node_308_t_scratch_0_ptr_f32, node_308_t_in_0_shape_ch_const_u32, node_308_t_out_0_shape_ch_const_u32, node_308_t_in_0_shape_w_const_u32, node_308_t_in_0_shape_h_const_u32, node_308_t_out_0_shape_w_const_u32, node_308_t_out_0_shape_h_const_u32, node_308_t_weight_0_shape_w_const_u32, node_308_t_weight_0_shape_h_const_u32, node_308_l_pad_W_0_const_s32, node_308_l_pad_H_0_const_s32, node_308_l_stride_1_const_u16, node_308_l_stride_0_const_u16, 1, 1, node_308_l_dilation_H_const_u16, node_308_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(74, 1, {(stai_ptr) node_308_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_308 */
  /* LITE_KERNEL_SECTION BEGIN node_309 */
  {
      const ai_float* node_309_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 32364);
    ai_float* node_309_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 38764);
    const ai_u8* node_309_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 293820);
    const ai_u8* node_309_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 293856);
    ai_float* node_309_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(75, 1, {(stai_ptr) node_309_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_309_t_in_0_ptr_const_f32, node_309_t_out_0_ptr_f32, node_309_t_weight_0_ptr_const_u8, node_309_t_weight_1_ptr_const_u8, node_309_t_scratch_0_ptr_f32, node_309_t_in_0_shape_ch_const_u32, node_309_t_out_0_shape_ch_const_u32, node_309_t_in_0_shape_w_const_u32, node_309_t_in_0_shape_h_const_u32, node_309_t_out_0_shape_w_const_u32, node_309_t_out_0_shape_h_const_u32, node_309_t_weight_0_shape_w_const_u32, node_309_t_weight_0_shape_h_const_u32, node_309_l_pad_W_0_const_s32, node_309_l_pad_H_0_const_s32, node_309_l_stride_1_const_u16, node_309_l_stride_0_const_u16, 3, 3, node_309_l_dilation_H_const_u16, node_309_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(75, 1, {(stai_ptr) node_309_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_309 */
  /* LITE_KERNEL_SECTION BEGIN obj_8 */
  {
      ai_handle obj_8_t_out_0_ptr_handle = (ai_handle)(net_ctx->_outputs[3] + 0);
    const ai_handle obj_8_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 38764);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(97, 1, {(stai_ptr) obj_8_t_in_0_ptr_const_handle});
    
  forward_lite_nl_sigmoid_if32of32(obj_8_t_out_0_ptr_handle, obj_8_t_in_0_ptr_const_handle, obj_8_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(97, 1, {(stai_ptr) obj_8_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END obj_8 */
  /* LITE_KERNEL_SECTION BEGIN node_302 */
  {
      const ai_float* node_302_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 1638400);
    ai_float* node_302_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[1] + 1612800);
    const ai_u8* node_302_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 293860);
    const ai_u8* node_302_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 294884);
    ai_float* node_302_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(68, 1, {(stai_ptr) node_302_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_302_t_in_0_ptr_const_f32, node_302_t_out_0_ptr_f32, node_302_t_weight_0_ptr_const_u8, node_302_t_weight_1_ptr_const_u8, node_302_t_scratch_0_ptr_f32, node_302_t_in_0_shape_ch_const_u32, node_302_t_out_0_shape_ch_const_u32, node_302_t_in_0_shape_w_const_u32, node_302_t_in_0_shape_h_const_u32, node_302_t_out_0_shape_w_const_u32, node_302_t_out_0_shape_h_const_u32, node_302_t_weight_0_shape_w_const_u32, node_302_t_weight_0_shape_h_const_u32, node_302_l_pad_W_0_const_s32, node_302_l_pad_H_0_const_s32, node_302_l_stride_1_const_u16, node_302_l_stride_0_const_u16, 1, 1, node_302_l_dilation_H_const_u16, node_302_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(68, 1, {(stai_ptr) node_302_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_302 */
  /* LITE_KERNEL_SECTION BEGIN node_303 */
  {
      const ai_float* node_303_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 1612800);
    ai_float* node_303_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_outputs[6] + 0);
    const ai_u8* node_303_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 294900);
    const ai_u8* node_303_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 295044);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(69, 1, {(stai_ptr) node_303_t_in_0_ptr_const_f32});
    
  forward_lite_dw_if32of32wf32(node_303_t_in_0_ptr_const_f32, node_303_t_out_0_ptr_f32, node_303_t_weight_0_ptr_const_u8, node_303_t_weight_1_ptr_const_u8, node_303_t_in_0_shape_ch_const_u32, node_303_t_out_0_shape_ch_const_u32, node_303_t_in_0_shape_w_const_u32, node_303_t_in_0_shape_h_const_u32, node_303_t_out_0_shape_w_const_u32, node_303_t_out_0_shape_h_const_u32, node_303_t_weight_0_shape_w_const_u32, node_303_t_weight_0_shape_h_const_u32, node_303_l_pad_W_0_const_s32, node_303_l_pad_H_0_const_s32, node_303_l_stride_1_const_u16, node_303_l_stride_0_const_u16, 3, 3, node_303_l_dilation_H_const_u16, node_303_l_dilation_W_const_u16, (ai_size)(4));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(69, 1, {(stai_ptr) node_303_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_303 */
  /* LITE_KERNEL_SECTION BEGIN node_296 */
  {
      const ai_float* node_296_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[1] + 1638400);
    ai_float* node_296_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 38764);
    const ai_u8* node_296_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 295060);
    const ai_u8* node_296_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 295316);
    ai_float* node_296_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(62, 1, {(stai_ptr) node_296_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_296_t_in_0_ptr_const_f32, node_296_t_out_0_ptr_f32, node_296_t_weight_0_ptr_const_u8, node_296_t_weight_1_ptr_const_u8, node_296_t_scratch_0_ptr_f32, node_296_t_in_0_shape_ch_const_u32, node_296_t_out_0_shape_ch_const_u32, node_296_t_in_0_shape_w_const_u32, node_296_t_in_0_shape_h_const_u32, node_296_t_out_0_shape_w_const_u32, node_296_t_out_0_shape_h_const_u32, node_296_t_weight_0_shape_w_const_u32, node_296_t_weight_0_shape_h_const_u32, node_296_l_pad_W_0_const_s32, node_296_l_pad_H_0_const_s32, node_296_l_stride_1_const_u16, node_296_l_stride_0_const_u16, 1, 1, node_296_l_dilation_H_const_u16, node_296_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(62, 1, {(stai_ptr) node_296_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_296 */
  /* LITE_KERNEL_SECTION BEGIN node_297 */
  {
      const ai_float* node_297_t_in_0_ptr_const_f32 = (ai_float*)(net_ctx->_activations[0] + 38764);
    ai_float* node_297_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 45164);
    const ai_u8* node_297_t_weight_0_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 295320);
    const ai_u8* node_297_t_weight_1_ptr_const_u8 = (ai_u8*)(net_ctx->_weights[0] + 295356);
    ai_float* node_297_t_scratch_0_ptr_f32 = (ai_float*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(63, 1, {(stai_ptr) node_297_t_in_0_ptr_const_f32});
    
  forward_lite_conv2d_if32of32wf32(node_297_t_in_0_ptr_const_f32, node_297_t_out_0_ptr_f32, node_297_t_weight_0_ptr_const_u8, node_297_t_weight_1_ptr_const_u8, node_297_t_scratch_0_ptr_f32, node_297_t_in_0_shape_ch_const_u32, node_297_t_out_0_shape_ch_const_u32, node_297_t_in_0_shape_w_const_u32, node_297_t_in_0_shape_h_const_u32, node_297_t_out_0_shape_w_const_u32, node_297_t_out_0_shape_h_const_u32, node_297_t_weight_0_shape_w_const_u32, node_297_t_weight_0_shape_h_const_u32, node_297_l_pad_W_0_const_s32, node_297_l_pad_H_0_const_s32, node_297_l_stride_1_const_u16, node_297_l_stride_0_const_u16, 3, 3, node_297_l_dilation_H_const_u16, node_297_l_dilation_W_const_u16, (ai_size)(1));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(63, 1, {(stai_ptr) node_297_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END node_297 */
  /* LITE_KERNEL_SECTION BEGIN cls_8 */
  {
      ai_handle cls_8_t_out_0_ptr_handle = (ai_handle)(net_ctx->_outputs[0] + 0);
    const ai_handle cls_8_t_in_0_ptr_const_handle = (ai_handle)(net_ctx->_activations[0] + 45164);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(88, 1, {(stai_ptr) cls_8_t_in_0_ptr_const_handle});
    
  forward_lite_nl_sigmoid_if32of32(cls_8_t_out_0_ptr_handle, cls_8_t_in_0_ptr_const_handle, cls_8_t_in_0_shape_ch_h_prod_const_s32, NULL);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(88, 1, {(stai_ptr) cls_8_t_out_0_ptr_handle});
  }
  /* LITE_KERNEL_SECTION END cls_8 */
  return net_ctx->_return_code;
}

/*****************************************************************************/
/*  Getters APIs Section  */
STAI_API_ENTRY
stai_size stai_network_get_context_size()
{
  return (stai_size)STAI_NETWORK_CONTEXT_SIZE;
}

#if defined(HAVE_NETWORK_INFO)
STAI_API_ENTRY
stai_return_code stai_network_get_info(
  stai_network* network,
  stai_network_info* info)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, info==NULL, STAI_ERROR_NETWORK_INVALID_INFO, net_ctx->_return_code)

  // Copy of network info struct
  *info = g_network_info;

  return STAI_SUCCESS;
}
#endif


STAI_API_ENTRY
stai_return_code stai_network_get_activations(
  stai_network* network, stai_ptr* activations, stai_size* n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, !n_activations, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_activations = STAI_NETWORK_ACTIVATIONS_NUM;
for (stai_size idx=0; activations && (idx<STAI_NETWORK_ACTIVATIONS_NUM); idx++) {
    // get address of the activations buffers
    activations[idx] = net_ctx->_activations[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_weights(
  stai_network* network, stai_ptr* weights, stai_size* n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_weights, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_weights = STAI_NETWORK_WEIGHTS_NUM;
for (stai_size idx=0; weights && (idx<STAI_NETWORK_WEIGHTS_NUM); idx++) {
    // get address of the weights buffers
    weights[idx] = net_ctx->_weights[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_inputs(
  stai_network* network, stai_ptr* inputs, stai_size* n_inputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_inputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_inputs = STAI_NETWORK_IN_NUM;
  for (stai_size idx=0; inputs && (idx<STAI_NETWORK_IN_NUM); idx++) {
    inputs[idx] = net_ctx->_inputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_outputs(
  stai_network* network, stai_ptr* outputs, stai_size* n_outputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_outputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_outputs = STAI_NETWORK_OUT_NUM;
  for (stai_size idx=0; outputs && (idx<STAI_NETWORK_OUT_NUM); idx++) {
    outputs[idx] = net_ctx->_outputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_error(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /* return 1st generated error or STAI_SUCCESS if no errors so far */
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_states(
  stai_network* network, stai_ptr* states, stai_size* n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_states, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  /* get the number of internals states (supporting multi-heap also for internal states) */
  *n_states = STAI_NETWORK_STATES_NUM;

  STAI_UNUSED(states)
return net_ctx->_return_code;
}


/*****************************************************************************/
/*  Setters APIs Section  */

STAI_API_ENTRY
stai_return_code stai_network_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _activations_alignment[] = STAI_NETWORK_ACTIVATIONS_ALIGNMENTS;
  STAI_PRINT("  [stai_network_set_activations] network(%p) activations[%d]: %p\n\n", net_ctx, n_activations, activations)
  _STAI_SET_ERROR(net_ctx, !activations,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_activations!=STAI_NETWORK_ACTIVATIONS_NUM,
                  STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_NUM, net_ctx->_return_code)

  for (stai_size idx=0; activations && idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    STAI_PRINT("  activation[%d]: %p\n", idx, activations[idx])
    _STAI_SET_ERROR(net_ctx, activations[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)activations[idx]) & (_activations_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_activations[idx] = activations[idx];
  }
  net_ctx->_inputs[0] = activations[1] + 0;

  net_ctx->_outputs[0] = activations[0] + 38764;

  net_ctx->_outputs[1] = activations[0] + 1964;

  net_ctx->_outputs[2] = activations[0] + 53164;

  net_ctx->_outputs[3] = activations[0] + 32364;

  net_ctx->_outputs[4] = activations[0] + 364;

  net_ctx->_outputs[5] = activations[0] + 51564;

  net_ctx->_outputs[6] = activations[1] + 64000;

  net_ctx->_outputs[7] = activations[0] + 8364;

  net_ctx->_outputs[8] = activations[0] + 53564;

  net_ctx->_outputs[9] = activations[1] + 0;

  net_ctx->_outputs[10] = activations[0] + 16364;

  net_ctx->_outputs[11] = activations[0] + 55564;
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _weights_alignment[] = STAI_NETWORK_WEIGHTS_ALIGNMENTS;
  _STAI_SET_ERROR(net_ctx, !weights,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_weights!=STAI_NETWORK_WEIGHTS_NUM,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_NUM, net_ctx->_return_code)
  for (stai_size idx=0; weights && idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    STAI_PRINT("  weight[%d]: %p\n", idx, weights[idx])
    _STAI_SET_ERROR(net_ctx, weights[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)weights[idx]) & (_weights_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_weights[idx] = weights[idx];
  }_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs)
{
  const uintptr_t _inputs_alignment[] = STAI_NETWORK_IN_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !inputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_inputs!=STAI_NETWORK_IN_NUM,
                  STAI_ERROR_NETWORK_INVALID_IN_NUM, net_ctx->_return_code)

  for (stai_size idx=0; inputs && idx<STAI_NETWORK_IN_NUM; idx++) {
    STAI_PRINT("  input[%d]: %p\n", idx, inputs[idx])
    _STAI_SET_ERROR(net_ctx, inputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)inputs[idx]) & (_inputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_inputs[idx] = inputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs)
{
  const uintptr_t _outputs_alignment[] = STAI_NETWORK_OUT_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !outputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_outputs!=STAI_NETWORK_OUT_NUM,
                  STAI_ERROR_NETWORK_INVALID_OUT_NUM, net_ctx->_return_code)

  for (stai_size idx=0; outputs && idx<n_outputs; idx++) {
    STAI_PRINT("  output[%d]: %p\n", idx, outputs[idx])
    _STAI_SET_ERROR(net_ctx, outputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)outputs[idx]) & (_outputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_outputs[idx] = outputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  STAI_UNUSED(states)
  STAI_UNUSED(n_states)
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}

STAI_API_ENTRY
stai_return_code stai_network_set_callback(
  stai_network* network, const stai_event_cb cb, void* cb_cookie)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  STAI_PRINT("  set_callback %p cb %p cookie %p\n", net_ctx, cb, cb_cookie)
  // _STAI_SET_ERROR(net_ctx, cb==NULL, STAI_ERROR_NETWORK_INVALID_CALLBACK, net_ctx->_return_code)
  net_ctx->_callback = cb;
  net_ctx->_callback_cookie = cb_cookie;
  return net_ctx->_return_code;
}

#undef _STAI_SET_ERROR
#undef _STAI_CONTEXT_ALIGNMENT
#undef _STAI_CONTEXT_ACQUIRE
#undef _STAI_NETWORK_EVENT_NODE_START_CB
#undef _STAI_NETWORK_EVENT_NODE_STOP_CB
#undef _STAI_NETWORK_MODEL_SIGNATURE
#undef _STAI_NETWORK_DATETIME
#undef _STAI_NETWORK_COMPILE_DATETIME

