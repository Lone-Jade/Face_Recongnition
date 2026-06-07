/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-06-07T17:03:49+0800
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
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
#ifndef STAI_NETWORK_DETAILS_H
#define STAI_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_network_details = {
  .tensors = (const stai_tensor[93]) {
   { .size_bytes = 1228800, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 3, 320, 320}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_output" },
   { .size_bytes = 1228800, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 320, 320, 3}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "input_Transpose_output" },
   { .size_bytes = 1638400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 160, 160, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_458_output" },
   { .size_bytes = 1638400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 160, 160, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_211_output" },
   { .size_bytes = 1638400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 160, 160, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_212_output" },
   { .size_bytes = 1638400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 160, 160, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_461_output" },
   { .size_bytes = 1638400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 160, 160, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_215_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 80, 80, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_216_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 80, 80, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_217_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 80, 80, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_464_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 80, 80, 16}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_220_output" },
   { .size_bytes = 1638400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 80, 80, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_221_output" },
   { .size_bytes = 1638400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 80, 80, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_467_output" },
   { .size_bytes = 1638400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 80, 80, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_224_output" },
   { .size_bytes = 1638400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 80, 80, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_225_output" },
   { .size_bytes = 1638400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 80, 80, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_470_output" },
   { .size_bytes = 1638400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 80, 80, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_228_output" },
   { .size_bytes = 1638400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 80, 80, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_229_output" },
   { .size_bytes = 1638400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 80, 80, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_473_output" },
   { .size_bytes = 1638400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 80, 80, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_232_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_233_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_234_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_476_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_237_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_238_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_479_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_241_output" },
   { .size_bytes = 102400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_242_output" },
   { .size_bytes = 102400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_243_output" },
   { .size_bytes = 102400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_482_output" },
   { .size_bytes = 102400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_246_output" },
   { .size_bytes = 102400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_247_output" },
   { .size_bytes = 102400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_485_output" },
   { .size_bytes = 102400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_250_output" },
   { .size_bytes = 25600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_251_output" },
   { .size_bytes = 25600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_252_output" },
   { .size_bytes = 25600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_488_output" },
   { .size_bytes = 25600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_255_output" },
   { .size_bytes = 25600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_256_output" },
   { .size_bytes = 25600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_491_output" },
   { .size_bytes = 25600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_259_output" },
   { .size_bytes = 25600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_260_output" },
   { .size_bytes = 25600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_494_output" },
   { .size_bytes = 25600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_263_output" },
   { .size_bytes = 25600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_292_output" },
   { .size_bytes = 25600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_509_output" },
   { .size_bytes = 25600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_295_output" },
   { .size_bytes = 4000, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 10}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_318_output" },
   { .size_bytes = 4000, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 10}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_319_output" },
   { .size_bytes = 400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_312_output" },
   { .size_bytes = 400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_313_output" },
   { .size_bytes = 400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 100}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "obj_32_output" },
   { .size_bytes = 1600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_306_output" },
   { .size_bytes = 1600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_307_output" },
   { .size_bytes = 400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_300_output" },
   { .size_bytes = 400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 10, 10, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_301_output" },
   { .size_bytes = 400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 100}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "cls_32_output" },
   { .size_bytes = 102400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_268_output" },
   { .size_bytes = 102400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_269_output" },
   { .size_bytes = 102400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_270_output" },
   { .size_bytes = 102400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_497_output" },
   { .size_bytes = 102400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_273_output" },
   { .size_bytes = 102400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_288_output" },
   { .size_bytes = 102400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_506_output" },
   { .size_bytes = 102400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_291_output" },
   { .size_bytes = 16000, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 10}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_316_output" },
   { .size_bytes = 16000, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 10}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_317_output" },
   { .size_bytes = 1600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_310_output" },
   { .size_bytes = 1600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_311_output" },
   { .size_bytes = 1600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 400}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "obj_16_output" },
   { .size_bytes = 6400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_304_output" },
   { .size_bytes = 6400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_305_output" },
   { .size_bytes = 1600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_298_output" },
   { .size_bytes = 1600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 20, 20, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_299_output" },
   { .size_bytes = 1600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 400}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "cls_16_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_278_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_279_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_280_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_500_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_283_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_284_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_503_output" },
   { .size_bytes = 409600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_287_output" },
   { .size_bytes = 64000, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 10}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_314_output" },
   { .size_bytes = 64000, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 10}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_315_output" },
   { .size_bytes = 6400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_308_output" },
   { .size_bytes = 6400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_309_output" },
   { .size_bytes = 6400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1600}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "obj_8_output" },
   { .size_bytes = 25600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_302_output" },
   { .size_bytes = 25600, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_303_output" },
   { .size_bytes = 6400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_296_output" },
   { .size_bytes = 6400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {4, (const int32_t[4]){1, 40, 40, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "node_297_output" },
   { .size_bytes = 6400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 1600}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "cls_8_output" }
  },
  .nodes = (const stai_node_details[92]){
    {.id = 2, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* input_Transpose */
    {.id = 1, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* node_458 */
    {.id = 2, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* node_211 */
    {.id = 3, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* node_212 */
    {.id = 4, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* node_461 */
    {.id = 5, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* node_215 */
    {.id = 6, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* node_216 */
    {.id = 7, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* node_217 */
    {.id = 8, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} }, /* node_464 */
    {.id = 9, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){9}}, .output_tensors = {1, (const int32_t[1]){10}} }, /* node_220 */
    {.id = 10, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){10}}, .output_tensors = {1, (const int32_t[1]){11}} }, /* node_221 */
    {.id = 11, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){11}}, .output_tensors = {1, (const int32_t[1]){12}} }, /* node_467 */
    {.id = 12, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){12}}, .output_tensors = {1, (const int32_t[1]){13}} }, /* node_224 */
    {.id = 13, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){13}}, .output_tensors = {1, (const int32_t[1]){14}} }, /* node_225 */
    {.id = 14, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){14}}, .output_tensors = {1, (const int32_t[1]){15}} }, /* node_470 */
    {.id = 15, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){15}}, .output_tensors = {1, (const int32_t[1]){16}} }, /* node_228 */
    {.id = 16, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){16}}, .output_tensors = {1, (const int32_t[1]){17}} }, /* node_229 */
    {.id = 17, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){17}}, .output_tensors = {1, (const int32_t[1]){18}} }, /* node_473 */
    {.id = 18, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){18}}, .output_tensors = {1, (const int32_t[1]){19}} }, /* node_232 */
    {.id = 19, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){19}}, .output_tensors = {1, (const int32_t[1]){20}} }, /* node_233 */
    {.id = 20, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){20}}, .output_tensors = {1, (const int32_t[1]){21}} }, /* node_234 */
    {.id = 21, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){21}}, .output_tensors = {1, (const int32_t[1]){22}} }, /* node_476 */
    {.id = 22, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){22}}, .output_tensors = {1, (const int32_t[1]){23}} }, /* node_237 */
    {.id = 23, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){23}}, .output_tensors = {1, (const int32_t[1]){24}} }, /* node_238 */
    {.id = 24, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){24}}, .output_tensors = {1, (const int32_t[1]){25}} }, /* node_479 */
    {.id = 25, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){25}}, .output_tensors = {1, (const int32_t[1]){26}} }, /* node_241 */
    {.id = 26, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){26}}, .output_tensors = {1, (const int32_t[1]){27}} }, /* node_242 */
    {.id = 27, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){27}}, .output_tensors = {1, (const int32_t[1]){28}} }, /* node_243 */
    {.id = 28, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){28}}, .output_tensors = {1, (const int32_t[1]){29}} }, /* node_482 */
    {.id = 29, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){29}}, .output_tensors = {1, (const int32_t[1]){30}} }, /* node_246 */
    {.id = 30, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){30}}, .output_tensors = {1, (const int32_t[1]){31}} }, /* node_247 */
    {.id = 31, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){31}}, .output_tensors = {1, (const int32_t[1]){32}} }, /* node_485 */
    {.id = 32, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){32}}, .output_tensors = {1, (const int32_t[1]){33}} }, /* node_250 */
    {.id = 33, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){33}}, .output_tensors = {1, (const int32_t[1]){34}} }, /* node_251 */
    {.id = 34, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){34}}, .output_tensors = {1, (const int32_t[1]){35}} }, /* node_252 */
    {.id = 35, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){35}}, .output_tensors = {1, (const int32_t[1]){36}} }, /* node_488 */
    {.id = 36, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){36}}, .output_tensors = {1, (const int32_t[1]){37}} }, /* node_255 */
    {.id = 37, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){37}}, .output_tensors = {1, (const int32_t[1]){38}} }, /* node_256 */
    {.id = 38, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){38}}, .output_tensors = {1, (const int32_t[1]){39}} }, /* node_491 */
    {.id = 39, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){39}}, .output_tensors = {1, (const int32_t[1]){40}} }, /* node_259 */
    {.id = 40, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){40}}, .output_tensors = {1, (const int32_t[1]){41}} }, /* node_260 */
    {.id = 41, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){41}}, .output_tensors = {1, (const int32_t[1]){42}} }, /* node_494 */
    {.id = 42, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){42}}, .output_tensors = {1, (const int32_t[1]){43}} }, /* node_263 */
    {.id = 59, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){43}}, .output_tensors = {1, (const int32_t[1]){44}} }, /* node_292 */
    {.id = 60, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){44}}, .output_tensors = {1, (const int32_t[1]){45}} }, /* node_509 */
    {.id = 61, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){45}}, .output_tensors = {1, (const int32_t[1]){46}} }, /* node_295 */
    {.id = 84, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){46}}, .output_tensors = {1, (const int32_t[1]){47}} }, /* node_318 */
    {.id = 85, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){47}}, .output_tensors = {1, (const int32_t[1]){48}} }, /* node_319 */
    {.id = 78, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){46}}, .output_tensors = {1, (const int32_t[1]){49}} }, /* node_312 */
    {.id = 79, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){49}}, .output_tensors = {1, (const int32_t[1]){50}} }, /* node_313 */
    {.id = 103, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){50}}, .output_tensors = {1, (const int32_t[1]){51}} }, /* obj_32 */
    {.id = 72, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){46}}, .output_tensors = {1, (const int32_t[1]){52}} }, /* node_306 */
    {.id = 73, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){52}}, .output_tensors = {1, (const int32_t[1]){53}} }, /* node_307 */
    {.id = 66, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){46}}, .output_tensors = {1, (const int32_t[1]){54}} }, /* node_300 */
    {.id = 67, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){54}}, .output_tensors = {1, (const int32_t[1]){55}} }, /* node_301 */
    {.id = 94, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){55}}, .output_tensors = {1, (const int32_t[1]){56}} }, /* cls_32 */
    {.id = 43, .type = AI_LAYER_UPSAMPLE_TYPE, .input_tensors = {1, (const int32_t[1]){43}}, .output_tensors = {1, (const int32_t[1]){57}} }, /* node_268 */
    {.id = 44, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){33, 57}}, .output_tensors = {1, (const int32_t[1]){58}} }, /* node_269 */
    {.id = 45, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){58}}, .output_tensors = {1, (const int32_t[1]){59}} }, /* node_270 */
    {.id = 46, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){59}}, .output_tensors = {1, (const int32_t[1]){60}} }, /* node_497 */
    {.id = 47, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){60}}, .output_tensors = {1, (const int32_t[1]){61}} }, /* node_273 */
    {.id = 56, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){61}}, .output_tensors = {1, (const int32_t[1]){62}} }, /* node_288 */
    {.id = 57, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){62}}, .output_tensors = {1, (const int32_t[1]){63}} }, /* node_506 */
    {.id = 58, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){63}}, .output_tensors = {1, (const int32_t[1]){64}} }, /* node_291 */
    {.id = 82, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){64}}, .output_tensors = {1, (const int32_t[1]){65}} }, /* node_316 */
    {.id = 83, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){65}}, .output_tensors = {1, (const int32_t[1]){66}} }, /* node_317 */
    {.id = 76, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){64}}, .output_tensors = {1, (const int32_t[1]){67}} }, /* node_310 */
    {.id = 77, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){67}}, .output_tensors = {1, (const int32_t[1]){68}} }, /* node_311 */
    {.id = 100, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){68}}, .output_tensors = {1, (const int32_t[1]){69}} }, /* obj_16 */
    {.id = 70, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){64}}, .output_tensors = {1, (const int32_t[1]){70}} }, /* node_304 */
    {.id = 71, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){70}}, .output_tensors = {1, (const int32_t[1]){71}} }, /* node_305 */
    {.id = 64, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){64}}, .output_tensors = {1, (const int32_t[1]){72}} }, /* node_298 */
    {.id = 65, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){72}}, .output_tensors = {1, (const int32_t[1]){73}} }, /* node_299 */
    {.id = 91, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){73}}, .output_tensors = {1, (const int32_t[1]){74}} }, /* cls_16 */
    {.id = 48, .type = AI_LAYER_UPSAMPLE_TYPE, .input_tensors = {1, (const int32_t[1]){61}}, .output_tensors = {1, (const int32_t[1]){75}} }, /* node_278 */
    {.id = 49, .type = AI_LAYER_ELTWISE_TYPE, .input_tensors = {2, (const int32_t[2]){26, 75}}, .output_tensors = {1, (const int32_t[1]){76}} }, /* node_279 */
    {.id = 50, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){76}}, .output_tensors = {1, (const int32_t[1]){77}} }, /* node_280 */
    {.id = 51, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){77}}, .output_tensors = {1, (const int32_t[1]){78}} }, /* node_500 */
    {.id = 52, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){78}}, .output_tensors = {1, (const int32_t[1]){79}} }, /* node_283 */
    {.id = 53, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){79}}, .output_tensors = {1, (const int32_t[1]){80}} }, /* node_284 */
    {.id = 54, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){80}}, .output_tensors = {1, (const int32_t[1]){81}} }, /* node_503 */
    {.id = 55, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){81}}, .output_tensors = {1, (const int32_t[1]){82}} }, /* node_287 */
    {.id = 80, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){82}}, .output_tensors = {1, (const int32_t[1]){83}} }, /* node_314 */
    {.id = 81, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){83}}, .output_tensors = {1, (const int32_t[1]){84}} }, /* node_315 */
    {.id = 74, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){82}}, .output_tensors = {1, (const int32_t[1]){85}} }, /* node_308 */
    {.id = 75, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){85}}, .output_tensors = {1, (const int32_t[1]){86}} }, /* node_309 */
    {.id = 97, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){86}}, .output_tensors = {1, (const int32_t[1]){87}} }, /* obj_8 */
    {.id = 68, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){82}}, .output_tensors = {1, (const int32_t[1]){88}} }, /* node_302 */
    {.id = 69, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){88}}, .output_tensors = {1, (const int32_t[1]){89}} }, /* node_303 */
    {.id = 62, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){82}}, .output_tensors = {1, (const int32_t[1]){90}} }, /* node_296 */
    {.id = 63, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){90}}, .output_tensors = {1, (const int32_t[1]){91}} }, /* node_297 */
    {.id = 88, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){91}}, .output_tensors = {1, (const int32_t[1]){92}} } /* cls_8 */
  },
  .n_nodes = 92
};
#endif

