/**
  ******************************************************************************
  * @file           : ai_detection.h
  * @brief          : AI face detection header
  ******************************************************************************
  */

#ifndef __AI_DETECTION_H__
#define __AI_DETECTION_H__

#include <stdint.h>
#include "app_x-cube-ai.h"  /* stai_ptr */

/* YuNet anchor configuration for 320x320 input.
   Anchors are at grid CENTERS: ((j+0.5)*stride, (i+0.5)*stride) */

/* ---- SDRAM Buffer Layout (32MB @ 0xD0000000) ---- */
#define SDRAM_BASE              0xD0000000
#define LCD_FRAME_BUFFER_ADDR   0xD0000000  /* Frame buffer: 800×480×4 ≈ 1.5MB */
#define AI_IMG_BUF_ADDR         0xD0400000  /* Raw image buffer (USART receive) */
#define AI_CAM_BUF_ADDR         0xD0600000  /* Camera frame buffer */
#define AI_ACTIVATION_2_ADDR    0xD0800000  /* AI activation buf2: 2,457,600B (2.34MB).
                                               MUST be after frame buffer + camera buffer.
                                               Set in app_x-cube-ai.c data_activations[]. */
#define NUM_STRIDES       3
#define YU_STRIDES        {8, 16, 32}
#define YU_GRID_SIZES     {40, 20, 10}

/* ---- Detection Results ---- */
#define MAX_DETECTIONS    50
#define DET_THRESHOLD     0.5f    /* Score threshold (cls * obj) */
#define NMS_THRESHOLD     0.45f   /* IoU threshold for NMS */

typedef struct {
    float x1, y1, x2, y2;
    float score;
} Detection;

/* Exported arrays */
extern const int   yu_strides[3];
extern const int   yu_grid_sizes[3];

/* Exported functions */
void ai_preprocess(uint32_t *src, int src_w, int src_h,
                   float *dst, int dst_w, int dst_h);

int  ai_postprocess(stai_ptr *outputs, Detection *dets, int max_dets,
                    float threshold, float nms_threshold);

void ai_draw_detections(Detection *dets, int ndet, uint32_t *fb,
                        int fb_w, int off_x, int off_y,
                        int img_w, int img_h);

#endif /* __AI_DETECTION_H__ */
