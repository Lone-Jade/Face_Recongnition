/**
  ******************************************************************************
  * @file           : ai_detection.h
  * @brief          : AI face detection - preprocessing, postprocessing, drawing
  ******************************************************************************
  */

#ifndef __AI_DETECTION_H__
#define __AI_DETECTION_H__

#include <stdint.h>
#include "app_x-cube-ai.h"  /* stai_ptr */

/* ---- SDRAM Buffer Layout (32MB @ 0xD0000000) ---- */
#define SDRAM_BASE              0xD0000000
#define LCD_FRAME_BUFFER_ADDR   0xD0000000  /* Frame buffer: 800x480x4 ¡Ö 1.5MB */
#define AI_IMG_BUF_ADDR         0xD0400000  /* Raw image buffer (USART receive) */
#define AI_CAM_BUF_ADDR         0xD0600000  /* Camera frame buffer (future) */
#define AI_ACTIVATION_2_ADDR    0xD0800000  /* AI activation buf2: 2,457,600B.
                                               Must match data_activations[]
                                               in AI/App/app_x-cube-ai.c. */
#define NUM_STRIDES       3
#define YU_STRIDES        {8, 16, 32}
#define YU_GRID_SIZES     {40, 20, 10}

/* ---- Detection Results ----
   Tuned for ST AI INT8 model. Float32 ONNX may use lower thresholds.
   INT8 quantization noise creates many tiny (7-18px) false positives
   that require min-box filtering + higher confidence threshold. */
#define MAX_DETECTIONS    50
#define DET_THRESHOLD     0.40f    /* Score threshold - lower to catch smaller/blurrier faces */
#define NMS_THRESHOLD     0.40f    /* IoU threshold for NMS */
#define MIN_BOX_SIZE      10.0f    /* Min box width/height in 320x320 input pixels */

typedef struct {
    float x1, y1, x2, y2;
    float score;
} Detection;

/* Exported arrays */
extern const int   yu_strides[3];
extern const int   yu_grid_sizes[3];

/* Exported functions */

/**
  * @brief  Convert ARGB8888 image to float32 NCHW tensor.
  *         Model expects BGR channel order with CHANNEL_FIRST layout:
  *         dst[c][y][x] = dst[c*H*W + y*W + x]
  *         c=0->Blue, c=1->Green, c=2->Red.  Values in [0, 255], no rescale.
  * @param  src    Pointer to ARGB8888 image (src_w x src_h)
  * @param  src_w  Source width
  * @param  src_h  Source height
  * @param  dst    Pointer to float32 NCHW tensor (3 x dst_h x dst_w)
  * @param  dst_w  Model input width (320 for YuNet-320)
  * @param  dst_h  Model input height (320 for YuNet-320)
  */
void ai_preprocess(uint32_t *src, int src_w, int src_h,
                   float *dst, int dst_w, int dst_h);

/**
  * @brief  Post-process YuNet 12-output tensors into detected faces.
  *         Scores = cls * obj (both already sigmoid-ed by model).
  *         Applies score threshold + IoU NMS.
  * @param  outputs       Array of 12 output tensor pointers (from stai_output[])
  * @param  dets          Output detection array
  * @param  max_dets      Max number of detections
  * @param  threshold     Score threshold (cls * obj)
  * @param  nms_threshold IoU threshold for NMS
  * @retval Number of valid detections after NMS.
  */
int  ai_postprocess(stai_ptr *outputs, Detection *dets, int max_dets,
                    float threshold, float nms_threshold,
                    float min_box_size);

/**
  * @brief  Draw green bounding boxes on ARGB8888 framebuffer.
  * @param  dets / ndet  Detection array and its size
  * @param  fb / fb_w    Frame buffer base and width (usually 800)
  * @param  off_x / off_y  Offset of image region within framebuffer
  * @param  img_w / img_h  Display size of the image (for coordinate scaling)
  */
void ai_draw_detections(Detection *dets, int ndet, uint32_t *fb,
                        int fb_w, int off_x, int off_y,
                        int img_w, int img_h);

#endif /* __AI_DETECTION_H__ */
