/**
  ******************************************************************************
  * @file           : ai_detection.c
  * @brief          : AI face detection - preprocessing, postprocessing, drawing
  ******************************************************************************
  */

#include "ai_detection.h"
#include <math.h>

/* YuNet anchor configuration for 320x320 input (stride only) */
const int yu_strides[3]    = YU_STRIDES;
const int yu_grid_sizes[3] = YU_GRID_SIZES;

/* ========== Preprocessing ========== */

/**
  * @brief  Convert ARGB8888 image to float32 NCHW tensor (1, 3, H, W).
  *         Model expects BGR channel order, values in [0, 255].
  *         Resizes by integer-ratio pixel sampling.
  *
  *         IMPORTANT: The model uses STAI_FLAG_CHANNEL_FIRST, meaning
  *         data layout MUST be NCHW: input[c][y][x] = dst[c*H*W + y*W + x].
  *         Channel order is BGR (c=0→Blue, c=1→Green, c=2→Red).
  */
void ai_preprocess(uint32_t *src, int src_w, int src_h,
                   float *dst, int dst_w, int dst_h)
{
  int c, y, x, sy, sx;
  uint8_t r, g, b;
  int x_ratio = ((src_w << 16) / dst_w) + 1;
  int y_ratio = ((src_h << 16) / dst_h) + 1;
  int plane_size = dst_h * dst_w;  /* Size of one channel plane */

  for (c = 0; c < 3; c++)
  {
    float *dst_c = dst + c * plane_size;  /* NCHW: point to channel c start */
    for (y = 0; y < dst_h; y++)
    {
      sy = (y * y_ratio) >> 16;
      if (sy >= src_h) sy = src_h - 1;
      for (x = 0; x < dst_w; x++)
      {
        sx = (x * x_ratio) >> 16;
        if (sx >= src_w) sx = src_w - 1;

        uint32_t pixel = src[sy * src_w + sx];
        b = pixel & 0xFF;
        g = (pixel >> 8) & 0xFF;
        r = (pixel >> 16) & 0xFF;

        /* BGR channel order, values in [0, 255] — no normalization */
        float val;
        if (c == 0)      val = (float)b;
        else if (c == 1) val = (float)g;
        else             val = (float)r;

        dst_c[y * dst_w + x] = val;  /* NCHW: channel-first, planar BGR */
      }
    }
  }
}

/* ========== Postprocessing: YuNet output decode + NMS ========== */

/**
  * @brief  Decode YuNet 12-output tensors into face Detection structs.
  *         Anchors are at grid CENTERS ((j+0.5)*stride, (i+0.5)*stride),
  *         matching standard YuNet convention. Stride is used for both
  *         anchor center positioning and box size decoding.
  *         Applies score threshold (cls * obj) and IoU-based NMS.
  * @retval Number of valid detections.
  */
int ai_postprocess(stai_ptr *outputs, Detection *dets, int max_dets,
                   float threshold, float nms_threshold)
{
  int det_count = 0;
  int k, i, j, loc;
  int gs;
  float stride;
  float *cls, *obj, *bbox;

  for (k = 0; k < NUM_STRIDES; k++)
  {
    stride = (float)yu_strides[k];
    gs     = yu_grid_sizes[k];

    /* cls=k, obj=k+3, bbox=k+6
       ST AI compiled output order:
         outputs[0..2]  = cls_8, cls_16, cls_32   (sigmoid applied by model)
         outputs[3..5]  = obj_8, obj_16, obj_32   (sigmoid applied by model)
         outputs[6..8]  = bbox_8, bbox_16, bbox_32 (4 x H x W in CHW layout)
         outputs[9..11] = kps_8, kps_16, kps_32   (10 x H x W, unused here)
    */
    cls  = (float *)outputs[k];
    obj  = (float *)outputs[k + 3];
    bbox = (float *)outputs[k + 6];

    for (i = 0; i < gs; i++)
    {
      for (j = 0; j < gs; j++)
      {
        loc = i * gs + j;

        /* Score = cls * obj (both already sigmoid-ed by model) */
        float score = cls[loc] * obj[loc];
        if (score < threshold) continue;

        /* Bbox regression (CHW layout: c0=dx, c1=dy, c2=dw, c3=dh) */
        float dx = bbox[0 * gs * gs + loc];
        float dy = bbox[1 * gs * gs + loc];
        float dw = bbox[2 * gs * gs + loc];
        float dh = bbox[3 * gs * gs + loc];

        /* Decode: anchor at grid CENTER ((j+0.5)*stride, (i+0.5)*stride).
           Standard YuNet convention: regression targets are relative
           to anchor center, scaled by stride.
           bbox channels (CHW): c0=dx, c1=dy, c2=dw, c3=dh */
        float cx = ((float)j + 0.5f + dx) * stride;
        float cy = ((float)i + 0.5f + dy) * stride;
        float bw = expf(dw) * stride;
        float bh = expf(dh) * stride;

        /* Convert from [cx, cy, w, h] to [x1, y1, x2, y2] */
        float x1 = cx - bw * 0.5f;
        float y1 = cy - bh * 0.5f;
        float x2 = cx + bw * 0.5f;
        float y2 = cy + bh * 0.5f;

        /* Clip to image boundaries [0, 320] */
        if (x1 < 0.0f) x1 = 0.0f;
        if (y1 < 0.0f) y1 = 0.0f;
        if (x2 > 320.0f) x2 = 320.0f;
        if (y2 > 320.0f) y2 = 320.0f;

        if (x2 <= x1 || y2 <= y1) continue;

        if (det_count < max_dets)
        {
          dets[det_count].x1    = x1;
          dets[det_count].y1    = y1;
          dets[det_count].x2    = x2;
          dets[det_count].y2    = y2;
          dets[det_count].score = score;
          det_count++;
        }
      }
    }
  }

  /* ---------- Sort by score (descending, bubble sort) ---------- */
  if (det_count > 1)
  {
    int a, b;
    for (a = 0; a < det_count - 1; a++)
      for (b = a + 1; b < det_count; b++)
        if (dets[b].score > dets[a].score) {
          Detection t = dets[a]; dets[a] = dets[b]; dets[b] = t;
        }

    /* ---------- NMS ---------- */
    for (a = 0; a < det_count; a++)
    {
      if (dets[a].score <= 0.0f) continue;
      float area_a = (dets[a].x2 - dets[a].x1) * (dets[a].y2 - dets[a].y1);
      for (b = a + 1; b < det_count; b++)
      {
        if (dets[b].score <= 0.0f) continue;
        float xx1 = dets[a].x1 > dets[b].x1 ? dets[a].x1 : dets[b].x1;
        float yy1 = dets[a].y1 > dets[b].y1 ? dets[a].y1 : dets[b].y1;
        float xx2 = dets[a].x2 < dets[b].x2 ? dets[a].x2 : dets[b].x2;
        float yy2 = dets[a].y2 < dets[b].y2 ? dets[a].y2 : dets[b].y2;
        float iw = xx2 - xx1;
        float ih = yy2 - yy1;
        if (iw <= 0.0f || ih <= 0.0f) continue;
        float inter = iw * ih;
        float area_b = (dets[b].x2 - dets[b].x1) * (dets[b].y2 - dets[b].y1);
        float iou = inter / (area_a + area_b - inter);
        if (iou > nms_threshold) dets[b].score = -1.0f;
      }
    }

    /* ---------- Compact ---------- */
    int wi = 0;
    for (a = 0; a < det_count; a++)
      if (dets[a].score > 0.0f) {
        if (wi != a) dets[wi] = dets[a];
        wi++;
      }
    det_count = wi;
  }

  return det_count;
}

/* ========== Drawing ========== */

/**
  * @brief  Draw green bounding boxes for detected faces on ARGB8888 framebuffer.
  * @param  dets / ndet: Detection array and its size
  * @param  fb / fb_w: Frame buffer base and width in pixels (800)
  * @param  off_x / off_y: Offset of image region within framebuffer
  * @param  img_w / img_h: Display size of the image (for coordinate scaling)
  */
void ai_draw_detections(Detection *dets, int ndet, uint32_t *fb,
                        int fb_w, int off_x, int off_y,
                        int img_w, int img_h)
{
  int d, t, x, y;
  float scale_x = (float)img_w / 320.0f;  /* Model input was 320x320 */
  float scale_y = (float)img_h / 320.0f;

  for (d = 0; d < ndet; d++)
  {
    int x1 = (int)(dets[d].x1 * scale_x) + off_x;
    int y1 = (int)(dets[d].y1 * scale_y) + off_y;
    int x2 = (int)(dets[d].x2 * scale_x) + off_x;
    int y2 = (int)(dets[d].y2 * scale_y) + off_y;

    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 >= fb_w) x2 = fb_w - 1;
    if (y2 >= 480) y2 = 479;

    uint32_t green = 0xFF00FF00;  /* ARGB8888 Green */

    for (t = 0; t < 2; t++)  /* 2-pixel thick */
    {
      int bx1 = x1 + t, by1 = y1 + t;
      int bx2 = x2 - t, by2 = y2 - t;
      if (bx2 <= bx1 || by2 <= by1) break;

      /* Top & bottom edges */
      for (x = bx1; x <= bx2; x++) {
        fb[by1 * fb_w + x] = green;
        fb[by2 * fb_w + x] = green;
      }
      /* Left & right edges */
      for (y = by1; y <= by2; y++) {
        fb[y * fb_w + bx1] = green;
        fb[y * fb_w + bx2] = green;
      }
    }
  }
}
