/**
  ******************************************************************************
  * @file           : ai_detection.c
  * @brief          : AI face detection — preprocessing, postprocessing, drawing
  ******************************************************************************
  */

#include "ai_detection.h"
#include <math.h>
#include <stdio.h>

/* YuNet anchor configuration for 320×320 input (stride only) */
const int yu_strides[3]    = YU_STRIDES;
const int yu_grid_sizes[3] = YU_GRID_SIZES;

/* Debug flag: set to 1 to enable verbose serial output, 0 to disable.
   NOTE: renamed from AI_DEBUG to AI_DBG to avoid conflict with ST AI macro. */
#define AI_DBG 1

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
  *         Anchors at grid CORNER (j*stride, i*stride) — matching
  *         STM32 model zoo convention. Regression targets are offsets
  *         from grid top-left corner in stride units.
  *         Applies: score threshold, min-box-size filter, IoU NMS.
  *
  *         ST AI compiled output order:
  *           outputs[0..2]  = cls_8, cls_16, cls_32   (sigmoid already applied)
  *           outputs[3..5]  = obj_8, obj_16, obj_32   (sigmoid already applied)
  *           outputs[6..8]  = bbox_8, bbox_16, bbox_32 (4×H×W CHW flat)
  *           outputs[9..11] = kps_8, kps_16, kps_32   (10×H×W CHW flat, unused)
  *
  * @retval Number of valid detections after NMS.
  */
int ai_postprocess(stai_ptr *outputs, Detection *dets, int max_dets,
                   float threshold, float nms_threshold,
                   float min_box_size)
{
  int det_count = 0;
  int k, i, j, loc;
  int gs;
  float stride;
  float *cls, *obj, *bbox;

#if AI_DBG
  printf("\r\n=== AI Postprocess ===\r\n");
  printf("threshold=%.3f nms=%.3f minbox=%.1f max_dets=%d\r\n",
         threshold, nms_threshold, min_box_size, max_dets);
  printf("Output tensor pointers:\r\n");
  for (k = 0; k < 12; k++) {
    printf("  out[%d]=%p\r\n", k, (void*)outputs[k]);
  }
#endif

  for (k = 0; k < NUM_STRIDES; k++)
  {
    stride = (float)yu_strides[k];
    gs     = yu_grid_sizes[k];

    cls  = (float *)outputs[k];
    obj  = (float *)outputs[k + 3];
    bbox = (float *)outputs[k + 6];

#if AI_DBG
    /* Sample first 5 cls/obj values to check if model outputs look sane */
    printf("Stride %d (gs=%d): cls[0..4]=", (int)stride, gs);
    for (i = 0; i < 5 && i < gs*gs; i++)
      printf("%.4f ", cls[i]);
    printf(" obj[0..4]=");
    for (i = 0; i < 5 && i < gs*gs; i++)
      printf("%.4f ", obj[i]);
    printf("\r\n");
    /* Sample bbox[0..3] for loc=0 using BOTH indexing schemes */
    printf("  bbox[0,0] CHW: dx=%.4f dy=%.4f dw=%.4f dh=%.4f\r\n",
           bbox[0*gs*gs+0], bbox[1*gs*gs+0],
           bbox[2*gs*gs+0], bbox[3*gs*gs+0]);
    printf("  bbox[0,0] IL:  dx=%.4f dy=%.4f dw=%.4f dh=%.4f\r\n",
           bbox[0*4+0], bbox[0*4+1], bbox[0*4+2], bbox[0*4+3]);
#endif

    int stride_candidates = 0;
    float top_scores[5] = {0};
    for (i = 0; i < gs; i++)
    {
      for (j = 0; j < gs; j++)
      {
        loc = i * gs + j;

        float score = cls[loc] * obj[loc];
        if (score < threshold) continue;

        /* ST AI outputs bbox in interleaved format (N×4), NOT CHW (4×N).
           Verified by USART debug: IL indexing produces correct 74×140 box
           matching PC ONNX, while CHW produces wrong 23×24 box. */
        float dx = bbox[loc * 4 + 0];
        float dy = bbox[loc * 4 + 1];
        float dw = bbox[loc * 4 + 2];
        float dh = bbox[loc * 4 + 3];

        /* Anchor at grid CORNER: cx = (j + dx) * stride */
        float cx = ((float)j + dx) * stride;
        float cy = ((float)i + dy) * stride;
        float bw = expf(dw) * stride;
        float bh = expf(dh) * stride;

        /* Filter INT8 quantization noise: reject tiny boxes (< min_box_size pixels) */
        if (bw < min_box_size || bh < min_box_size) continue;

        stride_candidates++;

        /* Track top scores for debug */
        if (score > top_scores[4]) {
          top_scores[4] = score;
          int s;
          for (s = 3; s >= 0; s--) {
            if (top_scores[s] < top_scores[s+1]) {
              float tmp = top_scores[s];
              top_scores[s] = top_scores[s+1];
              top_scores[s+1] = tmp;
            }
          }
        }

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

#if AI_DBG
    printf("  Stride %d raw candidates: %d (threshold=%.3f)\r\n",
           (int)stride, stride_candidates, threshold);
    printf("  Top-5 scores: %.4f %.4f %.4f %.4f %.4f\r\n",
           top_scores[0], top_scores[1], top_scores[2],
           top_scores[3], top_scores[4]);
    /* Print bbox at the best candidate's position using both indexing schemes */
    if (stride_candidates > 0 && top_scores[0] > 0.0f) {
      /* Find the grid position of the highest-scoring candidate */
      int best_loc = -1, best_i = -1, best_j = -1;
      float best_score = 0.0f;
      for (i = 0; i < gs; i++) {
        for (j = 0; j < gs; j++) {
          loc = i * gs + j;
          float s = cls[loc] * obj[loc];
          if (s > best_score) { best_score = s; best_loc = loc; best_i = i; best_j = j; }
        }
      }
      if (best_loc >= 0) {
        printf("  Best candidate grid=(%d,%d) loc=%d score=%.4f\r\n",
               best_i, best_j, best_loc, best_score);
        printf("  bbox CHW[loc=%d]: dx=%.4f dy=%.4f dw=%.4f dh=%.4f\r\n",
               best_loc,
               bbox[0*gs*gs+best_loc], bbox[1*gs*gs+best_loc],
               bbox[2*gs*gs+best_loc], bbox[3*gs*gs+best_loc]);
        printf("  bbox IL [loc=%d]: dx=%.4f dy=%.4f dw=%.4f dh=%.4f\r\n",
               best_loc,
               bbox[best_loc*4+0], bbox[best_loc*4+1],
               bbox[best_loc*4+2], bbox[best_loc*4+3]);
        /* Show what box each would produce */
        float dcx, dcy, dbw, dbh;
        /* CHW decode */
        dcx = ((float)best_j + bbox[0*gs*gs+best_loc]) * stride;
        dcy = ((float)best_i + bbox[1*gs*gs+best_loc]) * stride;
        dbw = expf(bbox[2*gs*gs+best_loc]) * stride;
        dbh = expf(bbox[3*gs*gs+best_loc]) * stride;
        printf("  -> CHW box: (%.0f,%.0f)-(%.0f,%.0f) %.0fx%.0f\r\n",
               dcx-dbw*0.5f, dcy-dbh*0.5f, dcx+dbw*0.5f, dcy+dbh*0.5f, dbw, dbh);
        /* Interleaved decode */
        dcx = ((float)best_j + bbox[best_loc*4+0]) * stride;
        dcy = ((float)best_i + bbox[best_loc*4+1]) * stride;
        dbw = expf(bbox[best_loc*4+2]) * stride;
        dbh = expf(bbox[best_loc*4+3]) * stride;
        printf("  -> IL  box: (%.0f,%.0f)-(%.0f,%.0f) %.0fx%.0f\r\n",
               dcx-dbw*0.5f, dcy-dbh*0.5f, dcx+dbw*0.5f, dcy+dbh*0.5f, dbw, dbh);
      }
    }
#endif
  }

#if AI_DBG
  printf("Total raw candidates (all strides): %d\r\n", det_count);
#endif

  /* ---------- Sort by score (descending) ---------- */
  if (det_count > 1)
  {
    int a, b;

#if AI_DBG
    {
      int show_n = det_count < 5 ? det_count : 5;
      int ta, tb, ti;
      /* Simple partial sort on the first 'show_n' elements to find top scores */
      for (ta = 0; ta < show_n - 1; ta++)
        for (tb = ta + 1; tb < show_n; tb++)
          if (dets[tb].score > dets[ta].score) {
            Detection t = dets[ta]; dets[ta] = dets[tb]; dets[tb] = t;
          }
      printf("Sorting %d raw detections...\r\n", det_count);
      printf("Top-5 raw detections (pre-NMS):\r\n");
      for (ti = 0; ti < show_n; ti++)
        printf("  #%d: score=%.4f box=(%.1f,%.1f,%.1f,%.1f) size=%.0fx%.0f\r\n",
               ti, dets[ti].score,
               dets[ti].x1, dets[ti].y1, dets[ti].x2, dets[ti].y2,
               dets[ti].x2-dets[ti].x1, dets[ti].y2-dets[ti].y1);
    }
#endif

    for (a = 0; a < det_count - 1; a++)
      for (b = a + 1; b < det_count; b++)
        if (dets[b].score > dets[a].score) {
          Detection t = dets[a]; dets[a] = dets[b]; dets[b] = t;
        }

    /* ---------- NMS ---------- */
#if AI_DBG
    int nms_suppressed = 0;
#endif
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
        /* Containment filter: if >75% of the smaller box is overlapped,
           suppress regardless of IoU. Catches small boxes inside large ones. */
        float smaller_area = (area_a < area_b) ? area_a : area_b;
        if (iou > nms_threshold || inter > 0.75f * smaller_area) {
          dets[b].score = -1.0f;
#if AI_DBG
          nms_suppressed++;
#endif
        }
      }
    }

/* ---------- Compact ---------- */
    { int wi = 0;
    for (a = 0; a < det_count; a++)
      if (dets[a].score > 0.0f) {
        if (wi != a) dets[wi] = dets[a];
        wi++;
      }

#if AI_DBG
    printf("NMS: suppressed %d, remaining %d\r\n", nms_suppressed, wi);
#endif
    det_count = wi;
    }
  }
#if AI_DBG
  else {
    printf("Only %d raw detections (no sorting/NMS needed)\r\n", det_count);
  }
#endif

#if AI_DBG
  {
    int di;
    printf("=== Final detections: %d ===\r\n", det_count);
    for (di = 0; di < det_count; di++) {
      printf("  #%d: score=%.4f box=(%.1f,%.1f)-(%.1f,%.1f) w=%.0f h=%.0f\r\n",
             di, dets[di].score,
             dets[di].x1, dets[di].y1, dets[di].x2, dets[di].y2,
             dets[di].x2-dets[di].x1, dets[di].y2-dets[di].y1);
    }
  }
#endif

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
  float scale_x = (float)img_w / 320.0f;  /* Model input was 320×320 */
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
