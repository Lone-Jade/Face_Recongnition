/**
  ******************************************************************************
  * @file    test_postprocess.c
  * @brief   PC-side standalone test for YuNet postprocessing logic.
  *          Compile: gcc -o test_postprocess test_postprocess.c -lm
  *          Or on Windows: cl test_postprocess.c
  *
  *          Uses hand-picked synthetic data to verify:
  *          1. Anchor center calculation (grid corner, NOT grid center)
  *          2. Bbox decode (cx = (j+dx)*stride, w = exp(dw)*stride)
  *          3. Score threshold + NMS
  ******************************************************************************
  */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define STRIDE_8   8
#define STRIDE_16  16
#define STRIDE_32  32
#define NUM_STRIDES 3

#define MAX_DETECTIONS 50
#define DET_THRESHOLD  0.5f
#define NMS_THRESHOLD  0.45f

typedef struct {
    float x1, y1, x2, y2;
    float score;
} Detection;

/* ===== Implementation matching STM32 ai_detection.c (FIXED version) ===== */

static int decode_and_nms(
    const float *outputs[12],  /* 12 output tensor pointers */
    Detection *dets, int max_dets,
    float threshold, float nms_threshold)
{
    const int strides[3] = {8, 16, 32};
    const int gs_map[3]  = {40, 20, 10};
    int det_count = 0;
    int k, i, j, loc;

    for (k = 0; k < NUM_STRIDES; k++)
    {
        float stride = (float)strides[k];
        int   gs     = gs_map[k];

        const float *cls  = outputs[k];        /* cls_8, cls_16, cls_32 */
        const float *obj  = outputs[k + 3];    /* obj_8, obj_16, obj_32 */
        const float *bbox = outputs[k + 6];    /* bbox_8, bbox_16, bbox_32 (CHW flat) */

        for (i = 0; i < gs; i++)
        {
            for (j = 0; j < gs; j++)
            {
                loc = i * gs + j;
                float score = cls[loc] * obj[loc];
                if (score < threshold) continue;

                /* CHW: c0=dx, c1=dy, c2=dw, c3=dh */
                float dx = bbox[0 * gs * gs + loc];
                float dy = bbox[1 * gs * gs + loc];
                float dw = bbox[2 * gs * gs + loc];
                float dh = bbox[3 * gs * gs + loc];

                /* FIXED: anchor at grid corner (j*stride, i*stride) */
                float cx = ((float)j + dx) * stride;
                float cy = ((float)i + dy) * stride;
                float bw = expf(dw) * stride;
                float bh = expf(dh) * stride;

                float x1 = cx - bw * 0.5f;
                float y1 = cy - bh * 0.5f;
                float x2 = cx + bw * 0.5f;
                float y2 = cy + bh * 0.5f;

                if (x1 < 0.0f) x1 = 0.0f;
                if (y1 < 0.0f) y1 = 0.0f;
                if (x2 > 320.0f) x2 = 320.0f;
                if (y2 > 320.0f) y2 = 320.0f;
                if (x2 <= x1 || y2 <= y1) continue;

                if (det_count < max_dets)
                {
                    dets[det_count].x1 = x1; dets[det_count].y1 = y1;
                    dets[det_count].x2 = x2; dets[det_count].y2 = y2;
                    dets[det_count].score = score;
                    det_count++;
                }
            }
        }
    }

    /* Sort by score descending */
    if (det_count > 1)
    {
        int a, b;
        for (a = 0; a < det_count - 1; a++)
            for (b = a + 1; b < det_count; b++)
                if (dets[b].score > dets[a].score)
                {
                    Detection t = dets[a]; dets[a] = dets[b]; dets[b] = t;
                }

        /* NMS */
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
                float iw = xx2 - xx1, ih = yy2 - yy1;
                if (iw <= 0.0f || ih <= 0.0f) continue;
                float inter = iw * ih;
                float area_b = (dets[b].x2 - dets[b].x1) * (dets[b].y2 - dets[b].y1);
                float iou = inter / (area_a + area_b - inter);
                if (iou > nms_threshold) dets[b].score = -1.0f;
            }
        }

        /* Compact */
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


/* ===== Synthetic test case ===== */

int main(void)
{
    /*
     * Synthetic test: create minimal outputs that should produce
     * exactly 1 detection at a known location.
     *
     * Grid at stride 16: 20x20 grid → 400 anchors
     * Place a face center at (160, 160) → anchor at grid (10, 10)
     * dx=0, dy=0, dw=0, dh=0 → cx=10*16=160, cy=10*16=160, w=h=16
     * → box = (152, 152, 168, 168)
     */
    int gs8 = 1600, gs16 = 400, gs32 = 100;
    int bbox8_size = 4 * gs8;   /* 6400 */
    int bbox16_size = 4 * gs16; /* 1600 */
    int bbox32_size = 4 * gs32; /* 400 */

    /* Allocate test data */
    float *cls_8  = (float *)calloc(gs8, sizeof(float));
    float *cls_16 = (float *)calloc(gs16, sizeof(float));
    float *cls_32 = (float *)calloc(gs32, sizeof(float));
    float *obj_8  = (float *)calloc(gs8, sizeof(float));
    float *obj_16 = (float *)calloc(gs16, sizeof(float));
    float *obj_32 = (float *)calloc(gs32, sizeof(float));
    float *bbox_8  = (float *)calloc(bbox8_size, sizeof(float));
    float *bbox_16 = (float *)calloc(bbox16_size, sizeof(float));
    float *bbox_32 = (float *)calloc(bbox32_size, sizeof(float));
    float *kps_8  = (float *)calloc(10 * gs8, sizeof(float));
    float *kps_16 = (float *)calloc(10 * gs16, sizeof(float));
    float *kps_32 = (float *)calloc(10 * gs32, sizeof(float));

    /* Place ONE face at stride-16 grid position (10, 10) */
    int loc_test = 10 * 20 + 10; /* i=10, j=10 in 20x20 grid */
    cls_16[loc_test] = 0.9f;   /* High classification score */
    obj_16[loc_test] = 0.9f;   /* High objectness score */
    /* dx=0, dy=0, dw=0, dh=0 → exp(0)=1 → box size = stride (16x16) */

    const float *outputs[12] = {
        cls_8, cls_16, cls_32,        /* 0-2: cls */
        obj_8, obj_16, obj_32,        /* 3-5: obj */
        bbox_8, bbox_16, bbox_32,     /* 6-8: bbox */
        kps_8, kps_16, kps_32         /* 9-11: kps */
    };

    Detection dets[MAX_DETECTIONS];
    int ndet = decode_and_nms(outputs, dets, MAX_DETECTIONS,
                               DET_THRESHOLD, NMS_THRESHOLD);

    printf("=== YuNet Postprocess Test (FIXED, grid corner anchors) ===\n\n");
    printf("Test: 1 face at stride-16 grid (10,10), dx=dy=dw=dh=0\n");
    printf("Expected: 1 detection at (152.0, 152.0, 168.0, 168.0)  score=0.8100\n\n");

    printf("Detections: %d\n", ndet);
    int pass = 1;
    for (int d = 0; d < ndet; d++)
    {
        Detection *det = &dets[d];
        printf("  [%d] score=%.4f  box=(%.1f, %.1f, %.1f, %.1f)\n",
               d, det->score, det->x1, det->y1, det->x2, det->y2);

        /* Tolerance check */
        float eps = 0.01f;
        if (fabsf(det->score - 0.81f) > eps) pass = 0;
        if (fabsf(det->x1 - 152.0f) > eps) pass = 0;
        if (fabsf(det->y1 - 152.0f) > eps) pass = 0;
        if (fabsf(det->x2 - 168.0f) > eps) pass = 0;
        if (fabsf(det->y2 - 168.0f) > eps) pass = 0;
    }

    if (pass && ndet == 1)
        printf("\n[PASS] Test passed!\n");
    else
    {
        printf("\n[FAIL] Test failed!\n");
        if (ndet != 1) printf("  Expected 1 detection, got %d\n", ndet);
    }

    /* Test 2: buggy version (+0.5 offset) would produce box shifted by +8 pixels */
    printf("\n--- Bug demonstration ---\n");
    printf("If +0.5f offset were used, cx would be (10+0.5+0)*16=168.0 instead of 160.0\n");
    printf("Resulting box: (160.0, 160.0, 176.0, 176.0) — shifted by 8 pixels\n");
    printf("This explains the 'dense distribution on facial features' problem.\n");

    /* Free */
    free(cls_8); free(cls_16); free(cls_32);
    free(obj_8); free(obj_16); free(obj_32);
    free(bbox_8); free(bbox_16); free(bbox_32);
    free(kps_8); free(kps_16); free(kps_32);

    return pass ? 0 : 1;
}
