/**
  ******************************************************************************
  * @file    onnx_infer.c
  * @brief   PC-side C program: ONNX Runtime inference + YuNet face detection.
  *
  * Build (Windows, MSVC):
  *   cl /O2 /I"%ONNXRUNTIME_DIR%\include" onnx_infer.c ^
  *      /link "%ONNXRUNTIME_DIR%\lib\onnxruntime.lib"
  *
  * Build (MinGW):
  *   gcc -O2 -I"$ONNXRUNTIME_DIR/include" onnx_infer.c -o onnx_infer.exe ^
  *       -L"$ONNXRUNTIME_DIR/lib" -lonnxruntime -lm
  *
  * Usage:
  *   onnx_infer.exe <model.onnx> <image.jpg> [conf_thresh] [nms_thresh]
  *
  * First, install ONNX Runtime:
  *   pip install onnxruntime
  *   # Headers/Libs are in site-packages/onnxruntime/
  ******************************************************************************
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ONNX Runtime C API */
#include "onnxruntime_c_api.h"

/* ---- Image helpers (minimal, no external deps) ---- */
/* We'll use stb_image for simplicity — include from single-header lib */
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

/* ---- YuNet parameters ---- */
#define INPUT_W    320
#define INPUT_H    320
#define NUM_STRIDES 3
#define MAX_DETECTIONS 100

static const int STRIDES[3]    = {8, 16, 32};
static const int GRID_SIZES[3] = {40, 20, 10};

typedef struct {
    float x1, y1, x2, y2, score;
} Detection;

/* ---- ONNX Runtime globals ---- */
static const OrtApi* ort = NULL;
static OrtEnv*        env = NULL;
static OrtSession*    session = NULL;
static OrtMemoryInfo* mem_info = NULL;

/* ---- Init ONNX Runtime ---- */
static int init_onnx(const char* model_path)
{
    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (ort == NULL) {
        fprintf(stderr, "Failed to get ONNX Runtime API\n");
        return -1;
    }

    OrtStatus* status;
    status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "yunet_test", &env);
    if (status) { fprintf(stderr, "CreateEnv: %s\n", ort->GetErrorMessage(status)); return -1; }

    OrtSessionOptions* sess_opts;
    ort->CreateSessionOptions(&sess_opts);
    ort->SetIntraOpNumThreads(sess_opts, 4);
    ort->SetSessionGraphOptimizationLevel(sess_opts, ORT_ENABLE_ALL);

    status = ort->CreateSession(env, model_path, sess_opts, &session);
    ort->ReleaseSessionOptions(sess_opts);
    if (status) { fprintf(stderr, "CreateSession: %s\n", ort->GetErrorMessage(status)); return -1; }

    ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
    printf("ONNX Runtime initialized.\n");
    return 0;
}

/* ---- Preprocess image ---- */
static float* preprocess_image(const char* img_path)
{
    int w, h, c;
    unsigned char* data = stbi_load(img_path, &w, &h, &c, 3);  /* Force BGR (OpenCV order) */
    if (!data) { fprintf(stderr, "Cannot load: %s\n", img_path); return NULL; }
    printf("Loaded: %dx%d %dch\n", w, h, c);

    /* Resize to 320x320 using nearest-neighbor (simple, matches C on embedded) */
    float* input = (float*)malloc(1 * 3 * INPUT_H * INPUT_W * sizeof(float));
    int x_ratio = ((w << 16) / INPUT_W) + 1;
    int y_ratio = ((h << 16) / INPUT_H) + 1;

    float* r_plane = input + 0 * INPUT_H * INPUT_W;  /* BGR: c0=B */
    float* g_plane = input + 1 * INPUT_H * INPUT_W;  /* c1=G */
    float* b_plane = input + 2 * INPUT_H * INPUT_W;  /* c2=R (wait, BGR → R is last) */

    for (int y = 0; y < INPUT_H; y++) {
        int sy = (y * y_ratio) >> 16;
        if (sy >= h) sy = h - 1;
        for (int x = 0; x < INPUT_W; x++) {
            int sx = (x * x_ratio) >> 16;
            if (sx >= w) sx = w - 1;
            int idx = (sy * w + sx) * 3;
            b_plane[y * INPUT_W + x] = (float)data[idx + 0];  /* B */
            g_plane[y * INPUT_W + x] = (float)data[idx + 1];  /* G */
            r_plane[y * INPUT_W + x] = (float)data[idx + 2];  /* R */
        }
    }
    stbi_image_free(data);
    printf("Preprocessed: BGR NCHW 1x3x%d×%d\n", INPUT_H, INPUT_W);
    return input;
}

/* ---- Run inference ---- */
static int run_inference(float* input, float** outputs)
{
    OrtStatus* status;
    size_t input_tensor_size = 1 * 3 * INPUT_H * INPUT_W;
    int64_t input_shape[] = {1, 3, INPUT_H, INPUT_W};

    OrtValue* input_tensor = NULL;
    status = ort->CreateTensorWithDataAsOrtValue(
        mem_info, input, input_tensor_size * sizeof(float),
        input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    if (status) { fprintf(stderr, "CreateTensor: %s\n", ort->GetErrorMessage(status)); return -1; }

    /* Get input/output names */
    OrtAllocator* allocator;
    ort->GetAllocatorWithDefaultOptions(&allocator);

    char* input_name;
    ort->SessionGetInputName(session, 0, allocator, &input_name);

    size_t num_outputs;
    ort->SessionGetOutputCount(session, &num_outputs);
    printf("Model has %zu outputs\n", num_outputs);

    char** output_names = (char**)malloc(num_outputs * sizeof(char*));
    for (size_t i = 0; i < num_outputs; i++)
        ort->SessionGetOutputName(session, (int)i, allocator, &output_names[i]);

    /* Run */
    OrtValue* output_tensors[12];
    status = ort->Run(session, NULL,
                      (const char* const*)&input_name, (const OrtValue* const*)&input_tensor, 1,
                      (const char* const*)output_names, num_outputs, output_tensors);
    if (status) { fprintf(stderr, "Run: %s\n", ort->GetErrorMessage(status)); return -1; }

    /* Copy outputs to flat float arrays */
    for (size_t i = 0; i < num_outputs; i++) {
        float* out_data;
        ort->GetTensorMutableData(output_tensors[i], (void**)&out_data);
        OrtTensorTypeAndShapeInfo* shape_info;
        ort->GetTensorTypeAndShape(output_tensors[i], &shape_info);
        size_t num_elems;
        ort->GetTensorShapeElementCount(shape_info, &num_elems);
        ort->ReleaseTensorTypeAndShapeInfo(shape_info);

        outputs[i] = (float*)malloc(num_elems * sizeof(float));
        memcpy(outputs[i], out_data, num_elems * sizeof(float));

        ort->ReleaseValue(output_tensors[i]);
    }

    /* Cleanup */
    ort->ReleaseValue(input_tensor);
    allocator->Free(allocator, input_name);
    for (size_t i = 0; i < num_outputs; i++) allocator->Free(allocator, output_names[i]);
    free(output_names);
    ort->ReleaseAllocator(allocator);

    return (int)num_outputs;
}

/* ---- YuNet decode (matches STM32 ai_detection.c) ---- */
static int decode_and_nms(float** outputs, Detection* dets, int max_dets,
                           float threshold, float nms_threshold, float min_box_size)
{
    int det_count = 0;
    int k, i, j, loc;

    for (k = 0; k < NUM_STRIDES; k++) {
        int gs = GRID_SIZES[k];
        float stride = (float)STRIDES[k];

        /* STM32 order: cls[k], obj[k+3], bbox[k+6] */
        float* cls  = outputs[k];       /* Flat: gs*gs elements */
        float* obj  = outputs[k + 3];
        float* bbox = outputs[k + 6];   /* CHW flat: 4*gs*gs elements */

        /* ONNX outputs are INTERLEAVED (N×4), need to convert to CHW */
        /* For now, access as: bbox[loc*4 + ch] */
        int is_onnx_interleaved = 1;  /* Set to 0 if using ST AI outputs */
        int bbox_channels = 4;

        for (i = 0; i < gs; i++) {
            for (j = 0; j < gs; j++) {
                loc = i * gs + j;
                float score = cls[loc] * obj[loc];
                if (score < threshold) continue;

                float dx, dy, dw, dh;
                if (is_onnx_interleaved) {
                    dx = bbox[loc * bbox_channels + 0];
                    dy = bbox[loc * bbox_channels + 1];
                    dw = bbox[loc * bbox_channels + 2];
                    dh = bbox[loc * bbox_channels + 3];
                } else {
                    dx = bbox[0 * gs * gs + loc];
                    dy = bbox[1 * gs * gs + loc];
                    dw = bbox[2 * gs * gs + loc];
                    dh = bbox[3 * gs * gs + loc];
                }

                /* Anchor at grid corner */
                float cx = ((float)j + dx) * stride;
                float cy = ((float)i + dy) * stride;
                float bw = expf(dw) * stride;
                float bh = expf(dh) * stride;

                /* Minimum box size filter */
                if (bw < min_box_size || bh < min_box_size) continue;

                float x1 = cx - bw * 0.5f, y1 = cy - bh * 0.5f;
                float x2 = cx + bw * 0.5f, y2 = cy + bh * 0.5f;
                if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
                if (x2 > INPUT_W) x2 = (float)INPUT_W;
                if (y2 > INPUT_H) y2 = (float)INPUT_H;
                if (x2 <= x1 || y2 <= y1) continue;

                if (det_count < max_dets) {
                    dets[det_count].x1 = x1; dets[det_count].y1 = y1;
                    dets[det_count].x2 = x2; dets[det_count].y2 = y2;
                    dets[det_count].score = score;
                    det_count++;
                }
            }
        }
    }

    /* Sort descending */
    for (int a = 0; a < det_count - 1; a++)
        for (int b = a + 1; b < det_count; b++)
            if (dets[b].score > dets[a].score) {
                Detection t = dets[a]; dets[a] = dets[b]; dets[b] = t;
            }

    /* NMS */
    for (int a = 0; a < det_count; a++) {
        if (dets[a].score <= 0) continue;
        float aa = (dets[a].x2 - dets[a].x1) * (dets[a].y2 - dets[a].y1);
        for (int b = a + 1; b < det_count; b++) {
            if (dets[b].score <= 0) continue;
            float xx1 = dets[a].x1 > dets[b].x1 ? dets[a].x1 : dets[b].x1;
            float yy1 = dets[a].y1 > dets[b].y1 ? dets[a].y1 : dets[b].y1;
            float xx2 = dets[a].x2 < dets[b].x2 ? dets[a].x2 : dets[b].x2;
            float yy2 = dets[a].y2 < dets[b].y2 ? dets[a].y2 : dets[b].y2;
            float iw = xx2 - xx1, ih = yy2 - yy1;
            if (iw <= 0 || ih <= 0) continue;
            float ab = (dets[b].x2 - dets[b].x1) * (dets[b].y2 - dets[b].y1);
            float iou = iw * ih / (aa + ab - iw * ih);
            if (iou > nms_threshold) dets[b].score = -1.0f;
        }
    }

    /* Compact */
    int w = 0;
    for (int a = 0; a < det_count; a++)
        if (dets[a].score > 0) { if (w != a) dets[w] = dets[a]; w++; }
    return w;
}

/* ---- Draw boxes ---- */
static void draw_boxes(unsigned char* img, int w, int h,
                        Detection* dets, int ndet, const char* out_path)
{
    for (int d = 0; d < ndet; d++) {
        float sx = (float)w / INPUT_W, sy = (float)h / INPUT_H;
        int x1 = (int)(dets[d].x1 * sx), y1 = (int)(dets[d].y1 * sy);
        int x2 = (int)(dets[d].x2 * sx), y2 = (int)(dets[d].y2 * sy);
        if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
        if (x2 >= w) x2 = w - 1; if (y2 >= h) y2 = h - 1;

        /* Green box, 3px thick */
        for (int t = 0; t < 3; t++) {
            for (int x = x1 + t; x <= x2 - t && x < w; x++) {
                if (y1 + t < h) { int idx = ((y1 + t) * w + x) * 3; img[idx]=0; img[idx+1]=255; img[idx+2]=0; }
                if (y2 - t >= 0 && y2 - t < h) { int idx = ((y2 - t) * w + x) * 3; img[idx]=0; img[idx+1]=255; img[idx+2]=0; }
            }
            for (int y = y1 + t; y <= y2 - t && y < h; y++) {
                if (x1 + t < w) { int idx = (y * w + x1 + t) * 3; img[idx]=0; img[idx+1]=255; img[idx+2]=0; }
                if (x2 - t >= 0 && x2 - t < w) { int idx = (y * w + x2 - t) * 3; img[idx]=0; img[idx+1]=255; img[idx+2]=0; }
            }
        }
    }
    stbi_write_jpg(out_path, w, h, 3, img, 90);
    printf("Saved: %s (%d faces)\n", out_path, ndet);
}

/* ---- Main ---- */
int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Usage: %s <model.onnx> <image.jpg> [conf_thresh] [nms_thresh] [min_box]\n", argv[0]);
        printf("  e.g.: %s yunetn_320.onnx face.jpg 0.6 0.45 15\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* image_path = argv[2];
    float conf_thresh = (argc > 3) ? (float)atof(argv[3]) : 0.6f;
    float nms_thresh  = (argc > 4) ? (float)atof(argv[4]) : 0.45f;
    float min_box     = (argc > 5) ? (float)atof(argv[5]) : 15.0f;

    printf("=== ONNX Runtime YuNet Face Detection ===\n");
    printf("Model: %s\nImage: %s\nConf: %.2f  NMS: %.2f  MinBox: %.0f\n\n",
           model_path, image_path, conf_thresh, nms_thresh, min_box);

    /* Init */
    clock_t t0 = clock();
    if (init_onnx(model_path) != 0) return 1;

    /* Preprocess */
    float* input = preprocess_image(image_path);
    if (!input) return 1;

    /* Inference */
    clock_t t1 = clock();
    float* outputs[12] = {NULL};
    int num_out = run_inference(input, outputs);
    if (num_out < 0) return 1;
    clock_t t2 = clock();
    printf("Inference: %.1f ms\n", (double)(t2 - t1) * 1000.0 / CLOCKS_PER_SEC);

    /* Decode + NMS */
    Detection dets[MAX_DETECTIONS];
    int ndet = decode_and_nms(outputs, dets, MAX_DETECTIONS,
                               conf_thresh, nms_thresh, min_box);
    printf("Detections: %d\n", ndet);
    for (int i = 0; i < ndet; i++)
        printf("  Face %d: score=%.4f  (%.1f, %.1f, %.1f, %.1f)\n",
               i, dets[i].score, dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2);

    /* Draw */
    int iw, ih, ic;
    unsigned char* img = stbi_load(image_path, &iw, &ih, &ic, 3);
    if (img) {
        char out_path[512];
        snprintf(out_path, sizeof(out_path), "%s_result.jpg", image_path);
        draw_boxes(img, iw, ih, dets, ndet, out_path);
        stbi_image_free(img);
    }

    /* Cleanup */
    free(input);
    for (int i = 0; i < num_out; i++) free(outputs[i]);
    ort->ReleaseMemoryInfo(mem_info);
    ort->ReleaseSession(session);
    ort->ReleaseEnv(env);

    clock_t te = clock();
    printf("\nTotal: %.1f ms\n", (double)(te - t0) * 1000.0 / CLOCKS_PER_SEC);
    return 0;
}
