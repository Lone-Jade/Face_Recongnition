#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare STM32 preprocessing vs bilinear vs letterbox for face detection.

This test uses the EXACT same ARGB8888 pixel data from the built-in C header
files to replicate STM32 preprocessing in Python, then runs ONNX inference
to compare detection results.

Usage:
    python PC_test/compare_preprocessing.py
"""

import os, sys, time
import numpy as np
import onnxruntime as ort
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model_stm32", "yunetn_320.onnx")

STRIDES = [8, 16, 32]
GS_MAP = {8: 40, 16: 20, 32: 10}


def parse_c_header_array(filepath):
    """Parse uint32_t face_img_X[N] = { ... } from a C header file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # Find the array content between { and }
    match = re.search(r'\{([^}]+)\}', content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find array in {filepath}")
    # Extract hex values
    hex_vals = re.findall(r'0x[0-9A-Fa-f]{8}', match.group(1))
    data = np.array([int(v, 16) for v in hex_vals], dtype=np.uint32)
    return data


def argb8888_to_bgr_nchw_320x320_stm32(pixels, src_w=320, src_h=240):
    """
    Replicate STM32 ai_preprocess() EXACTLY:
    - Integer-ratio nearest-neighbor resize to 320x320
    - BGR channel order, NCHW layout, values in [0, 255]
    """
    dst_w, dst_h = 320, 320
    x_ratio = ((src_w << 16) // dst_w) + 1
    y_ratio = ((src_h << 16) // dst_h) + 1

    # Reshape pixels to 2D for nearest-neighbor sampling
    img_2d = pixels.reshape(src_h, src_w)

    # Build destination image (HWC, BGR, uint8)
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.float32)

    for y in range(dst_h):
        sy = (y * y_ratio) >> 16
        if sy >= src_h:
            sy = src_h - 1
        for x in range(dst_w):
            sx = (x * x_ratio) >> 16
            if sx >= src_w:
                sx = src_w - 1

            p = img_2d[sy, sx]
            b = p & 0xFF
            g = (p >> 8) & 0xFF
            r = (p >> 16) & 0xFF
            dst_img[y, x, 0] = float(b)
            dst_img[y, x, 1] = float(g)
            dst_img[y, x, 2] = float(r)

    # Convert to NCHW (channel-first): (1, 3, 320, 320)
    nchw = np.transpose(dst_img, (2, 0, 1))
    nchw = np.expand_dims(nchw, axis=0).astype(np.float32)
    return nchw


def argb8888_to_bgr_nchw_320x320_letterbox(pixels, src_w=320, src_h=240):
    """
    Letterbox: 1:1 pixel mapping for top 240 rows, pad bottom 80 rows with 0.
    Preserves face aspect ratio (no vertical stretching).
    """
    dst_w, dst_h = 320, 320

    img_2d = pixels.reshape(src_h, src_w)
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.float32)

    for y in range(src_h):       # Only map source rows
        for x in range(dst_w):
            p = img_2d[y, x]
            b = p & 0xFF
            g = (p >> 8) & 0xFF
            r = (p >> 16) & 0xFF
            dst_img[y, x, 0] = float(b)
            dst_img[y, x, 1] = float(g)
            dst_img[y, x, 2] = float(r)
    # Rows 240..319 remain 0 (black padding)

    nchw = np.transpose(dst_img, (2, 0, 1))
    nchw = np.expand_dims(nchw, axis=0).astype(np.float32)
    return nchw


def decode_stm32(onnx_outputs, conf_thresh, nms_thresh, min_box_size):
    """STM32 C logic faithfully replicated."""
    outputs = list(onnx_outputs)
    for i in range(12):
        arr = outputs[i]
        if arr.ndim == 3 and arr.shape[2] in (4, 10):
            outputs[i] = np.transpose(arr, (0, 2, 1))

    all_dets = []
    for k, stride in enumerate(STRIDES):
        gs = GS_MAP[stride]
        cls = outputs[k].flatten()
        obj = outputs[k + 3].flatten()
        bbox = outputs[k + 6].flatten()

        for i in range(gs):
            for j in range(gs):
                loc = i * gs + j
                score = cls[loc] * obj[loc]
                if score < conf_thresh:
                    continue

                dx = bbox[0 * gs * gs + loc]
                dy = bbox[1 * gs * gs + loc]
                dw = bbox[2 * gs * gs + loc]
                dh = bbox[3 * gs * gs + loc]

                cx = (float(j) + dx) * stride
                cy = (float(i) + dy) * stride
                bw = np.exp(dw) * stride
                bh = np.exp(dh) * stride

                if bw < min_box_size or bh < min_box_size:
                    continue

                x1 = max(0.0, min(cx - bw * 0.5, 320.0))
                y1 = max(0.0, min(cy - bh * 0.5, 320.0))
                x2 = max(0.0, min(cx + bw * 0.5, 320.0))
                y2 = max(0.0, min(cy + bh * 0.5, 320.0))
                if x2 <= x1 or y2 <= y1:
                    continue

                all_dets.append((x1, y1, x2, y2, score))

    # Sort + NMS
    all_dets.sort(key=lambda d: -d[4])
    suppressed = [False] * len(all_dets)
    for a in range(len(all_dets)):
        if suppressed[a]:
            continue
        aa = (all_dets[a][2] - all_dets[a][0]) * (all_dets[a][3] - all_dets[a][1])
        for b in range(a + 1, len(all_dets)):
            if suppressed[b]:
                continue
            xx1 = max(all_dets[a][0], all_dets[b][0])
            yy1 = max(all_dets[a][1], all_dets[b][1])
            xx2 = min(all_dets[a][2], all_dets[b][2])
            yy2 = min(all_dets[a][3], all_dets[b][3])
            iw, ih = xx2 - xx1, yy2 - yy1
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            ab = (all_dets[b][2] - all_dets[b][0]) * (all_dets[b][3] - all_dets[b][1])
            smaller = min(aa, ab)
            iou = inter / (aa + ab - inter)
            if iou > nms_thresh or inter > 0.75 * smaller:
                suppressed[b] = True

    return [all_dets[i] for i in range(len(all_dets)) if not suppressed[i]]


def main():
    conf = 0.7
    nms_thresh = 0.3
    minbox = 15.0

    # Find header files
    img0_path = os.path.join(BASE_DIR, "BSP", "Inc", "face_img_0.h")
    img1_path = os.path.join(BASE_DIR, "BSP", "Inc", "face_img_1.h")

    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    for img_idx, img_path in enumerate([img0_path, img1_path]):
        print(f"\n{'='*70}")
        print(f"  face_img_{img_idx}: {img_path}")
        print(f"{'='*70}")

        pixels = parse_c_header_array(img_path)
        print(f"  Pixels loaded: {len(pixels)} (320x240={76800})")
        print(f"  First 5 pix: {[f'0x{p:08X}' for p in pixels[:5]]}")

        # --- Method 1: STM32 nearest-neighbor ---
        t0 = time.time()
        input_stm32 = argb8888_to_bgr_nchw_320x320_stm32(pixels)
        outputs_stm32 = session.run(None, {input_name: input_stm32})
        t1 = time.time()
        dets_stm32 = decode_stm32(outputs_stm32, conf, nms_thresh, minbox)
        print(f"\n  [STM32 nearest-neighbor] inference: {(t1-t0)*1000:.1f}ms, detections: {len(dets_stm32)}")
        for d in dets_stm32:
            print(f"    score={d[4]:.4f}  box=({d[0]:.0f},{d[1]:.0f})-({d[2]:.0f},{d[3]:.0f})  w={d[2]-d[0]:.0f} h={d[3]-d[1]:.0f}")

        # --- Method 2: Letterbox ---
        t0 = time.time()
        input_lb = argb8888_to_bgr_nchw_320x320_letterbox(pixels)
        outputs_lb = session.run(None, {input_name: input_lb})
        t1 = time.time()
        dets_lb = decode_stm32(outputs_lb, conf, nms_thresh, minbox)
        print(f"\n  [Letterbox 1:1]      inference: {(t1-t0)*1000:.1f}ms, detections: {len(dets_lb)}")
        for d in dets_lb:
            print(f"    score={d[4]:.4f}  box=({d[0]:.0f},{d[1]:.0f})-({d[2]:.0f},{d[3]:.0f})  w={d[2]-d[0]:.0f} h={d[3]-d[1]:.0f}")

        # --- Compare: are the model outputs different? ---
        print(f"\n  --- Output comparison: STM32-NN vs Letterbox ---")
        for i in range(12):
            diff = np.max(np.abs(outputs_stm32[i] - outputs_lb[i]))
            cos = np.dot(outputs_stm32[i].flatten(), outputs_lb[i].flatten()) / (
                np.linalg.norm(outputs_stm32[i].flatten()) * np.linalg.norm(outputs_lb[i].flatten()) + 1e-8
            )
            flag = " *** DIFFERENT ***" if diff > 0.1 else ""
            print(f"    out[{i:2d}]: max|diff|={diff:.6f}  cos_sim={cos:.6f}{flag}")

        # --- Also show input tensor differences ---
        diff_input = np.max(np.abs(input_stm32 - input_lb))
        print(f"\n  Input tensor max|diff| (STM32-NN vs Letterbox): {diff_input:.2f}")
        # Sample middle region of input
        mid_y = 160
        print(f"  Input row {mid_y} (B channel, x=140..160):")
        print(f"    STM32-NN: {input_stm32[0,0,mid_y,140:160]}")
        print(f"    Letterbox: {input_lb[0,0,mid_y,140:160]}")

    print("\nDone!")


if __name__ == "__main__":
    main()
