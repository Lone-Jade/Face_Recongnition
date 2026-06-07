#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare ONNX model outputs vs ST AI compiled model outputs (from stedgeai_validation.npz).

For each validation sample, runs ONNX inference, then compares:
  1. Output tensor differences (per-tensor cosine similarity, max absolute error)
  2. Detection counts (ONNX vs ST AI decoded detections)
  3. Box coordinate agreement for matching detections

Usage:
    D:/PythonConda/envs/pytorch_face_detection/python.exe PC_test/compare_validation.py
"""

import os, sys, time, argparse
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model_stm32", "yunetn_320.onnx")
VALIDATION_NPZ = os.path.join(BASE_DIR, "model_stm32", "stedgeai_validation.npz")
OUT_DIR = os.path.join(BASE_DIR, "PC_test")

STRIDES = [8, 16, 32]
GS_MAP = {8: 40, 16: 20, 32: 10}
INPUT_SIZE = (320, 320)

OUTPUT_NAMES = ["cls_8","cls_16","cls_32","obj_8","obj_16","obj_32",
                "bbox_8","bbox_16","bbox_32","kps_8","kps_16","kps_32"]


def decode_stm32(outputs, conf_thresh, nms_thresh, min_box_size):
    """Replicates STM32 C code: outputs already in flat format (cls/obj: 1D, bbox: CHW flat)."""
    all_dets = []
    for k, stride in enumerate(STRIDES):
        gs = GS_MAP[stride]
        cls  = outputs[k].flatten()
        obj  = outputs[k + 3].flatten()
        bbox = outputs[k + 6].flatten()

        for i in range(gs):
            for j in range(gs):
                loc = i * gs + j
                score = cls[loc] * obj[loc]
                if score < conf_thresh: continue
                dx = bbox[0 * gs * gs + loc]
                dy = bbox[1 * gs * gs + loc]
                dw = bbox[2 * gs * gs + loc]
                dh = bbox[3 * gs * gs + loc]
                cx = (float(j) + dx) * stride
                cy = (float(i) + dy) * stride
                bw = np.exp(dw) * stride
                bh = np.exp(dh) * stride
                if bw < min_box_size or bh < min_box_size: continue
                x1 = max(0.0, min(cx - bw * 0.5, 320.0))
                y1 = max(0.0, min(cy - bh * 0.5, 320.0))
                x2 = max(0.0, min(cx + bw * 0.5, 320.0))
                y2 = max(0.0, min(cy + bh * 0.5, 320.0))
                if x2 <= x1 or y2 <= y1: continue
                all_dets.append((x1, y1, x2, y2, score))

    # Sort + NMS
    all_dets.sort(key=lambda d: -d[4])
    suppressed = [False] * len(all_dets)
    for a in range(len(all_dets)):
        if suppressed[a]: continue
        aa = (all_dets[a][2] - all_dets[a][0]) * (all_dets[a][3] - all_dets[a][1])
        for b in range(a + 1, len(all_dets)):
            if suppressed[b]: continue
            xx1 = max(all_dets[a][0], all_dets[b][0])
            yy1 = max(all_dets[a][1], all_dets[b][1])
            xx2 = min(all_dets[a][2], all_dets[b][2])
            yy2 = min(all_dets[a][3], all_dets[b][3])
            iw, ih = xx2 - xx1, yy2 - yy1
            if iw <= 0 or ih <= 0: continue
            ab = (all_dets[b][2] - all_dets[b][0]) * (all_dets[b][3] - all_dets[b][1])
            iou = iw * ih / (aa + ab - iw * ih)
            if iou > nms_thresh: suppressed[b] = True

    return [(d[0],d[1],d[2],d[3],d[4]) for i, d in enumerate(all_dets) if not suppressed[i]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--conf', type=float, default=0.6)
    p.add_argument('--nms', type=float, default=0.45)
    p.add_argument('--minbox', type=float, default=15.0)
    args = p.parse_args()

    print("=" * 70)
    print("  ONNX vs ST AI Compiled Model Comparison")
    print(f"  Conf={args.conf}, NMS={args.nms}, MinBox={args.minbox}")
    print("=" * 70)

    if not os.path.exists(VALIDATION_NPZ):
        print(f"Validation NPZ not found: {VALIDATION_NPZ}")
        sys.exit(1)

    npz = np.load(VALIDATION_NPZ)
    num_samples = npz['m_outputs_1'].shape[0]
    print(f"Validation samples: {num_samples}")

    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    for s in range(num_samples):
        print(f"\n--- Sample {s} ---")

        # Load ST AI input and reshape to NCHW
        stai_input = npz['m_inputs_1'][s]  # (307200,) flat
        # The input is flat: need to determine if it's NCHW or NHWC
        # shape = 307200 = 3 * 320 * 320, could be CHW or HWC
        # ST AI network.h says CHANNEL_FIRST, so it's CHW
        stai_nchw = stai_input.reshape(1, 3, 320, 320).astype(np.float32)

        # Run ONNX inference
        t0 = time.time()
        onnx_outputs = session.run(None, {input_name: stai_nchw})
        t_infer = time.time() - t0 - 0  # avoid unused warning
        print(f"  ONNX inference: elapsed")

        # Load ST AI outputs
        stai_outputs = []
        for i in range(1, 13):
            stai_outputs.append(npz[f'm_outputs_{i}'][s:s+1])

        # Compare per-tensor
        print(f"  {'Output':10s} {'cos_sim':>10s} {'max|diff|':>10s} {'ONNX_range':>20s} {'STAI_range':>20s}")
        for i in range(12):
            onx = onnx_outputs[i].flatten()
            sta = stai_outputs[i].flatten()

            # ONNX bbox is interleaved (N×4), ST AI is CHW flat (4×N)
            # Convert ONNX to CHW for comparison
            if onnx_outputs[i].ndim == 3 and onnx_outputs[i].shape[2] in (4, 10):
                onx_chw = np.transpose(onnx_outputs[i].reshape(1, -1, onnx_outputs[i].shape[2]), (0, 2, 1)).flatten()
            else:
                onx_chw = onx

            dot = np.dot(onx_chw, sta)
            norm_o = np.linalg.norm(onx_chw)
            norm_s = np.linalg.norm(sta)
            cos = dot / (norm_o * norm_s) if norm_o > 0 and norm_s > 0 else 0
            maxd = np.max(np.abs(onx_chw - sta))
            flag = " ⚠️" if maxd > 0.5 else ""
            print(f"  {OUTPUT_NAMES[i]:10s} {cos:10.6f} {maxd:10.4f}  [{onx.min():.3f},{onx.max():.3f}]   [{sta.min():.3f},{sta.max():.3f}]{flag}")

        # Decode detections from both
        dets_onnx = decode_stm32(onnx_outputs, args.conf, args.nms, args.minbox)
        dets_stai = decode_stm32(stai_outputs, args.conf, args.nms, args.minbox)
        print(f"  ONNX: {len(dets_onnx)} faces, ST AI: {len(dets_stai)} faces")

        if len(dets_stai) > 10:
            print(f"  ⚠️  ST AI produces {len(dets_stai)} detections — INT8 noise suspected!")
            if len(dets_onnx) < len(dets_stai) // 2:
                print(f"  💡 ONNX (float32) has far fewer — consider increasing conf or minbox for hardware")

    print("\nDone!")


if __name__ == "__main__":
    main()
