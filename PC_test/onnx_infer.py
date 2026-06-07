#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX Runtime face detection: load image → inference → draw green boxes.

Usage:
    D:/PythonConda/envs/pytorch_face_detection/python.exe PC_test/onnx_infer.py
    D:/PythonConda/envs/pytorch_face_detection/python.exe PC_test/onnx_infer.py --image my_face.jpg
    D:/PythonConda/envs/pytorch_face_detection/python.exe PC_test/onnx_infer.py --conf 0.6 --nms 0.45 --minbox 15

This faithfully replicates the STM32 C postprocessing logic for comparison.
"""

import os, sys, time, argparse
import numpy as np
import cv2
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model_stm32", "yunetn_320.onnx")
TEST_IMG = os.path.join(BASE_DIR,
    "WIDERFace", "WIDER_test", "images", "29--Students_Schoolkids",
    "29_Students_Schoolkids_Students_Schoolkids_29_1.jpg")
OUT_DIR = os.path.join(BASE_DIR, "PC_test")

STRIDES = [8, 16, 32]
GS_MAP = {8: 40, 16: 20, 32: 10}
INPUT_SIZE = (320, 320)


def preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    original = img.copy()
    h, w = img.shape[:2]
    resized = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    chw = np.transpose(resized, (2, 0, 1))  # HWC→CHW, BGR
    nchw = np.expand_dims(chw, axis=0).astype(np.float32)
    return nchw, original, (w, h)


def decode_stm32(onnx_outputs, conf_thresh, nms_thresh, min_box_size):
    """STM32 C logic: ONNX bbox interleaved→CHW, anchor at grid corner."""
    # Convert ONNX bbox/kps from interleaved (N×C) to CHW flat (C×N)
    outputs = list(onnx_outputs)
    for i in range(12):
        arr = outputs[i]
        if arr.ndim == 3 and arr.shape[2] in (4, 10):
            outputs[i] = np.transpose(arr, (0, 2, 1))

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
            if iou > nms_thresh:
                suppressed[b] = True

    return [all_dets[i] for i in range(len(all_dets)) if not suppressed[i]]


def draw_boxes(img, dets, orig_w, orig_h, out_path):
    out = img.copy()
    sx, sy = orig_w / 320.0, orig_h / 320.0
    for det in dets:
        x1, y1, x2, y2, score = det
        x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{score:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(out, f"Faces: {len(dets)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(out_path, out)
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description="ONNX YuNet face detection")
    p.add_argument('--image', default=TEST_IMG, help='Input image path')
    p.add_argument('--conf', type=float, default=0.6, help='Confidence threshold')
    p.add_argument('--nms', type=float, default=0.45, help='NMS IoU threshold')
    p.add_argument('--minbox', type=float, default=15.0, help='Min box size (px)')
    args = p.parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        sys.exit(1)

    print(f"=== ONNX YuNet Face Detection ===")
    print(f"Image: {args.image}")
    print(f"Conf={args.conf}, NMS={args.nms}, MinBox={args.minbox}")

    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Preprocess
    input_data, orig_img, (ow, oh) = preprocess(args.image)
    print(f"Original: {ow}x{oh}")

    # Inference
    t0 = time.time()
    outputs = session.run(None, {input_name: input_data})
    t_infer = time.time() - t0
    print(f"Inference: {t_infer*1000:.1f} ms")

    # Decode
    dets = decode_stm32(outputs, args.conf, args.nms, args.minbox)
    print(f"Detections: {len(dets)}")
    for d in dets:
        print(f"  score={d[4]:.4f}  ({d[0]:.1f}, {d[1]:.1f}, {d[2]:.1f}, {d[3]:.1f})")

    # Draw
    out_path = os.path.join(OUT_DIR, "onnx_infer_result.jpg")
    draw_boxes(orig_img, dets, ow, oh, out_path)
    print("Done!")


if __name__ == "__main__":
    main()
