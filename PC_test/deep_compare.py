#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep-dive: compare model output values from Python STM32-NN preprocessing
with USART output from STM32 hardware.
"""

import os, sys
import numpy as np
import onnxruntime as ort
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model_stm32", "yunetn_320.onnx")
OUTPUT_NAMES = ["cls_8","cls_16","cls_32","obj_8","obj_16","obj_32",
                "bbox_8","bbox_16","bbox_32","kps_8","kps_16","kps_32"]


def parse_c_header_array(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r'\{([^}]+)\}', content, re.DOTALL)
    hex_vals = re.findall(r'0x[0-9A-Fa-f]{8}', match.group(1))
    return np.array([int(v, 16) for v in hex_vals], dtype=np.uint32)


def preprocess_stm32(pixels, src_w=320, src_h=240):
    """Exact replica of STM32 ai_preprocess() — per-channel loops."""
    dst_w, dst_h = 320, 320
    x_ratio = ((src_w << 16) // dst_w) + 1
    y_ratio = ((src_h << 16) // dst_h) + 1
    plane_size = dst_h * dst_w

    dst = np.zeros((3 * plane_size), dtype=np.float32)
    img_2d = pixels.reshape(src_h, src_w)

    for c in range(3):
        dst_c = c * plane_size
        for y in range(dst_h):
            sy = (y * y_ratio) >> 16
            if sy >= src_h:
                sy = src_h - 1
            for x in range(dst_w):
                sx = (x * x_ratio) >> 16
                if sx >= src_w:
                    sx = src_w - 1

                p = img_2d[sy, sx]
                if c == 0:
                    val = float(p & 0xFF)          # B
                elif c == 1:
                    val = float((p >> 8) & 0xFF)   # G
                else:
                    val = float((p >> 16) & 0xFF)  # R
                dst[dst_c + y * dst_w + x] = val

    # Reshape to NCHW: (1, 3, 320, 320)
    return dst.reshape(1, 3, dst_h, dst_w).astype(np.float32)


def main():
    img0_path = os.path.join(BASE_DIR, "BSP", "Inc", "face_img_0.h")

    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    for img_idx, img_path in enumerate([img0_path]):
        pixels = parse_c_header_array(img_path)

        # Preprocess with STM32 method
        input_data = preprocess_stm32(pixels)

        # Print input tensor samples (matching USART output format)
        print("=== Input tensor samples (matching USART) ===")
        print(f"B[0..4]: {input_data[0,0,0,0]:.1f} {input_data[0,0,0,1]:.1f} {input_data[0,0,0,2]:.1f} {input_data[0,0,0,3]:.1f} {input_data[0,0,0,4]:.1f}")
        print(f"G[0..4]: {input_data[0,1,0,0]:.1f} {input_data[0,1,0,1]:.1f} {input_data[0,1,0,2]:.1f} {input_data[0,1,0,3]:.1f} {input_data[0,1,0,4]:.1f}")
        print(f"R[0..4]: {input_data[0,2,0,0]:.1f} {input_data[0,2,0,1]:.1f} {input_data[0,2,0,2]:.1f} {input_data[0,2,0,3]:.1f} {input_data[0,2,0,4]:.1f}")

        # Run ONNX inference
        outputs = session.run(None, {input_name: input_data})

        # Print model output samples (matching USART format)
        print("\n=== Model output samples (matching USART) ===")
        strides = [8, 16, 32]
        gs_map = {8: 40, 16: 20, 32: 10}
        for k, stride in enumerate(strides):
            gs = gs_map[stride]
            cls = outputs[k].flatten()
            obj = outputs[k + 3].flatten()
            bbox = outputs[k + 6].flatten()

            print(f"Stride {stride} (gs={gs}): cls[0..4]={cls[0]:.4f} {cls[1]:.4f} {cls[2]:.4f} {cls[3]:.4f} {cls[4]:.4f}  obj[0..4]={obj[0]:.4f} {obj[1]:.4f} {obj[2]:.4f} {obj[3]:.4f} {obj[4]:.4f}")
            print(f"  bbox[0,0]: dx={bbox[0*gs*gs+0]:.4f} dy={bbox[1*gs*gs+0]:.4f} dw={bbox[2*gs*gs+0]:.4f} dh={bbox[3*gs*gs+0]:.4f}")

        # Print full detection details
        print("\n=== All raw candidates with score > 0.5 ===")
        all_raw = []
        for k, stride in enumerate(strides):
            gs = gs_map[stride]
            cls = outputs[k].flatten()
            obj = outputs[k + 3].flatten()
            bbox = outputs[k + 6].flatten()
            for i in range(gs):
                for j in range(gs):
                    loc = i * gs + j
                    score = cls[loc] * obj[loc]
                    if score > 0.5:
                        dx = bbox[0*gs*gs+loc]; dy = bbox[1*gs*gs+loc]
                        dw = bbox[2*gs*gs+loc]; dh = bbox[3*gs*gs+loc]
                        cx = (j + dx) * stride; cy = (i + dy) * stride
                        bw = np.exp(dw) * stride; bh = np.exp(dh) * stride
                        all_raw.append((score, stride, i, j, cx, cy, bw, bh))

        all_raw.sort(key=lambda x: -x[0])
        print(f"Total candidates with score > 0.5: {len(all_raw)}")
        for r in all_raw[:20]:
            print(f"  score={r[0]:.4f} stride={r[1]} grid=({r[2]},{r[3]}) "
                  f"center=({r[4]:.1f},{r[5]:.1f}) size={r[6]:.0f}x{r[7]:.0f}")

        # Compare with USART output
        print("\n=== COMPARISON with USART output ===")
        print("USART Frame 1 reported:")
        print("  Stride 8:  cls[0..4]=0.5756 0.5835 0.5952 0.5720 0.5172  obj[0..4]=0.0000 0.0000 0.0000 0.0000 0.0000")
        print("  Stride 16: cls[0..4]=0.6299 0.6207 0.5949 0.5732 0.5660  obj[0..4]=0.0002 0.0000 0.0000 0.0000 0.0000")
        print("  bbox[0,0]: dx=0.7080 dy=0.5722 dw=0.4949 dh=0.5376")
        print("  Stride 16 raw candidates: 8, scores: 0.8563 0.8521 0.8499 0.8476 0.8404")
        print(f"\nPython ONNX reports:")
        print(f"  Stride 8:  cls[0..4]={outputs[0].flatten()[0]:.4f} {outputs[0].flatten()[1]:.4f} {outputs[0].flatten()[2]:.4f} {outputs[0].flatten()[3]:.4f} {outputs[0].flatten()[4]:.4f}  obj[0..4]={outputs[3].flatten()[0]:.4f} {outputs[3].flatten()[1]:.4f} {outputs[3].flatten()[2]:.4f} {outputs[3].flatten()[3]:.4f} {outputs[3].flatten()[4]:.4f}")
        gs16 = 20
        print(f"  Stride 16: cls[0..4]={outputs[1].flatten()[0]:.4f} {outputs[1].flatten()[1]:.4f} {outputs[1].flatten()[2]:.4f} {outputs[1].flatten()[3]:.4f} {outputs[1].flatten()[4]:.4f}  obj[0..4]={outputs[4].flatten()[0]:.4f} {outputs[4].flatten()[1]:.4f} {outputs[4].flatten()[2]:.4f} {outputs[4].flatten()[3]:.4f} {outputs[4].flatten()[4]:.4f}")
        b16 = outputs[7].flatten()
        print(f"  bbox[0,0]: dx={b16[0*gs16*gs16+0]:.4f} dy={b16[1*gs16*gs16+0]:.4f} dw={b16[2*gs16*gs16+0]:.4f} dh={b16[3*gs16*gs16+0]:.4f}")

        # Check if they match!
        cls16_usart = np.array([0.6299, 0.6207, 0.5949, 0.5732, 0.5660])
        cls16_python = outputs[1].flatten()[:5]
        cls16_diff = np.max(np.abs(cls16_usart - cls16_python))
        print(f"\n  cls_16 max|diff| USART vs Python: {cls16_diff:.4f}")
        if cls16_diff > 0.01:
            print("  *** MISMATCH! STM32 model outputs differ from ONNX! ***")
        else:
            print("  Match. Model outputs are identical.")

    print("\nDone!")


if __name__ == "__main__":
    main()
