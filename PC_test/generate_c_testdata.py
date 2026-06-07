#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate C-compatible test data from stedgeai_validation.npz for PC-side
verification of the STM32 C postprocessing code.

Output:
  PC_test/test_data.h   — C header with input tensor and expected outputs
  PC_test/test_data.c   — C source with actual data arrays

Usage:
  D:/PythonConda/envs/pytorch_face_detection/python.exe PC_test/generate_c_testdata.py
"""

import os, sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ_PATH = os.path.join(BASE_DIR, "model_stm32", "stedgeai_validation.npz")
OUT_DIR = os.path.join(BASE_DIR, "PC_test")

def main():
    print("Loading validation data...")
    data = np.load(NPZ_PATH)
    print(f"Keys: {list(data.keys())}")

    # ST AI Studio validation format: m_inputs_1, m_outputs_1..m_outputs_12
    sample_input = data['m_inputs_1']  # shape: (3, 320, 320) NCHW or (1, 3, 320, 320)
    if sample_input.ndim == 4:
        sample_input = sample_input[0]  # Remove batch dim
    print(f"Sample input: shape={sample_input.shape}, range=[{sample_input.min():.2f}, {sample_input.max():.2f}]")

    # Parse outputs in ST AI compiled order (matching network.h OUT_1..OUT_12)
    sample_outputs = []
    for i in range(1, 13):
        key = f'm_outputs_{i}'
        arr = data[key]  # May have batch dim
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        sample_outputs.append(arr)
        print(f"  {key}: shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]")

    # Generate C header
    hdr = """/* Auto-generated test data for YuNet postprocessing verification.
   Generated from stedgeai_validation.npz
   Input: NCHW BGR float32, 320x320
   Outputs: 12 tensors (cls, obj, bbox, kps × 3 strides)
*/
#ifndef TEST_DATA_H
#define TEST_DATA_H

#include <stdint.h>
#define TEST_IMG_H 320
#define TEST_IMG_W 320
#define TEST_IMG_C 3
#define TEST_NUM_OUTPUTS 12

extern const float test_input[TEST_IMG_C * TEST_IMG_H * TEST_IMG_W];
"""
    for i in range(12):
        sz = sample_outputs[i].size
        hdr += f"extern const float test_output_{i}[{sz}];\n"
    hdr += "\n#endif /* TEST_DATA_H */\n"

    # Generate C source
    src = '/* Auto-generated test data */\n#include "test_data.h"\n\n'

    # Input
    src += f"// Input tensor: {sample_input.shape} (BGR NCHW)\n"
    src += "const float test_input[TEST_IMG_C * TEST_IMG_H * TEST_IMG_W] = {\n"
    flat_input = sample_input.flatten()
    for i in range(0, len(flat_input), 8):
        src += "  " + ", ".join(f"{v:8.2f}f" for v in flat_input[i:i+8]) + ",\n"
    src += "};\n\n"

    # Outputs
    for i, arr in enumerate(sample_outputs):
        flat = arr.flatten()
        src += f"// Output {i}: shape {arr.shape}\n"
        src += f"const float test_output_{i}[{len(flat)}] = {{\n"
        for j in range(0, len(flat), 8):
            src += "  " + ", ".join(f"{v:12.8f}f" for v in flat[j:j+8]) + ",\n"
        src += "};\n\n"

    # Save files
    with open(os.path.join(OUT_DIR, "test_data.h"), "w") as f:
        f.write(hdr)
    with open(os.path.join(OUT_DIR, "test_data.c"), "w") as f:
        f.write(src)

    print(f"\nSaved: {os.path.join(OUT_DIR, 'test_data.h')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'test_data.c')}")
    print(f"Input bytes: {len(flat_input) * 4}")
    total_out = sum(s.size for s in sample_outputs)
    print(f"Total output bytes: {total_out * 4}")
    print("Done!")

if __name__ == "__main__":
    main()
