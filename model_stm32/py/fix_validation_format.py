#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复验证数据集格式 - 严格按照 ST Edge AI 文档规范生成。

文档规范要点 (ref_post_proc_support.md):
  npz 文件支持的 key pair:
    - m_inputs  / m_outputs          (单输入单输出)
    - m_inputs_<idx> / m_outputs_<idx>  (多输入输出, idx 从 1 开始)
    - c_inputs_<idx> / c_outputs_<idx>  (如果 m_ 未定义则使用)

  数组格式:
    - shape 必须为 (batch_size, -1)，即扁平化的张量
    - 加载时 Cube.AI 会根据 c-model 的 shape 自动 reshape 回去

生成文件:
  validation_dataset/
  --- stedgeai_validation.npz    ← - 正确的 key: m_inputs_1, m_outputs_1..12
  --- stedgeai_validation_c.npz   ← 备选格式: c_inputs_1, c_outputs_1..12
  --- stedgeai_val_input.npy     ← 单独输入文件 [N, 1228800]
  --- stedgeai_val_output.npz    ← 单独输出文件 [N, -1] × 12

用法:
  D:/PythonConda/envs/pytorch_face_detection/python.exe fix_validation_format.py
"""

import os
import sys
import numpy as np
import onnxruntime as ort
from generate_validation_dataset import (
    parse_wider_face_annotations,
    select_by_criteria,
    preprocess_for_yunet,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "Public_pretrainedmodel_public_dataset",
    "widerface", "yunetn_320", "yunetn_320.onnx"
)
WIDER_FACE_DIR = os.path.join(BASE_DIR, "datasets", "WIDERFace")
ANNOTATION_DIR = os.path.join(WIDER_FACE_DIR, "wider_face_split")
OUTPUT_DIR = os.path.join(BASE_DIR, "validation_dataset")
INPUT_SIZE = (320, 320)


def main():
    # =====================================================================
    # 1. 选取样本
    # =====================================================================
    images_dir = os.path.join(WIDER_FACE_DIR, "WIDER_val", "images")
    ann_file = os.path.join(ANNOTATION_DIR, "wider_face_val_bbx_gt.txt")
    annotations = parse_wider_face_annotations(ann_file, images_dir)
    samples = select_by_criteria(annotations, 10, 'diverse')
    print(f"Selected {len(samples)} images for validation")

    # =====================================================================
    # 2. ONNX 推理，收集输入和参考输出
    # =====================================================================
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name    # "input"
    output_names = [o.name for o in session.get_outputs()]

    all_inputs = []
    all_outputs_raw = []

    for idx, sample in enumerate(samples):
        inp, _, _ = preprocess_for_yunet(sample['image_path'], INPUT_SIZE)
        all_inputs.append(inp)                               # [1, 3, 320, 320]
        outputs = session.run(output_names, {input_name: inp})
        all_outputs_raw.append(outputs)                      # list of 12 arrays
        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx+1}/{len(samples)}")

    print(f"Done: {len(all_inputs)} samples")

    # =====================================================================
    # 3. 合并 + 扁平化（ST Edge AI 要求 (N, -1) 格式）
    # =====================================================================
    N = len(all_inputs)

    # 合并所有输入 - 扁平化: [N, 1228800] = [N, 3*320*320]
    combined_input = np.concatenate(all_inputs, axis=0)    # [N, 3, 320, 320]
    flat_input = combined_input.reshape(N, -1)              # [N, 1228800]
    print(f"\nInput:  {combined_input.shape} - flat {flat_input.shape}")

    # 合并所有输出 - 扁平化: [N, -1] for each
    flat_outputs = {}           # key: "m_outputs_<idx>", value: [N, flat_size]
    c_flat_outputs = {}         # 备选: c_outputs_<idx>

    for i, name in enumerate(output_names):
        combined = np.concatenate([out[i] for out in all_outputs_raw], axis=0)
        flat = combined.reshape(N, -1)
        idx_str = f"{i+1}"
        flat_outputs[f"m_outputs_{idx_str}"] = flat
        c_flat_outputs[f"c_outputs_{idx_str}"] = flat
        print(f"  m_outputs_{idx_str} ({name}): {combined.shape} - flat {flat.shape}")

    # =====================================================================
    # 4. 保存 - 格式 A：m_inputs_1 + m_outputs_<idx>（ST 标准格式）
    # =====================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    npz_data_m = {"m_inputs_1": flat_input}
    npz_data_m.update(flat_outputs)

    path_a = os.path.join(OUTPUT_DIR, "stedgeai_validation.npz")
    np.savez(path_a, **npz_data_m)
    size_mb = os.path.getsize(path_a) / 1024 / 1024
    print(f"\n[Format A - m_inputs/m_outputs] {path_a} ({size_mb:.1f} MB)")
    print(f"   Keys: {list(npz_data_m.keys())}")
    print(f"   Usage: Place this file in Cube-AI-Studio Validation data field")

    # =====================================================================
    # 5. 保存 - 格式 B：c_inputs_1 + c_outputs_<idx>（备选格式）
    # =====================================================================
    npz_data_c = {"c_inputs_1": flat_input}
    npz_data_c.update(c_flat_outputs)

    path_b = os.path.join(OUTPUT_DIR, "stedgeai_validation_c.npz")
    np.savez(path_b, **npz_data_c)
    size_mb = os.path.getsize(path_b) / 1024 / 1024
    print(f"\n[Format B - c_inputs/c_outputs] {path_b} ({size_mb:.1f} MB)")
    print(f"   Keys: {list(npz_data_c.keys())}")

    # =====================================================================
    # 6. 保存 - 单独 .npy 输入 + .npz 输出（CLI 模式）
    # =====================================================================
    input_npy = os.path.join(OUTPUT_DIR, "stedgeai_val_input.npy")
    np.save(input_npy, flat_input)
    print(f"\n- [CLI input] {input_npy} - shape {flat_input.shape}")

    output_npz = os.path.join(OUTPUT_DIR, "stedgeai_val_output.npz")
    np.savez(output_npz, **flat_outputs)
    size_mb = os.path.getsize(output_npz) / 1024 / 1024
    print(f"- [CLI output] {output_npz} ({size_mb:.1f} MB)")
    print(f"   Keys: {list(flat_outputs.keys())}")

    # =====================================================================
    # 使用说明
    # =====================================================================
    print(f"""
{'='*60}
USAGE IN CUBE-AI-STUDIO
{'='*60}

  Validation data file:  {path_a}

  Cube-AI-Studio 验证步骤:
    1. 打开 Cube-AI-Studio, 进入 Validate 模式
    2. 在 Validation data 区点击 "Browse..."
    3. 选择: {path_a}
    4. Cube-AI-Studio 会自动:
       - 识别 m_inputs_1 - reshape 到 [10, 3, 320, 320]
       - 识别 m_outputs_1..12 - 分别 reshape 到各输出张量
       - 运行对比验证 (ONNX Runtime reference vs C-model)

  CLI 验证命令:
    stedgeai.exe validate ^
        --model yunetn_320.onnx ^
        --mode host ^
        --files-input {input_npy} ^
        --files-output {output_npz}

  CLI 板卡验证命令:
    stedgeai.exe validate ^
        --model yunetn_320.onnx ^
        --mode target ^
        --files-input {input_npy} ^
        --files-output {output_npz} ^
        --target stm32h7 ^
        --desc serial:COM4:115200
{'='*60}
""")


if __name__ == "__main__":
    main()
