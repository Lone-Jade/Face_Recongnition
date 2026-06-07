#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WIDER Face → Cube-AI-Studio Validation Dataset Generator
=========================================================
从 WIDER Face 数据集选取图片，生成 Cube-AI-Studio 可用的验证数据集。

输出格式：
  - CSV 清单 (validation_manifest.csv)  — 供 Cube-AI-Studio 导入
  - NPZ 文件  (validation_data.npz)     — 预处理好的输入 + ONNX 参考输出
  - NPY 文件  (inputs/*.npy, outputs/*.npy) — 单样本文件
  - 可视化   (previews/*.jpg)          — 预处理后的预览图

用法：
  D:/PythonConda/envs/pytorch_face_detection/python.exe generate_validation_dataset.py
  D:/PythonConda/envs/pytorch_face_detection/python.exe generate_validation_dataset.py --num-samples 20
  D:/PythonConda/envs/pytorch_face_detection/python.exe generate_validation_dataset.py --dataset val --num-samples 15

Cube-AI-Studio 使用方式：
  1. 打开 Cube-AI-Studio → Validate
  2. 在 "Validation Data" 中选择 validation_manifest.csv
  3. 或在 stedgeai CLI 中：
     stedgeai.exe validate --model yunetn_320.onnx --mode target \
         --files-input validation_data.npz
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import cv2
import onnxruntime as ort

# ============================================================================
# 路径配置
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "Public_pretrainedmodel_public_dataset",
    "widerface", "yunetn_320", "yunetn_320.onnx"
)
WIDER_FACE_DIR = os.path.join(BASE_DIR, "datasets", "WIDERFace")
ANNOTATION_DIR = os.path.join(WIDER_FACE_DIR, "wider_face_split")
OUTPUT_DIR = os.path.join(BASE_DIR, "validation_dataset")

# 模型参数
INPUT_SIZE = (320, 320)          # (width, height) — 模型输入尺寸
INPUT_SHAPE = (1, 3, 320, 320)   # NCHW, BGR
STRIDES = [8, 16, 32]
NUM_KEYPOINTS = 5

# 类别名（WIDER Face 只有 "face" 一个类别）
CLASS_NAMES = ["face"]


# ============================================================================
# 1. 解析 WIDER Face 标注
# ============================================================================
def parse_wider_face_annotations(annotation_file, images_dir):
    """
    解析 WIDER Face 标注文件。

    标注格式：
      <image_path>
      <num_faces>
      <x y w h blur expression illumination invalid occlusion pose>
      ...

    Returns:
      list of dict: [{
          'image_path': str,        # 图片完整路径
          'num_faces': int,
          'boxes': [[x, y, w, h], ...],  # 边界框
          'blur': [int, ...],
          'expression': [int, ...],
          'illumination': [int, ...],
          'occlusion': [int, ...],
          'pose': [int, ...],
      }]
    """
    annotations = []
    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # 第一行：图片路径
        img_rel_path = line
        img_full_path = os.path.join(images_dir, img_rel_path)
        if not os.path.exists(img_full_path):
            # 跳过不存在的图片
            i += 1
            num_faces = int(lines[i].strip())
            i += 1 + num_faces
            continue

        i += 1
        # 第二行：人脸数量
        num_faces = int(lines[i].strip())
        i += 1

        boxes = []
        blur = []
        expression = []
        illumination = []
        occlusion = []
        pose = []

        for j in range(num_faces):
            vals = list(map(int, lines[i].strip().split()))
            if len(vals) >= 10:
                boxes.append(vals[:4])      # x, y, w, h
                blur.append(vals[4])
                expression.append(vals[5])
                illumination.append(vals[6])
                if len(vals) > 7:
                    occlusion.append(vals[7] if len(vals) > 7 else 0)
                if len(vals) > 8:
                    pose.append(vals[8] if len(vals) > 8 else 0)
            i += 1

        annotations.append({
            'image_path': img_full_path,
            'num_faces': num_faces,
            'boxes': boxes,
            'blur': blur,
            'expression': expression,
            'illumination': illumination,
            'occlusion': occlusion if occlusion else [0]*num_faces,
            'pose': pose if pose else [0]*num_faces,
        })

    return annotations


# ============================================================================
# 2. 图片选取策略
# ============================================================================
def select_diverse_samples(annotations, num_samples=20, seed=42):
    """
    从标注中选取多样化的样本，确保覆盖不同的场景类别和人脸数量。
    """
    rng = np.random.RandomState(seed)

    # 按子目录（场景类别）分组
    by_category = {}
    for ann in annotations:
        category = os.path.basename(os.path.dirname(ann['image_path']))
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(ann)

    print(f"共 {len(by_category)} 个场景类别, {len(annotations)} 张图片")

    # 确保每个类别至少选 1 张
    categories = sorted(by_category.keys())
    samples = []

    # 给每个类别分配配额（按类别图片数比例，但至少 1 张）
    total_images = len(annotations)
    for cat in categories:
        quota = max(1, int(num_samples * len(by_category[cat]) / total_images))
        cat_samples = rng.choice(by_category[cat], min(quota, len(by_category[cat])), replace=False)
        samples.extend(cat_samples)

    # 如果不够 num_samples，随机补足
    if len(samples) < num_samples:
        remaining = [a for a in annotations if a not in samples]
        extra = rng.choice(remaining, min(num_samples - len(samples), len(remaining)), replace=False)
        samples.extend(extra)

    # 如果超过 num_samples，随机裁剪
    if len(samples) > num_samples:
        samples = rng.choice(samples, num_samples, replace=False).tolist()

    return list(samples)


def select_by_criteria(annotations, num_samples=20, criteria='diverse'):
    """
    按不同标准选取图片。

    criteria:
      - 'diverse': 覆盖不同场景
      - 'easy':    清晰正面人脸
      - 'hard':    模糊/遮挡/侧脸
      - 'crowd':   多人脸场景
    """
    if criteria == 'diverse':
        return select_diverse_samples(annotations, num_samples)
    elif criteria == 'easy':
        # 选清晰、正面、无遮挡的
        easy = []
        for ann in annotations:
            if ann['num_faces'] > 0 and ann['num_faces'] <= 3:
                avg_blur = np.mean(ann['blur']) if ann['blur'] else 0
                avg_occlusion = np.mean(ann['occlusion']) if ann['occlusion'] else 0
                if avg_blur <= 1 and avg_occlusion == 0:
                    easy.append(ann)
        rng = np.random.RandomState(42)
        return rng.choice(easy, min(num_samples, len(easy)), replace=False).tolist()
    elif criteria == 'hard':
        # 选模糊/遮挡/极端光照的
        hard = []
        for ann in annotations:
            if ann['num_faces'] > 0:
                has_hard = any(b >= 2 for b in ann['blur']) or \
                           any(o >= 2 for o in ann['occlusion'])
                if has_hard:
                    hard.append(ann)
        rng = np.random.RandomState(42)
        return rng.choice(hard, min(num_samples, len(hard)), replace=False).tolist()
    elif criteria == 'crowd':
        # 选人脸数量 ≥ 5 的场景
        crowd = [ann for ann in annotations if ann['num_faces'] >= 5]
        rng = np.random.RandomState(42)
        return rng.choice(crowd, min(num_samples, len(crowd)), replace=False).tolist()
    else:
        return select_diverse_samples(annotations, num_samples)


# ============================================================================
# 3. 图像预处理（匹配 YuNet 输入格式）
# ============================================================================
def preprocess_for_yunet(image_path, input_size=(320, 320)):
    """
    预处理图片以匹配 YuNet 模型输入格式。

    模型期望的格式（来自 Cube.AI 分析）：
      - 输入名: 'input'
      - Shape: [1, 3, 320, 320] (NCHW)
      - 格式: BGR
      - 像素值范围: [0, 255] (scale=1, offset=0, float32)

    Returns:
      input_array: np.ndarray, shape [1, 3, 320, 320], dtype=float32
      original_img: 原始图片
      resized_img: 缩放后图片 [320, 320, 3]
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    original_img = img.copy()

    # 缩放到 320×320
    img_resized = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)

    # HWC → CHW, 保持 BGR
    img_chw = np.transpose(img_resized, (2, 0, 1))

    # 添加 batch 维度 [C, H, W] → [1, C, H, W]
    img_nchw = np.expand_dims(img_chw, axis=0).astype(np.float32)

    return img_nchw, original_img, img_resized


def preprocess_batch(image_paths, input_size=(320, 320)):
    """批量预处理，返回 numpy 数组。"""
    batch = []
    originals = []
    resized = []
    for path in image_paths:
        inp, orig, res = preprocess_for_yunet(path, input_size)
        batch.append(inp)
        originals.append(orig)
        resized.append(res)
    return np.concatenate(batch, axis=0), originals, resized


# ============================================================================
# 4. ONNX 推理（生成参考输出）
# ============================================================================
def run_onnx_inference(session, input_name, input_data):
    """运行 ONNX 模型推理。"""
    outputs = session.run(None, {input_name: input_data})
    return outputs


# ============================================================================
# 5. 生成 Cube-AI-Studio 验证数据
# ============================================================================
def generate_validation_dataset(samples, output_dir, model_path,
                                num_samples=None, criteria='diverse'):
    """
    生成完整的验证数据集。

    输出文件结构：
      validation_dataset/
      ├── validation_manifest.csv       # CSV 清单（Cube-AI-Studio 导入用）
      ├── validation_data.npz           # 所有预处理数据（可直接加载）
      ├── inputs/                       # 单张图片的 .npy 输入文件
      │   ├── sample_001_input.npy
      │   └── ...
      ├── outputs/                      # 对应的 ONNX 参考输出
      │   ├── sample_001_output.npz
      │   └── ...
      └── previews/                     # 预处理后的可视化图
          ├── sample_001_preview.jpg
          └── ...
    """
    os.makedirs(output_dir, exist_ok=True)
    inputs_dir = os.path.join(output_dir, "inputs")
    outputs_dir = os.path.join(output_dir, "outputs")
    previews_dir = os.path.join(output_dir, "previews")
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(previews_dir, exist_ok=True)

    # 加载 ONNX 模型
    print(f"Loading ONNX model: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    if num_samples:
        samples = samples[:num_samples]

    print(f"\nGenerating validation dataset for {len(samples)} images...")
    print(f"Selection criteria: {criteria}")
    print(f"Output directory: {output_dir}\n")

    # CSV 行
    csv_rows = []
    # 所有输入/输出的累积列表（用于 .npz）
    all_inputs = []
    all_outputs = {name: [] for name in output_names}

    for idx, sample in enumerate(samples):
        img_path = sample['image_path']
        sample_id = f"sample_{idx+1:03d}"
        img_name = os.path.basename(img_path)
        category = os.path.basename(os.path.dirname(img_path))

        print(f"[{idx+1}/{len(samples)}] {category}/{img_name}")

        try:
            # 预处理
            t0 = time.time()
            input_data, original_img, resized_img = preprocess_for_yunet(img_path, INPUT_SIZE)
            prep_time = (time.time() - t0) * 1000

            # ONNX 推理
            t0 = time.time()
            outputs = run_onnx_inference(session, input_name, input_data)
            infer_time = (time.time() - t0) * 1000

            # 保存输入 .npy
            input_npy_path = os.path.join(inputs_dir, f"{sample_id}_input.npy")
            np.save(input_npy_path, input_data)

            # 保存输出 .npz
            output_npz_path = os.path.join(outputs_dir, f"{sample_id}_output.npz")
            output_dict = {f"output_{i:02d}_{name}": outputs[i]
                          for i, name in enumerate(output_names)}
            np.savez_compressed(output_npz_path, **output_dict)

            # 保存预处理后的预览图（BGR → RGB 用于查看）
            preview_path = os.path.join(previews_dir, f"{sample_id}_preview.jpg")
            preview_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

            # 在预览图上绘制 GT 标注框
            h, w = resized_img.shape[:2]
            orig_h, orig_w = original_img.shape[:2]
            for box in sample['boxes']:
                x, y, bw, bh = box
                # 缩放到 320×320
                x_scaled = int(x * w / orig_w)
                y_scaled = int(y * h / orig_h)
                bw_scaled = int(bw * w / orig_w)
                bh_scaled = int(bh * h / orig_h)
                cv2.rectangle(preview_rgb, (x_scaled, y_scaled),
                             (x_scaled + bw_scaled, y_scaled + bh_scaled),
                             (0, 255, 0), 1)

            cv2.imwrite(preview_path, cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))

            # 累积到批次
            all_inputs.append(input_data)
            for i, name in enumerate(output_names):
                all_outputs[name].append(outputs[i])

            # CSV 行信息
            num_gt_faces = sample['num_faces']
            avg_blur = np.mean(sample['blur']) if sample['blur'] else 0
            avg_occlusion = np.mean(sample['occlusion']) if sample['occlusion'] else 0

            csv_rows.append({
                'sample_id': sample_id,
                'image_path': img_path,
                'category': category,
                'image_name': img_name,
                'orig_width': orig_w,
                'orig_height': orig_h,
                'num_gt_faces': num_gt_faces,
                'avg_blur': f"{avg_blur:.1f}",
                'avg_occlusion': f"{avg_occlusion:.1f}",
                'prep_time_ms': f"{prep_time:.1f}",
                'infer_time_ms': f"{infer_time:.1f}",
                'input_npy': input_npy_path,
                'output_npz': output_npz_path,
                'preview_jpg': preview_path,
            })

            print(f"    GT faces: {num_gt_faces}, "
                  f"prep: {prep_time:.1f}ms, "
                  f"infer: {infer_time:.1f}ms")

        except Exception as e:
            print(f"    [ERROR] {e}")
            continue

    # =========================================================================
    # 保存 CSV 清单
    # =========================================================================
    csv_path = os.path.join(output_dir, "validation_manifest.csv")
    fieldnames = [
        'sample_id', 'category', 'image_name', 'image_path',
        'orig_width', 'orig_height', 'num_gt_faces',
        'avg_blur', 'avg_occlusion',
        'prep_time_ms', 'infer_time_ms',
        'input_npy', 'output_npz', 'preview_jpg',
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nCSV manifest saved: {csv_path}")

    # =========================================================================
    # 保存组合 NPZ（所有样本合并）
    # =========================================================================
    npz_path = os.path.join(output_dir, "validation_data.npz")
    npz_data = {}

    # 合并所有输入
    if all_inputs:
        npz_data['inputs'] = np.concatenate(all_inputs, axis=0)

    # 合并所有输出（每个输出一个键）
    for i, name in enumerate(output_names):
        if all_outputs[name]:
            npz_data[f'output_{i:02d}_{name}'] = np.concatenate(all_outputs[name], axis=0)

    np.savez_compressed(npz_path, **npz_data)
    npz_size = os.path.getsize(npz_path) / 1024 / 1024

    print(f"Combined NPZ saved: {npz_path} ({npz_size:.1f} MB)")

    # =========================================================================
    # 保存 Cube-AI-Studio 简化 CSV（仅图片路径，供 Studio 直接导入）
    # =========================================================================
    studio_csv_path = os.path.join(output_dir, "cubeai_studio_input.csv")
    with open(studio_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Cube-AI-Studio 期望的格式：每行一个图片路径
        writer.writerow(['image_path'])
        for row in csv_rows:
            writer.writerow([row['image_path']])

    print(f"Cube-AI-Studio CSV: {studio_csv_path}")

    # =========================================================================
    # 打印统计摘要
    # =========================================================================
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples:       {len(csv_rows)}")
    print(f"Categories covered:  {len(set(r['category'] for r in csv_rows))}")

    gt_counts = [int(r['num_gt_faces']) for r in csv_rows]
    print(f"Total GT faces:      {sum(gt_counts)}")
    print(f"Avg faces/image:     {np.mean(gt_counts):.1f}")
    print(f"Max faces/image:     {max(gt_counts)}")
    print(f"Min faces/image:     {min(gt_counts)}")

    if csv_rows:
        infer_times = [float(r['infer_time_ms']) for r in csv_rows]
        print(f"Avg ONNX infer:      {np.mean(infer_times):.1f} ms")

    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── validation_manifest.csv    ← 完整 CSV 清单")
    print(f"  ├── cubeai_studio_input.csv    ← Cube-AI-Studio 直接导入")
    print(f"  ├── validation_data.npz        ← 合并 NPZ（{npz_size:.1f} MB）")
    print(f"  ├── inputs/                    ← {len(csv_rows)} 个 .npy 输入文件")
    print(f"  ├── outputs/                   ← {len(csv_rows)} 个 .npz 参考输出文件")
    print(f"  └── previews/                  ← {len(csv_rows)} 张预处理预览图")
    print("=" * 60)

    return csv_path, npz_path, studio_csv_path


# ============================================================================
# 6. 统计信息
# ============================================================================
def print_dataset_stats(annotations):
    """打印 WIDER Face 数据集统计信息。"""
    print("\n" + "=" * 60)
    print("WIDER Face Dataset Statistics")
    print("=" * 60)

    total_images = len(annotations)
    total_faces = sum(a['num_faces'] for a in annotations)

    # 按类别统计
    categories = {}
    for ann in annotations:
        cat = os.path.basename(os.path.dirname(ann['image_path']))
        if cat not in categories:
            categories[cat] = {'images': 0, 'faces': 0}
        categories[cat]['images'] += 1
        categories[cat]['faces'] += ann['num_faces']

    print(f"Total images: {total_images}")
    print(f"Total faces:  {total_faces}")
    print(f"Categories:   {len(categories)}")
    print(f"Avg faces/img: {total_faces/total_images:.1f}")
    print(f"\nTop 10 categories by image count:")
    sorted_cats = sorted(categories.items(), key=lambda x: -x[1]['images'])
    for i, (cat, stats) in enumerate(sorted_cats[:10]):
        print(f"  {i+1}. {cat}: {stats['images']} images, {stats['faces']} faces")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate validation dataset for Cube-AI-Studio from WIDER Face"
    )
    parser.add_argument('--dataset', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='WIDER Face subset (default: val)')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of images to select (default: 20)')
    parser.add_argument('--criteria', type=str, default='diverse',
                        choices=['diverse', 'easy', 'hard', 'crowd'],
                        help='Selection criteria (default: diverse)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: ./validation_dataset)')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='Path to ONNX model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only print dataset statistics, no generation')
    args = parser.parse_args()

    # 确定数据集路径
    dataset_key = {
        'train': 'WIDER_train',
        'val': 'WIDER_val',
        'test': 'WIDER_test',
    }[args.dataset]

    images_dir = os.path.join(WIDER_FACE_DIR, dataset_key, "images")

    # 标注文件
    if args.dataset == 'train':
        annotation_file = os.path.join(ANNOTATION_DIR, "wider_face_train_bbx_gt.txt")
    elif args.dataset == 'val':
        annotation_file = os.path.join(ANNOTATION_DIR, "wider_face_val_bbx_gt.txt")
    else:
        # test 集没有公开标注，用 filelist
        annotation_file = None

    output_dir = args.output or os.path.join(BASE_DIR, "validation_dataset")

    print("=" * 60)
    print("  Cube-AI-Studio Validation Dataset Generator")
    print(f"  Dataset: WIDER Face ({args.dataset})")
    print(f"  Images dir: {images_dir}")
    print(f"  Output dir: {output_dir}")
    print("=" * 60)

    if annotation_file and os.path.exists(annotation_file):
        print(f"\nParsing annotations: {os.path.basename(annotation_file)}")
        annotations = parse_wider_face_annotations(annotation_file, images_dir)
        print(f"Found {len(annotations)} images with annotations")

        if args.stats_only:
            print_dataset_stats(annotations)
            return

        # 选取样本
        samples = select_by_criteria(annotations, args.num_samples, args.criteria)
    else:
        # test 集没有标注 — 仅使用图片文件列表
        print("\nNo annotations available (test set). Using random images.")
        # 收集所有图片文件
        all_images = []
        for root, dirs, files in os.walk(images_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(root, f))

        rng = np.random.RandomState(args.seed)
        selected = rng.choice(all_images, min(args.num_samples, len(all_images)), replace=False)

        samples = []
        for img_path in selected:
            samples.append({
                'image_path': img_path,
                'num_faces': 0,
                'boxes': [],
                'blur': [],
                'expression': [],
                'illumination': [],
                'occlusion': [],
                'pose': [],
            })

    if args.stats_only:
        return

    # 生成验证数据集
    csv_path, npz_path, studio_csv_path = generate_validation_dataset(
        samples, output_dir, args.model,
        num_samples=args.num_samples, criteria=args.criteria
    )

    print(f"\nDone! Use the following in Cube-AI-Studio:")
    print(f"  1. CSV import: {studio_csv_path}")
    print(f"  2. NPZ import: {npz_path}")
    print(f"\nOr use with stedgeai CLI:")
    print(f"  stedgeai.exe validate --model yunetn_320.onnx \\")
    print(f"      --mode target --files-input {os.path.join(output_dir, 'inputs')} \\")
    print(f"      --files-output {os.path.join(output_dir, 'outputs')}")


if __name__ == "__main__":
    main()
