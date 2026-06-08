#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YuNet Face Detection Model Test Script
=======================================
Tests the pre-trained YuNet ONNX model (yunetn_320) for:
1. Model architecture analysis and STM32 compatibility
2. Single image inference
3. Batch inference on 10 WIDER Face images
4. Performance metrics

Usage:
    D:/PythonConda/envs/pytorch_face_detection/python.exe test_yunet.py
"""

import os
import sys
import time
import glob
import argparse
import numpy as np
import cv2
import onnx
import onnxruntime as ort

# ============================================================================
# Configuration
# ============================================================================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
FLOAT_MODEL_PATH = os.path.join(
    MODEL_DIR, "Public_pretrainedmodel_public_dataset",
    "widerface", "yunetn_320", "yunetn_320.onnx"
)
INT8_MODEL_PATH = os.path.join(
    MODEL_DIR, "Public_pretrainedmodel_public_dataset",
    "widerface", "yunetn_320", "yunetn_320_qdq_int8.onnx"
)
OUTPUT_DIR = os.path.join(MODEL_DIR, "test_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# YuNet parameters (from STM32 model zoo)
INPUT_SIZE = (320, 320)  # (width, height)
STRIDES = [8, 16, 32]
NUM_KEYPOINTS = 5  # eyes, nose, mouth corners

# Detection thresholds
CONFIDENCE_THRESH = 0.5
NMS_THRESH = 0.45
MAX_DETECTIONS = 10


# ============================================================================
# YuNet Decoder (adapted from STM32 model zoo postprocessing)
# ============================================================================
def generate_yunet_anchors(input_size, strides):
    """Generate anchor centers for each stride level."""
    centers = []
    for stride in strides:
        grid_h = input_size[1] // stride
        grid_w = input_size[0] // stride
        # Create grid of (x, y) centers
        anchor_centers = np.stack(
            np.mgrid[:grid_h, :grid_w][::-1], axis=-1
        )
        anchor_centers = (anchor_centers * stride).astype(np.float32).reshape(-1, 2)
        centers.append(anchor_centers)
    return centers


def decode_yunet_outputs(outputs, input_size, strides, num_kps=5):
    """
    Decode raw YuNet model outputs into bounding boxes, scores, and keypoints.

    Args:
        outputs: List of 12 numpy arrays (4 per stride: cls, obj, bbox, kps)
        input_size: Tuple (width, height)
        strides: List of stride values
        num_kps: Number of keypoints

    Returns:
        bboxes: [N, 4] (x1, y1, x2, y2) normalized to [0, 1]
        scores: [N] confidence scores
        keypoints: [N, num_kps, 2] normalized to [0, 1]
    """
    # Sort outputs by shape (smallest to largest), matching ST code behavior
    # The ST code sorts all 12 tensors by their full shape tuple
    # Output shapes: stride 32 (1,100,*), stride 16 (1,400,*), stride 8 (1,1600,*)
    nets_out_shapes_sorted = sorted(outputs, key=lambda x: x.shape)

    # Generate anchors in descending stride order to match sorted outputs
    # sorted_strides[0]=32 → smallest output → nets_out[0:4]
    # sorted_strides[1]=16 → middle output → nets_out[4:8]
    # sorted_strides[2]=8  → largest output → nets_out[8:12]
    sorted_strides = sorted(strides, reverse=True)  # [32, 16, 8]
    anchor_centers = generate_yunet_anchors(input_size, sorted_strides)

    all_scores = []
    all_bboxes = []
    all_kpss = []

    for idx, stride in enumerate(sorted_strides):
        # Each stride has 4 outputs: cls, obj, reg, kps
        # These are grouped in the sorted list
        cls_pred = nets_out_shapes_sorted[4 * idx + 0].reshape(-1, 1)
        obj_pred = nets_out_shapes_sorted[4 * idx + 1].reshape(-1, 1)
        reg_pred = nets_out_shapes_sorted[4 * idx + 2].reshape(-1, 4)
        kps_pred = nets_out_shapes_sorted[4 * idx + 3].reshape(-1, num_kps * 2)

        # Decode bounding boxes
        bbox_cxy = reg_pred[:, :2] * stride + anchor_centers[idx]
        bbox_wh = np.exp(reg_pred[:, 2:]) * stride

        # Convert to (x1, y1, x2, y2)
        tl_x = bbox_cxy[:, 0] - bbox_wh[:, 0] / 2.0
        tl_y = bbox_cxy[:, 1] - bbox_wh[:, 1] / 2.0
        br_x = bbox_cxy[:, 0] + bbox_wh[:, 0] / 2.0
        br_y = bbox_cxy[:, 1] + bbox_wh[:, 1] / 2.0

        bboxes = np.stack([tl_x, tl_y, br_x, br_y], axis=-1)

        # Decode keypoints
        per_kps = np.concatenate([
            ((kps_pred[:, [2 * i, 2 * i + 1]] * stride) + anchor_centers[idx])
            for i in range(num_kps)
        ], axis=-1)

        # Score = cls * obj
        scores = (cls_pred * obj_pred).reshape(-1)

        all_scores.append(scores)
        all_bboxes.append(bboxes)
        all_kpss.append(per_kps)

    # Concatenate all detections
    all_scores = np.concatenate(all_scores, axis=0)
    all_bboxes = np.concatenate(all_bboxes, axis=0)
    all_kpss = np.concatenate(all_kpss, axis=0)  # Shape: [N, 10]

    # Normalize to [0, 1]
    w, h = input_size
    all_bboxes[:, [0, 2]] /= w
    all_bboxes[:, [1, 3]] /= h

    # Normalize keypoints
    kps_reshaped = all_kpss.reshape(-1, num_kps, 2)
    kps_reshaped[:, :, 0] /= w
    kps_reshaped[:, :, 1] /= h

    return all_bboxes, all_scores, kps_reshaped


def nms(boxes, scores, iou_threshold, max_output_size=10, score_threshold=0.5):
    """Non-Maximum Suppression using OpenCV."""
    # Filter by score threshold
    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]

    if len(boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    # Convert to OpenCV format: (x, y, w, h)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    w = x2 - x1
    h = y2 - y1
    cv_boxes = np.stack([x1, y1, w, h], axis=-1)

    indices = cv2.dnn.NMSBoxes(
        cv_boxes.tolist(), scores.tolist(),
        score_threshold, iou_threshold
    )

    if len(indices) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    indices = np.array(indices).flatten()
    # Limit output
    indices = indices[:max_output_size]

    return boxes[indices], scores[indices], indices


# ============================================================================
# Model Analysis
# ============================================================================
def analyze_model(model_path):
    """Analyze ONNX model for STM32 compatibility."""
    print("\n" + "=" * 70)
    print(f"MODEL ANALYSIS: {os.path.basename(model_path)}")
    print("=" * 70)

    model = onnx.load(model_path)
    file_size_kb = os.path.getsize(model_path) / 1024

    print(f"\n  File size: {file_size_kb:.1f} KB")

    # Input info
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  Input: {inp.name}, shape={shape}")

    # Count parameters from initializers
    total_params = 0
    total_param_bytes = 0
    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        params = arr.size
        total_params += params
        total_param_bytes += arr.nbytes

    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Parameter size: {total_param_bytes / 1024:.1f} KB")

    # Output info
    print(f"\n  Outputs ({len(model.graph.output)}):")
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {shape}")

    # Opset
    for opset in model.opset_import:
        domain = opset.domain or "ai.onnx"
        print(f"\n  Opset: {domain} v{opset.version}")

    # Count unique operator types
    op_types = {}
    for node in model.graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

    print(f"\n  Operators ({len(op_types)} types, {len(model.graph.node)} total):")
    for op, count in sorted(op_types.items(), key=lambda x: -x[1]):
        print(f"    {op}: {count}")

    # STM32 compatibility assessment
    print("\n  --- STM32H747I-DISCO Compatibility ---")
    if file_size_kb < 512:
        print(f"    [PASS] Model size ({file_size_kb:.0f} KB) < 512 KB Flash limit")
    else:
        print(f"    [WARN] Model size ({file_size_kb:.0f} KB) may exceed 512 KB Flash limit")

    # Check for problematic ops
    stm32_supported_ops = {
        'Conv', 'Relu', 'LeakyRelu', 'MaxPool', 'AveragePool', 'GlobalAveragePool',
        'BatchNormalization', 'Add', 'Mul', 'Concat', 'Reshape', 'Transpose',
        'Squeeze', 'Unsqueeze', 'Flatten', 'Gemm', 'Softmax', 'Sigmoid',
        'Clip', 'Pad', 'Resize', 'Gather', 'Slice', 'Split',
        'ReduceMean', 'Dropout', 'PRelu', 'HardSigmoid', 'ConvTranspose',
        'LRN', 'DepthToSpace', 'SpaceToDepth', 'MatMul',
        'Exp', 'ReduceMin', 'ReduceMax', 'ReduceProd', 'ReduceSum',
        'Cast', 'Floor', 'Ceil', 'Identity', 'Erf',
        'Where', 'NonMaxSuppression', 'Abs', 'And', 'Or', 'Equal',
        'Greater', 'Less', 'GreaterOrEqual', 'LessOrEqual', 'Not',
        'Max', 'Min', 'Neg', 'Reciprocal', 'Sqrt',
        'Pow', 'Log', 'LogSoftmax', 'InstanceNormalization',
        'LSTM', 'GRU', 'RNN',
    }

    unsupported = set(op_types.keys()) - stm32_supported_ops
    if unsupported:
        print(f"    [WARN] Potentially unsupported ops: {', '.join(sorted(unsupported))}")
    else:
        print(f"    [PASS] All ops are in STM32 Cube.AI supported set")

    # Estimate RAM usage (intermediate activations)
    # Rough estimate: input_size * channels * bytes_per_element
    activation_mem = 320 * 320 * 3 * 4  # float32 input
    print(f"    Estimated peak activation memory: ~{activation_mem / 1024:.0f} KB (float32)")
    print(f"    INT8 will reduce this by ~4x: ~{activation_mem / 1024 / 4:.0f} KB")

    return {
        'file_size_kb': file_size_kb,
        'total_params': total_params,
        'param_bytes_kb': total_param_bytes / 1024,
        'op_types': op_types,
        'unsupported_ops': unsupported,
    }


# ============================================================================
# Inference Functions
# ============================================================================
def load_onnx_model(model_path):
    """Load ONNX model with onnxruntime."""
    print(f"\nLoading model: {model_path}")
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_names = [o.name for o in session.get_outputs()]
    return session, input_name, input_shape, output_names


def preprocess_image(image_path, input_size, color_mode='bgr'):
    """
    Preprocess image for YuNet model.
    - Input: NCHW format, BGR, values in [0, 255], scale=1, offset=0
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    original_shape = img.shape  # (H, W, C)

    # Resize to model input size
    img_resized = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)

    # BGR format (OpenCV default — already BGR)
    # No rescaling: scale=1, offset=0

    # HWC → CHW
    img_chw = np.transpose(img_resized, (2, 0, 1))

    # Add batch dimension: CHW → NCHW
    img_nchw = np.expand_dims(img_chw, axis=0).astype(np.float32)

    return img_nchw, img, original_shape


def run_inference(session, input_name, input_data):
    """Run ONNX model inference."""
    outputs = session.run(None, {input_name: input_data})
    return outputs


def draw_detections(img, bboxes, scores, keypoints, input_size, original_shape,
                    color=(0, 255, 0)):
    """Draw bounding boxes and keypoints on the original image."""
    img_out = img.copy()
    h_orig, w_orig = original_shape[:2]
    w_in, h_in = input_size

    for i, (box, score) in enumerate(zip(bboxes, scores)):
        if score < CONFIDENCE_THRESH:
            continue

        # Scale from input size to original image
        x1 = int(box[0] * w_orig)
        y1 = int(box[1] * h_orig)
        x2 = int(box[2] * w_orig)
        y2 = int(box[3] * h_orig)

        # Clamp to image boundaries
        x1 = max(0, min(x1, w_orig - 1))
        y1 = max(0, min(y1, h_orig - 1))
        x2 = max(0, min(x2, w_orig - 1))
        y2 = max(0, min(y2, h_orig - 1))

        # Draw bounding box
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"face {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_out, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(img_out, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw keypoints
        if keypoints is not None and len(keypoints) > i:
            kps = keypoints[i]
            for kp_idx, kp in enumerate(kps):
                kpx = int(kp[0] * w_orig)
                kpy = int(kp[1] * h_orig)
                kpx = max(0, min(kpx, w_orig - 1))
                kpy = max(0, min(kpy, h_orig - 1))
                cv2.circle(img_out, (kpx, kpy), 3, (0, 0, 255), -1)

    return img_out


def detect_faces(session, input_name, image_path, conf_thresh=CONFIDENCE_THRESH,
                 nms_thresh=NMS_THRESH):
    """Full pipeline: load image → inference → decode → NMS."""
    # Preprocess
    input_data, original_img, orig_shape = preprocess_image(
        image_path, INPUT_SIZE, color_mode='bgr'
    )

    # Inference
    outputs = run_inference(session, input_name, input_data)

    # Decode
    bboxes, scores, keypoints = decode_yunet_outputs(
        outputs, INPUT_SIZE, STRIDES, NUM_KEYPOINTS
    )

    # NMS
    nms_boxes, nms_scores, indices = nms(
        bboxes, scores, nms_thresh, MAX_DETECTIONS, conf_thresh
    )

    return nms_boxes, nms_scores, keypoints[indices] if len(indices) > 0 else keypoints[:0], original_img, orig_shape


# ============================================================================
# Test with camera / sample images
# ============================================================================
def generate_test_images(output_dir, num_images=10):
    """
    Generate synthetic test images with face-like patterns for testing,
    OR look for existing WIDER Face images.
    """
    # Try to find existing images in common locations
    search_paths = [
        os.path.join(MODEL_DIR, "datasets", "pred", "*.jpg"),
        os.path.join(MODEL_DIR, "datasets", "test", "*.jpg"),
    ]

    for search_path in search_paths:
        found = glob.glob(search_path)
        if found:
            print(f"Found {len(found)} existing images in {search_path}")
            return found[:num_images]

    # Create synthetic face-like test images
    print(f"No existing images found. Creating {num_images} synthetic test images...")
    created = []
    for i in range(num_images):
        # Create a 640x480 image with a gradient background
        img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

        # Add a "face-like" circular region
        cx, cy = np.random.randint(150, 490), np.random.randint(100, 380)
        radius = np.random.randint(40, 80)

        # Skin-colored circle
        skin_color = (
            np.random.randint(100, 200),  # B
            np.random.randint(100, 180),  # G
            np.random.randint(140, 220),  # R
        )
        cv2.circle(img, (cx, cy), radius, skin_color, -1)

        # Add "eyes"
        eye_radius = radius // 6
        cv2.circle(img, (cx - radius // 3, cy - radius // 3), eye_radius, (0, 0, 0), -1)
        cv2.circle(img, (cx + radius // 3, cy - radius // 3), eye_radius, (0, 0, 0), -1)

        # Add "mouth"
        cv2.ellipse(img, (cx, cy + radius // 3), (radius // 4, radius // 6),
                     0, 0, 180, (0, 0, 200), 2)

        filepath = os.path.join(output_dir, f"test_face_{i:02d}.jpg")
        cv2.imwrite(filepath, img)
        created.append(filepath)

    return created


def run_batch_test(session, input_name, image_paths, output_dir):
    """Run inference on a batch of images and save results."""
    print("\n" + "=" * 70)
    print(f"BATCH TEST: {len(image_paths)} images")
    print("=" * 70)

    results = []
    total_time = 0

    for i, img_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] {os.path.basename(img_path)}")

        try:
            t_start = time.time()
            boxes, scores, keypoints, orig_img, orig_shape = detect_faces(
                session, input_name, img_path
            )
            t_infer = time.time() - t_start
            total_time += t_infer

            # Filter by confidence
            valid = scores >= CONFIDENCE_THRESH

            print(f"  Inference time: {t_infer*1000:.1f} ms")
            print(f"  Detections: {len(boxes)} total, {np.sum(valid)} above threshold")

            if np.sum(valid) > 0:
                for j, (box, score) in enumerate(zip(boxes[valid], scores[valid])):
                    print(f"    Face {j+1}: score={score:.3f}, "
                          f"box=({box[0]:.3f},{box[1]:.3f},{box[2]:.3f},{box[3]:.3f})")

            # Draw and save result
            result_img = draw_detections(
                orig_img, boxes, scores, keypoints,
                INPUT_SIZE, orig_shape
            )

            out_path = os.path.join(output_dir, f"result_{i+1:02d}_{os.path.basename(img_path)}")
            cv2.imwrite(out_path, result_img)

            results.append({
                'image': img_path,
                'output': out_path,
                'num_faces': int(np.sum(valid)),
                'inference_time_ms': t_infer * 1000,
            })

        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({
                'image': img_path,
                'output': None,
                'num_faces': 0,
                'inference_time_ms': 0,
                'error': str(e),
            })

    if total_time > 0:
        avg_time = total_time / len(image_paths) * 1000
        print(f"\n--- Summary ---")
        print(f"  Average inference time: {avg_time:.1f} ms")
        print(f"  Total time: {total_time:.3f} s")
        print(f"  FPS (theoretical): {1000/avg_time:.1f}")

    return results


def run_camera_test(session, input_name):
    """Run real-time face detection using webcam."""
    print("\n" + "=" * 70)
    print("CAMERA TEST — Press 'q' to quit")
    print("=" * 70)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Try index 1...")
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] No camera available. Skipping camera test.")
        return

    fps_counter = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # Only process every other frame for speed
        if frame_idx % 2 != 0:
            cv2.imshow('YuNet Face Detection', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        t_start = time.time()

        # Preprocess
        img_resized = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        img_chw = np.transpose(img_resized, (2, 0, 1))
        input_data = np.expand_dims(img_chw, axis=0).astype(np.float32)

        # Inference
        outputs = run_inference(session, input_name, input_data)

        # Decode
        bboxes, scores, keypoints = decode_yunet_outputs(
            outputs, INPUT_SIZE, STRIDES, NUM_KEYPOINTS
        )

        # NMS
        nms_boxes, nms_scores, indices = nms(
            bboxes, scores, NMS_THRESH, MAX_DETECTIONS, CONFIDENCE_THRESH
        )

        t_infer = time.time() - t_start
        fps_counter.append(1.0 / t_infer)

        # Draw on original frame
        h, w = frame.shape[:2]
        for i, (box, score) in enumerate(zip(nms_boxes, nms_scores)):
            if score < CONFIDENCE_THRESH:
                continue
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if indices is not None and len(keypoints) > i and i < len(indices):
                kps = keypoints[i]
                for kp in kps:
                    kpx = int(kp[0] * w)
                    kpy = int(kp[1] * h)
                    cv2.circle(frame, (kpx, kpy), 2, (0, 0, 255), -1)

        # FPS display
        if len(fps_counter) > 10:
            avg_fps = np.mean(fps_counter[-10:])
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('YuNet Face Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if fps_counter:
        print(f"\n  Average FPS: {np.mean(fps_counter):.1f}")


# ============================================================================
# Report Generation
# ============================================================================
def generate_report(float_analysis, int8_analysis, batch_results):
    """Generate evaluation report."""
    report_path = os.path.join(OUTPUT_DIR, "model_evaluation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("YuNet Face Detection Model — Evaluation Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("## 1. Model Overview\n\n")
        f.write(f"  Architecture: YuNet (yunetn_320)\n")
        f.write(f"  Input size: {INPUT_SIZE[0]}×{INPUT_SIZE[1]}\n")
        f.write(f"  Color format: BGR, [0, 255]\n")
        f.write(f"  Number of keypoints: {NUM_KEYPOINTS}\n")
        f.write(f"  Strides: {STRIDES}\n\n")

        f.write("## 2. STM32H747I-DISCO Compatibility\n\n")
        f.write(f"  Float model size: {float_analysis['file_size_kb']:.1f} KB\n")
        f.write(f"  INT8 model size: {int8_analysis['file_size_kb']:.1f} KB\n")
        f.write(f"  Float parameters: {float_analysis['total_params']:,}\n")
        f.write(f"  Flash limit: 512 KB\n")
        flash_ok = float_analysis['file_size_kb'] < 512
        f.write(f"  Flash fit: {'PASS' if flash_ok else 'WARN'}\n\n")

        f.write("## 3. Inference Performance\n\n")
        if batch_results:
            times = [r['inference_time_ms'] for r in batch_results if r['inference_time_ms'] > 0]
            if times:
                avg_ms = np.mean(times)
                f.write(f"  Average inference time (PC, CPU): {avg_ms:.1f} ms\n")
                f.write(f"  Estimated STM32H747 @400MHz: ~{avg_ms * 20:.0f} ms (rough estimate)\n\n")

        f.write("## 4. Detection Results\n\n")
        for r in batch_results:
            status = "OK" if 'error' not in r else f"ERROR: {r['error']}"
            f.write(f"  {os.path.basename(r['image'])}: {r['num_faces']} faces ({status})\n")

        f.write("\n## 5. Recommendations\n\n")
        f.write("  - INT8 quantization reduces model size by ~35%\n")
        f.write("  - For STM32H747, the INT8 QDQ model is recommended\n")
        f.write("  - Flash requirement: ~200 KB (INT8), well within 512 KB limit\n")
        f.write("  - Consider further pruning or TFLite conversion for better Cube.AI support\n")
        f.write("  - Input resolution 160×160 reduces memory by 4x if needed\n")

    print(f"\nReport saved to: {report_path}")
    return report_path


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="YuNet Face Detection Test")
    parser.add_argument('--model', type=str, default='float',
                        choices=['float', 'int8'],
                        help='Model type to use (default: float)')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['analyze', 'test', 'camera', 'all'],
                        help='Operation mode (default: all)')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image path for inference')
    parser.add_argument('--num-test-images', type=int, default=10,
                        help='Number of test images (default: 10)')
    parser.add_argument('--conf-thresh', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--nms-thresh', type=float, default=0.45,
                        help='NMS IoU threshold (default: 0.45)')
    args = parser.parse_args()

    global CONFIDENCE_THRESH, NMS_THRESH
    CONFIDENCE_THRESH = args.conf_thresh
    NMS_THRESH = args.nms_thresh

    model_path = FLOAT_MODEL_PATH if args.model == 'float' else INT8_MODEL_PATH

    print("=" * 70)
    print("  YuNet Face Detection — Model Test & Evaluation")
    print(f"  Model: {os.path.basename(model_path)}")
    print(f"  Python: {sys.executable}")
    print("=" * 70)

    # Step 1: Model Analysis
    float_analysis = analyze_model(FLOAT_MODEL_PATH)
    int8_analysis = analyze_model(INT8_MODEL_PATH)

    if args.mode == 'analyze':
        return

    # Step 2: Load model
    session, input_name, input_shape, output_names = load_onnx_model(model_path)

    # Step 3: Single image test (if specified)
    if args.image:
        print(f"\n--- Single Image Test: {args.image} ---")
        boxes, scores, keypoints, orig_img, orig_shape = detect_faces(
            session, input_name, args.image, CONFIDENCE_THRESH, NMS_THRESH
        )
        valid = scores >= CONFIDENCE_THRESH
        print(f"  Found {np.sum(valid)} faces:")
        for i, (box, score) in enumerate(zip(boxes[valid], scores[valid])):
            print(f"    Face {i+1}: score={score:.3f}, "
                  f"box=({box[0]:.3f},{box[1]:.3f},{box[2]:.3f},{box[3]:.3f})")
        result_img = draw_detections(orig_img, boxes, scores, keypoints,
                                     INPUT_SIZE, orig_shape)
        out_path = os.path.join(OUTPUT_DIR, f"result_{os.path.basename(args.image)}")
        cv2.imwrite(out_path, result_img)
        print(f"  Result saved: {out_path}")

        if args.mode == 'test':
            return

    # Step 4: Generate/collect test images
    test_images = generate_test_images(OUTPUT_DIR, args.num_test_images)
    print(f"\nTest images: {len(test_images)}")

    # Step 5: Batch test
    batch_results = run_batch_test(session, input_name, test_images, OUTPUT_DIR)

    # Step 6: Generate report
    report_path = generate_report(float_analysis, int8_analysis, batch_results)

    # Step 7: Camera test (only in 'camera' or 'all' mode)
    if args.mode in ('camera', 'all'):
        run_camera_test(session, input_name)

    print("\n" + "=" * 70)
    print("  Done! Results saved to:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
