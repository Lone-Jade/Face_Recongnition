#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PC test: verify YuNet postprocessing logic.
Compares the STM32 C implementation against the correct Python reference.

Usage:
    D:/PythonConda/envs/pytorch_face_detection/python.exe PC_test/verify_postprocess.py
    D:/PythonConda/envs/pytorch_face_detection/python.exe PC_test/verify_postprocess.py --image path/to/face.jpg
"""

import os, sys, time, argparse
import numpy as np
import cv2
import onnxruntime as ort

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model_stm32", "yunetn_320.onnx")
VALIDATION_NPZ = os.path.join(BASE_DIR, "model_stm32", "stedgeai_validation.npz")

INPUT_SIZE = (320, 320)
STRIDES = [8, 16, 32]
CONF_THRESH = 0.5
NMS_THRESH = 0.45

# ============================================================
# 1. Correct Python reference postprocessing (from test_yunet.py)
# ============================================================
def decode_yunet_correct(outputs):
    """
    CORRECT implementation: outputs sorted by shape, anchors at grid_top_left * stride.
    """
    # Sort outputs by shape (same as ST model zoo code)
    nets_out_sorted = sorted(outputs, key=lambda x: x.shape)

    sorted_strides = sorted(STRIDES, reverse=True)  # [32, 16, 8]
    # Generate anchor centers for each stride
    anchor_centers = []
    for stride in sorted_strides:
        grid_h = INPUT_SIZE[1] // stride
        grid_w = INPUT_SIZE[0] // stride
        centers = np.stack(np.mgrid[:grid_h, :grid_w][::-1], axis=-1)
        centers = (centers * stride).astype(np.float32).reshape(-1, 2)
        anchor_centers.append(centers)

    all_scores = []
    all_bboxes = []

    for idx, stride in enumerate(sorted_strides):
        cls_pred = nets_out_sorted[4 * idx + 0].reshape(-1, 1)
        obj_pred = nets_out_sorted[4 * idx + 1].reshape(-1, 1)
        reg_pred = nets_out_sorted[4 * idx + 2].reshape(-1, 4)

        bbox_cxy = reg_pred[:, :2] * stride + anchor_centers[idx]
        bbox_wh  = np.exp(reg_pred[:, 2:]) * stride

        tl_x = bbox_cxy[:, 0] - bbox_wh[:, 0] / 2.0
        tl_y = bbox_cxy[:, 1] - bbox_wh[:, 1] / 2.0
        br_x = bbox_cxy[:, 0] + bbox_wh[:, 0] / 2.0
        br_y = bbox_cxy[:, 1] + bbox_wh[:, 1] / 2.0

        bboxes = np.stack([tl_x, tl_y, br_x, br_y], axis=-1)
        scores = (cls_pred * obj_pred).reshape(-1)

        all_scores.append(scores)
        all_bboxes.append(bboxes)

    all_scores = np.concatenate(all_scores, axis=0)
    all_bboxes = np.concatenate(all_bboxes, axis=0)
    return all_bboxes, all_scores


# ============================================================
# 2. Convert ONNX outputs to ST AI format (CHW for spatial tensors)
# ============================================================
def onnx_to_stai_format(onnx_outputs):
    """
    ONNX output format:
      cls_X:  (1, N, 1)  — same as ST AI
      obj_X:  (1, N, 1)  — same
      bbox_X: (1, N, 4)  — ONNX uses interleaved [dx,dy,dw,dh] per anchor
      kps_X:  (1, N, 10) — ONNX uses interleaved keypoints per anchor

    ST AI format (CHW):
      cls_X:  (1, N, 1)  — same
      obj_X:  (1, N, 1)  — same
      bbox_X: (1, 4, N)  — CHW: channel-first planar
      kps_X:  (1, 10, N) — CHW: channel-first planar

    This function converts ONNX → ST AI format.
    """
    stai_outputs = []
    for out in onnx_outputs:
        if out.ndim == 3 and out.shape[2] in (4, 10):
            # bbox (N, 4) or kps (N, 10) → transpose to (C, N) then reshape
            # out shape (1, N, C) → (1, C, N)
            stai_outputs.append(np.transpose(out, (0, 2, 1)))
        else:
            stai_outputs.append(out)
    return stai_outputs


# ============================================================
# 3. STM32 C implementation (buggy version - with +0.5 offset)
# ============================================================
def decode_yunet_stm32_buggy(onnx_outputs):
    """
    BUGGY implementation (current STM32 C code):
    outputs assumed in [cls_8, cls_16, cls_32, obj_8, obj_16, obj_32, bbox_8, ...] order,
    anchors at grid CENTER ((j+0.5)*stride).
    """
    outputs = onnx_to_stai_format(onnx_outputs)  # Match ST AI CHW layout
    gs_map = {0: 40, 1: 20, 2: 10}
    all_scores = []
    all_bboxes = []

    for k, stride in enumerate(STRIDES):  # [8, 16, 32]
        gs = gs_map[k]
        cls  = outputs[k].reshape(-1)       # (1, N, 1) → (N,)
        obj  = outputs[k + 3].reshape(-1)
        bbox = outputs[k + 6].flatten()     # (1, 4, N) → flat 1D (match C pointer access)

        for i in range(gs):
            for j in range(gs):
                loc = i * gs + j
                score = cls[loc] * obj[loc]
                if score < CONF_THRESH:
                    continue

                dx = bbox[0 * gs * gs + loc]
                dy = bbox[1 * gs * gs + loc]
                dw = bbox[2 * gs * gs + loc]
                dh = bbox[3 * gs * gs + loc]

                # BUG: +0.5f offset!
                cx = (float(j) + 0.5 + dx) * stride
                cy = (float(i) + 0.5 + dy) * stride
                bw = np.exp(dw) * stride
                bh = np.exp(dh) * stride

                x1 = cx - bw * 0.5
                y1 = cy - bh * 0.5
                x2 = cx + bw * 0.5
                y2 = cy + bh * 0.5

                x1 = max(0.0, min(x1, 320.0))
                y1 = max(0.0, min(y1, 320.0))
                x2 = max(0.0, min(x2, 320.0))
                y2 = max(0.0, min(y2, 320.0))

                if x2 <= x1 or y2 <= y1:
                    continue

                all_scores.append(score)
                all_bboxes.append([x1, y1, x2, y2])

    return np.array(all_bboxes), np.array(all_scores)


# ============================================================
# 3. STM32 C implementation (fixed - anchor at grid corner)
# ============================================================
def decode_yunet_stm32_fixed(onnx_outputs):
    """
    FIXED implementation: same output order assumption, but anchors at grid corner.
    """
    outputs = onnx_to_stai_format(onnx_outputs)  # Match ST AI CHW layout
    gs_map = {0: 40, 1: 20, 2: 10}
    all_scores = []
    all_bboxes = []

    for k, stride in enumerate(STRIDES):
        gs = gs_map[k]
        cls  = outputs[k].reshape(-1)
        obj  = outputs[k + 3].reshape(-1)
        bbox = outputs[k + 6].flatten()     # (1, 4, N) → flat 1D

        for i in range(gs):
            for j in range(gs):
                loc = i * gs + j
                score = cls[loc] * obj[loc]
                if score < CONF_THRESH:
                    continue

                dx = bbox[0 * gs * gs + loc]
                dy = bbox[1 * gs * gs + loc]
                dw = bbox[2 * gs * gs + loc]
                dh = bbox[3 * gs * gs + loc]

                # FIXED: anchor at (j, i) * stride, NOT (j+0.5, i+0.5) * stride
                cx = (float(j) + dx) * stride
                cy = (float(i) + dy) * stride
                bw = np.exp(dw) * stride
                bh = np.exp(dh) * stride

                x1 = cx - bw * 0.5
                y1 = cy - bh * 0.5
                x2 = cx + bw * 0.5
                y2 = cy + bh * 0.5

                x1 = max(0.0, min(x1, 320.0))
                y1 = max(0.0, min(y1, 320.0))
                x2 = max(0.0, min(x2, 320.0))
                y2 = max(0.0, min(y2, 320.0))

                if x2 <= x1 or y2 <= y1:
                    continue

                all_scores.append(score)
                all_bboxes.append([x1, y1, x2, y2])

    return np.array(all_bboxes), np.array(all_scores)


# ============================================================
# 4. NMS
# ============================================================
def apply_nms(bboxes, scores, iou_thresh=0.45, conf_thresh=0.5, max_dets=50):
    keep = scores >= conf_thresh
    bboxes = bboxes[keep]
    scores = scores[keep]
    if len(bboxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,))

    x1 = bboxes[:, 0]; y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]; y2 = bboxes[:, 3]
    w = x2 - x1; h = y2 - y1
    cv_boxes = np.stack([x1, y1, w, h], axis=-1)

    indices = cv2.dnn.NMSBoxes(cv_boxes.tolist(), scores.tolist(),
                                conf_thresh, iou_thresh)
    if len(indices) == 0:
        return np.zeros((0, 4)), np.zeros((0,))
    indices = np.array(indices).flatten()[:max_dets]
    return bboxes[indices], scores[indices]


# ============================================================
# 5. Main test
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Verify YuNet postprocessing")
    parser.add_argument('--image', type=str, default=None, help="Test image path")
    args = parser.parse_args()

    print("=" * 70)
    print("  YuNet Postprocessing Verification")
    print("=" * 70)

    # Load ONNX model
    print(f"\nLoading model: {MODEL_PATH}")
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    onnx_outputs_names = [o.name for o in session.get_outputs()]
    print(f"  ONNX output names: {onnx_outputs_names}")

    # Find or use test image
    if args.image and os.path.exists(args.image):
        test_img_path = args.image
    else:
        # Use the first built-in image if possible, otherwise create a synthetic one
        img_paths = [
            os.path.join(BASE_DIR, "WIDERFace", "datasets", "pred", "*.jpg"),
        ]
        import glob
        found = []
        for p in img_paths:
            found.extend(glob.glob(p))
        if found:
            test_img_path = found[0]
        else:
            # Create a simple gradient with a dark center region
            test_img_path = os.path.join(BASE_DIR, "PC_test", "_test_input.jpg")
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img[:] = [180, 160, 140]
            cv2.circle(img, (300, 240), 100, (80, 60, 40), -1)
            cv2.circle(img, (270, 200), 15, (20, 20, 20), -1)
            cv2.circle(img, (330, 200), 15, (20, 20, 20), -1)
            cv2.imwrite(test_img_path, img)
            print(f"\nCreated synthetic test image: {test_img_path}")

    print(f"Test image: {test_img_path}")

    # Preprocess
    img = cv2.imread(test_img_path)
    img_resized = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    img_chw = np.transpose(img_resized, (2, 0, 1))
    input_data = np.expand_dims(img_chw, axis=0).astype(np.float32)

    # Run inference
    t0 = time.time()
    onnx_outputs = session.run(None, {input_name: input_data})
    print(f"Inference time: {(time.time()-t0)*1000:.1f} ms")

    # Print ONNX output shapes and names
    print("\nONNX outputs (raw order):")
    for i, (name, out) in enumerate(zip(onnx_outputs_names, onnx_outputs)):
        print(f"  [{i}] {name}: shape={out.shape}, range=[{out.min():.4f}, {out.max():.4f}]")

    # 1. Correct decoding
    bboxes_correct, scores_correct = decode_yunet_correct(onnx_outputs)
    nms_boxes_corr, nms_scores_corr = apply_nms(bboxes_correct, scores_correct,
                                                  NMS_THRESH, CONF_THRESH)
    print(f"\n--- CORRECT (Python ref) ---")
    print(f"  Raw detections: {len(scores_correct)}")
    print(f"  After NMS: {len(nms_scores_corr)}")
    for i in range(len(nms_scores_corr)):
        b = nms_boxes_corr[i]; s = nms_scores_corr[i]
        print(f"  Face {i}: score={s:.3f}, box=({b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f})")

    # 2. Buggy STM32 decoding
    bboxes_bug, scores_bug = decode_yunet_stm32_buggy(onnx_outputs)
    nms_boxes_bug, nms_scores_bug = apply_nms(bboxes_bug, scores_bug,
                                                NMS_THRESH, CONF_THRESH)
    print(f"\n--- BUGGY (STM32 current, anchor +0.5) ---")
    print(f"  Raw detections: {len(scores_bug)}")
    print(f"  After NMS: {len(nms_scores_bug)}")
    for i in range(min(len(nms_scores_bug), 20)):
        b = nms_boxes_bug[i]; s = nms_scores_bug[i]
        print(f"  Box {i}: score={s:.3f}, ({b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f})")

    # 3. Fixed STM32 decoding
    bboxes_fix, scores_fix = decode_yunet_stm32_fixed(onnx_outputs)
    nms_boxes_fix, nms_scores_fix = apply_nms(bboxes_fix, scores_fix,
                                                NMS_THRESH, CONF_THRESH)
    print(f"\n--- FIXED (STM32 corrected, anchor at grid corner) ---")
    print(f"  Raw detections: {len(scores_fix)}")
    print(f"  After NMS: {len(nms_scores_fix)}")
    for i in range(len(nms_scores_fix)):
        b = nms_boxes_fix[i]; s = nms_scores_fix[i]
        print(f"  Face {i}: score={s:.3f}, box=({b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f})")

    # Draw results
    result_correct = img_resized.copy()
    result_buggy = img_resized.copy()
    result_fixed = img_resized.copy()

    for box, score in zip(nms_boxes_corr, nms_scores_corr):
        if score < CONF_THRESH: continue
        x1,y1,x2,y2 = map(int, [box[0],box[1],box[2],box[3]])
        cv2.rectangle(result_correct, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(result_correct, f"{score:.2f}", (x1,y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    for box, score in zip(nms_boxes_bug, nms_scores_bug):
        if score < CONF_THRESH: continue
        x1,y1,x2,y2 = map(int, [box[0],box[1],box[2],box[3]])
        cv2.rectangle(result_buggy, (x1,y1), (x2,y2), (0,0,255), 1)
        cv2.putText(result_buggy, f"{score:.2f}", (x1,y1-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)

    for box, score in zip(nms_boxes_fix, nms_scores_fix):
        if score < CONF_THRESH: continue
        x1,y1,x2,y2 = map(int, [box[0],box[1],box[2],box[3]])
        cv2.rectangle(result_fixed, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(result_fixed, f"{score:.2f}", (x1,y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    out_dir = os.path.join(BASE_DIR, "PC_test")
    cv2.imwrite(os.path.join(out_dir, "result_correct_python_ref.jpg"), result_correct)
    cv2.imwrite(os.path.join(out_dir, "result_buggy_stm32_before_fix.jpg"), result_buggy)
    cv2.imwrite(os.path.join(out_dir, "result_fixed_stm32_after_fix.jpg"), result_fixed)

    print(f"\nResults saved to {out_dir}/")
    print(f"  result_correct_python_ref.jpg     — Green boxes (correct)")
    print(f"  result_buggy_stm32_before_fix.jpg — Red boxes (buggy: many small, shifted)")
    print(f"  result_fixed_stm32_after_fix.jpg  — Blue boxes (fixed)")
    print("\nDone!")

    # Summary
    if len(nms_scores_corr) > 0 and len(nms_scores_bug) > 0:
        bug_density = len(nms_scores_bug) / max(len(nms_scores_corr), 1)
        print(f"\nDiagnosis: buggy version produces {bug_density:.1f}x as many detections")
        print(f"Root cause: anchor center uses (j+0.5)*stride instead of j*stride")

if __name__ == "__main__":
    main()
