#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep diagnosis: why 1 face → 81 detections on STM32?

1. ONNX inference on test image → decode using EXACT STM32 C logic
2. Compare ONNX outputs vs ST AI validation NPZ outputs
3. Per-stride breakdown: positive anchors, scores, NMS stats
4. Draw all intermediate results

Usage:
    D:/PythonConda/envs/pytorch_face_detection/python.exe PC_test/diagnose_detections.py
"""

import os, sys, time, argparse, struct
import numpy as np
import cv2
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model_stm32", "yunetn_320.onnx")
VALIDATION_NPZ = os.path.join(BASE_DIR, "model_stm32", "stedgeai_validation.npz")
TEST_IMG = os.path.join(BASE_DIR,
    "WIDERFace", "WIDER_test", "images", "29--Students_Schoolkids",
    "29_Students_Schoolkids_Students_Schoolkids_29_1.jpg")
OUT_DIR = os.path.join(BASE_DIR, "PC_test")

INPUT_SIZE = (320, 320)
STRIDES = [8, 16, 32]
GS_MAP = {8: 40, 16: 20, 32: 10}

# ============================================================
# 1. STM32 C logic — faithfully replicated in Python
# ============================================================
def onnx_to_chw(onnx_outputs):
    """Convert ONNX bbox/kps from interleaved (NxC) to CHW flat (CxN)."""
    result = list(onnx_outputs)
    for i in range(12):
        arr = onnx_outputs[i]
        # bbox_8/16/32 have shape (1, N, 4) → need (1, 4, N)
        # kps_8/16/32 have shape (1, N, 10) → need (1, 10, N)
        if arr.ndim == 3 and arr.shape[2] in (4, 10):
            result[i] = np.transpose(arr, (0, 2, 1))
    return result


def decode_stm32(onnx_outputs, conf_thresh, sort_and_nms=True, nms_thresh=0.45):
    """
    STM32 C logic: outputs in [cls_8,cls_16,cls_32, obj_8,obj_16,obj_32,
                              bbox_8,bbox_16,bbox_32, kps_8,kps_16,kps_32]
    Converts ONNX format (interleaved) to CHW flat first, matching ST AI output.
    Anchor at grid corner (j*stride, i*stride).
    Returns list of (x1,y1,x2,y2,score,stride_idx).
    """
    outputs = onnx_to_chw(onnx_outputs)  # Convert ONNX interleaved → CHW
    all_dets = []

    for k, stride in enumerate(STRIDES):
        gs = GS_MAP[stride]
        cls  = outputs[k].flatten()
        obj  = outputs[k + 3].flatten()
        bbox = outputs[k + 6].flatten()  # Now in CHW flat layout

        pos_count = 0
        for i in range(gs):
            for j in range(gs):
                loc = i * gs + j
                score = cls[loc] * obj[loc]
                if score < conf_thresh:
                    continue
                pos_count += 1

                dx = bbox[0 * gs * gs + loc]
                dy = bbox[1 * gs * gs + loc]
                dw = bbox[2 * gs * gs + loc]
                dh = bbox[3 * gs * gs + loc]

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

                all_dets.append((x1, y1, x2, y2, score, k))

        print(f"  stride {stride:2d} (gs={gs:2d}): score>={conf_thresh} → {pos_count} raw")

    if not sort_and_nms:
        return all_dets

    # Sort by score descending
    all_dets.sort(key=lambda d: -d[4])

    # NMS
    keep = []
    suppressed = [False] * len(all_dets)
    for a in range(len(all_dets)):
        if suppressed[a]:
            continue
        keep.append(a)
        area_a = (all_dets[a][2] - all_dets[a][0]) * (all_dets[a][3] - all_dets[a][1])
        for b in range(a + 1, len(all_dets)):
            if suppressed[b]:
                continue
            xx1 = max(all_dets[a][0], all_dets[b][0])
            yy1 = max(all_dets[a][1], all_dets[b][1])
            xx2 = min(all_dets[a][2], all_dets[b][2])
            yy2 = min(all_dets[a][3], all_dets[b][3])
            iw = xx2 - xx1
            ih = yy2 - yy1
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            area_b = (all_dets[b][2] - all_dets[b][0]) * (all_dets[b][3] - all_dets[b][1])
            iou = inter / (area_a + area_b - inter)
            if iou > nms_thresh:
                suppressed[b] = True

    result = [all_dets[i] for i in keep]
    return result


def decode_python_ref(outputs, conf_thresh, nms_thresh=0.45):
    """Python reference: sorted outputs, proper grouping."""
    nets_out_sorted = sorted(outputs, key=lambda x: x.shape)
    sorted_strides = sorted(STRIDES, reverse=True)

    anchor_centers = []
    for stride in sorted_strides:
        grid_h = INPUT_SIZE[1] // stride
        grid_w = INPUT_SIZE[0] // stride
        centers = np.stack(np.mgrid[:grid_h, :grid_w][::-1], axis=-1)
        centers = (centers * stride).astype(np.float32).reshape(-1, 2)
        anchor_centers.append(centers)

    all_scores, all_bboxes = [], []
    for idx, stride in enumerate(sorted_strides):
        cls_pred = nets_out_sorted[4 * idx + 0].reshape(-1, 1)
        obj_pred = nets_out_sorted[4 * idx + 1].reshape(-1, 1)
        reg_pred = nets_out_sorted[4 * idx + 2].reshape(-1, 4)
        bbox_cxy = reg_pred[:, :2] * stride + anchor_centers[idx]
        bbox_wh = np.exp(reg_pred[:, 2:]) * stride
        tl_x = bbox_cxy[:, 0] - bbox_wh[:, 0] / 2.0
        tl_y = bbox_cxy[:, 1] - bbox_wh[:, 1] / 2.0
        br_x = bbox_cxy[:, 0] + bbox_wh[:, 0] / 2.0
        br_y = bbox_cxy[:, 1] + bbox_wh[:, 1] / 2.0
        bboxes = np.stack([tl_x, tl_y, br_x, br_y], axis=-1)
        scores = (cls_pred * obj_pred).reshape(-1)
        all_scores.append(scores)
        all_bboxes.append(bboxes)

    all_scores = np.concatenate(all_scores)
    all_bboxes = np.concatenate(all_bboxes)

    keep = all_scores >= conf_thresh
    boxes = all_bboxes[keep]; scores = all_scores[keep]
    if len(boxes) == 0:
        return []
    x1,y1,x2,y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    indices = cv2.dnn.NMSBoxes(
        np.stack([x1,y1,x2-x1,y2-y1], axis=-1).tolist(),
        scores.tolist(), conf_thresh, nms_thresh)
    if len(indices) == 0:
        return []
    indices = np.array(indices).flatten()[:50]
    result = []
    for i in indices:
        result.append((float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                       float(scores[i]), 0))
    return result


# ============================================================
# 2. Draw
# ============================================================
def draw_all_dets(img, dets_raw, dets_nms, title, out_path):
    """Draw ALL raw detections in red, NMS survivors in green."""
    out = img.copy()
    for det in dets_raw:
        x1,y1,x2,y2,score,k = det
        # Very faint red for raw
        cv2.rectangle(out, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,200), 1)
    for det in dets_nms:
        x1,y1,x2,y2,score,k = det
        cv2.rectangle(out, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.putText(out, f"{score:.2f}", (int(x1), int(y1)-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    cv2.putText(out, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    if out_path:
        cv2.imwrite(out_path, out)
    return out


# ============================================================
# 3. Compare ONNX vs ST AI validation
# ============================================================
def compare_with_validation(onnx_outputs, npz_data, sample_idx=0):
    """Compare ONNX inference outputs with ST AI compiled outputs from NPZ."""
    print("\n" + "=" * 70)
    print(f"COMPARISON: ONNX vs ST AI validation (sample {sample_idx})")
    print("=" * 70)

    onnx_outs_flat = []
    stai_outs_flat = []

    for i in range(12):
        onnx_arr = onnx_outputs[i].flatten()
        stai_key = f'm_outputs_{i+1}'
        stai_arr = npz_data[stai_key][sample_idx].flatten()

        onnx_outs_flat.append(onnx_arr)
        stai_outs_flat.append(stai_arr)

        diff = onnx_arr - stai_arr
        max_abs = np.max(np.abs(diff))
        mean_abs = np.mean(np.abs(diff))
        # Cosine similarity
        dot = np.dot(onnx_arr, stai_arr)
        norm_o = np.linalg.norm(onnx_arr)
        norm_s = np.linalg.norm(stai_arr)
        cos_sim = dot / (norm_o * norm_s) if norm_o > 0 and norm_s > 0 else 0

        name = ["cls_8","cls_16","cls_32","obj_8","obj_16","obj_32",
                "bbox_8","bbox_16","bbox_32","kps_8","kps_16","kps_32"][i]
        flag = "⚠️  DIFFER!" if max_abs > 0.1 else "✓ OK"
        print(f"  {name:8s}: max|diff|={max_abs:.4f}, mean|diff|={mean_abs:.4f}, cos_sim={cos_sim:.6f}  {flag}")

    return onnx_outs_flat, stai_outs_flat


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Diagnose face detection")
    parser.add_argument('--image', type=str, default=TEST_IMG,
                        help='Test image path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.45, help='NMS threshold')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        print("Trying alternative...")
        alt = os.path.join(BASE_DIR, "WIDERFace", "WIDER_val", "images",
                           "0--Parade", "0_Parade_marchingband_1_1004.jpg")
        if os.path.exists(alt):
            args.image = alt
            print(f"Using: {alt}")
        else:
            sys.exit(1)

    print("=" * 70)
    print("  YuNet Detection Diagnosis")
    print(f"  Image: {args.image}")
    print(f"  Conf threshold: {args.conf}, NMS threshold: {args.nms}")
    print("=" * 70)

    # Load ONNX
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Preprocess
    img = cv2.imread(args.image)
    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    img_chw = np.transpose(img_resized, (2, 0, 1))
    input_data = np.expand_dims(img_chw, axis=0).astype(np.float32)

    # ONNX inference
    t0 = time.time()
    onnx_outputs = session.run(None, {input_name: input_data})
    print(f"\nONNX inference: {(time.time()-t0)*1000:.1f} ms")

    # ---- STM32 C logic decode ----
    print("\n--- STM32 C logic (our implementation) ---")
    dets_raw = decode_stm32(onnx_outputs, args.conf, sort_and_nms=False)
    dets_nms = decode_stm32(onnx_outputs, args.conf, sort_and_nms=True, nms_thresh=args.nms)

    print(f"\n  Total raw detections (score>={args.conf}): {len(dets_raw)}")
    print(f"  After NMS: {len(dets_nms)}")
    for d in dets_nms[:20]:
        print(f"  score={d[4]:.4f}  ({d[0]:.1f},{d[1]:.1f},{d[2]:.1f},{d[3]:.1f})  stride={STRIDES[d[5]]}")

    if len(dets_raw) > 0:
        scores = [d[4] for d in dets_raw]
        print(f"\n  Score stats: min={min(scores):.4f}, max={max(scores):.4f}, "
              f"mean={np.mean(scores):.4f}, median={np.median(scores):.4f}")

    # ---- Python reference decode ----
    print("\n--- Python reference logic ---")
    dets_ref = decode_python_ref(onnx_outputs, args.conf, args.nms)
    print(f"  After NMS: {len(dets_ref)}")
    for d in dets_ref:
        print(f"  score={d[4]:.4f}  ({d[0]:.1f},{d[1]:.1f},{d[2]:.1f},{d[3]:.1f})")

    # ---- Draw ----
    out_raw_path = os.path.join(OUT_DIR, "diagnose_raw_detections.jpg")
    out_nms_path = os.path.join(OUT_DIR, "diagnose_nms_result.jpg")

    draw_all_dets(img_resized, dets_raw, dets_nms,
                  f"STM32 C logic: {len(dets_raw)} raw → {len(dets_nms)} after NMS",
                  out_nms_path)

    # Scale to original image
    scale_x = orig_w / 320.0
    scale_y = orig_h / 320.0
    draw_all_dets(img, dets_raw, dets_nms,
                  f"Original: {len(dets_raw)} raw → {len(dets_nms)} NMS (conf={args.conf}, nms={args.nms})",
                  os.path.join(OUT_DIR, "diagnose_original.jpg"))

    print(f"\nSaved: {out_nms_path}")

    # ---- Compare with ST AI validation NPZ ----
    if os.path.exists(VALIDATION_NPZ):
        npz_data = np.load(VALIDATION_NPZ)

        # Find closest matching input in validation set
        # (the first input in NPZ might not match our test image exactly,
        #  but we can still compare output statistics)
        print("\n--- Validation NPZ analysis ---")
        for i in range(12):
            key = f'm_outputs_{i+1}'
            arr = npz_data[key]
            name = ["cls_8","cls_16","cls_32","obj_8","obj_16","obj_32",
                    "bbox_8","bbox_16","bbox_32","kps_8","kps_16","kps_32"][i]
            print(f"  {key} ({name}): shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]")

        # Run STM32 decode on each validation sample
        print("\n--- STM32 decode on ST AI validation outputs ---")
        for s in range(min(5, npz_data['m_outputs_1'].shape[0])):
            stai_outputs = []
            for i in range(12):
                key = f'm_outputs_{i+1}'
                arr = npz_data[key][s:s+1]  # Keep batch dim for consistency
                stai_outputs.append(arr)
            dets_stai = decode_stm32(stai_outputs, args.conf, sort_and_nms=True, nms_thresh=args.nms)
            print(f"  Sample {s}: {len(dets_stai)} faces after NMS", end="")
            for d in dets_stai[:5]:
                print(f"  [{d[4]:.3f} @ ({d[0]:.0f},{d[1]:.0f},{d[2]:.0f},{d[3]:.0f})]", end="")
            print()

        print("\nIMPORTANT: If ST AI validation shows few faces (1-5) but STM32 hardware")
        print("shows many (81), the issue may be:")
        print("  1. Model compilation differences (INT8 quantization)")
        print("  2. Different pre-processing on STM32")
        print("  3. Memory corruption or alignment issues on hardware")
        print("  4. D-Cache coherency (SDRAM not flushed before AI reads)")

    print("\nDone!")


if __name__ == "__main__":
    main()
