"""
test_image.py — 单张图片人脸检测测试
=====================================
加载训练好的模型, 对单张图片/图片文件夹进行人脸检测,
绘制检测框并保存结果。

用法:
  python test_image.py --image test.jpg
  python test_image.py --image test.jpg --model best_model.pth
  python test_image.py --dir ./test_images/
"""

import os
import sys
import argparse
import time
from glob import glob
from typing import Tuple

import cv2
import numpy as np
import torch

from config import *
from models import create_face_detector


def load_model(model_path: str):
    """加载训练好的模型"""
    print(f"[Model] Loading: {model_path}")
    model = create_face_detector(width_mult=BACKBONE_WIDTH_MULTIPLIER)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # 处理可能的 key 前缀不匹配
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image_path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    预处理输入图片。

    Returns:
        input_tensor: [1, C, H, W] 模型输入
        original_image: [H, W, C] BGR 原始图片
        original_size: (W, H) 原始尺寸
    """
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    H, W = original.shape[:2]

    # Resize 到模型输入尺寸
    resized = cv2.resize(original, (IMAGE_SIZE, IMAGE_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)  # [C, H, W] → [1, C, H, W]

    return tensor, original, (W, H)


def detect_faces(model, image_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    对预处理后的图片运行人脸检测。

    Returns:
        boxes:  [K, 4]  像素坐标 [x1, y1, x2, y2]
        scores: [K]     置信度
    """
    with torch.no_grad():
        boxes, scores = model.predict(image_tensor)

    return boxes.numpy(), scores.numpy()


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    original_size: Tuple[int, int],
) -> np.ndarray:
    """
    在图片上绘制检测框。

    Args:
        image:         [H, W, C] BGR 原始图片
        boxes:         [K, 4] 像素坐标 (模型输入尺寸)
        scores:        [K] 置信度
        original_size: (W, H)

    Returns:
        带检测框的图片
    """
    H, W = original_size[1], original_size[0]
    result = image.copy()

    # 缩放到原始尺寸
    scale_x = W / IMAGE_SIZE
    scale_y = H / IMAGE_SIZE

    for box, score in zip(boxes, scores):
        if score < SCORE_THRESHOLD:
            continue

        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        # 绘制矩形框 (绿色)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制置信度
        label = f"{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(
            result, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

    # 在顶部显示检测数量
    num_faces = (scores > SCORE_THRESHOLD).sum()
    cv2.putText(
        result,
        f"Faces detected: {num_faces}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Face Detection on Images")
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--dir", type=str, default=None, help="Directory of images")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(MODEL_DIR, "best_model.pth"),
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save output images"
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Don't save, just display stats",
    )
    args = parser.parse_args()

    if not args.image and not args.dir:
        print("[Error] Please specify --image or --dir")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"[Error] Model not found: {args.model}")
        print("        Run train.py first, or specify --model path")
        sys.exit(1)

    # 加载模型
    model = load_model(args.model)
    print(f"[Model] Loaded. Ready for inference.\n")

    # 收集图片列表
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.dir:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_paths.extend(glob(os.path.join(args.dir, ext)))
            image_paths.extend(glob(os.path.join(args.dir, ext.upper())))

    if not image_paths:
        print("[Error] No images found")
        sys.exit(1)

    # 推断
    total_time = 0
    total_faces = 0

    for img_path in sorted(image_paths):
        print(f"[Detect] {img_path}")

        # 预处理
        tensor, original, orig_size = preprocess_image(img_path)

        # 检测
        t_start = time.time()
        boxes, scores = detect_faces(model, tensor)
        t_elapsed = time.time() - t_start

        total_time += t_elapsed
        num_faces = (scores > SCORE_THRESHOLD).sum()
        total_faces += num_faces

        print(f"  Time: {t_elapsed*1000:.1f} ms")
        print(f"  Faces: {num_faces}")

        for box, score in zip(boxes, scores):
            if score > SCORE_THRESHOLD:
                print(f"    [{score:.2f}] Box: {box.astype(int).tolist()}")

        # 绘制
        if args.save:
            result = draw_detections(original, boxes, scores, orig_size)

            if args.output:
                os.makedirs(args.output, exist_ok=True)
                out_name = os.path.basename(img_path)
                out_path = os.path.join(args.output, f"detected_{out_name}")
            else:
                name, ext = os.path.splitext(img_path)
                out_path = f"{name}_detected{ext}"

            cv2.imwrite(out_path, result)
            print(f"  Saved: {out_path}")
        print()

    # 总结
    avg_time = total_time / len(image_paths) * 1000 if image_paths else 0
    print(f"{'='*50}")
    print(f"Summary:")
    print(f"  Images:      {len(image_paths)}")
    print(f"  Total faces: {total_faces}")
    print(f"  Avg time:    {avg_time:.1f} ms/image")
    print(f"  FPS:         {1000/avg_time:.1f}" if avg_time > 0 else "  FPS: N/A")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
