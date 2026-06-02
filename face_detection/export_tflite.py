"""
export_tflite.py — 导出 TensorFlow Lite 模型
=============================================
将 ONNX 模型转换为 TFLite INT8 格式, 用于 Cube.AI 部署。

转换路径:
  PyTorch → ONNX (opset 11) → TensorFlow SavedModel → TFLite (INT8)

或使用 onnx2tf 直接转换:
  ONNX → TFLite (一条命令)

Cube.AI 对 TFLite 的支持通常优于 ONNX (原生 TFLite Micro 运行时)。
"""

import os
import sys
import argparse
import subprocess

from config import *

# 检查 TFLite 转换工具
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

# 检查 onnx2tf
try:
    import onnx2tf
    HAS_ONNX2TF = True
except ImportError:
    HAS_ONNX2TF = False


def convert_onnx_to_tflite_via_tf(onnx_path: str, tflite_path: str):
    """
    方法A: ONNX → TensorFlow SavedModel → TFLite
    需要: onnx-tf, tensorflow
    """
    print("[Convert] Using onnx-tf path: ONNX → TF SavedModel → TFLite")

    try:
        from onnx_tf.backend import prepare
    except ImportError:
        print("[Error] onnx-tf not installed.")
        print("        Install: pip install onnx-tf tensorflow")
        return False

    # Step 1: ONNX → TF
    import onnx
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    saved_model_dir = os.path.join(MODEL_DIR, "tf_saved_model")
    tf_rep.export_graph(saved_model_dir)
    print(f"  ✓ TF SavedModel saved to: {saved_model_dir}")

    # Step 2: TF → TFLite (INT8)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"  ✓ TFLite model saved: {tflite_path} ({size_kb:.1f} KB)")
    return True


def convert_onnx_to_tflite_via_onnx2tf(onnx_path: str, tflite_path: str):
    """
    方法B: 使用 onnx2tf 直接转换 ONNX → TFLite
    需要: onnx2tf
    """
    print("[Convert] Using onnx2tf path: ONNX → TFLite")

    output_dir = os.path.join(MODEL_DIR, "onnx2tf_output")

    cmd = [
        "onnx2tf",
        "-i", onnx_path,
        "-o", output_dir,
        "-oiqt",               # INT8量化
        "-osd",                 # 不检查输出shape (兼容部分算子)
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[Error] onnx2tf failed:")
        print(result.stderr[:1000])
        return False

    # onnx2tf 输出的 TFLite 文件
    output_tflite = os.path.join(output_dir, "face_detector_float32.tflite")

    if os.path.exists(output_tflite):
        # 移动到目标路径
        if os.path.exists(tflite_path):
            os.remove(tflite_path)
        os.rename(output_tflite, tflite_path)
        size_kb = os.path.getsize(tflite_path) / 1024
        print(f"  ✓ TFLite model saved: {tflite_path} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"[Error] onnx2tf output not found: {output_tflite}")
        return False


def validate_tflite(tflite_path: str):
    """验证 TFLite 模型"""
    if not HAS_TF:
        print("[Warning] TensorFlow not installed, skipping TFLite validation")
        return

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\n[TFLite Validate]")
    print(f"  Input:  {input_details[0]['shape']} ({input_details[0]['dtype']})")
    for i, out in enumerate(output_details):
        print(f"  Output {i}: {out['shape']} ({out['dtype']})")

    # 检查算子
    # (TFLite 没有直接列出算子的 API, 通过图遍历)
    print(f"  Tensors: {len(interpreter.get_tensor_details())}")


def main():
    parser = argparse.ArgumentParser(description="Export to TFLite")
    parser.add_argument("--onnx", type=str,
                        default=os.path.join(MODEL_DIR, "face_detector.onnx"),
                        help="Path to ONNX model")
    parser.add_argument("--output", type=str,
                        default=os.path.join(MODEL_DIR, "face_detector.tflite"),
                        help="Output TFLite path")
    parser.add_argument("--method", type=str, default="auto",
                        choices=["auto", "onnx2tf", "onnx-tf"],
                        help="Conversion method")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f"[Error] ONNX model not found: {args.onnx}")
        print("        Run export_onnx.py first.")
        sys.exit(1)

    print(f"[TFLite Export] Converting: {args.onnx}")
    print(f"  Available tools: TF={HAS_TF}, onnx2tf={HAS_ONNX2TF}")

    success = False

    # 自动选择最佳转换路径
    if args.method == "auto":
        if HAS_ONNX2TF:
            success = convert_onnx_to_tflite_via_onnx2tf(args.onnx, args.output)
        elif HAS_TF:
            success = convert_onnx_to_tflite_via_tf(args.onnx, args.output)
        else:
            print("[Error] Neither onnx2tf nor tensorflow installed.")
            print("        Install one of:")
            print("          pip install onnx2tf tensorflow")
            print("          pip install onnx-tf tensorflow")
    elif args.method == "onnx2tf":
        success = convert_onnx_to_tflite_via_onnx2tf(args.onnx, args.output)
    elif args.method == "onnx-tf":
        success = convert_onnx_to_tflite_via_tf(args.onnx, args.output)

    if success:
        validate_tflite(args.output)
        print(f"\n[Done] TFLite model ready for Cube.AI import:")
        print(f"       {args.output}")


if __name__ == "__main__":
    main()
