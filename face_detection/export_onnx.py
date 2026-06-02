"""
export_onnx.py — 导出 ONNX 模型
================================
将训练好的 PyTorch 人脸检测器导出为 ONNX 格式,
供 Cube.AI / STM32Cube.AI 使用。

导出策略:
  1. 导出基础模型 (不包含 NMS 后处理) → Cube.AI 部署
  2. 可选: 导出含 NMS 的完整模型 → PC 端 ONNX Runtime 验证

Cube.AI 兼容性:
  - Opset 11 (推荐)
  - 标准算子: Conv, Relu, Sigmoid, Reshape, Concat
  - 避免: 动态形状, 控制流, NonMaxSuppression (Cube.AI 不支持)
"""

import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn

from config import *
from models import create_face_detector

# 尝试导入 onnx (如果已安装)
try:
    import onnx
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print(
        "[Warning] onnx/onnxruntime not installed. " "Run: pip install onnx onnxruntime"
    )


def export_basic_model(model: nn.Module, output_path: str, opset_version: int = 11):
    """
    导出基础检测头模型 (不含 NMS)。

    输入:  [1, 3, 160, 160]  图像
    输出: [1, 1550, 1]      分类预测 (logits)
          [1, 1550, 4]      回归预测 (编码偏移)

    这是 Cube.AI 兼容的导出格式。
    NMS 后处理在 MCU 端用 C 实现。
    """
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    print(f"[Export] Exporting basic model (opset {opset_version})...")
    print(f"  Input:  {dummy_input.shape}")
    print(f"  Output: cls_preds + reg_preds")

    with torch.no_grad():
        cls_preds, reg_preds = model(dummy_input)
        print(f"  cls output: {cls_preds.shape}")
        print(f"  reg output: {reg_preds.shape}")

    # 导出一个包装器, 输出 concat 后的两个张量
    class ExportWrapper(nn.Module):
        def __init__(self, detector):
            super().__init__()
            self.detector = detector

        def forward(self, x):
            cls, reg = self.detector(x)
            # 输出: sigmoid 后的置信度 + 解码后的框
            scores = torch.sigmoid(cls)
            return scores, reg

    wrapper = ExportWrapper(model)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["scores", "boxes"],
        dynamic_axes=None,  # 固定 batch size
        verbose=False,
    )

    print(f"  ✓ Exported to: {output_path}")

    # 验证 ONNX 模型
    if HAS_ONNX:
        _validate_onnx(output_path, dummy_input, wrapper)
        _print_onnx_info(output_path)


def _validate_onnx(onnx_path: str, dummy_input: torch.Tensor, model_ref: nn.Module):
    """验证导出的 ONNX 模型"""
    print("\n[Validate] Checking ONNX model...")

    # 1. 结构验证
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model structure valid")

    # 2. 推理验证
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}

    # 在 PyTorch 中运行
    with torch.no_grad():
        pt_scores, pt_boxes = model_ref(dummy_input)

    # 在 ONNX Runtime 中运行
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_scores = ort_outputs[0]
    ort_boxes = ort_outputs[1]

    # 比较输出 (允许小误差)
    score_diff = np.abs(pt_scores.cpu().numpy() - ort_scores).max()
    box_diff = np.abs(pt_boxes.cpu().numpy() - ort_boxes).max()

    print(f"  ✓ ONNX Runtime inference OK")
    print(f"    score max diff: {score_diff:.6f}")
    print(f"    box max diff:   {box_diff:.6f}")
    print(f"    (differences < 1e-5 expected for FP32)")

    # 模型大小
    import os

    file_size_kb = os.path.getsize(onnx_path) / 1024
    print(f"  ONNX file size: {file_size_kb:.1f} KB")


def _print_onnx_info(onnx_path: str):
    """打印 ONNX 模型信息 (算子列表等)"""
    onnx_model = onnx.load(onnx_path)

    # 收集算子类型
    ops = set()
    for node in onnx_model.graph.node:
        ops.add(node.op_type)

    print(f"\n[ONNX Info]")
    print(f"  Opset: {onnx_model.opset_import[0].version}")
    print(f"  Operators ({len(ops)}): {sorted(ops)}")
    print(
        f"  Input:  {[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in onnx_model.graph.input]}"
    )
    print(
        f"  Output: {[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in onnx_model.graph.output]}"
    )


def main():
    parser = argparse.ArgumentParser(description="Export to ONNX")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=os.path.join(MODEL_DIR, "best_model.pth"),
        help="Path to trained model",
    )
    parser.add_argument("--qat", action="store_true", help="Use QAT model")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(MODEL_DIR, "face_detector.onnx"),
        help="Output ONNX path",
    )
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    args = parser.parse_args()

    if not HAS_ONNX:
        print("[Error] onnx/onnxruntime not installed")
        sys.exit(1)

    # 加载模型
    model_path = args.pretrained
    if args.qat:
        qat_path = os.path.join(MODEL_DIR, "best_model_qat.pth")
        if os.path.exists(qat_path):
            model_path = qat_path
        else:
            print(f"[Warning] QAT model not found, using FP32: {model_path}")

    print(f"[Model] Loading: {model_path}")
    model = create_face_detector(width_mult=BACKBONE_WIDTH_MULTIPLIER)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

    # 导出
    export_basic_model(model, args.output, args.opset)

    print(f"\n[Done] ONNX model ready for Cube.AI import:")
    print(f"       {args.output}")


if __name__ == "__main__":
    main()
