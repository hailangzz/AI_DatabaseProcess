from io import BytesIO

import onnx
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import head

from segment_onnx_export.model_structure import \
    myself_model_struct as rk_head

try:
    import onnxsim
except ImportError:
    onnxsim = None


def export_rknn_onnx(
    weights,
    opset=18,
    simplify=False,
    input_shape=(1, 3, 640, 640),
    device="cpu"
):

    setattr(
        head.Segment,
        "forward",
        rk_head.segment_forward
    )

    setattr(
        head.Detect,
        "forward",
        rk_head.detect_forward
    )

    model_wrapper = YOLO(weights)

    model = (
        model_wrapper
        .model
        .fuse()
        .eval()
    )

    model.to(device)

    # ======== EXPORT PATCH START ========
    from ultralytics.nn.modules.head import Segment
    from ultralytics.nn.modules.block import C2f

    for m in model.modules():

        # 1. Detect / Segment export 模式（必须）
        if isinstance(m, Segment):
            m.export = True
            m.dynamic = False
            m.training = False

        # 2. Focus export 模式（关键）
        if m.__class__.__name__ == "Focus":
            m.export = True

        # 3. C2f ONNX safe（避免 split graph 问题）
        if isinstance(m, C2f):
            if hasattr(m, "forward_split"):
                m.forward = m.forward_split

    fake_input = torch.randn(
        input_shape
    ).to(device)

    for _ in range(2):
        model(fake_input)

    save_path = weights.replace(
        ".pt",
        ".onnx"
    )

    output_names = [
        "yolov8_output0_box",
        "yolov8_output0_class",
        "yolov8_output0_class_sum",
        "yolov8_output0_mask",

        "yolov8_output1_box",
        "yolov8_output1_class",
        "yolov8_output1_class_sum",
        "yolov8_output1_mask",

        "yolov8_output2_box",
        "yolov8_output2_class",
        "yolov8_output2_class_sum",
        "yolov8_output2_mask",

        "yolov8_proto",
    ]

    with BytesIO() as f:

        torch.onnx.export(
            model,
            fake_input,
            f,
            opset_version=opset,
            input_names=["images"],
            output_names=output_names,
        )

        f.seek(0)

        onnx_model = onnx.load(f)

    onnx.checker.check_model(
        onnx_model
    )

    if simplify and onnxsim:

        onnx_model, check = (
            onnxsim.simplify(onnx_model)
        )

        assert check

    onnx.save(
        onnx.shape_inference.infer_shapes(
            onnx_model
        ),
        save_path
    )

    print(
        f"Export success: {save_path}"
    )

    return save_path