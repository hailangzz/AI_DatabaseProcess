import argparse

from segment_onnx_export.exporter import \
    export_rknn_onnx


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-w",
        "--weights",
        required=True
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=18
    )

    parser.add_argument(
        "--sim",
        action="store_true"
    )

    parser.add_argument(
        "--device",
        default="cpu"
    )

    parser.add_argument(
        "--input-shape",
        nargs="+",
        type=int,
        default=[1, 3, 640, 640]
    )

    args = parser.parse_args()

    export_rknn_onnx(
        weights=args.weights,
        opset=args.opset,
        simplify=args.sim,
        input_shape=tuple(args.input_shape),
        device=args.device
    )


if __name__ == "__main__":
    main()