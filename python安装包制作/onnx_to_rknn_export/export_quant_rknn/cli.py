import argparse

from .converter import convert_onnx_to_rknn


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_path")

    parser.add_argument("platform")

    parser.add_argument("--dataset", required=True)

    parser.add_argument("--output2", default="model.rknn")

    parser.add_argument("--quant", action="store_true")

    args = parser.parse_args()

    convert_onnx_to_rknn(
        model_path=args.model_path,
        platform=args.platform,
        dataset_path=args.dataset,
        output_path=args.output,
        do_quant=args.quant,
    )


if __name__ == "__main__":
    main()
