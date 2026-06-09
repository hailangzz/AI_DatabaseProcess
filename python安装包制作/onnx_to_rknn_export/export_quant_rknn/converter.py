from rknn.api import RKNN


def convert_onnx_to_rknn(
    model_path,
    platform,
    dataset_path,
    output_path,
    do_quant=True
):
    rknn = RKNN(verbose=False)

    print('--> Config model')

    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=platform
    )

    print('--> Loading model')

    ret = rknn.load_onnx(model=model_path)

    if ret != 0:
        raise RuntimeError("Load ONNX failed")

    print('--> Building model')

    ret = rknn.build(
        do_quantization=do_quant,
        dataset=dataset_path
    )

    if ret != 0:
        raise RuntimeError("Build RKNN failed")

    print('--> Export RKNN')

    ret = rknn.export_rknn(output_path)

    if ret != 0:
        raise RuntimeError("Export RKNN failed")

    print("RKNN export success")

    rknn.release()