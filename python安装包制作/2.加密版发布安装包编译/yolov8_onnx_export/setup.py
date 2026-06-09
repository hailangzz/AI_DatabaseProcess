from setuptools import setup, find_packages

setup(
    name="yolov8_onnx_export",
    version="0.1.0",

    packages=find_packages(),

    include_package_data=True,

    package_data={
        "segment_onnx_export": ["*.so", "*.cpython-*.so"],
        "segment_onnx_export.model_structure": ["*.so", "*.cpython-*.so"],
    },

    entry_points={
        "console_scripts": [
            "yolo_onnx_export=segment_onnx_export.cli:main"
        ]
    },
)