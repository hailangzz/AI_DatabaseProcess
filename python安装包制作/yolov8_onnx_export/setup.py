from setuptools import setup, find_packages

setup(
    name="yolov8_onnx_export",
    version="0.1.0",

    packages=find_packages(),

    install_requires=[],
    # install_requires=[
    # "numpy>=1.23,<2.0",
    # "torch",
    # "ultralytics",
    # "onnx",
    # ],

    entry_points={
        "console_scripts": [
            "yolo_onnx_export="
            "segment_onnx_export.cli:main"
        ]
    },
)