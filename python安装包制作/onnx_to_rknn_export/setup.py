from setuptools import setup, find_packages

setup(
    name="export_quant_rknn",
    version="0.1.0",
    author="zz",
    description="Used for quantization of ONNX models to RKNN",
    packages=find_packages(),
    install_requires=[],
    entry_points={
            "console_scripts": [
                "rknn_convert=export_quant_rknn.cli:main"
            ]
        },
    # install_requires=["numpy", "opencv-python", ],
)