from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        [
            "segment_onnx_export/exporter.pyx",
            "segment_onnx_export/model_structure/myself_model_struct.pyx"
        ],
        language_level="3"
    )
)