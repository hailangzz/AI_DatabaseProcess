from setuptools import setup, find_packages

setup(
    name="label-checker",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "PyQt5",
        "opencv-python",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "label_checker = label_checker.app:main"
        ]
    },
)