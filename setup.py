from setuptools import setup, find_packages

setup(
    name="safe",
    version="0.1.0",
    description="SAFE: Feature Extraction for Anomaly Detection",
    author="Phat Nguyen",
    author_email="tienphat.nguyen122@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "torch>=1.10.0",
        "h5py",
        "scipy",
        "pillow",
        "tqdm",
        "detectron2"
    ],
)