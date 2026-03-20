"""
DAOS-RFF: Domain-Adaptive Open-Set RF Fingerprinting

Setup script for the DAOS-RFF package.
"""

from setuptools import setup, find_packages

setup(
    name="daos-rff",
    version="0.1.0",
    description="Domain-Adaptive Open-Set RF Fingerprinting using Evidential Deep Learning",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "torchvision>=0.15.0",
        "torchmetrics>=1.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "h5py>=3.8.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
)
