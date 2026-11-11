from setuptools import setup, find_packages

setup(
    name="bca_vit",
    version="1.0.0",
    description="Bidirectional Cross-Attention Adapters for Parameter-Efficient Vision Transformer Transfer Learning",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
    ],
)
