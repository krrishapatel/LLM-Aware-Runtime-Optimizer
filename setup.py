#!/usr/bin/env python3
"""
Setup script for LLM-Aware Runtime Optimizer
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements_minimal.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-runtime-optimizer",
    version="0.1.0",
    author="LLM Optimizer Team",
    author_email="support@llm-optimizer.com",
    description="MLIR-based runtime optimizer for quantized transformer LLMs",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/llm-optimizer/llm-runtime-optimizer",
    project_urls={
        "Bug Reports": "https://github.com/llm-optimizer/llm-runtime-optimizer/issues",
        "Source": "https://github.com/llm-optimizer/llm-runtime-optimizer",
        "Documentation": "https://llm-optimizer.readthedocs.io/",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "ipykernel>=6.15.0",
            "notebook>=6.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-optimize=llm_optimizer.cli:main",
            "llm-benchmark=llm_optimizer.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="mlir, llm, transformer, optimization, quantization, tensorrt, cuda",
)
