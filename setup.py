#!/usr/bin/env python3
"""Setup script.

The previous version could not install. It read requirements from
`requirements_minimal.txt`, which does not exist, and used
`find_packages(where="src")` with `package_dir={"": "src"}` in a repo that has no
src/ directory, so it found no packages.
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent

# The runtime dependencies, kept here as the single source of truth so
# `pip install -e .` and `pip install -r requirements.txt` cannot drift apart.
INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
]

setup(
    name="llm-runtime-optimizer",
    version="0.2.0",
    author="Krrisha Patel",
    description="Quantize a PyTorch transformer and measure whether it got faster",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/krrishapatel/LLM-Aware-Runtime-Optimizer",
    project_urls={
        "Source": "https://github.com/krrishapatel/LLM-Aware-Runtime-Optimizer",
        "Issues": (
            "https://github.com/krrishapatel/LLM-Aware-Runtime-Optimizer/issues"
        ),
    },
    license="MIT",
    packages=find_packages(include=["llm_optimizer", "llm_optimizer.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # 3.9 is the floor because the type hints use the typing module rather than
    # builtin generics, and dataclasses need 3.7 at minimum.
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        # ONNX export and graph optimization. Optional: llm_optimizer imports
        # onnx inside the functions that need it, not at module level.
        "onnx": ["onnx>=1.14.0", "onnxruntime>=1.15.0", "onnxscript>=0.1.0"],
        "dev": [
            "pytest>=7.0.0",
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
            "onnxscript>=0.1.0",
            "psutil>=5.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-optimize=llm_optimizer.cli:main",
        ],
    },
    zip_safe=False,
    keywords="pytorch quantization transformer benchmarking onnx sagemaker",
)
