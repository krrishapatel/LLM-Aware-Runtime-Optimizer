"""Quantize a PyTorch transformer, then measure whether it got faster.

Imports are plain and unguarded. The previous version wrapped each one in
try/except ImportError and set a MODULE_AVAILABLE flag, which hid two real
faults: `deployment` exported SageMakerDeployer while __init__ asked for
DeploymentManager, and `quantization` raised AttributeError on torch.qint16.
Both showed up as a flag quietly set to False.

`onnx_export` is not imported here, because onnx is an optional dependency.
Import it directly when you need it.
"""

__version__ = "0.2.0"

from . import analysis, benchmark
from .core import LLMOptimizer
from .packaging import SageMakerPackageBuilder
from .quantization import QuantizationPipeline
from .utils import OptimizationConfig, get_system_info, setup_logging, validate_environment

__all__ = [
    "LLMOptimizer",
    "OptimizationConfig",
    "QuantizationPipeline",
    "SageMakerPackageBuilder",
    "analysis",
    "benchmark",
    "get_system_info",
    "setup_logging",
    "validate_environment",
]
