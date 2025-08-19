"""
LLM-Aware Runtime Optimizer
"""

__version__ = "0.1.0"
__author__ = "LLM Optimizer Team"
__email__ = "support@llm-optimizer.com"

# Import core modules
from .core import LLMOptimizer
from .utils import OptimizationConfig, PerformanceMetrics, setup_logging

# Import other modules only if they're available
try:
    from .quantization import QuantizationPipeline
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QuantizationPipeline = None
    QUANTIZATION_AVAILABLE = False

try:
    from .deployment import DeploymentManager
    DEPLOYMENT_AVAILABLE = True
except ImportError:
    DeploymentManager = None
    DEPLOYMENT_AVAILABLE = False

# MLIR, ONNX, TensorRT are not directly imported here due to macOS compatibility
MLIR_AVAILABLE = False
ONNX_AVAILABLE = False
TENSORRT_AVAILABLE = False

__all__ = [
    "LLMOptimizer",
    "OptimizationConfig",
    "PerformanceMetrics",
    "setup_logging",
]

if QUANTIZATION_AVAILABLE:
    __all__.append("QuantizationPipeline")
if DEPLOYMENT_AVAILABLE:
    __all__.append("DeploymentManager")
