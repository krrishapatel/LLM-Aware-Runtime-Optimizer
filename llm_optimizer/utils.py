"""Config dataclasses, logging setup, and environment reporting."""

import logging
import os
import platform
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

MIN_RECOMMENDED_MEMORY_BYTES = 2 * 1024**3


@dataclass
class OptimizationConfig:
    """Settings for a run."""

    quantization: str = "dynamic"
    target_device: str = "cpu"
    num_benchmark_runs: int = 50
    warmup_runs: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """Configure the root logger. Called by the CLI, not on import."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers,
        force=True,
    )


def _memory_info() -> Dict[str, Optional[int]]:
    """Total and available memory, or None if psutil is not installed.

    psutil used to be a module-level import, which made every part of the
    package fail to import on a machine without it.
    """
    try:
        import psutil
    except ImportError:
        return {"memory_total": None, "memory_available": None}
    virtual = psutil.virtual_memory()
    return {"memory_total": virtual.total, "memory_available": virtual.available}


def get_system_info() -> Dict[str, Any]:
    """Platform, Python, CPU count, memory, and torch device availability."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        **_memory_info(),
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        info["quantized_engines"] = list(torch.backends.quantized.supported_engines)
    except ImportError:
        info["torch_version"] = None

    return info


def validate_environment() -> Dict[str, Any]:
    """Check the environment and return a result with a `ready` boolean.

    Returns `ready` explicitly. The old version returned a dict that was always
    non-empty and the CLI tested it with `if validate_environment():`, so the
    check printed "Environment is ready" no matter what it found.
    """
    info = get_system_info()
    problems = []
    warnings = []

    if info.get("torch_version") is None:
        problems.append("torch is not installed.")

    available = info.get("memory_available")
    if available is not None and available < MIN_RECOMMENDED_MEMORY_BYTES:
        warnings.append(
            f"Only {available / 1024**3:.1f} GB of memory available; "
            f"{MIN_RECOMMENDED_MEMORY_BYTES / 1024**3:.0f} GB recommended."
        )
    elif available is None:
        warnings.append("psutil not installed, so memory was not checked.")

    if not info.get("quantized_engines"):
        warnings.append(
            "torch reports no quantized engines, so int8 quantization will fail."
        )

    return {
        "ready": not problems,
        "problems": problems,
        "warnings": warnings,
        "system": info,
    }
