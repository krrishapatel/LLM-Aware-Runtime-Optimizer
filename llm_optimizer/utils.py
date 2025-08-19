#!/usr/bin/env python3
"""
Utility functions and classes for LLM Optimizer.
"""

import logging
import os
import platform
import psutil
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

@dataclass
class OptimizationConfig:
    """Configuration for optimization."""
    optimization_level: str = "balanced"
    target_device: str = "cpu"
    enable_monitoring: bool = False
    max_memory_usage: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    optimization_time: float = 0.0
    model_size_original: int = 0
    model_size_optimized: int = 0
    latency_improvement: float = 0.0
    memory_usage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """Setup logging configuration."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )

def validate_environment() -> Dict[str, Any]:
    """Validate the environment for LLM optimization."""
    validation_results = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "status": "ready"
    }
    
    # Check for potential issues
    if validation_results["memory_available"] < 2 * 1024**3:  # Less than 2GB
        validation_results["status"] = "warning"
        validation_results["warning"] = "Low memory available"
    
    return validation_results

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "memory_percent": psutil.virtual_memory().percent
    }
