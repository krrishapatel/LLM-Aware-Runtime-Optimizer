#!/usr/bin/env python3
"""
Basic tests for LLM Optimizer.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_import():
    """Test that the package can be imported."""
    try:
        from llm_optimizer import LLMOptimizer
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import LLMOptimizer: {e}")

def test_optimizer_creation():
    """Test optimizer creation."""
    try:
        from llm_optimizer import LLMOptimizer
        optimizer = LLMOptimizer("gpt2", target_device="cpu")
        assert optimizer.model_name == "gpt2"
        assert optimizer.target_device == "cpu"
    except Exception as e:
        pytest.fail(f"Failed to create optimizer: {e}")

def test_utils_import():
    """Test utils import."""
    try:
        from llm_optimizer.utils import OptimizationConfig, PerformanceMetrics
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import utils: {e}")

def test_config_creation():
    """Test config creation."""
    try:
        from llm_optimizer.utils import OptimizationConfig
        config = OptimizationConfig()
        assert config.optimization_level == "balanced"
        assert config.target_device == "cpu"
    except Exception as e:
        pytest.fail(f"Failed to create config: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
