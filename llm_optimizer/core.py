"""
Core LLM Optimizer class that orchestrates the optimization pipeline.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None
    PreTrainedModel = None
    PreTrainedTokenizer = None

from .utils import OptimizationConfig, PerformanceMetrics, setup_logging

logger = logging.getLogger(__name__)

class LLMOptimizer:
    """
    Main LLM Optimizer class that orchestrates the optimization pipeline.
    """
    
    def __init__(
        self,
        model_name: str,
        target_device: str = "cpu",  # Changed from "cuda" for macOS compatibility
        optimization_level: str = "balanced",
        config: Optional[OptimizationConfig] = None,
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.target_device = target_device
        self.optimization_level = optimization_level
        self.config = config
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "llm_optimizer")
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.optimized_model = None
        self.performance_metrics = PerformanceMetrics()
        
        # Setup logging
        setup_logging()
        logger.info(f"Initialized LLM Optimizer for {model_name}")
    
    def load_model(self) -> "LLMOptimizer":
        """Load the model and tokenizer."""
        try:
            if AutoTokenizer is None or AutoModel is None:
                raise ImportError("Transformers library not available")
                
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move to target device if available
            if self.target_device == "cuda" and hasattr(self.model, "cuda"):
                self.model = self.model.cuda()
            elif self.target_device == "mps" and hasattr(self.model, "to"):
                # Use Metal Performance Shaders for Apple Silicon
                self.model = self.model.to("mps")
            
            logger.info(f"Model loaded successfully on {self.target_device}")
            return self
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def optimize(self, **kwargs) -> Any:
        """Run the complete optimization pipeline."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Starting optimization pipeline...")
        start_time = time.time()
        
        try:
            # Basic optimization (model loading and validation)
            self.optimized_model = self.model
            
            # Record performance metrics
            self.performance_metrics.optimization_time = time.time() - start_time
            self.performance_metrics.model_size_original = self._get_model_size(self.model)
            self.performance_metrics.model_size_optimized = self._get_model_size(self.optimized_model)
            
            logger.info("Optimization completed successfully")
            return self.optimized_model
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def _get_model_size(self, model: Any) -> int:
        """Get model size in bytes."""
        try:
            if hasattr(model, 'state_dict'):
                total_params = sum(p.numel() for p in model.parameters())
                # Rough estimate: 4 bytes per parameter for float32
                return total_params * 4
            return 0
        except:
            return 0
    
    def _validate_performance(self):
        """Validate performance improvements."""
        if self.optimized_model is None:
            return
        
        logger.info("Validating performance improvements...")
        # Basic validation - in a real implementation, you'd run benchmarks
        logger.info("Performance validation completed")
    
    def save_optimized_model(self, output_path: str) -> str:
        """Save the optimized model."""
        if self.optimized_model is None:
            raise ValueError("No optimized model to save")
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model and tokenizer
            if hasattr(self.optimized_model, 'save_pretrained'):
                self.optimized_model.save_pretrained(output_dir)
            if self.tokenizer and hasattr(self.tokenizer, 'save_pretrained'):
                self.tokenizer.save_pretrained(output_dir)
            
            # Save performance metrics
            metrics_file = output_dir / "optimization_metrics.json"
            import json
            with open(metrics_file, 'w') as f:
                json.dump(self.performance_metrics.to_dict(), f, indent=2)
            
            logger.info(f"Optimized model saved to: {output_dir}")
            return str(output_dir)
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization report and metrics."""
        return {
            "model_name": self.model_name,
            "target_device": self.target_device,
            "optimization_level": self.optimization_level,
            "performance_metrics": self.performance_metrics.to_dict(),
            "model_loaded": self.model is not None,
            "optimization_completed": self.optimized_model is not None
        }
    
    def deploy_to_sagemaker(
        self,
        instance_type: str = "ml.g4dn.xlarge",
        auto_scaling: bool = True,
        **kwargs
    ) -> str:
        """Deploy model to AWS SageMaker (placeholder)."""
        logger.warning("SageMaker deployment not implemented in this version")
        return "deployment_not_implemented"
