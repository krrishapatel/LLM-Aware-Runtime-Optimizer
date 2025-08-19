#!/usr/bin/env python3
"""
Basic LLM optimization example.
"""

import logging
import time
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_device():
    """Detect the best available device for the current system."""
    try:
        import torch
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            return "cuda"
        
        # Check if MPS (Metal Performance Shaders) is available for Apple Silicon
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        
        # Fallback to CPU
        return "cpu"
        
    except ImportError:
        return "cpu"

def main():
    """Main example function."""
    logger.info("Starting LLM optimization example...")
    
    try:
        # Detect the best available device
        device = detect_device()
        logger.info(f"Detected device: {device}")
        
        # Import the optimizer
        from llm_optimizer import LLMOptimizer
        
        logger.info("Initializing LLM optimizer...")
        
        # Initialize optimizer with detected device
        optimizer = LLMOptimizer(
            model_name="gpt2",
            target_device=device,
            optimization_level="balanced"
        )
        
        logger.info("Loading GPT-2 model...")
        
        # Load the model
        optimizer.load_model()
        
        logger.info("Running optimization...")
        
        # Run optimization
        start_time = time.time()
        optimized_model = optimizer.optimize()
        optimization_time = time.time() - start_time
        
        logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
        
        # Get optimization report
        report = optimizer.get_optimization_report()
        
        logger.info("Optimization Report:")
        for key, value in report.items():
            if key == "performance_metrics":
                logger.info(f"  {key}:")
                for metric_key, metric_value in value.items():
                    if isinstance(metric_value, float):
                        logger.info(f"    {metric_key}: {metric_value:.6f}")
                    else:
                        logger.info(f"    {metric_key}: {metric_value}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Save the optimized model
        output_path = "./optimized_gpt2_example"
        logger.info(f"Saving optimized model to {output_path}...")
        
        saved_path = optimizer.save_optimized_model(output_path)
        logger.info(f"Model saved to: {saved_path}")
        
        # Compare model sizes
        if hasattr(optimizer, 'model') and hasattr(optimizer, 'optimized_model'):
            original_size = sum(p.numel() for p in optimizer.model.parameters()) * 4 / (1024**2)
            optimized_size = sum(p.numel() for p in optimizer.optimized_model.parameters()) * 4 / (1024**2)
            
            logger.info(f"Original model size: {original_size:.1f} MB")
            logger.info(f"Optimized model size: {optimized_size:.1f} MB")
            
            if original_size > 0:
                size_reduction = ((original_size - optimized_size) / original_size) * 100
                logger.info(f"Size reduction: {size_reduction:.1f}%")
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise

if __name__ == "__main__":
    main()
