#!/usr/bin/env python3
"""
Quantization example demonstrating the quantization pipeline.

This example shows:
1. Different quantization types (dynamic, static, mixed)
2. Quantization planning and analysis
3. Performance impact of quantization
4. Model size reduction
"""

import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run quantization example."""
    logger.info("Starting quantization example...")
    
    try:
        # Import required components
        from llm_optimizer import QuantizationPipeline
        from transformers import AutoModel, AutoTokenizer
        
        # Load a model for quantization
        logger.info("Loading BERT model for quantization...")
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        logger.info(f"Model loaded: {model_name}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test different quantization approaches
        quantization_types = ["dynamic", "static", "mixed"]
        
        for qtype in quantization_types:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing {qtype.upper()} quantization")
            logger.info(f"{'='*50}")
            
            # Create quantization pipeline
            quantizer = QuantizationPipeline(
                quantization_type=qtype,
                target_precision="int8",
                enable_qat=False
            )
            
            # Create quantization plan
            logger.info("Creating quantization plan...")
            plan = quantizer.create_quantization_plan(model)
            
            logger.info("Quantization Plan:")
            logger.info(f"  Total Layers: {plan['model_info']['total_layers']}")
            logger.info(f"  Quantizable Layers: {plan['model_info']['quantizable_layers']}")
            logger.info(f"  Attention Layers: {plan['model_info']['attention_layers']}")
            logger.info(f"  Linear Layers: {plan['model_info']['linear_layers']}")
            
            if plan['recommendations']:
                logger.info("  Recommendations:")
                for rec in plan['recommendations']:
                    logger.info(f"    • {rec}")
            
            # Run quantization
            logger.info(f"Running {qtype} quantization...")
            start_time = time.time()
            
            try:
                quantized_model = quantizer.quantize(
                    model=model,
                    target_size_reduction=0.75
                )
                
                quantization_time = time.time() - start_time
                logger.info(f"Quantization completed in {quantization_time:.2f} seconds")
                
                # Get quantization stats
                stats = quantizer.get_quantization_stats()
                logger.info(f"Quantization Stats:")
                logger.info(f"  Type: {stats['quantization_type']}")
                logger.info(f"  Precision: {stats['target_precision']}")
                logger.info(f"  QAT Enabled: {stats['enable_qat']}")
                
                # Compare model sizes
                original_size = get_model_size(model)
                quantized_size = get_model_size(quantized_model)
                size_reduction = ((original_size - quantized_size) / original_size) * 100
                
                logger.info(f"Model Size Comparison:")
                logger.info(f"  Original: {original_size:.2f} MB")
                logger.info(f"  Quantized: {quantized_size:.2f} MB")
                logger.info(f"  Reduction: {size_reduction:.1f}%")
                
                # Test model functionality
                logger.info("Testing quantized model functionality...")
                test_model_functionality(quantized_model, tokenizer)
                
            except Exception as e:
                logger.error(f"{qtype} quantization failed: {e}")
                continue
        
        logger.info("\nQuantization example completed!")
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.info("Make sure you have installed the package: pip install -e .")
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise

def get_model_size(model):
    """Get approximate model size in MB."""
    try:
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            # Assume float32 (4 bytes per parameter)
            size_bytes = total_params * 4
            return size_bytes / (1024 * 1024)  # Convert to MB
        else:
            return 0.0
    except:
        return 0.0

def test_model_functionality(model, tokenizer):
    """Test if the quantized model still works."""
    try:
        # Create test input
        test_text = "Hello, this is a test sentence."
        inputs = tokenizer(test_text, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        logger.info("  ✓ Model inference successful")
        
        # Check output shape
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
            logger.info(f"  ✓ Output shape: {hidden_states.shape}")
        
    except Exception as e:
        logger.error(f"  ✗ Model functionality test failed: {e}")

if __name__ == "__main__":
    # Import torch here to avoid import errors
    import torch
    main()
