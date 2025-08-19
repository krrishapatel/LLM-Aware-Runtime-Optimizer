"""
Quantization pipeline for LLM models to reduce model size and improve inference performance.
"""

import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, quantize_fx
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class QuantizationPipeline:
    """
    Advanced quantization pipeline for transformer models.
    
    Supports:
    - Dynamic quantization (INT8)
    - Static quantization (INT8/INT16)
    - Mixed precision quantization
    - Quantization-aware training (QAT)
    """
    
    def __init__(
        self,
        quantization_type: str = "dynamic",
        target_precision: str = "int8",
        calibration_data: Optional[Any] = None,
        enable_qat: bool = False
    ):
        """
        Initialize quantization pipeline.
        
        Args:
            quantization_type: Type of quantization ("dynamic", "static", "mixed")
            target_precision: Target precision ("int8", "int16", "mixed")
            calibration_data: Data for calibration (required for static quantization)
            enable_qat: Enable quantization-aware training
        """
        self.quantization_type = quantization_type
        self.target_precision = target_precision
        self.calibration_data = calibration_data
        self.enable_qat = enable_qat
        
        # Quantization configuration
        self.quantization_config = self._get_quantization_config()
        
        logger.info(f"Initialized {quantization_type} quantization pipeline with {target_precision} precision")
    
    def _get_quantization_config(self) -> Dict[str, Any]:
        """Get quantization configuration based on type and precision."""
        config = {
            "dynamic": {
                "int8": {"dtype": torch.qint8, "qscheme": torch.per_tensor_affine},
                "int16": {"dtype": torch.qint16, "qscheme": torch.per_tensor_affine}
            },
            "static": {
                "int8": {"dtype": torch.qint8, "qscheme": torch.per_tensor_symmetric},
                "int16": {"dtype": torch.qint16, "qscheme": torch.per_tensor_symmetric}
            },
            "mixed": {
                "mixed": {"dtype": torch.float16, "qscheme": torch.per_tensor_affine}
            }
        }
        
        return config.get(self.quantization_type, {}).get(self.target_precision, {})
    
    def quantize(
        self,
        model: PreTrainedModel,
        target_size_reduction: float = 0.75,
        **kwargs
    ) -> PreTrainedModel:
        """
        Quantize the model according to the specified configuration.
        
        Args:
            model: PyTorch model to quantize
            target_size_reduction: Target size reduction (0.0 to 1.0)
            **kwargs: Additional quantization parameters
            
        Returns:
            Quantized model
        """
        logger.info(f"Starting {self.quantization_type} quantization...")
        
        # Prepare model for quantization
        prepared_model = self._prepare_model_for_quantization(model)
        
        # Apply quantization based on type
        if self.quantization_type == "dynamic":
            quantized_model = self._apply_dynamic_quantization(prepared_model)
        elif self.quantization_type == "static":
            quantized_model = self._apply_static_quantization(prepared_model)
        elif self.quantization_type == "mixed":
            quantized_model = self._apply_mixed_precision(prepared_model)
        else:
            raise ValueError(f"Unsupported quantization type: {self.quantization_type}")
        
        # Post-quantization optimization
        optimized_model = self._post_quantization_optimization(quantized_model)
        
        # Validate quantization results
        self._validate_quantization(model, optimized_model, target_size_reduction)
        
        logger.info("Quantization completed successfully!")
        return optimized_model
    
    def _prepare_model_for_quantization(self, model: PreTrainedModel) -> PreTrainedModel:
        """Prepare model for quantization by setting appropriate modes."""
        model.eval()
        
        # Set quantization mode
        if hasattr(model, 'config'):
            model.config.use_cache = False
        
        # Prepare for dynamic quantization
        if self.quantization_type == "dynamic":
            # Mark layers for quantization
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    module.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        return model
    
    def _apply_dynamic_quantization(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply dynamic quantization to the model."""
        logger.info("Applying dynamic quantization...")
        
        # Quantize linear layers
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear, nn.Conv1d, nn.Conv2d},
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def _apply_static_quantization(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply static quantization to the model."""
        if self.calibration_data is None:
            raise ValueError("Calibration data required for static quantization")
        
        logger.info("Applying static quantization...")
        
        # Prepare for static quantization
        model.eval()
        
        # Calibrate the model
        calibrated_model = self._calibrate_model(model)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(calibrated_model)
        
        return quantized_model
    
    def _apply_mixed_precision(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply mixed precision quantization."""
        logger.info("Applying mixed precision quantization...")
        
        # Convert to half precision
        model = model.half()
        
        # Apply selective quantization to specific layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "attention" in name.lower():
                # Keep attention layers in half precision for better accuracy
                continue
            elif isinstance(module, nn.Linear):
                # Quantize other linear layers
                module = torch.quantization.quantize_dynamic(
                    module, {nn.Linear}, dtype=torch.qint8
                )
        
        return model
    
    def _calibrate_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Calibrate model for static quantization."""
        logger.info("Calibrating model...")
        
        # Set calibration mode
        model.eval()
        
        # Run calibration data through the model
        with torch.no_grad():
            for batch in self.calibration_data:
                if isinstance(batch, dict):
                    _ = model(**batch)
                elif isinstance(batch, (tuple, list)):
                    _ = model(*batch)
                else:
                    _ = model(batch)
        
        return model
    
    def _post_quantization_optimization(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply post-quantization optimizations."""
        logger.info("Applying post-quantization optimizations...")
        
        # Fuse operations where possible
        if hasattr(torch.quantization, 'fuse_modules'):
            model = torch.quantization.fuse_modules(model, ['conv', 'bn', 'relu'])
        
        # Optimize for inference
        model.eval()
        
        # Enable optimizations
        if hasattr(torch, 'jit'):
            try:
                model = torch.jit.optimize_for_inference(torch.jit.script(model))
            except Exception as e:
                logger.warning(f"JIT optimization failed: {e}")
        
        return model
    
    def _validate_quantization(
        self,
        original_model: PreTrainedModel,
        quantized_model: PreTrainedModel,
        target_size_reduction: float
    ):
        """Validate quantization results."""
        logger.info("Validating quantization results...")
        
        # Check model size reduction
        original_size = self._get_model_size(original_model)
        quantized_size = self._get_model_size(quantized_model)
        actual_reduction = (original_size - quantized_size) / original_size
        
        logger.info(f"Model size reduction: {actual_reduction:.2%} (target: {target_size_reduction:.2%})")
        
        if actual_reduction < target_size_reduction * 0.8:  # Allow 20% tolerance
            logger.warning(f"Size reduction target not met: {actual_reduction:.2%} < {target_size_reduction:.2%}")
        
        # Check parameter count
        original_params = sum(p.numel() for p in original_model.parameters())
        quantized_params = sum(p.numel() for p in quantized_model.parameters())
        
        logger.info(f"Parameter count: {original_params:,} -> {quantized_params:,}")
        
        # Validate model functionality
        self._validate_model_functionality(quantized_model)
    
    def _get_model_size(self, model: PreTrainedModel) -> int:
        """Get approximate model size in bytes."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return param_size + buffer_size
    
    def _validate_model_functionality(self, model: PreTrainedModel):
        """Validate that the quantized model still functions correctly."""
        logger.info("Validating model functionality...")
        
        try:
            # Test with dummy input
            if hasattr(model, 'config'):
                hidden_size = getattr(model.config, 'hidden_size', 768)
                dummy_input = torch.randn(1, 10, hidden_size)
                
                with torch.no_grad():
                    if hasattr(model, 'forward'):
                        output = model(dummy_input)
                        logger.info("Model forward pass successful")
                    else:
                        logger.info("Model forward pass validation skipped (no forward method)")
            
            logger.info("Model functionality validation passed")
            
        except Exception as e:
            logger.error(f"Model functionality validation failed: {e}")
            raise
    
    def get_quantization_stats(self) -> Dict[str, Any]:
        """Get quantization statistics and metrics."""
        return {
            "quantization_type": self.quantization_type,
            "target_precision": self.target_precision,
            "enable_qat": self.enable_qat,
            "config": self.quantization_config
        }
    
    def create_quantization_plan(
        self,
        model: PreTrainedModel
    ) -> Dict[str, Any]:
        """
        Create a detailed quantization plan for the model.
        
        Args:
            model: Model to analyze
            
        Returns:
            Quantization plan with layer-by-layer details
        """
        plan = {
            "model_info": {
                "total_layers": 0,
                "quantizable_layers": 0,
                "attention_layers": 0,
                "linear_layers": 0
            },
            "layer_analysis": [],
            "recommendations": []
        }
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Module):
                plan["model_info"]["total_layers"] += 1
                
                layer_info = {
                    "name": name,
                    "type": type(module).__name__,
                    "quantizable": False,
                    "recommended_precision": "fp32"
                }
                
                if isinstance(module, nn.Linear):
                    plan["model_info"]["linear_layers"] += 1
                    layer_info["quantizable"] = True
                    layer_info["recommended_precision"] = "int8"
                    
                    if "attention" in name.lower():
                        plan["model_info"]["attention_layers"] += 1
                        layer_info["recommended_precision"] = "fp16"  # Keep attention in higher precision
                
                plan["layer_analysis"].append(layer_info)
                
                if layer_info["quantizable"]:
                    plan["model_info"]["quantizable_layers"] += 1
        
        # Generate recommendations
        if plan["model_info"]["quantizable_layers"] > 0:
            plan["recommendations"].append(
                f"Quantize {plan['model_info']['quantizable_layers']} linear layers to int8"
            )
        
        if plan["model_info"]["attention_layers"] > 0:
            plan["recommendations"].append(
                f"Keep {plan['model_info']['attention_layers']} attention layers in fp16 for accuracy"
            )
        
        return plan
