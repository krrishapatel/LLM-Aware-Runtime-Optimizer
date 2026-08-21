"""The LLMOptimizer entry point: load, analyze, quantize, measure.

The version this replaces had an `optimize()` whose body was
`self.optimized_model = self.model`. It recorded a size before and after that
were the same number by construction, logged "Optimization completed
successfully", and returned the untouched model.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from . import analysis, benchmark
from .quantization import QuantizationPipeline
from .utils import setup_logging

logger = logging.getLogger(__name__)


class LLMOptimizer:
    """Quantize a transformer and measure what it cost.

    Either pass `model_name` and call `load_model()`, which needs transformers
    installed, or hand in an existing `nn.Module` as `model`.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model: Optional[nn.Module] = None,
        target_device: str = "cpu",
        quantization: str = "dynamic",
        cache_dir: Optional[str] = None,
        configure_logging: bool = False,
    ):
        if model_name is None and model is None:
            raise ValueError("Pass either model_name or model.")
        if quantization == "dynamic" and target_device != "cpu":
            # torch's int8 dynamic quantized kernels are CPU only. Running this
            # on cuda used to fail deep inside the forward pass, after the whole
            # pipeline had reported success.
            raise ValueError(
                "Dynamic int8 quantization runs on CPU only. Use "
                "target_device='cpu', or quantization='fp16' for CUDA."
            )

        self.model_name = model_name
        self.target_device = target_device
        self.quantization = quantization
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "llm_optimizer")

        self.model = model
        self.tokenizer = None
        self.optimized_model: Optional[nn.Module] = None
        self.quantization_report: Dict[str, Any] = {}
        self.benchmark_report: Dict[str, Any] = {}

        if configure_logging:
            # Off by default. A library that calls basicConfig on import hijacks
            # the logging setup of whatever imports it.
            setup_logging()

        logger.info("LLMOptimizer ready for %s", model_name or type(model).__name__)

    def load_model(self) -> "LLMOptimizer":
        """Load the model and tokenizer from HuggingFace."""
        if self.model_name is None:
            raise ValueError("No model_name to load.")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "load_model needs transformers: pip install transformers"
            ) from e

        logger.info("Loading %s", self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )

        # The tokenizer is only needed by save(). Quantizing and benchmarking
        # work without it, so a tokenizer that needs sentencepiece or tiktoken
        # to convert should not stop the run. Warn and carry on.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
        except Exception as e:
            logger.warning(
                "Could not load the tokenizer for %s (%s). Continuing without "
                "it; save() will not write tokenizer files.",
                self.model_name,
                e,
            )
            self.tokenizer = None
        if self.target_device != "cpu":
            self.model = self.model.to(self.target_device)
        return self

    def analyze(self) -> Dict[str, Any]:
        """Counts and suggestions for the loaded model."""
        self._require_model()
        return {
            "counts": analysis.analyze(self.model),
            "suggestions": analysis.suggest(self.model),
        }

    def optimize(self, calibration_data: Optional[Any] = None) -> nn.Module:
        """Quantize the model. Returns the quantized module."""
        self._require_model()
        pipeline = QuantizationPipeline(
            quantization_type=self.quantization,
            calibration_data=calibration_data,
        )
        self.optimized_model, self.quantization_report = pipeline.quantize(self.model)
        return self.optimized_model

    def benchmark(
        self,
        example_inputs: Any,
        num_runs: int = 50,
        warmup_runs: int = 5,
    ) -> Dict[str, Any]:
        """Time the original and the quantized model on the same input.

        Call after `optimize()`. The `significant` flag in the result is False
        when the difference is inside the run-to-run noise, which on a small
        model on a laptop is most of the time.
        """
        self._require_model()
        if self.optimized_model is None:
            raise ValueError("Call optimize() before benchmark().")

        self.benchmark_report = benchmark.compare(
            self.model,
            self.optimized_model,
            example_inputs,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
        )
        return self.benchmark_report

    def report(self) -> Dict[str, Any]:
        """Everything measured so far. Empty sections mean the step never ran."""
        return {
            "model_name": self.model_name,
            "target_device": self.target_device,
            "quantization": self.quantization,
            "model_loaded": self.model is not None,
            "optimized": self.optimized_model is not None,
            "quantization_report": self.quantization_report,
            "benchmark_report": self.benchmark_report,
        }

    def save(self, output_path: str) -> Path:
        """Save the quantized state dict and the measured report.

        Saves `state_dict()` rather than calling `save_pretrained`, because a
        quantized module is no longer a PreTrainedModel and does not have that
        method. The old code checked with hasattr and skipped the save when it
        was missing, so it wrote a metrics file and no model.
        """
        if self.optimized_model is None:
            raise ValueError("Nothing to save. Call optimize() first.")

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.optimized_model.state_dict(), output_dir / "model_state.pt")
        (output_dir / "optimization_report.json").write_text(
            json.dumps(self.report(), indent=2) + "\n"
        )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        logger.info("Saved to %s", output_dir)
        return output_dir

    def _require_model(self) -> None:
        if self.model is None:
            raise ValueError("No model. Call load_model() first.")
