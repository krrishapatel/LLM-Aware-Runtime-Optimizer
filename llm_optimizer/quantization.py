"""Quantization built on torch.ao.quantization.

The int8 conversion here is real PyTorch quantization, and `quantize` reports the
size it measured rather than the size it was asked for. The version this replaces
could not run at all: it referenced `torch.qint16`, which does not exist, so
constructing the pipeline raised AttributeError before any model was touched.
"""

import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.ao.quantization as tq

from .analysis import QUANTIZABLE_TYPES, analyze, serialized_size_bytes

logger = logging.getLogger(__name__)

QUANTIZATION_TYPES = ("dynamic", "static", "fp16")


def default_qengine() -> str:
    """Pick a quantized backend that this machine actually has.

    The previous code hardcoded 'fbgemm', which is x86 only. On Apple Silicon
    the supported engine is qnnpack, so every static path failed there.
    """
    supported = list(torch.backends.quantized.supported_engines)
    for engine in ("fbgemm", "qnnpack"):
        if engine in supported:
            return engine
    raise RuntimeError(
        f"No usable quantized engine. torch reports: {supported or 'none'}"
    )


class QuantizationPipeline:
    """Quantize a model and report what changed.

    Three modes:

    - `dynamic`: int8 weights on Linear and RNN layers, activations quantized at
      run time. Works on any model with no calibration data. CPU only.
    - `static`: int8 weights and activations, calibrated on real batches. Needs
      a model built with QuantStub/DeQuantStub around the quantized region;
      most HuggingFace models are not, so this raises rather than returning a
      model that produces garbage.
    - `fp16`: half precision for everything. Halves the size, and is fast on
      CUDA but usually slower than fp32 on CPU.
    """

    def __init__(
        self,
        quantization_type: str = "dynamic",
        calibration_data: Optional[Iterable[Any]] = None,
        qengine: Optional[str] = None,
    ):
        if quantization_type not in QUANTIZATION_TYPES:
            raise ValueError(
                f"Unsupported quantization type {quantization_type!r}. "
                f"Choose one of {', '.join(QUANTIZATION_TYPES)}."
            )
        if quantization_type == "static" and calibration_data is None:
            # Checked here, not partway through quantize(), so the failure lands
            # before a large model has been loaded.
            raise ValueError(
                "Static quantization needs calibration_data: an iterable of "
                "batches to run through the model."
            )

        self.quantization_type = quantization_type
        self.calibration_data = calibration_data
        self.qengine = qengine or default_qengine()
        self.last_report: Dict[str, Any] = {}

        logger.info(
            "Quantization pipeline ready: %s, engine %s",
            quantization_type,
            self.qengine,
        )

    def quantize(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Quantize the model and return it with a measured report.

        Returns a tuple, because the size reduction is the point and hiding it
        behind a separate accessor is how the old code got away with logging a
        target it never hit.
        """
        before = analyze(model)

        if self.quantization_type in ("dynamic", "static"):
            # torch.backends.quantized.engine defaults to 'none' on some builds,
            # and the int8 kernels then fail with "Didn't find engine for
            # operation quantized::linear_prepack NoQEngine" partway through the
            # conversion. Set it before touching the model.
            torch.backends.quantized.engine = self.qengine

        if self.quantization_type == "dynamic":
            quantized = self._dynamic(model)
        elif self.quantization_type == "static":
            quantized = self._static(model)
        else:
            quantized = self._fp16(model)

        # Measured on the serialized state_dict, not on parameters(). A
        # dynamically quantized module keeps its weights in _packed_params, so
        # summing parameters() gives 0 bytes and a reported 100% reduction.
        after_size = serialized_size_bytes(quantized)
        before_size = before["serialized_size_bytes"]
        report = {
            "quantization_type": self.quantization_type,
            "qengine": self.qengine,
            "size_bytes_before": before_size,
            "size_bytes_after": after_size,
            "size_reduction": (
                (before_size - after_size) / before_size if before_size else 0.0
            ),
            "quantizable_parameter_fraction": before["quantizable_parameter_fraction"],
            "converted_modules": self._count_quantized_modules(quantized),
        }
        self.last_report = report

        logger.info(
            "Measured size reduction: %.1f%% (%d -> %d bytes)",
            report["size_reduction"] * 100,
            before_size,
            after_size,
        )
        return quantized, report

    def _dynamic(self, model: nn.Module) -> nn.Module:
        """int8 dynamic quantization. Returns a new module, does not mutate."""
        model.eval()
        return tq.quantize_dynamic(model, set(QUANTIZABLE_TYPES), dtype=torch.qint8)

    def _static(self, model: nn.Module) -> nn.Module:
        """int8 static quantization with calibration."""
        if not any(isinstance(m, tq.QuantStub) for m in model.modules()):
            raise ValueError(
                "Static quantization needs the model to wrap its quantized "
                "region in QuantStub/DeQuantStub. Use quantization_type="
                "'dynamic' for an unmodified model."
            )

        model.eval()
        model.qconfig = tq.get_default_qconfig(self.qengine)

        prepared = tq.prepare(model, inplace=False)
        batches = 0
        with torch.no_grad():
            for batch in self.calibration_data:
                if isinstance(batch, dict):
                    prepared(**batch)
                elif isinstance(batch, (tuple, list)):
                    prepared(*batch)
                else:
                    prepared(batch)
                batches += 1

        if batches == 0:
            # Converting an uncalibrated model gives observers with no observed
            # range, which quantizes every activation to zero. Better to stop.
            raise ValueError(
                "calibration_data yielded no batches, so the observers saw no "
                "activations. Converting now would produce a model that "
                "outputs zeros."
            )

        logger.info("Calibrated on %d batch(es)", batches)
        return tq.convert(prepared, inplace=False)

    def _fp16(self, model: nn.Module) -> nn.Module:
        """Half precision.

        `.half()` mutates in place and returns self, so this copies first. The
        old implementation also ran a loop that assigned the result of
        quantize_dynamic to a local name and dropped it, which did nothing.
        """
        import copy

        return copy.deepcopy(model).eval().half()

    @staticmethod
    def _count_quantized_modules(model: nn.Module) -> int:
        """How many layers came out as a quantized type.

        Checks the class's module path for 'quantized' rather than listing every
        quantized class, because the set differs between the dynamic, static and
        fx paths.

        Skips `_packed_params`, which is a child module of each quantized Linear
        and is itself a quantized type. Counting it doubled the total, so two
        quantized Linear layers reported as four.
        """
        return sum(
            1
            for name, module in model.named_modules()
            if name and not name.endswith("_packed_params")
            and "quantized" in type(module).__module__
        )

    def plan(self, model: nn.Module) -> Dict[str, Any]:
        """Per-layer view of what quantization would touch."""
        report = analyze(model)
        layers = []
        for name, module in model.named_modules():
            if name == "":
                continue
            quantizable = isinstance(module, QUANTIZABLE_TYPES)
            layers.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "parameters": sum(
                        p.numel() for p in module.parameters(recurse=False)
                    ),
                    "quantizable": quantizable,
                }
            )
        return {
            "summary": report,
            "layers": layers,
            "quantization_type": self.quantization_type,
        }
