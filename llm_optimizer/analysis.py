"""Static analysis of a PyTorch model: what is in it, and what can be quantized.

This replaces an earlier `mlir.py` that was named after MLIR but never invoked
it. Nothing here compiles anything. It walks `named_modules()` and counts, which
is all the original code did too.
"""

import io
import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# torch.ao.quantization.quantize_dynamic replaces these module types with a
# quantized version. Anything else in the model is left at full precision, so
# these are the only layers that contribute to a size reduction.
QUANTIZABLE_TYPES = (nn.Linear, nn.LSTM, nn.GRU, nn.RNN)


def count_parameters(model: nn.Module) -> int:
    """Total number of parameters, including ones that are frozen."""
    return sum(p.numel() for p in model.parameters())


def tensor_size_bytes(model: nn.Module) -> int:
    """Size of the parameters and buffers reachable as tensors.

    Uses `element_size()` rather than assuming 4 bytes per parameter, so an fp16
    model reports its real footprint.

    Do not use this to measure a dynamically quantized model. Its weights live
    inside `_packed_params`, which is neither a parameter nor a buffer, so this
    returns 0 and a naive before/after comparison shows a 100% size reduction.
    Use `serialized_size_bytes` for that.
    """
    total = 0
    for param in model.parameters():
        total += param.nelement() * param.element_size()
    for buffer in model.buffers():
        total += buffer.nelement() * buffer.element_size()
    return total


def serialized_size_bytes(model: nn.Module) -> int:
    """Bytes the model's state_dict takes when saved.

    This is the number that holds for every quantization mode, because
    `state_dict()` includes the packed int8 weights that `parameters()` misses.
    It is a few hundred bytes larger than the raw tensor data because of the
    zip container torch.save writes.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def model_size_bytes(model: nn.Module) -> int:
    """Backwards-compatible alias for `tensor_size_bytes`."""
    return tensor_size_bytes(model)


def analyze(model: nn.Module) -> Dict[str, Any]:
    """Count the modules in a model by category.

    `quantizable_parameters` is the number that matters for size reduction:
    dynamic quantization only touches the layer types in QUANTIZABLE_TYPES, so
    a model whose weight is mostly embeddings will barely shrink no matter what
    the config says.
    """
    counts = {
        "total_modules": 0,
        "linear": 0,
        "embedding": 0,
        "layer_norm": 0,
        "conv": 0,
        "attention_named": 0,
        "quantizable_modules": 0,
    }
    quantizable_parameters = 0

    for name, module in model.named_modules():
        if name == "":
            # named_modules() yields the root under the empty name. Counting it
            # would inflate total_modules by one on every model.
            continue
        counts["total_modules"] += 1

        if isinstance(module, nn.Linear):
            counts["linear"] += 1
        elif isinstance(module, nn.Embedding):
            counts["embedding"] += 1
        elif isinstance(module, nn.LayerNorm):
            counts["layer_norm"] += 1
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            counts["conv"] += 1

        # A name match, not a type check. There is no nn.Attention, so this is a
        # heuristic on the module's path and it can miss or over-count.
        if "attention" in name.lower() or "attn" in name.lower():
            counts["attention_named"] += 1

        if isinstance(module, QUANTIZABLE_TYPES):
            counts["quantizable_modules"] += 1
            quantizable_parameters += sum(
                p.numel() for p in module.parameters(recurse=False)
            )

    total_parameters = count_parameters(model)
    return {
        **counts,
        "total_parameters": total_parameters,
        "quantizable_parameters": quantizable_parameters,
        "quantizable_parameter_fraction": (
            quantizable_parameters / total_parameters if total_parameters else 0.0
        ),
        "tensor_size_bytes": tensor_size_bytes(model),
        "serialized_size_bytes": serialized_size_bytes(model),
    }


def suggest(model: nn.Module) -> List[str]:
    """Plain-text suggestions based on the counts.

    Deliberately returns strings and not predicted speedups. The version of
    this file that shipped before returned a `latency_reduction` figure built
    by adding 0.15 per transformer layer and 0.20 per attention layer, which
    described nothing about the model or the hardware. If you want a number,
    run `benchmark.compare` and measure it.
    """
    report = analyze(model)
    notes: List[str] = []

    fraction = report["quantizable_parameter_fraction"]
    if report["quantizable_modules"] == 0:
        notes.append(
            "No Linear or RNN layers found, so dynamic quantization has nothing "
            "to convert and will not change the model size."
        )
    elif fraction < 0.25:
        notes.append(
            f"Only {fraction:.0%} of parameters are in quantizable layers. "
            f"Dynamic quantization can shrink at most that share of the weights."
        )
    else:
        notes.append(
            f"{fraction:.0%} of parameters are in {report['quantizable_modules']} "
            f"quantizable layers, so int8 dynamic quantization is worth trying."
        )

    if report["embedding"] > 0:
        notes.append(
            f"{report['embedding']} embedding layer(s) stay at full precision "
            f"under dynamic quantization. On small models these often dominate "
            f"the parameter count."
        )

    if report["conv"] > 0:
        notes.append(
            f"{report['conv']} conv layer(s) need static quantization with "
            f"calibration data. Dynamic quantization skips them."
        )

    # No claim about the direction of the latency change. On distilbert with the
    # qnnpack backend, dynamic int8 measured 0.67x at 128 tokens, so "usually
    # faster on CPU" is not something this function can assert.
    notes.append(
        "Measure before and after with benchmark.compare. Int8 dynamic "
        "quantization is CPU only, and whether it is faster depends on the "
        "backend and the sequence length, so do not assume it."
    )
    return notes
