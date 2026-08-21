"""Timed forward passes on a real model with real input.

This is the only place in the package that produces a latency number. Every
figure in the README comes from running this. The module it replaced timed a
function that built a dictionary and reported the result as TensorRT inference
latency.
"""

import logging
import statistics
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _forward(model: nn.Module, inputs: Any) -> Any:
    """Call the model with whatever shape the inputs came in as."""
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


def _synchronize(device: Optional[torch.device]) -> None:
    """Wait for queued GPU work before stopping the clock.

    CUDA and MPS launches are asynchronous. Without this a timing loop measures
    how fast Python can enqueue kernels, which comes out far too fast and looks
    like a huge speedup.
    """
    if device is None:
        return
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _model_device(model: nn.Module) -> Optional[torch.device]:
    for param in model.parameters():
        return param.device
    return None


def summarize(latencies_ms: List[float]) -> Dict[str, float]:
    """Turn a list of per-run latencies into statistics.

    Reports stdev alongside the mean. A mean on its own hides the case where
    the runs are so noisy that a comparison between two models means nothing.
    """
    if not latencies_ms:
        raise ValueError("No latencies to summarize.")
    ordered = sorted(latencies_ms)
    # min(..., len - 1) because int(0.95 * n) equals n for n = 1, which would
    # index off the end of the list.
    p95_index = min(int(0.95 * len(ordered)), len(ordered) - 1)
    p99_index = min(int(0.99 * len(ordered)), len(ordered) - 1)
    return {
        "runs": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "stdev_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p95_ms": ordered[p95_index],
        "p99_ms": ordered[p99_index],
    }


def measure_latency(
    model: nn.Module,
    inputs: Any,
    num_runs: int = 50,
    warmup_runs: int = 5,
) -> Dict[str, float]:
    """Time `num_runs` forward passes and return the statistics.

    Runs in eval mode under no_grad, because a model left in train mode pays for
    dropout and for building a graph it never uses.
    """
    if num_runs < 1:
        raise ValueError("num_runs must be at least 1.")
    if warmup_runs < 0:
        raise ValueError("warmup_runs cannot be negative.")

    was_training = model.training
    model.eval()
    device = _model_device(model)
    latencies_ms: List[float] = []

    try:
        with torch.no_grad():
            for _ in range(warmup_runs):
                _forward(model, inputs)
            _synchronize(device)

            for _ in range(num_runs):
                start = time.perf_counter()
                _forward(model, inputs)
                _synchronize(device)
                latencies_ms.append((time.perf_counter() - start) * 1000)
    finally:
        # Put the model back how it was found. A benchmark that silently leaves
        # a model in eval mode breaks the caller's next training step.
        model.train(was_training)

    return summarize(latencies_ms)


def compare(
    baseline: nn.Module,
    candidate: nn.Module,
    inputs: Any,
    num_runs: int = 50,
    warmup_runs: int = 5,
) -> Dict[str, Any]:
    """Time two models on the same input and report the measured difference.

    `significant` is False when the gap is inside the combined run-to-run noise.
    A 5% difference with a 10% standard deviation is not a result, and labelling
    it one is how an optimizer ends up claiming a speedup it does not have.
    """
    base = measure_latency(baseline, inputs, num_runs, warmup_runs)
    cand = measure_latency(candidate, inputs, num_runs, warmup_runs)

    base_ms = base["median_ms"]
    cand_ms = cand["median_ms"]
    combined_noise = base["stdev_ms"] + cand["stdev_ms"]

    return {
        "baseline": base,
        "candidate": cand,
        "speedup": base_ms / cand_ms if cand_ms > 0 else float("inf"),
        "latency_change": (cand_ms - base_ms) / base_ms if base_ms > 0 else 0.0,
        "significant": abs(cand_ms - base_ms) > combined_noise,
    }
