#!/usr/bin/env python3
"""Compare quantization modes on local models. No download, no network.

Run:  python examples/quantization_example.py

Shows the two results that are easy to get wrong:
  1. int8 makes a small enough model bigger, not smaller.
  2. fp16 is not exactly 50%, because torch.save's container does not halve.
"""

import torch
import torch.nn as nn
import torch.ao.quantization as tq

from llm_optimizer import QuantizationPipeline
from llm_optimizer.analysis import serialized_size_bytes


class MLP(nn.Module):
    """Two linear layers. `width` controls how much there is to quantize."""

    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)
        )

    def forward(self, x):
        return self.net(x)


class StaticMLP(nn.Module):
    """The same thing wrapped in stubs, which static quantization requires."""

    def __init__(self, width=64):
        super().__init__()
        self.quant = tq.QuantStub()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)
        )
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.net(self.quant(x)))


def show(label, before, after):
    change = 1 - after / before
    word = "smaller" if change > 0 else "LARGER"
    print(f"  {label:<28} {before:>8,} -> {after:>8,} bytes  "
          f"({abs(change):.1%} {word})")


def size_across_widths():
    # Quantization stores a scale and a zero point per tensor plus packing
    # metadata. On a small enough model that overhead is larger than the weights
    # it saves, so the checkpoint grows. A fixed "75% size reduction" claim is
    # false at this end of the range: the 75% only arrives above roughly 100k
    # parameters. Where the crossover sits depends on the mode, so both are here.
    print("Dynamic int8 by model width")
    for width in (8, 16, 32, 256, 1024):
        model = MLP(width)
        before = serialized_size_bytes(model)
        quantized, report = QuantizationPipeline("dynamic").quantize(model)
        show(f"width {width}", before, report["size_bytes_after"])
        assert quantized is not model

    print("\nStatic int8 by model width")
    for width in (16, 32, 64, 256):
        model = StaticMLP(width)
        before = serialized_size_bytes(model)
        calibration = [torch.randn(4, width) for _ in range(3)]
        _, report = QuantizationPipeline(
            "static", calibration_data=calibration
        ).quantize(model)
        show(f"width {width}", before, report["size_bytes_after"])


def modes_on_one_model():
    width = 256
    print(f"\nModes, width {width}")

    model = MLP(width)
    before = serialized_size_bytes(model)

    _, dynamic = QuantizationPipeline("dynamic").quantize(model)
    show("dynamic int8", before, dynamic["size_bytes_after"])
    print(f"    engine {dynamic['qengine']}, "
          f"{dynamic['converted_modules']} modules converted")

    # fp16 halves every stored float but not the ~1KB zip container torch.save
    # writes around them, so the reduction lands under 50%.
    _, fp16 = QuantizationPipeline("fp16").quantize(MLP(width))
    show("fp16", before, fp16["size_bytes_after"])

    # Static needs real activations to set the observer ranges. Converting
    # without them quantizes every activation to zero, so the pipeline raises
    # instead of returning a broken model.
    calibration = [torch.randn(8, width) for _ in range(10)]
    _, static = QuantizationPipeline(
        "static", calibration_data=calibration
    ).quantize(StaticMLP(width))
    show("static int8 (calibrated)", before, static["size_bytes_after"])


def errors_are_raised_early():
    print("\nRejected at construction, not halfway through")
    for kwargs, why in (
        ({"quantization_type": "int4"}, "int4 is not a mode"),
        ({"quantization_type": "static"}, "static without calibration data"),
    ):
        try:
            QuantizationPipeline(**kwargs)
        except ValueError as e:
            print(f"  {why}: {e}")


def main():
    size_across_widths()
    modes_on_one_model()
    errors_are_raised_early()


if __name__ == "__main__":
    main()
