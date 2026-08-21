#!/usr/bin/env python3
"""Quantize a real model and print what it measured.

Run:  python examples/basic_optimization.py

Downloads distilbert-base-uncased on first run, about 250 MB.
"""

import torch

from llm_optimizer import LLMOptimizer

MODEL_NAME = "distilbert-base-uncased"
SEQ_LEN = 32
RUNS = 100


def main():
    # Dynamic int8 is CPU only, so target_device is cpu. Passing cuda here
    # raises in the constructor rather than failing later inside forward().
    optimizer = LLMOptimizer(
        model_name=MODEL_NAME, quantization="dynamic", target_device="cpu"
    )
    optimizer.load_model()

    counts = optimizer.analyze()["counts"]
    print(f"{MODEL_NAME}")
    print(f"  parameters:       {counts['total_parameters']:,}")
    print(f"  quantizable:      {counts['quantizable_parameters']:,} "
          f"({counts['quantizable_parameter_fraction']:.0%})")
    for note in optimizer.analyze()["suggestions"]:
        print(f"  - {note}")

    optimizer.optimize()
    report = optimizer.quantization_report
    print("\nSize")
    print(f"  engine:    {report['qengine']}")
    print(f"  before:    {report['size_bytes_before'] / 1024**2:.2f} MB")
    print(f"  after:     {report['size_bytes_after'] / 1024**2:.2f} MB")
    print(f"  reduction: {report['size_reduction']:.1%}")

    vocab_size = optimizer.model.config.vocab_size
    example = torch.randint(0, vocab_size, (1, SEQ_LEN), dtype=torch.long)
    result = optimizer.benchmark(example, num_runs=RUNS)

    print(f"\nLatency, batch 1 x {SEQ_LEN} tokens, {RUNS} runs")
    print(f"  original:  {result['baseline']['median_ms']:.2f} ms "
          f"(stdev {result['baseline']['stdev_ms']:.2f})")
    print(f"  quantized: {result['candidate']['median_ms']:.2f} ms "
          f"(stdev {result['candidate']['stdev_ms']:.2f})")
    print(f"  speedup:   {result['speedup']:.2f}x")

    # The part that matters. A speedup number with significant=False is noise,
    # and reporting it as a win is how a benchmark table becomes fiction.
    if result["significant"]:
        direction = "faster" if result["speedup"] > 1 else "SLOWER"
        print(f"  the difference is real, and quantization made it {direction}")
    else:
        print("  the difference is inside the run-to-run noise: no measurable change")


if __name__ == "__main__":
    main()
