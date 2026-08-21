#!/usr/bin/env python3
"""Command line interface.

Every subcommand does what it says. The old CLI had `deploy` and `benchmark`
subcommands whose bodies printed "not implemented in this version", and an
`info` command whose environment check was `if validate_environment():` against
a function that always returned a non-empty dict.
"""

import argparse
import json
import logging
import sys
from typing import List, Optional

from .core import LLMOptimizer
from .utils import get_system_info, setup_logging, validate_environment

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-optimize",
        description="Quantize a transformer and measure the result.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  llm-optimize info\n"
            "  llm-optimize analyze prajjwal1/bert-tiny\n"
            "  llm-optimize optimize prajjwal1/bert-tiny --seq-len 32 --runs 30\n"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )
    parser.add_argument("--log-file", help="Also write logs to this file")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("info", help="Show system and environment information")

    analyze_parser = subparsers.add_parser(
        "analyze", help="Count a model's layers and print quantization suggestions"
    )
    analyze_parser.add_argument("model_name", help="HuggingFace model name or path")

    optimize_parser = subparsers.add_parser(
        "optimize", help="Quantize a model, benchmark it, and report measured numbers"
    )
    optimize_parser.add_argument("model_name", help="HuggingFace model name or path")
    optimize_parser.add_argument(
        "--quantization",
        default="dynamic",
        choices=["dynamic", "fp16"],
        help="Quantization mode (default: dynamic). Static needs calibration "
        "data and is only available through the Python API.",
    )
    optimize_parser.add_argument(
        "--seq-len", type=int, default=32, help="Sequence length for the benchmark input"
    )
    optimize_parser.add_argument(
        "--runs", type=int, default=30, help="Timed runs per model (default: 30)"
    )
    optimize_parser.add_argument("--output", help="Directory to save the result in")

    return parser


def show_info() -> int:
    result = validate_environment()

    print("Environment")
    print("-" * 40)
    print("ready:", "yes" if result["ready"] else "no")
    for problem in result["problems"]:
        print("  problem:", problem)
    for warning in result["warnings"]:
        print("  warning:", warning)

    print("\nSystem")
    print("-" * 40)
    for key, value in get_system_info().items():
        if key in ("memory_total", "memory_available") and value is not None:
            value = f"{value / 1024**3:.1f} GB"
        print(f"  {key}: {value}")

    return EXIT_OK if result["ready"] else EXIT_FAILED


def analyze_model(args) -> int:
    optimizer = LLMOptimizer(model_name=args.model_name).load_model()
    result = optimizer.analyze()

    print(f"Analysis of {args.model_name}")
    print("-" * 40)
    for key, value in result["counts"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    print("\nSuggestions")
    print("-" * 40)
    for note in result["suggestions"]:
        print(f"  - {note}")

    return EXIT_OK


def run_optimization(args) -> int:
    import torch

    optimizer = LLMOptimizer(
        model_name=args.model_name,
        quantization=args.quantization,
        target_device="cpu",
    ).load_model()

    optimizer.optimize()
    quantization = optimizer.quantization_report
    print(f"Quantization ({quantization['quantization_type']})")
    print("-" * 40)
    print(f"  size before: {quantization['size_bytes_before'] / 1024**2:.2f} MB")
    print(f"  size after:  {quantization['size_bytes_after'] / 1024**2:.2f} MB")
    print(f"  reduction:   {quantization['size_reduction']:.1%}")
    print(f"  converted modules: {quantization['converted_modules']}")

    vocab_size = getattr(optimizer.model.config, "vocab_size", 1000)
    example = torch.randint(0, vocab_size, (1, args.seq_len), dtype=torch.long)
    result = optimizer.benchmark(example, num_runs=args.runs)

    print(f"\nLatency over {args.runs} runs, batch 1 x {args.seq_len} tokens")
    print("-" * 40)
    print(
        f"  original:  {result['baseline']['median_ms']:.2f} ms median "
        f"(stdev {result['baseline']['stdev_ms']:.2f})"
    )
    print(
        f"  quantized: {result['candidate']['median_ms']:.2f} ms median "
        f"(stdev {result['candidate']['stdev_ms']:.2f})"
    )
    print(f"  speedup:   {result['speedup']:.2f}x")
    if not result["significant"]:
        print("  the difference is inside the run-to-run noise, so treat it as no change")

    if args.output:
        saved = optimizer.save(args.output)
        print(f"\nSaved to {saved}")
    else:
        print("\n" + json.dumps(optimizer.report()["quantization_report"], indent=2))

    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_argument_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_OK

    setup_logging(level=args.log_level, log_file=args.log_file)

    handlers = {
        "info": lambda: show_info(),
        "analyze": lambda: analyze_model(args),
        "optimize": lambda: run_optimization(args),
    }
    try:
        return handlers[args.command]()
    except (ImportError, ValueError, KeyError, OSError) as e:
        # Narrow, so a bug in the pipeline still shows its traceback instead of
        # being flattened into a one-line error message.
        print(f"error: {e}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
