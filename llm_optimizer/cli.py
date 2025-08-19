#!/usr/bin/env python3
"""
Command-line interface for LLM-Aware Runtime Optimizer.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .core import LLMOptimizer
from .utils import setup_logging, validate_environment, get_system_info

def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level, log_file=args.log_file)
    
    try:
        if args.command == "info":
            show_info()
        elif args.command == "optimize":
            run_optimization(args)
        elif args.command == "analyze":
            analyze_model(args)
        elif args.command == "deploy":
            deploy_model(args)
        elif args.command == "benchmark":
            run_benchmark(args)
        else:
            parser.print_help()
            
    except Exception as e:
        logging.error(f"Command failed: {e}")
        sys.exit(1)

def create_argument_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="LLM-Aware Runtime Optimizer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llm-optimize info                    # Show system information
  llm-optimize optimize gpt2           # Optimize GPT-2 model
  llm-optimize analyze bert-base       # Analyze BERT model
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        help="Log file path (optional)"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize a model")
    optimize_parser.add_argument("model_name", help="Model name or path")
    optimize_parser.add_argument("--device", default="cpu", help="Target device (default: cpu)")
    optimize_parser.add_argument("--level", default="balanced", 
                               choices=["conservative", "balanced", "aggressive"],
                               help="Optimization level (default: balanced)")
    optimize_parser.add_argument("--output", help="Output directory for optimized model")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a model")
    analyze_parser.add_argument("model_name", help="Model name or path")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy a model")
    deploy_parser.add_argument("model_path", help="Path to optimized model")
    deploy_parser.add_argument("--platform", default="local", help="Deployment platform")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark model performance")
    benchmark_parser.add_argument("model_path", help="Path to model")
    benchmark_parser.add_argument("--runs", type=int, default=100, help="Number of benchmark runs")
    
    return parser

def show_info():
    """Show system information."""
    print("🚀 LLM-Aware Runtime Optimizer")
    print("=" * 50)
    
    # Environment validation
    print("\n📋 Environment Validation:")
    if validate_environment():
        print("✅ Environment is ready for LLM optimization")
    else:
        print("❌ Environment has issues")
    
    # System information
    print("\n💻 System Information:")
    try:
        sys_info = get_system_info()
        for key, value in sys_info.items():
            if key == "memory_total" or key == "memory_available":
                value = f"{value / (1024**3):.1f} GB"
            print(f"  {key.replace('_', ' ').title()}: {value}")
    except Exception as e:
        print(f"  ❌ Could not get system info: {e}")
    
    # Package information
    print("\n📦 Package Information:")
    try:
        import llm_optimizer
        print(f"  Version: {llm_optimizer.__version__}")
        print(f"  Author: {llm_optimizer.__author__}")
    except Exception as e:
        print(f"  ❌ Could not get package info: {e}")

def run_optimization(args):
    """Run model optimization."""
    print(f"🚀 Starting optimization of {args.model_name}")
    
    try:
        optimizer = LLMOptimizer(
            model_name=args.model_name,
            target_device=args.device,
            optimization_level=args.level
        )
        
        print("📥 Loading model...")
        optimizer.load_model()
        
        print("⚡ Running optimization...")
        optimized_model = optimizer.optimize()
        
        if args.output:
            print(f"💾 Saving optimized model to {args.output}...")
            output_path = optimizer.save_optimized_model(args.output)
            print(f"✅ Model saved to: {output_path}")
        
        # Show optimization report
        report = optimizer.get_optimization_report()
        print("\n📊 Optimization Report:")
        for key, value in report.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
            
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        raise

def analyze_model(args):
    """Analyze a model."""
    print(f"🔍 Analyzing model: {args.model_name}")
    
    try:
        optimizer = LLMOptimizer(
            model_name=args.model_name,
            target_device="cpu"
        )
        
        print("📥 Loading model...")
        optimizer.load_model()
        
        # Basic model analysis
        if hasattr(optimizer.model, 'parameters'):
            total_params = sum(p.numel() for p in optimizer.model.parameters())
            print(f"📊 Total parameters: {total_params:,}")
            print(f"📊 Estimated model size: {total_params * 4 / (1024**2):.1f} MB")
        
        print("✅ Model analysis completed")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise

def deploy_model(args):
    """Deploy a model."""
    print(f"🚀 Deploying model from: {args.model_path}")
    print("⚠️  Deployment functionality not implemented in this version")
    print("   This is a placeholder for future implementation")

def run_benchmark(args):
    """Run model benchmarking."""
    print(f"⚡ Benchmarking model: {args.model_path}")
    print("⚠️  Benchmarking functionality not implemented in this version")
    print("   This is a placeholder for future implementation")

if __name__ == "__main__":
    main()
