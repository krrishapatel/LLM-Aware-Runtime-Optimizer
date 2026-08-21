# LLM Runtime Optimizer

Quantize a PyTorch transformer, then measure whether it actually got faster.

The point of this tool is the second half. Quantization is easy to apply and easy
to report as a win, because the size drop is real and immediate. Whether latency
improved is a separate question, and on a lot of hardware the answer is no. This
prints both numbers and tells you when the latency difference is inside the
run-to-run noise.

[![CI](https://github.com/krrishapatel/LLM-Aware-Runtime-Optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/krrishapatel/LLM-Aware-Runtime-Optimizer/actions/workflows/ci.yml)

## What it does

- Counts a model's layers and reports how much of it is quantizable
- Applies `torch.ao.quantization` dynamic int8, static int8, or fp16
- Measures size from the serialized `state_dict`, not from `parameters()`
- Times both models with warmup and device sync, and reports a significance flag
- Exports to ONNX and reports what onnxruntime's graph optimizer changed
- Writes a SageMaker deployment package and prints the AWS CLI commands to use it

## What it does not do

No MLIR. No TensorRT. No custom compiler passes. Nothing is deployed to AWS for
you. Earlier versions of this repo had modules named after those things which did
not call them, and a README quoting a 48% latency reduction that came from
hardcoded per-layer constants rather than a measurement. Those modules are gone.
See [the note below](#what-changed-in-020).

## Install

```bash
pip install -e .
pip install -r requirements-dev.txt   # tests, onnx, onnxruntime
```

Python 3.9+ and PyTorch 2.0+. No CUDA required.

## Use

```bash
llm-optimize info
llm-optimize analyze distilbert-base-uncased
llm-optimize optimize distilbert-base-uncased --seq-len 32 --runs 100
```

```python
from llm_optimizer import LLMOptimizer
import torch

opt = LLMOptimizer(model_name="distilbert-base-uncased", quantization="dynamic")
opt.load_model()
opt.optimize()

print(opt.quantization_report["size_reduction"])

result = opt.benchmark(torch.randint(0, 30522, (1, 32)), num_runs=100)
print(result["speedup"], result["significant"])
```

`significant` is False when the median gap is smaller than the two standard
deviations added together. When it is False, the speedup number means nothing.

## Measured results

`distilbert-base-uncased`, dynamic int8, batch 1. Apple Silicon (arm64), torch
2.13.0, Python 3.12.13, qnnpack backend. 100 timed runs after 5 warmup runs.

| Measurement | Before | After | Change |
|---|---|---|---|
| Serialized size | 253.19 MB | 131.71 MB | **48.0% smaller** |
| Median latency, 32 tokens | 10.87 ms | 9.79 ms | 1.11x, not significant |
| Median latency, 128 tokens | 20.20 ms | 30.23 ms | **0.67x, significantly slower** |

36 of 91 modules were converted, covering 64% of the parameters.

Read the third row. Dynamic int8 on this model on this machine is a real
slowdown at 128 tokens, reproducible across runs, and well outside the noise. At
32 tokens the apparent 1.11x speedup is inside the noise and the tool says so.
The size reduction is the only reliable win here.

That result is backend specific. qnnpack is what's available on Apple Silicon;
fbgemm on x86 generally does better on int8 matmul. The tool reports which
engine it used so you can tell the two situations apart. Run it on your own
hardware rather than trusting this table.

Two more measured results worth knowing, both covered by tests:

- **int8 can make a small model bigger.** Static int8 on a 544-parameter MLP
  grows it 6.6%; dynamic int8 on a 144-parameter one grows it 30%. Per-tensor
  scales, zero points and packing metadata cost more than the weights save. The
  familiar ~75% figure only arrives above roughly 100k parameters.
- **fp16 is not exactly 50%.** It measures 24% on a 4.5KB model, 46.7% on a 35KB
  one, and 49.8% at 500KB, because `torch.save`'s zip container is a fixed ~1KB
  that does not halve along with the weights.

Which is the general point: the size reduction depends on the model, and a
number written into a README ahead of time is a guess.

## Tests

```bash
python -m pytest
```

130 tests. They encode the measurements above, including the cases where
quantization loses. Three bugs in this rewrite were caught by these tests rather
than assumed away: a quantized model reporting 0 bytes because
`DynamicQuantizedLinear` keeps weights in `_packed_params`, a module count that
double-counted those same `_packed_params` children, and an assumption that graph
optimization always reduces the ONNX node count.

That last one is worth a note. On the same small transformer at level
`extended`, onnxruntime 1.29 takes the graph from 25 nodes to 28 while doing
genuine fusion: it folds nine Adds and nine MatMuls into seven Gemms, then
inserts fourteen Reshapes to give those Gemms 2D inputs. onnxruntime 1.19 goes
the other way, 30 nodes to 21, fusing into the contrib op `FusedMatMul` instead.
Both are correctly optimized graphs. Node count is not a quality metric, and
neither is the name of the fused operator, which is why CI runs both versions.

## SageMaker packaging

`SageMakerPackageBuilder` traces the model to TorchScript and writes the files
the SageMaker PyTorch container expects: `model.pt`, `config.json`, an
`inference.py` with the four handler functions, `requirements.txt`, `Dockerfile`,
and a `model.tar.gz`.

It does not call AWS. There is no `deploy()` method, no boto3 dependency, and no
credentials are read. `next_steps()` returns the `aws s3 cp` and
`aws sagemaker create-*` commands as strings for you to run yourself, including
the `delete-endpoint` reminder, since a forgotten `ml.g4dn.xlarge` endpoint is
$537 a month at $0.736 an hour. `cost_estimate()` prices the instance from a table dated
2026-08-17.

## What changed in 0.2.0

0.1.0 was 3032 lines that mostly did not work. `optimize()` was
`self.optimized_model = self.model`. `mlir.py` set `op["optimized"] = True` on a
dict of operation names. `tensorrt_integration.py` returned
`{"type": "tensorrt_engine", ...}` and benchmarked that dict.
`onnx_rewriter.py` had six passes whose entire body was `return onnx_model`.
`deployment.py` imported no boto3, logged fake AWS successes, wrote a file
called `model.placeholder`, and generated a `predict_fn` returning
`torch.randn(1, n, 50257)` decoded as text. `setup.py` could not install.
`requirements.txt` listed PyPI packages that do not exist.

All of that was deleted. What's here now is smaller and does what it says.

## License

MIT
