"""ONNX export, validation, and graph statistics.

`onnx` and `onnxruntime` are imported inside the functions, so the rest of the
package works without them installed.

What this does not do: rewrite the graph itself. The file this replaces had six
methods named after ONNX passes (constant folding, dead code elimination,
operator fusion) whose entire body was `return onnx_model` under a comment
saying it would be implemented later. The real graph rewriting lives in
onnxruntime, so `optimize_with_onnxruntime` hands the job to it and reports the
node counts before and after, which is a number you can check.
"""

import logging
import os
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# onnxruntime's own levels. 'all' includes layout changes that are hardware
# specific, so the optimized graph is not portable to a different machine.
GRAPH_OPTIMIZATION_LEVELS = ("disable", "basic", "extended", "all")


def export(
    model: nn.Module,
    example_inputs: Any,
    output_path: str,
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 17,
) -> str:
    """Export a model to ONNX at `output_path`.

    The caller supplies `example_inputs`. The old code guessed them from the
    config, built `torch.randint(0, 1000, ...)` token ids for every model, and
    hardcoded input_names of ['input_ids', 'attention_mask'], so exporting
    anything that was not a two-input language model produced a file whose
    inputs were mislabelled.
    """
    try:
        import onnx  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "ONNX export needs the onnx package: pip install onnx"
        ) from e

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                example_inputs,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
    finally:
        model.train(was_training)

    logger.info("Exported ONNX model to %s", output_path)
    return output_path


def validate(onnx_path: str) -> Dict[str, Any]:
    """Run the ONNX checker and return graph statistics.

    Raises whatever `onnx.checker` raises. An invalid model that is reported as
    valid is worse than a failed export.
    """
    import onnx

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    return graph_stats(model)


def graph_stats(onnx_model: Any) -> Dict[str, Any]:
    """Node counts by operator type, plus input and output names."""
    op_counts = Counter(node.op_type for node in onnx_model.graph.node)
    return {
        "nodes": len(onnx_model.graph.node),
        "op_counts": dict(sorted(op_counts.items())),
        "initializers": len(onnx_model.graph.initializer),
        "inputs": [i.name for i in onnx_model.graph.input],
        "outputs": [o.name for o in onnx_model.graph.output],
        "opset": [
            {"domain": o.domain, "version": o.version}
            for o in onnx_model.opset_import
        ],
    }


def optimize_with_onnxruntime(
    onnx_path: str,
    output_path: str,
    level: str = "extended",
) -> Tuple[str, Dict[str, Any]]:
    """Let onnxruntime rewrite the graph, and report what it changed.

    Returns the output path and a dict describing the difference. Read
    `ops_added` and `ops_removed`, not `nodes_removed`: a node count can go up
    while real fusion happens, and the direction is not even stable across
    onnxruntime versions. Measured on the same small transformer at level
    'extended':

      1.29: 25 -> 28 nodes. MatMul+Add folds into Gemm and the layer norms into
            SkipLayerNormalization, then fourteen Reshapes are inserted to give
            the Gemms 2D inputs.
      1.19: 30 -> 21 nodes, fusing the matmuls into the contrib op FusedMatMul.

    Both are correctly optimized graphs. Node count is not a quality metric, and
    neither is the name of the fused operator.
    """
    if level not in GRAPH_OPTIMIZATION_LEVELS:
        raise ValueError(
            f"level must be one of {', '.join(GRAPH_OPTIMIZATION_LEVELS)}, "
            f"got {level!r}"
        )
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "Graph optimization needs onnx and onnxruntime: "
            "pip install onnx onnxruntime"
        ) from e

    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    before = graph_stats(onnx.load(onnx_path))

    options = ort.SessionOptions()
    options.graph_optimization_level = levels[level]
    options.optimized_model_filepath = output_path
    # Creating the session is what writes the optimized file.
    ort.InferenceSession(onnx_path, options, providers=["CPUExecutionProvider"])

    if not os.path.exists(output_path):
        raise RuntimeError(
            f"onnxruntime did not write {output_path}. This happens when the "
            f"execution provider cannot serialize the optimized graph."
        )

    after = graph_stats(onnx.load(output_path))
    before_ops = before["op_counts"]
    after_ops = after["op_counts"]
    change = {
        "level": level,
        "nodes_before": before["nodes"],
        "nodes_after": after["nodes"],
        "nodes_removed": before["nodes"] - after["nodes"],
        "op_counts_before": before_ops,
        "op_counts_after": after_ops,
        # The signal worth reading. New operator types are the fusions that
        # happened; disappeared types are what got folded into them.
        "ops_added": sorted(set(after_ops) - set(before_ops)),
        "ops_removed": sorted(set(before_ops) - set(after_ops)),
    }
    logger.info(
        "onnxruntime %s: %d nodes -> %d nodes, added %s, removed %s",
        level,
        before["nodes"],
        after["nodes"],
        change["ops_added"] or "nothing",
        change["ops_removed"] or "nothing",
    )
    return output_path, change
