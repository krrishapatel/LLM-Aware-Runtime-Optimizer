"""ONNX export and graph statistics.

Skipped when onnx is not installed, which is why CI installs it: a skipped test
proves nothing, so the workflow includes onnx and onnxruntime in the dev
requirements.
"""

import pytest
import torch

from llm_optimizer import onnx_export

onnx = pytest.importorskip("onnx", reason="onnx is an optional dependency")


class TestExport:
    def test_a_model_exports_and_validates(self, mlp, tmp_path):
        path = onnx_export.export(
            mlp, torch.randn(1, 32), str(tmp_path / "model.onnx")
        )
        stats = onnx_export.validate(path)

        assert stats["nodes"] > 0
        assert stats["initializers"] > 0

    def test_the_caller_names_the_inputs(self, mlp, tmp_path):
        # The old code hardcoded input_names=['input_ids', 'attention_mask'] for
        # every model and built token ids with torch.randint regardless of what
        # the model took, so anything that was not a two-input language model
        # came out with mislabelled inputs.
        path = onnx_export.export(
            mlp,
            torch.randn(1, 32),
            str(tmp_path / "named.onnx"),
            input_names=["features"],
            output_names=["scores"],
        )
        stats = onnx_export.validate(path)

        assert stats["inputs"] == ["features"]
        assert stats["outputs"] == ["scores"]

    def test_the_model_is_left_in_the_mode_it_arrived_in(self, mlp, tmp_path):
        mlp.train()
        onnx_export.export(mlp, torch.randn(1, 32), str(tmp_path / "m.onnx"))

        assert mlp.training is True

    def test_a_language_model_exports_with_dynamic_axes(
        self, tiny_lm, tiny_lm_input, tmp_path
    ):
        path = onnx_export.export(
            tiny_lm,
            tiny_lm_input,
            str(tmp_path / "lm.onnx"),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
        )
        stats = onnx_export.validate(path)

        assert stats["inputs"] == ["input_ids"]
        assert "MatMul" in stats["op_counts"] or "Gemm" in stats["op_counts"]


class TestGraphStats:
    def test_operators_are_counted_by_type(self, mlp, tmp_path):
        path = onnx_export.export(mlp, torch.randn(1, 32), str(tmp_path / "m.onnx"))
        stats = onnx_export.validate(path)

        assert sum(stats["op_counts"].values()) == stats["nodes"]
        assert "Relu" in stats["op_counts"]

    def test_the_opset_is_reported(self, mlp, tmp_path):
        path = onnx_export.export(
            mlp, torch.randn(1, 32), str(tmp_path / "m.onnx"), opset_version=17
        )
        stats = onnx_export.validate(path)

        assert any(entry["version"] == 17 for entry in stats["opset"])


class TestOptimize:
    def test_an_invalid_level_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="level must be one of"):
            onnx_export.optimize_with_onnxruntime(
                str(tmp_path / "in.onnx"), str(tmp_path / "out.onnx"), level="maximum"
            )

    def test_optimization_reports_the_node_counts_it_measured(self, tiny_lm, tiny_lm_input, tmp_path):
        # The replacement for six methods that logged "operator fusion applied"
        # and returned the graph untouched. Whether fusion happened is now a
        # number the caller can check.
        pytest.importorskip("onnxruntime", reason="onnxruntime is optional")

        source = onnx_export.export(
            tiny_lm, tiny_lm_input, str(tmp_path / "lm.onnx")
        )
        _, change = onnx_export.optimize_with_onnxruntime(
            source, str(tmp_path / "lm_opt.onnx"), level="extended"
        )

        assert change["nodes_before"] > 0
        assert change["nodes_after"] > 0
        assert change["nodes_removed"] == change["nodes_before"] - change["nodes_after"]

    def test_real_fusion_shows_up_as_new_operator_types(
        self, tiny_lm, tiny_lm_input, tmp_path
    ):
        # The matmuls get folded into a single fused operator, but which one
        # depends on the onnxruntime version. CI measured both: 1.29 produces
        # Gemm, 1.19 produces the contrib op FusedMatMul. Asserting the exact
        # name passed locally and failed on the older runtime, so the assertion
        # is on the shape of the change rather than the operator's spelling.
        pytest.importorskip("onnxruntime", reason="onnxruntime is optional")

        source = onnx_export.export(tiny_lm, tiny_lm_input, str(tmp_path / "lm2.onnx"))
        _, change = onnx_export.optimize_with_onnxruntime(
            source, str(tmp_path / "lm2_opt.onnx"), level="extended"
        )

        assert change["ops_added"], "no new operator types, so nothing was fused"
        assert change["ops_removed"], "no operator types disappeared"
        assert any(
            "MatMul" in op or "Gemm" in op for op in change["ops_added"]
        ), f"the matmuls were not fused into anything: {change['ops_added']}"

    def test_the_node_count_direction_is_not_predictable(
        self, tiny_lm, tiny_lm_input, tmp_path
    ):
        # Why nodes_removed is not the success metric. Both directions were
        # measured on the same model and the same input:
        #
        #   onnxruntime 1.29: 25 -> 28 nodes. Nine Adds and nine MatMuls fold
        #     into seven Gemms, then fourteen Reshapes are inserted to give
        #     those Gemms 2D inputs, so the total rises.
        #   onnxruntime 1.19: 30 -> 21 nodes.
        #
        # A test asserting either direction is asserting a runtime version. So
        # this checks the arithmetic holds and that something changed, and the
        # caller is told to read ops_added instead.
        pytest.importorskip("onnxruntime", reason="onnxruntime is optional")

        source = onnx_export.export(tiny_lm, tiny_lm_input, str(tmp_path / "lm3.onnx"))
        _, change = onnx_export.optimize_with_onnxruntime(
            source, str(tmp_path / "lm3_opt.onnx"), level="extended"
        )

        assert change["nodes_removed"] == (
            change["nodes_before"] - change["nodes_after"]
        )
        assert change["nodes_after"] != change["nodes_before"]

    def test_disabling_optimization_fuses_nothing(
        self, tiny_lm, tiny_lm_input, tmp_path
    ):
        # Not "changes nothing": onnxruntime 1.19 at ORT_DISABLE_ALL still adds a
        # Constant node while converting the ONNX graph to its own format, so an
        # equality assertion here fails on that version. What the level actually
        # promises is that no fusion happens, which is what gets checked.
        pytest.importorskip("onnxruntime", reason="onnxruntime is optional")

        source = onnx_export.export(tiny_lm, tiny_lm_input, str(tmp_path / "lm4.onnx"))
        _, change = onnx_export.optimize_with_onnxruntime(
            source, str(tmp_path / "lm4_opt.onnx"), level="disable"
        )

        fused = [
            op
            for op in change["ops_added"]
            if "Fused" in op or "Gemm" in op or "SkipLayerNorm" in op
        ]
        assert fused == [], f"fusion happened with optimization disabled: {fused}"
        assert change["ops_removed"] == []
