"""Counts, sizes, and the suggestions built from them."""

import torch
import torch.nn as nn

from llm_optimizer import analysis


class TestCounts:
    def test_the_root_module_is_not_counted(self, mlp):
        # named_modules() yields the model itself under the empty name. Counting
        # it inflated every total_modules by one.
        report = analysis.analyze(mlp)
        assert report["total_modules"] == 3

    def test_layer_types_are_counted_separately(self, tiny_lm):
        report = analysis.analyze(tiny_lm)

        assert report["linear"] == 7
        assert report["embedding"] == 1
        assert report["layer_norm"] == 2
        assert report["conv"] == 0

    def test_attention_is_matched_on_the_layer_name(self, tiny_lm):
        report = analysis.analyze(tiny_lm)
        assert report["attention_named"] == 4

    def test_only_linear_and_rnn_layers_count_as_quantizable(self, tiny_lm):
        report = analysis.analyze(tiny_lm)
        # 7 Linear layers. The embedding and both LayerNorms are excluded.
        assert report["quantizable_modules"] == 7


class TestQuantizableFraction:
    def test_an_all_linear_model_is_fully_quantizable(self, mlp):
        report = analysis.analyze(mlp)
        assert report["quantizable_parameter_fraction"] == 1.0

    def test_an_embedding_heavy_model_is_mostly_not(self, embedding_heavy):
        report = analysis.analyze(embedding_heavy)

        # 2000 x 32 embedding against a 32 x 4 head. The fraction is what tells
        # you dynamic quantization will barely help here.
        assert report["quantizable_parameter_fraction"] < 0.01

    def test_a_model_with_no_parameters_does_not_divide_by_zero(self):
        report = analysis.analyze(nn.Sequential(nn.ReLU()))

        assert report["total_parameters"] == 0
        assert report["quantizable_parameter_fraction"] == 0.0


class TestSizes:
    def test_tensor_size_uses_the_real_element_size(self, mlp):
        # 32x64 + 64 + 64x32 + 32 = 4192 parameters at 4 bytes each.
        assert analysis.tensor_size_bytes(mlp) == 4192 * 4

    def test_half_precision_halves_the_tensor_size(self, mlp):
        before = analysis.tensor_size_bytes(mlp)
        after = analysis.tensor_size_bytes(mlp.half())

        assert after == before // 2

    def test_serialized_size_is_close_to_the_tensor_size(self, mlp):
        tensors = analysis.tensor_size_bytes(mlp)
        serialized = analysis.serialized_size_bytes(mlp)

        # Larger by the zip container torch.save writes, but not by much.
        assert tensors < serialized < tensors + 5000

    def test_serialized_size_sees_packed_quantized_weights(self, mlp):
        # The reason the report measures serialized bytes. A dynamically
        # quantized Linear holds its weights in _packed_params, so
        # tensor_size_bytes reports 0 and a before/after comparison would claim
        # a 100% size reduction.
        import torch.ao.quantization as tq

        torch.backends.quantized.engine = (
            torch.backends.quantized.supported_engines[-1]
        )
        quantized = tq.quantize_dynamic(mlp.eval(), {nn.Linear}, dtype=torch.qint8)

        assert analysis.tensor_size_bytes(quantized) == 0
        assert analysis.serialized_size_bytes(quantized) > 1000


class TestSuggestions:
    def test_a_model_with_no_linear_layers_is_told_so(self):
        notes = analysis.suggest(nn.Sequential(nn.ReLU(), nn.LayerNorm(8)))
        assert any("nothing to convert" in note for note in notes)

    def test_an_embedding_heavy_model_gets_a_warning(self, embedding_heavy):
        notes = analysis.suggest(embedding_heavy)

        assert any("Only" in note and "quantizable layers" in note for note in notes)
        assert any("embedding" in note for note in notes)

    def test_a_linear_model_is_told_to_try_int8(self, mlp):
        notes = analysis.suggest(mlp)
        assert any("worth trying" in note for note in notes)

    def test_the_report_holds_no_predicted_latency(self, tiny_lm):
        # The file this replaced returned a latency_reduction key built by adding
        # 0.15 per transformer layer and 0.20 per attention layer, with no
        # measurement behind it. Static analysis counts layers; the only latency
        # figure in the package comes out of benchmark.measure_latency.
        report = analysis.analyze(tiny_lm)

        for key in report:
            assert "latency" not in key
            assert "speedup" not in key
            assert "throughput" not in key

    def test_every_suggestion_mentions_measuring(self, mlp):
        notes = analysis.suggest(mlp)
        assert any("Measure" in note for note in notes)
