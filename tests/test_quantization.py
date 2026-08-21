"""Quantization: what it converts, what it measures, and what it refuses."""

import pytest
import torch
import torch.nn as nn

from llm_optimizer.quantization import QuantizationPipeline, default_qengine


class TestConstruction:
    def test_the_pipeline_can_be_built(self):
        # The version this replaces raised AttributeError here, because
        # _get_quantization_config referenced torch.qint16 and built the whole
        # config dict eagerly. torch has qint8, quint8 and qint32, no qint16, so
        # no instance of the class could ever be created.
        pipeline = QuantizationPipeline("dynamic")
        assert pipeline.quantization_type == "dynamic"

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported quantization type"):
            QuantizationPipeline("int4")

    def test_static_without_calibration_data_fails_at_construction(self):
        # Not partway through quantize(), which is after the model is loaded.
        with pytest.raises(ValueError, match="calibration_data"):
            QuantizationPipeline("static")

    def test_the_engine_comes_from_what_torch_supports(self):
        # The old code hardcoded 'fbgemm', which is x86 only, so every static
        # path failed on Apple Silicon where the engine is qnnpack.
        assert default_qengine() in torch.backends.quantized.supported_engines


class TestDynamic:
    def test_linear_layers_are_converted(self, mlp):
        _, report = QuantizationPipeline("dynamic").quantize(mlp)
        assert report["converted_modules"] == 2

    def test_packed_params_are_not_counted_as_extra_layers(self, tiny_lm):
        # Each quantized Linear owns a _packed_params child that is itself a
        # quantized type. Counting those doubled the total.
        _, report = QuantizationPipeline("dynamic").quantize(tiny_lm)
        assert report["converted_modules"] == 7

    def test_the_measured_size_reduction_is_believable(self, mlp):
        _, report = QuantizationPipeline("dynamic").quantize(mlp)

        # int8 weights against fp32, so somewhere near 75% minus overhead.
        assert 0.5 < report["size_reduction"] < 0.8
        assert report["size_bytes_after"] < report["size_bytes_before"]

    def test_the_size_after_is_not_zero(self, mlp):
        # It was. parameters() misses _packed_params, so summing tensors gave 0
        # bytes and the report claimed a 100% reduction. The report measures the
        # serialized state_dict instead.
        _, report = QuantizationPipeline("dynamic").quantize(mlp)

        assert report["size_bytes_after"] > 0
        assert report["size_reduction"] < 1.0

    def test_the_original_model_is_left_alone(self, mlp):
        before = [type(m).__name__ for m in mlp.modules()]
        QuantizationPipeline("dynamic").quantize(mlp)
        after = [type(m).__name__ for m in mlp.modules()]

        assert before == after

    def test_the_quantized_model_still_produces_close_output(self, mlp):
        torch.manual_seed(0)
        x = torch.randn(4, 32)
        expected = mlp.eval()(x)

        quantized, _ = QuantizationPipeline("dynamic").quantize(mlp)

        assert torch.allclose(expected, quantized(x), atol=0.05)

    def test_an_embedding_heavy_model_barely_shrinks(self, embedding_heavy):
        # The honest result. The embedding holds nearly all the weight and
        # dynamic quantization does not touch it, so claiming a fixed 75%
        # reduction for any model would be wrong here.
        _, report = QuantizationPipeline("dynamic").quantize(embedding_heavy)

        assert report["size_reduction"] < 0.05


class TestFp16:
    def test_the_size_roughly_halves(self, mlp):
        _, report = QuantizationPipeline("fp16").quantize(mlp)

        # Not exactly 50%. The measurement is on the serialized state_dict, and
        # torch.save's zip container costs a fixed ~1 KB that does not halve. On
        # this 16 KB model that drags the figure down to about 44%.
        assert 0.40 < report["size_reduction"] < 0.52

    def test_the_original_stays_fp32(self, mlp):
        # .half() mutates in place and returns self, so the implementation has
        # to copy first or it destroys the caller's model.
        QuantizationPipeline("fp16").quantize(mlp)
        assert next(mlp.parameters()).dtype == torch.float32

    def test_the_result_is_fp16(self, mlp):
        quantized, _ = QuantizationPipeline("fp16").quantize(mlp)
        assert next(quantized.parameters()).dtype == torch.float16


class TestStatic:
    def test_a_model_without_stubs_is_refused(self, mlp):
        # Better than the old path, which called torch.quantization.convert on a
        # model that was never prepared and returned it unchanged.
        pipeline = QuantizationPipeline("static", calibration_data=[torch.randn(2, 32)])

        with pytest.raises(ValueError, match="QuantStub"):
            pipeline.quantize(mlp)

    def test_empty_calibration_data_is_refused(self, stubbed_mlp):
        # Converting with no observed activations quantizes every activation
        # range to zero, so the model outputs zeros and reports success.
        pipeline = QuantizationPipeline("static", calibration_data=[])

        with pytest.raises(ValueError, match="no batches"):
            pipeline.quantize(stubbed_mlp)

    def test_a_stubbed_model_quantizes_and_shrinks(self, stubbed_mlp):
        batches = [torch.randn(4, 64) for _ in range(3)]
        pipeline = QuantizationPipeline("static", calibration_data=batches)

        quantized, report = pipeline.quantize(stubbed_mlp)

        assert report["size_reduction"] > 0.5
        assert quantized(torch.randn(2, 64)).shape == (2, 64)

    def test_the_calibrated_model_does_not_output_all_zeros(self, stubbed_mlp):
        batches = [torch.randn(4, 64) for _ in range(3)]
        quantized, _ = QuantizationPipeline(
            "static", calibration_data=batches
        ).quantize(stubbed_mlp)

        output = quantized(torch.randn(8, 64))

        assert output.abs().sum() > 0

    def test_int8_makes_a_very_small_model_larger(self, tiny_stubbed_mlp):
        # Measured, not assumed. A 544-parameter model gains per-tensor scales,
        # zero points and quantization metadata that outweigh the saving on the
        # weights, so the checkpoint grows by about 5%. This is why the report
        # returns what it measured instead of a target: a fixed "75% size
        # reduction" claim is wrong for models at this end of the range.
        batches = [torch.randn(4, 16) for _ in range(3)]

        _, report = QuantizationPipeline(
            "static", calibration_data=batches
        ).quantize(tiny_stubbed_mlp)

        assert report["size_reduction"] < 0


class TestPlan:
    def test_every_module_appears_once(self, tiny_lm):
        plan = QuantizationPipeline("dynamic").plan(tiny_lm)

        names = [layer["name"] for layer in plan["layers"]]
        assert len(names) == len(set(names))
        assert "" not in names

    def test_linear_layers_are_marked_quantizable(self, tiny_lm):
        plan = QuantizationPipeline("dynamic").plan(tiny_lm)

        by_name = {layer["name"]: layer for layer in plan["layers"]}
        assert by_name["head"]["quantizable"] is True
        assert by_name["embedding"]["quantizable"] is False

    def test_parameters_are_counted_without_recursing(self, tiny_lm):
        # recurse=True would count a child's parameters again under every
        # ancestor, so the column would not add up to the model total.
        plan = QuantizationPipeline("dynamic").plan(tiny_lm)

        total = sum(layer["parameters"] for layer in plan["layers"])
        assert total == sum(p.numel() for p in tiny_lm.parameters())
