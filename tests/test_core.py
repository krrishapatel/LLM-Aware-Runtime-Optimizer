"""The LLMOptimizer pipeline, driven with a local model so nothing downloads."""

import json

import pytest
import torch

from llm_optimizer import LLMOptimizer


class TestConstruction:
    def test_a_model_or_a_name_is_required(self):
        with pytest.raises(ValueError, match="model_name or model"):
            LLMOptimizer()

    def test_dynamic_quantization_on_cuda_is_refused_up_front(self, mlp):
        # torch's int8 dynamic kernels are CPU only. Accepting the config and
        # failing later meant the pipeline reported success on the size
        # reduction and then died in the forward pass.
        with pytest.raises(ValueError, match="CPU only"):
            LLMOptimizer(model=mlp, target_device="cuda", quantization="dynamic")

    def test_fp16_on_cuda_is_allowed(self, mlp):
        optimizer = LLMOptimizer(model=mlp, target_device="cuda", quantization="fp16")
        assert optimizer.target_device == "cuda"

    def test_importing_the_package_does_not_configure_logging(self, mlp):
        # A library that calls basicConfig on import takes over the logging setup
        # of whatever imported it. The old core.py called setup_logging() from
        # __init__ every time.
        import logging

        before = list(logging.getLogger().handlers)
        LLMOptimizer(model=mlp)

        assert list(logging.getLogger().handlers) == before


class TestPipeline:
    def test_optimize_returns_a_different_model(self, mlp):
        # The old optimize() body was `self.optimized_model = self.model`, so
        # this was the same object and the size before and after were equal by
        # construction.
        optimizer = LLMOptimizer(model=mlp)
        result = optimizer.optimize()

        assert result is not mlp
        assert optimizer.optimized_model is result

    def test_optimize_records_a_measured_reduction(self, mlp):
        optimizer = LLMOptimizer(model=mlp)
        optimizer.optimize()

        report = optimizer.quantization_report
        assert report["size_bytes_after"] < report["size_bytes_before"]
        assert 0 < report["size_reduction"] < 1

    def test_analyze_works_before_optimizing(self, tiny_lm):
        result = LLMOptimizer(model=tiny_lm).analyze()

        assert result["counts"]["linear"] == 7
        assert result["suggestions"]

    def test_benchmark_needs_optimize_first(self, mlp):
        optimizer = LLMOptimizer(model=mlp)

        with pytest.raises(ValueError, match="optimize"):
            optimizer.benchmark(torch.randn(1, 32))

    def test_benchmark_compares_both_models(self, mlp):
        optimizer = LLMOptimizer(model=mlp)
        optimizer.optimize()

        result = optimizer.benchmark(torch.randn(1, 32), num_runs=5, warmup_runs=1)

        assert "baseline" in result and "candidate" in result
        assert "significant" in result

    def test_analyze_without_a_model_is_an_error(self):
        optimizer = LLMOptimizer(model_name="does-not-matter")

        with pytest.raises(ValueError, match="load_model"):
            optimizer.analyze()

    def test_load_model_without_transformers_says_so(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fail_on_transformers(name, *args, **kwargs):
            if name == "transformers":
                raise ImportError("no transformers")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_on_transformers)
        optimizer = LLMOptimizer(model_name="prajjwal1/bert-tiny")

        with pytest.raises(ImportError, match="pip install transformers"):
            optimizer.load_model()


class TestReportAndSave:
    def test_the_report_shows_which_steps_have_run(self, mlp):
        optimizer = LLMOptimizer(model=mlp)

        assert optimizer.report()["optimized"] is False
        assert optimizer.report()["quantization_report"] == {}

        optimizer.optimize()
        assert optimizer.report()["optimized"] is True
        assert optimizer.report()["quantization_report"] != {}

    def test_saving_before_optimizing_is_an_error(self, mlp, tmp_path):
        with pytest.raises(ValueError, match="Nothing to save"):
            LLMOptimizer(model=mlp).save(str(tmp_path / "out"))

    def test_save_writes_the_weights_and_not_only_the_metrics(self, mlp, tmp_path):
        # The old save_pretrained path was guarded by hasattr, and a quantized
        # module does not have that method, so the check failed silently and the
        # directory came out holding a metrics file and no model.
        optimizer = LLMOptimizer(model=mlp)
        optimizer.optimize()

        output = optimizer.save(str(tmp_path / "out"))

        assert (output / "model_state.pt").exists()
        assert (output / "model_state.pt").stat().st_size > 1000
        assert (output / "optimization_report.json").exists()

    def test_the_saved_report_is_valid_json_with_the_measurements(self, mlp, tmp_path):
        optimizer = LLMOptimizer(model=mlp)
        optimizer.optimize()
        output = optimizer.save(str(tmp_path / "out"))

        report = json.loads((output / "optimization_report.json").read_text())

        assert report["quantization_report"]["size_reduction"] > 0
