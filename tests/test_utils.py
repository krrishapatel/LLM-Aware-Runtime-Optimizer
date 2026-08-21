"""Environment reporting, and the check that used to always pass."""

from llm_optimizer.utils import (
    OptimizationConfig,
    get_system_info,
    validate_environment,
)


class TestValidateEnvironment:
    def test_the_result_carries_an_explicit_ready_flag(self):
        # The whole point. The old function returned a dict that was always
        # non-empty, and the CLI tested it with `if validate_environment():`, so
        # "Environment is ready" printed no matter what was wrong.
        result = validate_environment()

        assert isinstance(result["ready"], bool)

    def test_ready_is_false_when_something_is_actually_broken(self, monkeypatch):
        monkeypatch.setattr(
            "llm_optimizer.utils.get_system_info",
            lambda: {"torch_version": None, "memory_available": None},
        )
        result = validate_environment()

        assert result["ready"] is False
        assert any("torch" in problem for problem in result["problems"])

    def test_low_memory_is_a_warning_and_not_a_failure(self, monkeypatch):
        monkeypatch.setattr(
            "llm_optimizer.utils.get_system_info",
            lambda: {
                "torch_version": "2.0.0",
                "memory_available": 512 * 1024**2,
                "quantized_engines": ["fbgemm"],
            },
        )
        result = validate_environment()

        assert result["ready"] is True
        assert any("memory available" in warning for warning in result["warnings"])

    def test_missing_psutil_is_reported_rather_than_ignored(self, monkeypatch):
        monkeypatch.setattr(
            "llm_optimizer.utils.get_system_info",
            lambda: {
                "torch_version": "2.0.0",
                "memory_available": None,
                "quantized_engines": ["fbgemm"],
            },
        )
        result = validate_environment()

        assert any("psutil" in warning for warning in result["warnings"])

    def test_no_quantized_engine_is_a_warning(self, monkeypatch):
        monkeypatch.setattr(
            "llm_optimizer.utils.get_system_info",
            lambda: {
                "torch_version": "2.0.0",
                "memory_available": 8 * 1024**3,
                "quantized_engines": [],
            },
        )
        result = validate_environment()

        assert any("quantized engines" in warning for warning in result["warnings"])


class TestSystemInfo:
    def test_it_works_without_psutil(self, monkeypatch):
        # psutil used to be a module-level import, so a machine without it could
        # not import any part of the package.
        import builtins

        real_import = builtins.__import__

        def fail_on_psutil(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("no psutil")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_on_psutil)
        info = get_system_info()

        assert info["memory_total"] is None
        assert info["platform"]

    def test_torch_details_are_included(self):
        info = get_system_info()

        assert info["torch_version"] is not None
        assert isinstance(info["cuda_available"], bool)
        assert isinstance(info["quantized_engines"], list)


class TestConfig:
    def test_the_defaults_match_what_the_package_supports(self):
        config = OptimizationConfig()

        assert config.quantization == "dynamic"
        assert config.target_device == "cpu"

    def test_it_round_trips_to_a_dict(self):
        config = OptimizationConfig(quantization="fp16", num_benchmark_runs=10)
        as_dict = config.to_dict()

        assert as_dict["quantization"] == "fp16"
        assert as_dict["num_benchmark_runs"] == 10
