"""The deployment package: what gets written, and what never gets called."""

import json
import tarfile
from pathlib import Path

import pytest
import torch

from llm_optimizer import packaging
from llm_optimizer.packaging import SageMakerPackageBuilder


class TestNoAwsCalls:
    def test_the_module_does_not_import_boto3(self):
        # The point of the rewrite. The old module was described as an "AWS
        # SageMaker Integration" and had deploy, _upload_model_to_s3,
        # _create_endpoint and delete_endpoint methods, none of which imported
        # boto3 or reached AWS. Each logged a success message under a comment
        # reading "For now, we'll simulate it".
        source = Path(packaging.__file__).read_text()

        assert "import boto3" not in source
        assert "boto3.client" not in source
        assert "sagemaker.Session" not in source

    def test_there_is_no_deploy_method(self):
        # A method named deploy that does not deploy is worse than no method.
        assert not hasattr(SageMakerPackageBuilder, "deploy")
        assert not hasattr(SageMakerPackageBuilder, "delete_endpoint")
        assert not hasattr(SageMakerPackageBuilder, "get_endpoint_status")

    def test_next_steps_returns_commands_without_running_them(self, tmp_path, mlp):
        builder = SageMakerPackageBuilder()
        builder.build(mlp, torch.randn(1, 32), str(tmp_path / "pkg"))

        steps = builder.next_steps("my-bucket", "arn:aws:iam::1:role/r", "my-endpoint")

        assert any(step.startswith("aws s3 cp") for step in steps)
        assert any("create-endpoint " in step for step in steps)
        # The caller is told what it will cost them to leave it running.
        assert any("delete-endpoint" in step for step in steps)

    def test_next_steps_needs_an_archive_first(self, tmp_path, mlp):
        builder = SageMakerPackageBuilder()
        builder.build(mlp, torch.randn(1, 32), str(tmp_path / "pkg"), create_archive=False)

        with pytest.raises(ValueError, match="create_archive"):
            builder.next_steps("b", "r", "e")


class TestBuild:
    def test_every_expected_file_is_written(self, tmp_path, mlp):
        builder = SageMakerPackageBuilder()
        package = builder.build(mlp, torch.randn(1, 32), str(tmp_path / "pkg"))

        assert (package / "inference.py").exists()
        assert (package / "requirements.txt").exists()
        assert (package / "Dockerfile").exists()
        assert (package / "model" / "model.pt").exists()
        assert (package / "model" / "config.json").exists()

    def test_the_saved_model_is_a_real_model(self, tmp_path, mlp):
        # Not a placeholder. The old builder wrote a file called
        # model.placeholder containing the text "This is a placeholder for the
        # actual model file".
        builder = SageMakerPackageBuilder()
        package = builder.build(mlp, torch.randn(1, 32), str(tmp_path / "pkg"))

        loaded = torch.jit.load(str(package / "model" / "model.pt"))
        x = torch.randn(3, 32)

        assert torch.allclose(loaded(x), mlp.eval()(x), atol=1e-5)

    def test_no_placeholder_file_is_created(self, tmp_path, mlp):
        package = SageMakerPackageBuilder().build(
            mlp, torch.randn(1, 32), str(tmp_path / "pkg")
        )

        assert not list(package.glob("**/*.placeholder"))

    def test_the_config_records_what_was_actually_packaged(self, tmp_path, mlp):
        # The old config.json hardcoded {"framework": "tensorrt",
        # "target_device": "cuda", "optimization_level": "aggressive"} for every
        # model, whatever had been built.
        package = SageMakerPackageBuilder().build(
            mlp, torch.randn(1, 32), str(tmp_path / "pkg")
        )
        config = json.loads((package / "model" / "config.json").read_text())

        assert config["parameter_count"] == sum(p.numel() for p in mlp.parameters())
        assert config["parameter_dtypes"] == ["torch.float32"]
        assert config["torch_version"] == torch.__version__

    def test_the_archive_holds_relative_paths(self, tmp_path, mlp):
        # SageMaker extracts into /opt/ml. An absolute or nested prefix puts the
        # files where the container will not look for them.
        builder = SageMakerPackageBuilder()
        builder.build(mlp, torch.randn(1, 32), str(tmp_path / "pkg"))

        with tarfile.open(builder.archive_path) as tar:
            names = tar.getnames()

        assert "inference.py" in names
        assert "model/model.pt" in names
        assert not any(name.startswith("/") for name in names)
        assert not any(name.startswith("pkg/") for name in names)

    def test_the_model_is_left_in_the_mode_it_arrived_in(self, tmp_path, mlp):
        mlp.train()
        SageMakerPackageBuilder().build(mlp, torch.randn(1, 32), str(tmp_path / "pkg"))

        assert mlp.training is True

    def test_a_quantized_model_can_be_packaged(self, tmp_path, mlp):
        from llm_optimizer.quantization import QuantizationPipeline

        quantized, _ = QuantizationPipeline("dynamic").quantize(mlp)
        package = SageMakerPackageBuilder().build(
            quantized, torch.randn(1, 32), str(tmp_path / "pkg")
        )

        assert (package / "model" / "model.pt").exists()


class TestGeneratedInferenceScript:
    def test_the_handlers_round_trip_a_request(self, tmp_path, mlp):
        # Runs the generated script rather than only checking it was written.
        # The old generator emitted a predict_fn whose body was
        # `outputs = torch.randn(1, len(inputs[0]), 50257)`, so a deployed
        # endpoint would have served random tokens.
        import importlib.util

        package = SageMakerPackageBuilder().build(
            mlp, torch.randn(1, 32), str(tmp_path / "pkg")
        )

        spec = importlib.util.spec_from_file_location(
            "generated_inference", package / "inference.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model = module.model_fn(str(package / "model"))
        tensor = module.input_fn(json.dumps({"inputs": [[0.5] * 32]}), "application/json")
        prediction = module.predict_fn(tensor, model)
        body, content_type = module.output_fn(prediction, "application/json")

        assert content_type == "application/json"
        outputs = json.loads(body)["outputs"]
        assert len(outputs[0]) == 32

    def test_the_same_input_gives_the_same_output(self, tmp_path, mlp):
        # A handler that returns torch.randn would fail this.
        import importlib.util

        package = SageMakerPackageBuilder().build(
            mlp, torch.randn(1, 32), str(tmp_path / "pkg")
        )
        spec = importlib.util.spec_from_file_location(
            "generated_inference_repeat", package / "inference.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model = module.model_fn(str(package / "model"))
        body = json.dumps({"inputs": [[0.5] * 32]})
        first = module.predict_fn(module.input_fn(body, "application/json"), model)
        second = module.predict_fn(module.input_fn(body, "application/json"), model)

        assert torch.equal(first, second)

    def test_a_request_without_an_inputs_key_is_rejected(self, tmp_path, mlp):
        import importlib.util

        package = SageMakerPackageBuilder().build(
            mlp, torch.randn(1, 32), str(tmp_path / "pkg")
        )
        spec = importlib.util.spec_from_file_location(
            "generated_inference_bad", package / "inference.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with pytest.raises(ValueError, match="inputs"):
            module.input_fn(json.dumps({"text": "hello"}), "application/json")


class TestCostEstimate:
    def test_a_known_instance_type_is_priced(self):
        estimate = SageMakerPackageBuilder(instance_type="ml.g4dn.xlarge").cost_estimate()

        assert estimate["hourly_usd"] == 0.736
        assert estimate["monthly_usd_min"] == pytest.approx(0.736 * 730, abs=0.01)

    def test_an_unknown_instance_type_raises(self):
        # The old table returned the g4dn.xlarge price for anything it did not
        # recognize, so an unsupported instance type was quoted a confident and
        # wrong number.
        builder = SageMakerPackageBuilder(instance_type="ml.p5.48xlarge")

        with pytest.raises(KeyError, match="No price recorded"):
            builder.cost_estimate()

    def test_the_estimate_says_when_the_prices_were_copied(self):
        estimate = SageMakerPackageBuilder().cost_estimate()
        assert estimate["prices_as_of"] == packaging.PRICES_AS_OF

    def test_the_range_scales_with_the_instance_count(self):
        estimate = SageMakerPackageBuilder(
            instance_type="ml.m5.large", min_instance_count=2, max_instance_count=6
        ).cost_estimate()

        assert estimate["monthly_usd_max"] == pytest.approx(
            estimate["monthly_usd_min"] * 3, abs=0.05
        )


class TestValidation:
    def test_a_max_below_the_min_is_rejected(self):
        with pytest.raises(ValueError, match="max_instance_count"):
            SageMakerPackageBuilder(min_instance_count=4, max_instance_count=2)

    def test_zero_instances_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            SageMakerPackageBuilder(min_instance_count=0)
