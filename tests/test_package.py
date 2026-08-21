"""The package surface: what imports, and what no longer exists."""

import importlib
from pathlib import Path

import pytest

import llm_optimizer


class TestImports:
    def test_everything_in_all_is_importable(self):
        # The old __init__ wrapped each import in try/except ImportError and set
        # a MODULE_AVAILABLE flag on failure. That hid two real faults: it asked
        # deployment for DeploymentManager, which was never defined, and
        # quantization raised AttributeError on torch.qint16. Both looked like a
        # flag quietly set to False.
        for name in llm_optimizer.__all__:
            assert hasattr(llm_optimizer, name), name

    def test_there_are_no_availability_flags(self):
        for name in dir(llm_optimizer):
            assert not name.endswith("_AVAILABLE")

    def test_the_optional_onnx_module_is_not_imported_eagerly(self):
        # onnx is an optional dependency, so importing the package must not
        # require it.
        assert "onnx_export" not in llm_optimizer.__all__

    def test_onnx_export_imports_without_onnx_installed(self):
        # The module itself must import; only its functions need onnx.
        module = importlib.import_module("llm_optimizer.onnx_export")
        assert hasattr(module, "export")


class TestRemovedModules:
    @pytest.mark.parametrize(
        "name", ["mlir", "tensorrt_integration", "onnx_rewriter", "deployment"]
    )
    def test_the_simulated_modules_are_gone(self, name):
        # mlir.py never invoked MLIR, tensorrt_integration.py never imported
        # tensorrt, and onnx_rewriter.py had six methods named after ONNX passes
        # whose whole body was `return onnx_model`.
        with pytest.raises(ImportError):
            importlib.import_module(f"llm_optimizer.{name}")

    def test_no_source_file_claims_to_simulate_a_result(self):
        # The marker for the code that was removed: 24 methods carried a comment
        # reading "For now, we'll simulate it" directly above a log line
        # reporting success.
        package_dir = Path(llm_optimizer.__file__).parent
        offenders = []
        for path in package_dir.glob("*.py"):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#") and "we'll simulate" in stripped:
                    offenders.append(f"{path.name}:{number}")

        assert offenders == []

    def test_no_committed_bytecode(self):
        # The repo tracked eleven .pyc files, three of them for modules whose
        # source had already been deleted.
        repo_root = Path(llm_optimizer.__file__).parent.parent
        tracked_pyc = [
            path
            for path in repo_root.rglob("*.pyc")
            if ".git" not in path.parts and ".venv" not in path.parts
        ]

        gitignore = (repo_root / ".gitignore").read_text()
        assert "__pycache__/" in gitignore
        assert all("__pycache__" in path.parts for path in tracked_pyc)
