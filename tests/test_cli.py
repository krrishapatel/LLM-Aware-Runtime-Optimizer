"""The CLI surface. No subcommand is a placeholder."""

import pytest

from llm_optimizer import cli


class TestParser:
    def test_no_command_prints_help_and_exits_clean(self, capsys):
        assert cli.main([]) == cli.EXIT_OK
        assert "usage:" in capsys.readouterr().out

    def test_the_subcommands_are_the_ones_that_work(self):
        parser = cli.create_argument_parser()
        actions = [
            action
            for action in parser._subparsers._group_actions
            if hasattr(action, "choices")
        ]
        commands = set(actions[0].choices)

        # `deploy` and `benchmark` are gone. Both existed as subcommands whose
        # entire body printed "not implemented in this version".
        assert commands == {"info", "analyze", "optimize"}

    def test_static_is_not_offered_on_the_command_line(self):
        # It needs calibration data, which a CLI flag cannot supply.
        parser = cli.create_argument_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["optimize", "gpt2", "--quantization", "static"])

    def test_logging_defaults_to_quiet(self):
        # The old default was INFO, so every run buried its own output under
        # progress logs from the library.
        args = cli.create_argument_parser().parse_args(["info"])
        assert args.log_level == "WARNING"


class TestInfo:
    def test_it_reports_the_environment(self, capsys):
        exit_code = cli.show_info()
        output = capsys.readouterr().out

        assert exit_code == cli.EXIT_OK
        assert "ready: yes" in output
        assert "torch_version" in output

    def test_a_broken_environment_exits_nonzero(self, capsys, monkeypatch):
        # The check that used to be `if validate_environment():` against a
        # function returning a non-empty dict, so it printed "Environment is
        # ready" unconditionally and always exited 0.
        monkeypatch.setattr(
            cli,
            "validate_environment",
            lambda: {
                "ready": False,
                "problems": ["torch is not installed."],
                "warnings": [],
                "system": {},
            },
        )

        exit_code = cli.show_info()
        output = capsys.readouterr().out

        assert exit_code == cli.EXIT_FAILED
        assert "ready: no" in output
        assert "problem: torch is not installed." in output

    def test_warnings_are_printed(self, capsys, monkeypatch):
        monkeypatch.setattr(
            cli,
            "validate_environment",
            lambda: {
                "ready": True,
                "problems": [],
                "warnings": ["psutil not installed, so memory was not checked."],
                "system": {},
            },
        )
        cli.show_info()

        assert "warning: psutil" in capsys.readouterr().out


class TestErrorHandling:
    def test_a_missing_dependency_becomes_a_clean_error(self, capsys, monkeypatch):
        def raise_import_error(_args):
            raise ImportError("load_model needs transformers")

        monkeypatch.setattr(cli, "analyze_model", raise_import_error)
        exit_code = cli.main(["analyze", "gpt2"])

        assert exit_code == cli.EXIT_FAILED
        assert "error: load_model needs transformers" in capsys.readouterr().err

    def test_an_unexpected_error_is_not_swallowed(self, monkeypatch):
        # The old CLI caught bare Exception and printed one line, so a real bug
        # in the pipeline lost its traceback.
        def raise_bug(_args):
            raise RuntimeError("a genuine bug")

        monkeypatch.setattr(cli, "analyze_model", raise_bug)

        with pytest.raises(RuntimeError, match="a genuine bug"):
            cli.main(["analyze", "gpt2"])
