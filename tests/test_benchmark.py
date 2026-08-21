"""Latency measurement, and the guard against reading noise as a speedup."""

import time

import pytest
import torch
import torch.nn as nn

from llm_optimizer import benchmark


class SlowModel(nn.Module):
    """Sleeps a fixed amount, so the measured latency is known in advance."""

    def __init__(self, delay_s: float):
        super().__init__()
        self.delay_s = delay_s
        self.calls = 0
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        self.calls += 1
        time.sleep(self.delay_s)
        return self.linear(x)


class TestSummarize:
    def test_statistics_are_computed_over_the_whole_list(self):
        result = benchmark.summarize([1.0, 2.0, 3.0, 4.0])

        assert result["runs"] == 4
        assert result["mean_ms"] == 2.5
        assert result["median_ms"] == 2.5
        assert result["min_ms"] == 1.0
        assert result["max_ms"] == 4.0

    def test_a_single_run_does_not_crash_on_the_percentiles(self):
        # int(0.95 * 1) is 0 but int(0.99 * 1) rounds to 0 too, and with a
        # different length the same arithmetic indexes off the end of the list.
        result = benchmark.summarize([7.0])

        assert result["p95_ms"] == 7.0
        assert result["p99_ms"] == 7.0
        assert result["stdev_ms"] == 0.0

    def test_percentiles_never_index_past_the_end(self):
        for count in range(1, 60):
            result = benchmark.summarize([float(i) for i in range(count)])
            assert result["p95_ms"] <= result["max_ms"]
            assert result["p99_ms"] <= result["max_ms"]

    def test_an_empty_list_is_an_error(self):
        with pytest.raises(ValueError, match="No latencies"):
            benchmark.summarize([])

    def test_stdev_is_reported(self):
        # A mean with no spread beside it hides the case where two models are
        # too noisy to tell apart.
        result = benchmark.summarize([1.0, 5.0])
        assert result["stdev_ms"] > 0


class TestMeasureLatency:
    def test_the_measured_time_matches_a_known_delay(self):
        model = SlowModel(delay_s=0.01)

        result = benchmark.measure_latency(
            model, torch.randn(1, 4), num_runs=5, warmup_runs=1
        )

        assert 9 < result["median_ms"] < 40
        assert result["runs"] == 5

    def test_warmup_runs_are_not_timed(self):
        model = SlowModel(delay_s=0.0)

        result = benchmark.measure_latency(
            model, torch.randn(1, 4), num_runs=4, warmup_runs=3
        )

        assert model.calls == 7
        assert result["runs"] == 4

    def test_the_model_is_left_in_the_mode_it_arrived_in(self, mlp):
        # A benchmark that quietly flips a model to eval breaks the caller's
        # next training step.
        mlp.train()
        benchmark.measure_latency(mlp, torch.randn(1, 32), num_runs=2, warmup_runs=0)

        assert mlp.training is True

    def test_an_eval_model_stays_in_eval(self, mlp):
        mlp.eval()
        benchmark.measure_latency(mlp, torch.randn(1, 32), num_runs=2, warmup_runs=0)

        assert mlp.training is False

    def test_dict_inputs_are_passed_as_keywords(self):
        class KeywordModel(nn.Module):
            def forward(self, first, second):
                return first + second

        result = benchmark.measure_latency(
            KeywordModel(),
            {"first": torch.ones(2), "second": torch.ones(2)},
            num_runs=2,
            warmup_runs=0,
        )

        assert result["runs"] == 2

    def test_tuple_inputs_are_unpacked(self):
        class TwoArgModel(nn.Module):
            def forward(self, a, b):
                return a * b

        result = benchmark.measure_latency(
            TwoArgModel(), (torch.ones(2), torch.ones(2)), num_runs=2, warmup_runs=0
        )

        assert result["runs"] == 2

    def test_zero_runs_is_an_error(self, mlp):
        with pytest.raises(ValueError, match="at least 1"):
            benchmark.measure_latency(mlp, torch.randn(1, 32), num_runs=0)

    def test_negative_warmup_is_an_error(self, mlp):
        with pytest.raises(ValueError, match="negative"):
            benchmark.measure_latency(mlp, torch.randn(1, 32), warmup_runs=-1)

    def test_no_gradients_are_built(self, mlp):
        benchmark.measure_latency(mlp, torch.randn(1, 32), num_runs=2, warmup_runs=0)

        assert all(p.grad is None for p in mlp.parameters())


class TestCompare:
    def test_a_real_difference_is_reported_as_significant(self):
        slow = SlowModel(delay_s=0.02)
        fast = SlowModel(delay_s=0.0)

        result = benchmark.compare(
            slow, fast, torch.randn(1, 4), num_runs=5, warmup_runs=1
        )

        assert result["speedup"] > 2
        assert result["latency_change"] < 0
        assert result["significant"] is True

    def test_comparing_a_model_with_itself_is_not_significant(self, mlp):
        # The check that stops the package reporting a speedup it does not have.
        # Two timings of the same model differ only by noise, so the gap has to
        # fall inside the combined standard deviation.
        result = benchmark.compare(
            mlp, mlp, torch.randn(1, 32), num_runs=30, warmup_runs=5
        )

        assert result["significant"] is False

    def test_both_models_are_timed_separately(self, mlp):
        result = benchmark.compare(
            mlp, mlp, torch.randn(1, 32), num_runs=3, warmup_runs=0
        )

        assert result["baseline"]["runs"] == 3
        assert result["candidate"]["runs"] == 3

    def test_speedup_is_the_ratio_of_the_medians(self):
        slow = SlowModel(delay_s=0.02)
        fast = SlowModel(delay_s=0.005)

        result = benchmark.compare(
            slow, fast, torch.randn(1, 4), num_runs=5, warmup_runs=1
        )
        expected = result["baseline"]["median_ms"] / result["candidate"]["median_ms"]

        assert result["speedup"] == pytest.approx(expected)
