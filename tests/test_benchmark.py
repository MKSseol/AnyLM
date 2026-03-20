"""Tests for the benchmark module."""

import json
import time

import pytest

from src.benchmark.metrics import (
    aggregate_metric_runs,
    compute_bleu,
    compute_model_size_mb,
    measure_latency,
    measure_peak_memory,
)


class TestComputeBleu:
    """Tests for BLEU score computation."""

    def test_perfect_translation(self) -> None:
        refs = ["The cat sat on the mat.", "Hello world."]
        hyps = ["The cat sat on the mat.", "Hello world."]
        score = compute_bleu(hyps, refs)
        assert score > 90.0  # Should be very high for perfect match

    def test_completely_wrong(self) -> None:
        refs = ["The cat sat on the mat."]
        hyps = ["Xyz abc def ghi jkl."]
        score = compute_bleu(hyps, refs)
        assert score < 10.0

    def test_empty_input(self) -> None:
        score = compute_bleu([], [])
        assert isinstance(score, float)


class TestMeasureLatency:
    """Tests for latency measurement."""

    def test_basic_latency(self) -> None:
        def slow_fn(x: str) -> str:
            time.sleep(0.01)
            return x.upper()

        inputs = ["hello"] * 5
        stats = measure_latency(slow_fn, inputs, num_repeats=1, warmup_count=1)

        assert "latency_p50_ms" in stats
        assert "latency_p99_ms" in stats
        assert "tokens_per_sec" in stats
        assert stats["latency_p50_ms"] > 0

    def test_latency_returns_correct_keys(self) -> None:
        stats = measure_latency(str.upper, ["test"] * 3, num_repeats=1, warmup_count=0)
        expected_keys = {"latency_p50_ms", "latency_p99_ms", "latency_mean_ms",
                         "latency_std_ms", "tokens_per_sec"}
        assert expected_keys == set(stats.keys())


class TestMeasurePeakMemory:
    """Tests for memory measurement."""

    def test_memory_measurement(self) -> None:
        def allocate_some_memory() -> None:
            _ = [0] * 100_000

        stats = measure_peak_memory(allocate_some_memory)
        assert "peak_memory_mb" in stats
        assert stats["peak_memory_mb"] >= 0


class TestComputeModelSize:
    """Tests for model size computation."""

    def test_file_size(self, tmp_path) -> None:
        test_file = tmp_path / "test_model.bin"
        test_file.write_bytes(b"0" * 1024 * 1024)  # 1 MB
        size = compute_model_size_mb(str(test_file))
        assert abs(size - 1.0) < 0.01

    def test_directory_size(self, tmp_path) -> None:
        (tmp_path / "file1.bin").write_bytes(b"0" * 512 * 1024)
        (tmp_path / "file2.bin").write_bytes(b"0" * 512 * 1024)
        size = compute_model_size_mb(str(tmp_path))
        assert abs(size - 1.0) < 0.01

    def test_nonexistent_path(self) -> None:
        with pytest.raises(FileNotFoundError):
            compute_model_size_mb("/nonexistent/path")


class TestAggregateMetricRuns:
    """Tests for metric aggregation."""

    def test_basic_aggregation(self) -> None:
        values = [10.0, 12.0, 11.0]
        result = aggregate_metric_runs(values)
        assert result["mean"] == pytest.approx(11.0)
        assert result["min"] == 10.0
        assert result["max"] == 12.0
        assert result["runs"] == values
        assert result["std"] > 0

    def test_single_value(self) -> None:
        result = aggregate_metric_runs([42.0])
        assert result["mean"] == 42.0
        assert result["std"] == 0.0
