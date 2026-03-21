"""Integration tests for the full benchmark pipeline.

Validates the end-to-end flow using a small dummy model,
without requiring HuggingFace Hub access.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import yaml

from src.benchmark.metrics import (
    aggregate_metric_runs,
    compute_bleu,
    measure_latency,
    measure_peak_memory,
)
from src.benchmark.runner import BenchmarkRunner
from src.quantization.methods.dynamic_int8 import DynamicInt8Quantizer

logger = logging.getLogger(__name__)


class _DummySeq2SeqModel(nn.Module):
    """Minimal seq2seq model for pipeline testing."""

    def __init__(self, vocab_size: int = 1000, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, hidden)
        self.decoder = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        embedded = self.encoder(input_ids)
        return self.decoder(embedded.mean(dim=1))

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        return torch.randint(0, 1000, (batch_size, 10))


class _DummyTokenizer:
    """Minimal tokenizer for pipeline testing."""

    def __init__(self) -> None:
        self.src_lang: str = ""
        self.eos_token = "</s>"

    def __call__(
        self, text: str, return_tensors: str = "pt", **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        # Simple character-level "tokenization"
        ids = [ord(c) % 1000 for c in text[:50]]
        return {"input_ids": torch.tensor([ids])}

    def batch_decode(
        self, token_ids: torch.Tensor, skip_special_tokens: bool = True
    ) -> list[str]:
        return ["dummy translation output"] * token_ids.shape[0]

    def convert_tokens_to_ids(self, token: str) -> int:
        return 42


class TestEndToEndPipeline:
    """End-to-end pipeline tests with mock model."""

    @pytest.fixture
    def dummy_model_and_tokenizer(self) -> tuple[nn.Module, _DummyTokenizer]:
        model = _DummySeq2SeqModel()
        model.eval()
        tokenizer = _DummyTokenizer()
        return model, tokenizer

    @pytest.fixture
    def smoke_config(self, tmp_path: Path) -> Path:
        """Create a minimal config for smoke testing."""
        config = {
            "experiment": {
                "name": "integration-smoke-test",
                "description": "E2E pipeline validation with dummy model",
                "date": "2026-03-21",
                "author": "test",
            },
            "model": {
                "name": "dummy-model",
                "revision": "main",
                "framework": "transformers",
            },
            "quantization": {
                "method": "none",
                "calibration_samples": 0,
                "group_size": 0,
            },
            "benchmark": {
                "language_pairs": ["kor_Hang-eng_Latn"],
                "test_dataset": "flores200",
                "num_samples": 5,
                "num_repeats": 2,
                "metrics": ["bleu", "latency_p50", "peak_memory_mb"],
                "batch_sizes": [1],
            },
            "hardware": {
                "device": "cpu",
                "max_memory_mb": 4096,
                "num_threads": 2,
            },
            "seed": 42,
        }
        config_file = tmp_path / "smoke_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        return config_file

    def test_metrics_pipeline(self) -> None:
        """Test that metrics computation works end-to-end."""
        hypotheses = ["Hello world.", "This is a test."]
        references = ["Hello world.", "This is a test."]

        bleu = compute_bleu(hypotheses, references)
        assert bleu > 0

        latency = measure_latency(
            lambda x: x.upper(), ["hello", "world"], num_repeats=1, warmup_count=1
        )
        assert "latency_p50_ms" in latency
        assert "tokens_per_sec" in latency

        memory = measure_peak_memory(lambda: [i**2 for i in range(1000)])
        assert "peak_memory_mb" in memory

        agg = aggregate_metric_runs([1.0, 2.0, 3.0])
        assert agg["mean"] == 2.0
        assert len(agg["runs"]) == 3

    def test_quantization_pipeline(self, dummy_model_and_tokenizer: tuple) -> None:
        """Test quantization → benchmark flow."""
        model, tokenizer = dummy_model_and_tokenizer

        quantizer = DynamicInt8Quantizer()
        original_size = quantizer._estimate_model_size(model)

        quantized = quantizer.quantize(model, {})
        quantized_size = quantizer.get_model_size_mb(quantized)

        # Quantized should be smaller
        assert quantized_size < original_size

        # Should still produce output
        x = torch.randint(0, 100, (1, 10))
        output = quantized(x)
        assert output.shape[0] == 1

    def test_quantization_save_and_report(
        self, dummy_model_and_tokenizer: tuple, tmp_path: Path
    ) -> None:
        """Test quantized model save produces files."""
        model, _ = dummy_model_and_tokenizer
        quantizer = DynamicInt8Quantizer()
        quantized = quantizer.quantize(model, {})

        save_dir = tmp_path / "quantized"
        quantizer.save(quantized, str(save_dir))

        assert (save_dir / "model_int8.pt").exists()
        assert (save_dir / "model_int8.pt").stat().st_size > 0

    def test_benchmark_runner_with_mock(
        self,
        smoke_config: Path,
        dummy_model_and_tokenizer: tuple,
        tmp_path: Path,
    ) -> None:
        """Test BenchmarkRunner end-to-end with a mocked model loader."""
        model, tokenizer = dummy_model_and_tokenizer

        runner = BenchmarkRunner(str(smoke_config), results_dir=str(tmp_path / "results"))

        with patch("src.benchmark.runner.ModelLoader") as MockLoader:
            mock_instance = MockLoader.return_value
            mock_instance.load.return_value = (model, tokenizer)
            results = runner.run()

        # Validate result structure
        assert "experiment" in results
        assert "timestamp" in results
        assert "config_hash" in results
        assert "results" in results
        assert results["experiment"] == "integration-smoke-test"

        # Check language pair results
        assert "kor_Hang-eng_Latn" in results["results"]
        pair_result = results["results"]["kor_Hang-eng_Latn"]
        assert "latency_p50_ms" in pair_result
        assert "latency_p99_ms" in pair_result
        assert "peak_memory_mb" in pair_result

        # Verify results file was saved
        results_dir = tmp_path / "results"
        result_files = list(results_dir.glob("*.json"))
        assert len(result_files) >= 1

        # Verify JSON is valid and complete
        with open(result_files[0]) as f:
            saved = json.load(f)
        assert saved["experiment"] == "integration-smoke-test"
        assert "hardware" in saved

    def test_test_data_loading(self) -> None:
        """Test that sample FLORES data files are loadable."""
        flores_dir = Path("data/test_sets/flores200")

        for lang in ["eng_Latn", "kor_Hang", "jpn_Jpan"]:
            test_file = flores_dir / f"{lang}.dev"
            assert test_file.exists(), f"Missing test data: {test_file}"

            with open(test_file) as f:
                lines = [line.strip() for line in f if line.strip()]
            assert len(lines) >= 10, f"Too few sentences for {lang}: {len(lines)}"


class TestDataFlowConsistency:
    """Tests for data flow between modules."""

    def test_config_to_runner_to_results(self, tmp_path: Path) -> None:
        """Verify config → runner → JSON results flow."""
        config = {
            "experiment": {"name": "flow-test", "description": "test", "date": "2026-03-21", "author": "test"},
            "model": {"name": "test-model", "framework": "transformers"},
            "quantization": {"method": "none", "calibration_samples": 0, "group_size": 0},
            "benchmark": {
                "language_pairs": ["eng_Latn-kor_Hang"],
                "test_dataset": "flores200",
                "num_samples": 3,
                "num_repeats": 1,
                "metrics": ["bleu"],
                "batch_sizes": [1],
            },
            "hardware": {"device": "cpu", "max_memory_mb": 4096, "num_threads": 2},
            "seed": 42,
        }
        config_file = tmp_path / "flow_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        runner = BenchmarkRunner(str(config_file), results_dir=str(tmp_path / "results"))

        # Verify config was loaded correctly
        assert runner.config["experiment"]["name"] == "flow-test"
        assert runner.config["benchmark"]["num_samples"] == 3
