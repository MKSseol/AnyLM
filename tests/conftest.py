"""Shared test fixtures for MAGA project."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_config() -> dict:
    """Return a test configuration dictionary."""
    return {
        "experiment": {
            "name": "test-experiment",
            "description": "Unit test experiment",
            "date": "2025-01-01",
            "author": "test",
        },
        "model": {
            "name": "facebook/nllb-200-distilled-600M",
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
            "num_samples": 10,
            "num_repeats": 1,
            "metrics": ["bleu", "latency_p50"],
            "batch_sizes": [1],
        },
        "hardware": {
            "device": "cpu",
            "max_memory_mb": 8192,
            "num_threads": 2,
        },
        "seed": 42,
    }


@pytest.fixture
def sample_sentences() -> list[dict[str, str]]:
    """Return 10 ko-en sentence pairs for testing."""
    return [
        {"ko": "안녕하세요.", "en": "Hello."},
        {"ko": "감사합니다.", "en": "Thank you."},
        {"ko": "오늘 날씨가 좋습니다.", "en": "The weather is nice today."},
        {"ko": "이것은 테스트입니다.", "en": "This is a test."},
        {"ko": "기계 번역은 빠르게 발전하고 있습니다.", "en": "Machine translation is advancing rapidly."},
        {"ko": "모델을 경량화합니다.", "en": "We optimize the model."},
        {"ko": "양자화는 모델 크기를 줄입니다.", "en": "Quantization reduces model size."},
        {"ko": "성능을 측정합니다.", "en": "We measure performance."},
        {"ko": "결과를 분석합니다.", "en": "We analyze the results."},
        {"ko": "연구를 계속합니다.", "en": "We continue the research."},
    ]


@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for test results."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir


@pytest.fixture
def tmp_config_file(tmp_path: Path, sample_config: dict) -> Path:
    """Write sample config to a temporary YAML file and return its path."""
    import yaml

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config, f)
    return config_file
