"""Tests for the export module."""

import pytest

from src.models.loader import ModelLoader


class TestModelLoader:
    """Tests for the ModelLoader class."""

    def test_unsupported_framework_raises(self, sample_config) -> None:
        sample_config["model"]["framework"] = "unsupported"
        with pytest.raises(ValueError, match="Unsupported framework"):
            ModelLoader(sample_config)

    def test_supported_frameworks(self, sample_config) -> None:
        for framework in ["transformers", "ctranslate2", "onnx"]:
            sample_config["model"]["framework"] = framework
            loader = ModelLoader(sample_config)
            assert loader._framework == framework

    def test_estimate_memory_known_model(self) -> None:
        mem = ModelLoader.estimate_memory(
            "facebook/nllb-200-distilled-600M", "none"
        )
        assert mem > 0

    def test_estimate_memory_unknown_model(self) -> None:
        mem = ModelLoader.estimate_memory("unknown/model", "none")
        assert mem > 0  # Should return a default estimate

    def test_onnx_not_implemented(self, sample_config) -> None:
        sample_config["model"]["framework"] = "onnx"
        loader = ModelLoader(sample_config)
        with pytest.raises(NotImplementedError):
            loader.load()

    def test_cache_dir_default(self, sample_config) -> None:
        loader = ModelLoader(sample_config)
        assert "maga" in str(loader.cache_dir)
