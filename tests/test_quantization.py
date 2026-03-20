"""Tests for the quantization module."""

import pytest
import torch
import torch.nn as nn

from src.quantization.base import BaseQuantizer
from src.quantization.methods.dynamic_int8 import DynamicInt8Quantizer
from src.quantization.methods.gptq import GPTQQuantizer
from src.quantization.methods.awq import AWQQuantizer
from src.quantization.methods.gguf import GGUFQuantizer


class _DummyModel(nn.Module):
    """Small model for testing quantization."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(x)))


class TestBaseQuantizer:
    """Tests for the BaseQuantizer ABC."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            BaseQuantizer()  # type: ignore[abstract]

    def test_subclass_must_implement_methods(self) -> None:
        class IncompleteQuantizer(BaseQuantizer):
            pass

        with pytest.raises(TypeError):
            IncompleteQuantizer()  # type: ignore[abstract]


class TestDynamicInt8Quantizer:
    """Tests for Dynamic INT8 quantization."""

    def test_quantize_reduces_size(self) -> None:
        model = _DummyModel()
        quantizer = DynamicInt8Quantizer()

        original_size = quantizer._estimate_model_size(model)
        quantized = quantizer.quantize(model, {})
        quantized_size = quantizer.get_model_size_mb(quantized)

        # INT8 should be smaller than FP32
        assert quantized_size < original_size

    def test_quantize_preserves_output_shape(self) -> None:
        model = _DummyModel()
        quantizer = DynamicInt8Quantizer()

        x = torch.randn(2, 64)
        original_output = model(x)

        quantized = quantizer.quantize(model, {})
        quantized_output = quantized(x)

        assert original_output.shape == quantized_output.shape

    def test_supported_bits(self) -> None:
        quantizer = DynamicInt8Quantizer()
        assert quantizer.supported_bits() == [8]

    def test_name_property(self) -> None:
        quantizer = DynamicInt8Quantizer()
        assert quantizer.name == "DynamicInt8Quantizer"

    def test_save_creates_file(self, tmp_path) -> None:
        model = _DummyModel()
        quantizer = DynamicInt8Quantizer()
        quantized = quantizer.quantize(model, {})

        output_dir = tmp_path / "quantized_model"
        quantizer.save(quantized, str(output_dir))

        assert (output_dir / "model_int8.pt").exists()


class TestPhase2Quantizers:
    """Tests for Phase 2+ quantizers (not yet implemented)."""

    def test_gptq_raises_not_implemented(self) -> None:
        quantizer = GPTQQuantizer()
        with pytest.raises(NotImplementedError):
            quantizer.quantize(None, {})

    def test_awq_raises_not_implemented(self) -> None:
        quantizer = AWQQuantizer()
        with pytest.raises(NotImplementedError):
            quantizer.quantize(None, {})

    def test_gguf_raises_not_implemented(self) -> None:
        quantizer = GGUFQuantizer()
        with pytest.raises(NotImplementedError):
            quantizer.quantize(None, {})

    def test_gptq_supported_bits(self) -> None:
        assert GPTQQuantizer().supported_bits() == [4, 8]

    def test_awq_supported_bits(self) -> None:
        assert AWQQuantizer().supported_bits() == [4]

    def test_gguf_supported_bits(self) -> None:
        assert GGUFQuantizer().supported_bits() == [2, 3, 4, 5, 6, 8]
