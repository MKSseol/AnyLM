"""Dynamic INT8 quantization using PyTorch's built-in quantization.

This is the simplest quantization approach — no calibration data required.
Weights are quantized to INT8 at load time, and activations are quantized
dynamically during inference.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import torch

from src.quantization.base import BaseQuantizer

logger = logging.getLogger(__name__)


class DynamicInt8Quantizer(BaseQuantizer):
    """Dynamic INT8 quantization via PyTorch.

    Applies torch.quantization.quantize_dynamic to quantize Linear layers
    to INT8. No calibration data needed.
    """

    def quantize(self, model: Any, config: dict) -> Any:
        """Apply dynamic INT8 quantization to the model.

        Args:
            model: A PyTorch nn.Module (e.g., HuggingFace model).
            config: Quantization configuration (unused for dynamic INT8).

        Returns:
            The dynamically quantized model.
        """
        logger.info("Applying Dynamic INT8 quantization...")

        original_size = self._estimate_model_size(model)
        logger.info("Original model size (params): %.1f MB", original_size)

        quantized_model = torch.ao.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )

        quantized_size = self._estimate_model_size(quantized_model)
        logger.info("Quantized model size (params): %.1f MB", quantized_size)
        logger.info(
            "Size reduction: %.1f%%",
            (1 - quantized_size / original_size) * 100 if original_size > 0 else 0,
        )

        return quantized_model

    def get_model_size_mb(self, quantized_model: Any) -> float:
        """Return the estimated size of the quantized model in MB.

        Args:
            quantized_model: The quantized PyTorch model.

        Returns:
            Estimated size in megabytes.
        """
        return self._estimate_model_size(quantized_model)

    def supported_bits(self) -> list[int]:
        """Return supported bit widths.

        Returns:
            [8] since this method only supports INT8.
        """
        return [8]

    def save(self, model: Any, output_dir: str) -> Path:
        """Save the quantized model to disk.

        Args:
            model: The quantized model.
            output_dir: Directory to save the model.

        Returns:
            Path to the saved model directory.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_file = output_path / "model_int8.pt"
        torch.save(model.state_dict(), model_file)
        logger.info("Quantized model saved to %s", model_file)

        return output_path

    @staticmethod
    def _estimate_model_size(model: Any) -> float:
        """Estimate model size in MB from parameters.

        Args:
            model: A PyTorch model.

        Returns:
            Estimated size in MB.
        """
        total_bytes = 0
        for param in model.parameters():
            total_bytes += param.nelement() * param.element_size()
        for buffer in model.buffers():
            total_bytes += buffer.nelement() * buffer.element_size()
        return total_bytes / (1024 * 1024)
