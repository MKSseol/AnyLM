"""GPTQ 4-bit quantization.

Post-training quantization using the GPTQ algorithm.
Requires calibration data for optimal results.
Planned for Phase 2 implementation.
"""

import logging
from typing import Any

from src.quantization.base import BaseQuantizer

logger = logging.getLogger(__name__)


class GPTQQuantizer(BaseQuantizer):
    """GPTQ post-training quantization.

    Uses calibration data to find optimal quantization parameters
    that minimize reconstruction error layer by layer.

    Note: Full implementation planned for Phase 2.
    """

    def quantize(self, model: Any, config: dict) -> Any:
        """Apply GPTQ quantization to the model.

        Args:
            model: The model to quantize.
            config: Quantization config with calibration_samples and group_size.

        Raises:
            NotImplementedError: Phase 2 feature.
        """
        raise NotImplementedError(
            "GPTQ quantization is planned for Phase 2. "
            "Use 'dynamic_int8' for Phase 1."
        )

    def get_model_size_mb(self, quantized_model: Any) -> float:
        """Return the size of the GPTQ-quantized model.

        Raises:
            NotImplementedError: Phase 2 feature.
        """
        raise NotImplementedError("GPTQ quantization is planned for Phase 2.")

    def supported_bits(self) -> list[int]:
        """Return supported bit widths for GPTQ."""
        return [4, 8]
