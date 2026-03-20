"""AWQ (Activation-aware Weight Quantization) 4-bit quantization.

Preserves important weights based on activation patterns.
Planned for Phase 2 implementation.
"""

import logging
from typing import Any

from src.quantization.base import BaseQuantizer

logger = logging.getLogger(__name__)


class AWQQuantizer(BaseQuantizer):
    """AWQ activation-aware quantization.

    Identifies and preserves salient weights based on activation
    magnitude, achieving better quality than uniform quantization.

    Note: Full implementation planned for Phase 2.
    """

    def quantize(self, model: Any, config: dict) -> Any:
        """Apply AWQ quantization to the model.

        Args:
            model: The model to quantize.
            config: Quantization config.

        Raises:
            NotImplementedError: Phase 2 feature.
        """
        raise NotImplementedError(
            "AWQ quantization is planned for Phase 2. "
            "Use 'dynamic_int8' for Phase 1."
        )

    def get_model_size_mb(self, quantized_model: Any) -> float:
        """Return the size of the AWQ-quantized model.

        Raises:
            NotImplementedError: Phase 2 feature.
        """
        raise NotImplementedError("AWQ quantization is planned for Phase 2.")

    def supported_bits(self) -> list[int]:
        """Return supported bit widths for AWQ."""
        return [4]
