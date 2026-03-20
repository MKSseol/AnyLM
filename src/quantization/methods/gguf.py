"""GGUF format quantization for llama.cpp compatibility.

Supports various quantization levels from Q2_K to Q8_0.
Planned for Phase 3 implementation.
"""

import logging
from typing import Any

from src.quantization.base import BaseQuantizer

logger = logging.getLogger(__name__)


class GGUFQuantizer(BaseQuantizer):
    """GGUF quantization for llama.cpp format.

    Converts models to GGUF format with various quantization
    levels (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0).

    Note: Full implementation planned for Phase 3.
    """

    def quantize(self, model: Any, config: dict) -> Any:
        """Convert and quantize model to GGUF format.

        Args:
            model: The model to quantize.
            config: Quantization config with target quantization level.

        Raises:
            NotImplementedError: Phase 3 feature.
        """
        raise NotImplementedError(
            "GGUF quantization is planned for Phase 3. "
            "Use 'dynamic_int8' for Phase 1."
        )

    def get_model_size_mb(self, quantized_model: Any) -> float:
        """Return the size of the GGUF-quantized model.

        Raises:
            NotImplementedError: Phase 3 feature.
        """
        raise NotImplementedError("GGUF quantization is planned for Phase 3.")

    def supported_bits(self) -> list[int]:
        """Return supported bit widths for GGUF."""
        return [2, 3, 4, 5, 6, 8]
