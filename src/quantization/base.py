"""Abstract base class for all quantization methods.

Every quantization implementation must inherit from BaseQuantizer
and implement the required interface methods.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseQuantizer(ABC):
    """Base class for all quantization methods.

    Subclasses must implement quantize(), get_model_size_mb(),
    and supported_bits() methods.

    Usage:
        quantizer = DynamicInt8Quantizer()
        quantized_model = quantizer.quantize(model, config)
        size_mb = quantizer.get_model_size_mb(quantized_model)
    """

    @abstractmethod
    def quantize(self, model: Any, config: dict) -> Any:
        """Quantize the given model.

        Args:
            model: The model to quantize.
            config: Quantization configuration dictionary.

        Returns:
            The quantized model.
        """
        ...

    @abstractmethod
    def get_model_size_mb(self, quantized_model: Any) -> float:
        """Return the disk size (MB) of the quantized model.

        Args:
            quantized_model: The quantized model.

        Returns:
            Model size in megabytes.
        """
        ...

    @abstractmethod
    def supported_bits(self) -> list[int]:
        """Return list of supported bit widths.

        Returns:
            List of integers, e.g., [4, 8].
        """
        ...

    @property
    def name(self) -> str:
        """Return the name of this quantization method."""
        return self.__class__.__name__
