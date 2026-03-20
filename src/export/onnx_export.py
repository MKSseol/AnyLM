"""ONNX model export.

Converts models to ONNX format for cross-platform inference
via ONNX Runtime (including ONNX Runtime Mobile).
Full implementation planned for Phase 2.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ONNXExporter:
    """Export models to ONNX format.

    Args:
        config: Export configuration.

    Note: Full implementation planned for Phase 2.
    """

    def __init__(self, config: dict) -> None:
        self._config = config

    def export(self, model: Any, tokenizer: Any, output_dir: str) -> Path:
        """Export model to ONNX format.

        Args:
            model: The model to export.
            tokenizer: The model tokenizer.
            output_dir: Directory to save the ONNX model.

        Raises:
            NotImplementedError: Phase 2 feature.
        """
        raise NotImplementedError("ONNX export is planned for Phase 2.")
