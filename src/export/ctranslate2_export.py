"""CTranslate2 model export.

Converts HuggingFace models to CTranslate2 format for efficient
CPU/desktop inference.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CTranslate2Exporter:
    """Export models to CTranslate2 format.

    Args:
        config: Export configuration.

    Usage:
        exporter = CTranslate2Exporter(config)
        output_path = exporter.export(model_name, output_dir)
    """

    def __init__(self, config: dict) -> None:
        self._config = config

    def export(
        self,
        model_name: str,
        output_dir: str,
        quantization: str = "float32",
    ) -> Path:
        """Convert a HuggingFace model to CTranslate2 format.

        Args:
            model_name: HuggingFace model identifier.
            output_dir: Directory to save the converted model.
            quantization: CTranslate2 quantization type
                (float32, float16, int8, int8_float16).

        Returns:
            Path to the converted model directory.
        """
        import ctranslate2

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Converting '%s' to CTranslate2 (%s)...", model_name, quantization)

        converter = ctranslate2.converters.TransformerConverter(
            model_name,
            copy_files=["tokenizer.json", "sentencepiece.bpe.model"],
        )
        converter.convert(str(output_path), quantization=quantization)

        logger.info("CTranslate2 export complete: %s", output_path)
        return output_path
