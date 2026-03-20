"""Unified model loader for multiple frameworks.

Supports HuggingFace Transformers, CTranslate2, and ONNX Runtime models
through a single interface.
"""

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "maga" / "models"
_CACHE_DIR = Path(os.environ.get("MAGA_MODEL_CACHE", str(_DEFAULT_CACHE_DIR)))

_FRAMEWORK_REGISTRY: dict[str, str] = {
    "transformers": "_load_transformers",
    "ctranslate2": "_load_ctranslate2",
    "onnx": "_load_onnx",
}


class ModelLoader:
    """Unified model loader.

    Supported frameworks:
    - transformers: HuggingFace models
    - ctranslate2: CTranslate2 converted models
    - onnx: ONNX Runtime models

    Args:
        config: Experiment configuration dictionary containing model settings.

    Usage:
        loader = ModelLoader(config)
        model, tokenizer = loader.load()
    """

    def __init__(self, config: dict) -> None:
        self._model_config = config["model"]
        self._hardware_config = config.get("hardware", {})
        self._model_name: str = self._model_config["name"]
        self._revision: str = self._model_config.get("revision", "main")
        self._framework: str = self._model_config.get("framework", "transformers")

        if self._framework not in _FRAMEWORK_REGISTRY:
            raise ValueError(
                f"Unsupported framework '{self._framework}'. "
                f"Supported: {list(_FRAMEWORK_REGISTRY.keys())}"
            )

    @property
    def cache_dir(self) -> Path:
        """Return the model cache directory."""
        return _CACHE_DIR

    def load(self) -> tuple[Any, Any]:
        """Load model and tokenizer based on the configured framework.

        Returns:
            Tuple of (model, tokenizer).
        """
        method_name = _FRAMEWORK_REGISTRY[self._framework]
        load_method = getattr(self, method_name)
        logger.info(
            "Loading model '%s' with framework '%s'",
            self._model_name,
            self._framework,
        )
        model, tokenizer = load_method()
        logger.info("Model loaded successfully.")
        return model, tokenizer

    def _load_transformers(self) -> tuple[Any, Any]:
        """Load a HuggingFace Transformers model."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = self._hardware_config.get("device", "cpu")

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            revision=self._revision,
            cache_dir=str(self.cache_dir),
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self._model_name,
            revision=self._revision,
            cache_dir=str(self.cache_dir),
        )

        if device == "cuda":
            import torch

            if torch.cuda.is_available():
                model = model.cuda()
                logger.info("Model moved to CUDA.")
            else:
                logger.warning("CUDA requested but not available. Using CPU.")
        elif device == "mps":
            import torch

            if torch.backends.mps.is_available():
                model = model.to("mps")
                logger.info("Model moved to MPS.")
            else:
                logger.warning("MPS requested but not available. Using CPU.")

        model.eval()
        return model, tokenizer

    def _load_ctranslate2(self) -> tuple[Any, Any]:
        """Load a CTranslate2 model.

        If the model hasn't been converted yet, converts from HuggingFace first.
        """
        import ctranslate2
        from transformers import AutoTokenizer

        ct2_model_dir = self.cache_dir / f"{self._model_name.replace('/', '_')}_ct2"

        if not ct2_model_dir.exists():
            logger.info("CTranslate2 model not found. Converting from HuggingFace...")
            self._convert_to_ctranslate2(ct2_model_dir)

        device = self._hardware_config.get("device", "cpu")
        if device == "mps":
            device = "cpu"  # CTranslate2 does not support MPS

        translator = ctranslate2.Translator(
            str(ct2_model_dir),
            device=device,
            inter_threads=self._hardware_config.get("num_threads", 4),
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            revision=self._revision,
            cache_dir=str(self.cache_dir),
        )

        return translator, tokenizer

    def _convert_to_ctranslate2(self, output_dir: Path) -> None:
        """Convert a HuggingFace model to CTranslate2 format."""
        import ctranslate2

        logger.info("Converting '%s' to CTranslate2 format...", self._model_name)
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        ctranslate2.converters.TransformerConverter(
            self._model_name,
            copy_files=["tokenizer.json", "sentencepiece.bpe.model"],
        ).convert(str(output_dir), quantization="float32")

        logger.info("Conversion complete: %s", output_dir)

    def _load_onnx(self) -> tuple[Any, Any]:
        """Load an ONNX Runtime model."""
        raise NotImplementedError(
            "ONNX loading is planned for Phase 2. "
            "Use 'transformers' or 'ctranslate2' framework for now."
        )

    @staticmethod
    def estimate_memory(model_name: str, quantization: str = "none") -> int:
        """Estimate memory usage in MB for a given model and quantization.

        Args:
            model_name: HuggingFace model identifier.
            quantization: Quantization method (none, dynamic_int8, gptq_4bit, etc.).

        Returns:
            Estimated peak memory in MB.
        """
        # Rough estimates based on parameter count
        # NLLB-200 Distilled 600M: ~600M params
        estimates: dict[str, dict[str, int]] = {
            "facebook/nllb-200-distilled-600M": {
                "none": 2400,       # FP32: ~2.4GB
                "fp16": 1200,       # FP16: ~1.2GB
                "dynamic_int8": 800, # INT8: ~800MB
                "gptq_4bit": 500,   # 4-bit: ~500MB
                "awq_4bit": 500,
            },
        }

        model_estimates = estimates.get(model_name, {})
        estimate = model_estimates.get(quantization, model_estimates.get("none", 4000))

        logger.info(
            "Estimated memory for %s (%s): %d MB",
            model_name, quantization, estimate,
        )
        return estimate
