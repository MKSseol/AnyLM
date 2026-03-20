"""Download and verify models from HuggingFace.

Usage:
    python scripts/download_model.py --model facebook/nllb-200-distilled-600M
    python scripts/download_model.py --model facebook/nllb-200-distilled-600M --cache-dir /path/to/cache
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "maga" / "models"


def main(model_name: str, cache_dir: str | None = None) -> None:
    """Download a model from HuggingFace Hub.

    Args:
        model_name: HuggingFace model identifier.
        cache_dir: Optional cache directory override.
    """
    from huggingface_hub import snapshot_download

    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading model: %s", model_name)
    logger.info("Cache directory: %s", cache)

    path = snapshot_download(
        repo_id=model_name,
        cache_dir=str(cache),
    )

    logger.info("Model downloaded to: %s", path)

    # Report size
    total_size = sum(
        f.stat().st_size for f in Path(path).rglob("*") if f.is_file()
    )
    logger.info("Total size: %.1f MB", total_size / (1024 * 1024))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and verify HuggingFace models."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model identifier (e.g., facebook/nllb-200-distilled-600M)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    args = parser.parse_args()
    main(args.model, args.cache_dir)
