"""Download FLORES-200 test data for benchmarking.

Supports two download methods:
1. HuggingFace datasets library (preferred)
2. Direct HTTPS download from FLORES GitHub releases

Usage:
    python scripts/download_flores.py
    python scripts/download_flores.py --langs kor_Hang eng_Latn jpn_Jpan
    python scripts/download_flores.py --split devtest --method direct
"""

import argparse
import io
import logging
import tarfile
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_LANGS = ["kor_Hang", "eng_Latn", "jpn_Jpan"]
OUTPUT_DIR = Path("data/test_sets/flores200")

# FLORES-200 direct download URL (GitHub release)
FLORES_URL = (
    "https://tinyurl.com/flores200dataset"
)


def download_via_huggingface(langs: list[str], split: str) -> None:
    """Download using HuggingFace datasets library.

    Args:
        langs: FLORES-200 language codes.
        split: Dataset split ('dev' or 'devtest').
    """
    from datasets import load_dataset

    logger.info("Loading FLORES-200 via HuggingFace (split=%s)...", split)
    ds = load_dataset("facebook/flores", "all", split=split)

    for lang in langs:
        col_name = f"sentence_{lang}"
        if col_name not in ds.column_names:
            col_name = lang
            if col_name not in ds.column_names:
                logger.warning("Language '%s' not found. Available: %s", lang, ds.column_names[:10])
                continue

        sentences = ds[col_name]
        _save_sentences(lang, sentences)


def download_via_direct(langs: list[str], split: str) -> None:
    """Download FLORES-200 directly from the official release.

    Args:
        langs: FLORES-200 language codes.
        split: Dataset split ('dev' or 'devtest').
    """
    logger.info("Downloading FLORES-200 directly from official release...")

    try:
        response = urllib.request.urlopen(FLORES_URL, timeout=120)
        data = response.read()
    except Exception as e:
        logger.error("Direct download failed: %s", e)
        logger.info(
            "Please download FLORES-200 manually:\n"
            "  1. Visit https://github.com/facebookresearch/flores\n"
            "  2. Download the dataset\n"
            "  3. Extract to %s/",
            OUTPUT_DIR,
        )
        raise

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for lang in langs:
            member_path = f"flores200_dataset/{split}/{lang}.{split}"
            try:
                f = tar.extractfile(member_path)
                if f is None:
                    logger.warning("File not found in archive: %s", member_path)
                    continue
                sentences = [line.decode("utf-8").strip() for line in f if line.strip()]
                _save_sentences(lang, sentences)
            except KeyError:
                logger.warning("Language '%s' not found in archive", lang)


def _save_sentences(lang: str, sentences: list[str]) -> None:
    """Save sentences to output file.

    Args:
        lang: Language code.
        sentences: List of sentences.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{lang}.dev"

    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence.strip() + "\n")

    logger.info("Saved %d sentences for '%s' → %s", len(sentences), lang, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download FLORES-200 test data for benchmarking."
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=DEFAULT_LANGS,
        help=f"Language codes to download (default: {DEFAULT_LANGS})",
    )
    parser.add_argument(
        "--split",
        default="devtest",
        choices=["dev", "devtest"],
        help="Dataset split (default: devtest)",
    )
    parser.add_argument(
        "--method",
        default="huggingface",
        choices=["huggingface", "direct"],
        help="Download method (default: huggingface)",
    )
    args = parser.parse_args()

    if args.method == "huggingface":
        download_via_huggingface(args.langs, args.split)
    else:
        download_via_direct(args.langs, args.split)

    logger.info("Done. Output directory: %s", OUTPUT_DIR)
