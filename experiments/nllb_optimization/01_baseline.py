"""
Experiment: NLLB-200 Distilled 600M baseline performance measurement.
Date: 2025-03-20
Config: configs/nllb_baseline.yaml
Depends on: None (first experiment)

Measures the original model's performance in FP32 mode across
multiple language pairs. Results serve as the baseline for all
subsequent optimization experiments.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.benchmark.runner import BenchmarkRunner
from src.models.loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    """Run baseline benchmark experiment.

    Args:
        config_path: Path to the YAML configuration file.
    """
    logger.info("=" * 60)
    logger.info("NLLB-200 Baseline Performance Measurement")
    logger.info("=" * 60)

    # Estimate memory before loading
    runner = BenchmarkRunner(config_path)
    config = runner.config

    estimated_mem = ModelLoader.estimate_memory(
        config["model"]["name"],
        config["quantization"]["method"],
    )
    logger.info("Estimated memory usage: %d MB", estimated_mem)

    # Run benchmark
    results = runner.run()

    # Print summary
    _print_summary(results)


def _print_summary(results: dict) -> None:
    """Print a human-readable summary of benchmark results."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("BASELINE RESULTS SUMMARY")
    logger.info("=" * 60)

    for lang_pair, metrics in results.get("results", {}).items():
        logger.info("")
        logger.info("Language pair: %s", lang_pair)
        logger.info("-" * 40)

        if "bleu" in metrics:
            bleu = metrics["bleu"]
            logger.info("  BLEU:        %.2f (std: %.2f)", bleu["mean"], bleu["std"])

        logger.info("  Latency p50: %.1f ms", metrics.get("latency_p50_ms", 0))
        logger.info("  Latency p99: %.1f ms", metrics.get("latency_p99_ms", 0))
        logger.info("  Tokens/sec:  %.1f", metrics.get("tokens_per_sec", 0))
        logger.info("  Peak memory: %.1f MB", metrics.get("peak_memory_mb", 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NLLB-200 baseline performance benchmark."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file (e.g., configs/nllb_baseline.yaml)",
    )
    args = parser.parse_args()
    main(args.config)
