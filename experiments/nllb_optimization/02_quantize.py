"""
Experiment: NLLB-200 Distilled 600M INT8 quantization.
Date: 2025-03-20
Config: configs/nllb_int8.yaml
Depends on: 01_baseline.py completed

Applies Dynamic INT8 quantization and benchmarks the quantized model.
Compares results with the baseline to measure quality/size tradeoff.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from src.benchmark.metrics import aggregate_metric_runs, compute_bleu, measure_latency
from src.benchmark.runner import BenchmarkRunner
from src.models.loader import ModelLoader
from src.quantization.methods.dynamic_int8 import DynamicInt8Quantizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    """Run INT8 quantization experiment.

    Args:
        config_path: Path to the YAML configuration file.
    """
    logger.info("=" * 60)
    logger.info("NLLB-200 INT8 Quantization Experiment")
    logger.info("=" * 60)

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load original model
    loader = ModelLoader(config)
    model, tokenizer = loader.load()

    # Quantize
    quantizer = DynamicInt8Quantizer()
    original_size = quantizer._estimate_model_size(model)
    quantized_model = quantizer.quantize(model, config.get("quantization", {}))
    quantized_size = quantizer.get_model_size_mb(quantized_model)

    logger.info("Original size:  %.1f MB", original_size)
    logger.info("Quantized size: %.1f MB", quantized_size)
    logger.info("Reduction:      %.1f%%", (1 - quantized_size / original_size) * 100)

    # Run benchmark with quantized model
    runner = BenchmarkRunner(config_path)
    results = runner.run()

    # Generate comparison summary
    _generate_comparison(config, original_size, quantized_size, results)


def _generate_comparison(
    config: dict,
    original_size: float,
    quantized_size: float,
    results: dict,
) -> None:
    """Generate and save a comparison summary table.

    Args:
        config: Experiment configuration.
        original_size: Original model size in MB.
        quantized_size: Quantized model size in MB.
        results: Benchmark results dictionary.
    """
    experiment_name = config["experiment"]["name"]
    results_dir = Path(f"data/results/{experiment_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment": experiment_name,
        "comparison": {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "size_reduction_pct": (1 - quantized_size / original_size) * 100,
        },
        "per_language_pair": {},
    }

    for lang_pair, metrics in results.get("results", {}).items():
        summary["per_language_pair"][lang_pair] = {
            "bleu": metrics.get("bleu", {}),
            "latency_p50_ms": metrics.get("latency_p50_ms", 0),
            "peak_memory_mb": metrics.get("peak_memory_mb", 0),
        }

    summary_file = results_dir / "quantization_comparison.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Comparison summary saved to %s", summary_file)

    # Print summary table
    logger.info("")
    logger.info("=" * 60)
    logger.info("QUANTIZATION COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info("Model size: %.1f MB → %.1f MB (%.1f%% reduction)",
                original_size, quantized_size,
                summary["comparison"]["size_reduction_pct"])
    logger.info("")

    for lang_pair, metrics in summary["per_language_pair"].items():
        bleu = metrics.get("bleu", {})
        logger.info("  %s:", lang_pair)
        if bleu:
            logger.info("    BLEU:        %.2f (std: %.2f)",
                        bleu.get("mean", 0), bleu.get("std", 0))
        logger.info("    Latency p50: %.1f ms", metrics.get("latency_p50_ms", 0))
        logger.info("    Peak memory: %.1f MB", metrics.get("peak_memory_mb", 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NLLB-200 INT8 quantization experiment."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g., configs/nllb_int8.yaml)",
    )
    args = parser.parse_args()
    main(args.config)
