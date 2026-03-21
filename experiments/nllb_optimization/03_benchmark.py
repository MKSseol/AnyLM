"""
Experiment: NLLB-200 device simulation benchmark.
Date: 2025-03-20
Config: configs/nllb_int8.yaml
Depends on: 02_quantize.py completed

Simulates memory-constrained environments (4GB, 6GB, 8GB RAM)
and measures performance with varying CPU thread limits and batch sizes.
"""

import argparse
import json
import logging
import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from src.benchmark.metrics import measure_latency, measure_peak_memory
from src.benchmark.profiler import HardwareProfiler
from src.models.loader import ModelLoader
from src.quantization.methods.dynamic_int8 import DynamicInt8Quantizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Memory limits to simulate (in MB)
MEMORY_LIMITS_MB = [4096, 6144, 8192]
# Thread counts to test
THREAD_COUNTS = [2, 4, 8]
# Batch sizes to test
BATCH_SIZES = [1, 4, 16]


def main(config_path: str) -> None:
    """Run device simulation benchmark.

    Args:
        config_path: Path to the YAML configuration file.
    """
    logger.info("=" * 60)
    logger.info("NLLB-200 Device Simulation Benchmark")
    logger.info("=" * 60)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    experiment_name = config["experiment"]["name"]
    results_dir = Path(f"data/results/{experiment_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {
        "experiment": f"{experiment_name}-device-simulation",
        "memory_limit_tests": {},
        "thread_count_tests": {},
        "batch_size_tests": {},
    }

    # Test 1: Memory limit simulation
    logger.info("--- Memory Limit Tests ---")
    for mem_limit_mb in MEMORY_LIMITS_MB:
        logger.info("Testing with %d MB memory limit...", mem_limit_mb)
        result = _test_memory_limit(config, mem_limit_mb)
        all_results["memory_limit_tests"][f"{mem_limit_mb}MB"] = result

    # Test 2: Thread count variation
    logger.info("--- Thread Count Tests ---")
    for num_threads in THREAD_COUNTS:
        logger.info("Testing with %d threads...", num_threads)
        result = _test_thread_count(config, num_threads)
        all_results["thread_count_tests"][f"{num_threads}_threads"] = result

    # Test 3: Batch size throughput
    logger.info("--- Batch Size Tests ---")
    for batch_size in BATCH_SIZES:
        logger.info("Testing with batch_size=%d...", batch_size)
        result = _test_batch_size(config, batch_size)
        all_results["batch_size_tests"][f"batch_{batch_size}"] = result

    # Save results
    output_file = results_dir / "device_simulation.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info("Device simulation results saved to %s", output_file)
    _print_summary(all_results)


def _test_memory_limit(config: dict, memory_limit_mb: int) -> dict:
    """Test if model fits within a memory limit.

    Args:
        config: Experiment configuration.
        memory_limit_mb: Maximum memory in MB.

    Returns:
        Dictionary with test results.
    """
    estimated = ModelLoader.estimate_memory(
        config["model"]["name"],
        config["quantization"]["method"],
    )

    fits = estimated <= memory_limit_mb
    return {
        "memory_limit_mb": memory_limit_mb,
        "estimated_usage_mb": estimated,
        "fits_in_memory": fits,
        "headroom_mb": memory_limit_mb - estimated if fits else 0,
    }


def _get_default_lang_pair(config: dict) -> tuple[str, str]:
    """Extract source and target language from the first configured language pair.

    Args:
        config: Experiment configuration.

    Returns:
        Tuple of (source_lang, target_lang).
    """
    lang_pair = config["benchmark"]["language_pairs"][0]
    src_lang, tgt_lang = lang_pair.split("-")
    return src_lang, tgt_lang


def _test_thread_count(config: dict, num_threads: int) -> dict:
    """Benchmark with a specific thread count.

    Args:
        config: Experiment configuration.
        num_threads: Number of CPU threads.

    Returns:
        Dictionary with performance metrics.
    """
    import torch

    torch.set_num_threads(num_threads)

    try:
        loader = ModelLoader(config)
        model, tokenizer = loader.load()

        if config["quantization"]["method"] == "dynamic_int8":
            quantizer = DynamicInt8Quantizer()
            model = quantizer.quantize(model, config.get("quantization", {}))

        src_lang, tgt_lang = _get_default_lang_pair(config)
        tokenizer.src_lang = src_lang
        sample_sentences = [f"This is test sentence number {i}." for i in range(20)]

        def translate(text: str) -> str:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                    max_new_tokens=128,
                )
            return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        stats = measure_latency(translate, sample_sentences, num_repeats=1, warmup_count=3)

        return {
            "num_threads": num_threads,
            "latency_p50_ms": stats["latency_p50_ms"],
            "latency_p99_ms": stats["latency_p99_ms"],
            "tokens_per_sec": stats["tokens_per_sec"],
        }
    except Exception as e:
        logger.error("Thread count test failed: %s", e)
        return {"num_threads": num_threads, "error": str(e)}


def _test_batch_size(config: dict, batch_size: int) -> dict:
    """Measure throughput with a specific batch size.

    Args:
        config: Experiment configuration.
        batch_size: Number of sentences per batch.

    Returns:
        Dictionary with throughput metrics.
    """
    import time

    import torch

    try:
        loader = ModelLoader(config)
        model, tokenizer = loader.load()

        if config["quantization"]["method"] == "dynamic_int8":
            quantizer = DynamicInt8Quantizer()
            model = quantizer.quantize(model, config.get("quantization", {}))

        src_lang, tgt_lang = _get_default_lang_pair(config)
        tokenizer.src_lang = src_lang
        sentences = [f"This is test sentence number {i}." for i in range(batch_size * 5)]

        # Process in batches
        total_time = 0.0
        total_sentences = 0

        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            )

            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                    max_new_tokens=128,
                )
            total_time += time.perf_counter() - t0
            total_sentences += len(batch)

        throughput = total_sentences / total_time if total_time > 0 else 0

        return {
            "batch_size": batch_size,
            "total_sentences": total_sentences,
            "total_time_sec": total_time,
            "sentences_per_sec": throughput,
        }
    except Exception as e:
        logger.error("Batch size test failed: %s", e)
        return {"batch_size": batch_size, "error": str(e)}


def _print_summary(results: dict) -> None:
    """Print a human-readable summary."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("DEVICE SIMULATION SUMMARY")
    logger.info("=" * 60)

    logger.info("")
    logger.info("Memory Limit Tests:")
    for name, data in results["memory_limit_tests"].items():
        status = "OK" if data.get("fits_in_memory") else "EXCEEDS LIMIT"
        logger.info("  %s: %s (estimated: %d MB)", name, status, data.get("estimated_usage_mb", 0))

    logger.info("")
    logger.info("Thread Count Tests:")
    for name, data in results["thread_count_tests"].items():
        if "error" not in data:
            logger.info("  %s: p50=%.1f ms, %.1f tok/s",
                        name, data.get("latency_p50_ms", 0), data.get("tokens_per_sec", 0))
        else:
            logger.info("  %s: ERROR - %s", name, data["error"])

    logger.info("")
    logger.info("Batch Size Tests:")
    for name, data in results["batch_size_tests"].items():
        if "error" not in data:
            logger.info("  %s: %.2f sent/sec", name, data.get("sentences_per_sec", 0))
        else:
            logger.info("  %s: ERROR - %s", name, data["error"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NLLB-200 device simulation benchmark."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g., configs/nllb_int8.yaml)",
    )
    args = parser.parse_args()
    main(args.config)
