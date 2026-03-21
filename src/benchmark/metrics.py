"""Benchmark metrics for translation model evaluation.

Provides functions for computing BLEU scores, measuring latency,
and tracking memory usage.
"""

import logging
import statistics
import time
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    """Compute BLEU score using sacrebleu.

    Args:
        hypotheses: List of model-generated translations.
        references: List of reference translations.

    Returns:
        BLEU score as a float.
    """
    import sacrebleu

    if not hypotheses or not references:
        logger.warning("Empty input for BLEU computation, returning 0.0")
        return 0.0

    result = sacrebleu.corpus_bleu(hypotheses, [references])
    logger.info("BLEU score: %.2f", result.score)
    return result.score


def compute_chrf(hypotheses: list[str], references: list[str]) -> float:
    """Compute chrF++ score using sacrebleu.

    Args:
        hypotheses: List of model-generated translations.
        references: List of reference translations.

    Returns:
        chrF++ score as a float.
    """
    import sacrebleu

    result = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    logger.info("chrF++ score: %.2f", result.score)
    return result.score


def compute_comet(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
) -> float:
    """Compute COMET score for translation quality evaluation.

    Requires the unbabel-comet package (Phase 2).

    Args:
        sources: List of source sentences.
        hypotheses: List of model-generated translations.
        references: List of reference translations.

    Returns:
        COMET score as a float.

    Raises:
        NotImplementedError: COMET is planned for Phase 2.
    """
    raise NotImplementedError(
        "COMET metric is planned for Phase 2. "
        "Install unbabel-comet and implement this function. "
        "For now, use compute_bleu() or compute_chrf()."
    )


def measure_latency(
    translate_fn: Callable[..., Any],
    inputs: list[str],
    num_repeats: int = 3,
    warmup_count: int = 10,
) -> dict[str, float]:
    """Measure translation latency with percentile statistics.

    Args:
        translate_fn: Callable that takes a string and returns a translation.
        inputs: List of source sentences.
        num_repeats: Number of measurement repetitions.
        warmup_count: Number of warm-up sentences to skip from measurement.

    Returns:
        Dictionary with latency statistics:
        - latency_p50_ms: median latency in milliseconds
        - latency_p99_ms: 99th percentile latency in milliseconds
        - tokens_per_sec: average throughput
    """
    # Warm-up phase
    warmup_inputs = inputs[:warmup_count] if len(inputs) >= warmup_count else inputs
    for sentence in warmup_inputs:
        translate_fn(sentence)

    all_latencies: list[float] = []
    total_tokens = 0

    for repeat in range(num_repeats):
        logger.info("Latency measurement run %d/%d", repeat + 1, num_repeats)
        for sentence in inputs:
            start = time.perf_counter()
            result = translate_fn(sentence)
            elapsed = time.perf_counter() - start
            all_latencies.append(elapsed * 1000)  # Convert to ms

            # Estimate token count from output
            if isinstance(result, str):
                total_tokens += len(result.split())

    latencies_array = np.array(all_latencies)
    total_time_sec = sum(all_latencies) / 1000

    stats = {
        "latency_p50_ms": float(np.percentile(latencies_array, 50)),
        "latency_p99_ms": float(np.percentile(latencies_array, 99)),
        "latency_mean_ms": float(np.mean(latencies_array)),
        "latency_std_ms": float(np.std(latencies_array)),
        "tokens_per_sec": total_tokens / total_time_sec if total_time_sec > 0 else 0,
    }

    logger.info(
        "Latency — p50: %.1f ms, p99: %.1f ms, tokens/sec: %.1f",
        stats["latency_p50_ms"],
        stats["latency_p99_ms"],
        stats["tokens_per_sec"],
    )
    return stats


def measure_peak_memory(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> dict[str, float]:
    """Measure peak memory usage during function execution.

    Args:
        func: Function to profile.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        Dictionary with memory statistics:
        - peak_memory_mb: peak memory usage in MB
        - memory_increment_mb: memory increase from baseline
    """
    import tracemalloc

    tracemalloc.start()

    # Run the function
    func(*args, **kwargs)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats = {
        "peak_memory_mb": peak / (1024 * 1024),
        "memory_increment_mb": current / (1024 * 1024),
    }

    logger.info("Peak memory: %.1f MB", stats["peak_memory_mb"])
    return stats


def compute_model_size_mb(model_path: str) -> float:
    """Compute the disk size of a model in MB.

    Args:
        model_path: Path to the model directory or file.

    Returns:
        Size in MB.
    """
    from pathlib import Path

    path = Path(model_path)
    if path.is_file():
        size_bytes = path.stat().st_size
    elif path.is_dir():
        size_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        raise FileNotFoundError(f"Model path not found: {model_path}")

    size_mb = size_bytes / (1024 * 1024)
    logger.info("Model size: %.1f MB (%s)", size_mb, model_path)
    return size_mb


def aggregate_metric_runs(values: list[float]) -> dict[str, float]:
    """Aggregate multiple measurement runs into summary statistics.

    Args:
        values: List of metric values from repeated runs.

    Returns:
        Dictionary with mean, std, min, max, and the raw runs.
    """
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "runs": values,
    }
