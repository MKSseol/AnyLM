"""Benchmark runner — orchestrates the full benchmark pipeline.

Takes a YAML config and runs model loading, translation, and metric
computation, then saves structured JSON results.
"""

import hashlib
import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.benchmark.metrics import (
    aggregate_metric_runs,
    compute_bleu,
    compute_chrf,
    measure_latency,
    measure_peak_memory,
)
from src.models.loader import ModelLoader

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs benchmark experiments based on YAML configuration.

    Args:
        config_path: Path to the YAML configuration file.
        results_dir: Directory to save results. Defaults to data/results/.

    Usage:
        runner = BenchmarkRunner("configs/nllb_baseline.yaml")
        results = runner.run()
    """

    def __init__(
        self,
        config_path: str,
        results_dir: str | None = None,
    ) -> None:
        self._config_path = Path(config_path)
        with open(self._config_path) as f:
            self._config: dict = yaml.safe_load(f)

        experiment_name = self._config["experiment"]["name"]
        self._results_dir = Path(
            results_dir or f"data/results/{experiment_name}"
        )
        self._results_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> dict:
        """Return the loaded configuration."""
        return self._config

    def run(self) -> dict[str, Any]:
        """Execute the full benchmark pipeline.

        Returns:
            Dictionary containing all benchmark results.
        """
        logger.info(
            "Starting benchmark: %s", self._config["experiment"]["name"]
        )

        # Load model
        loader = ModelLoader(self._config)
        model, tokenizer = loader.load()

        # Run benchmarks for each language pair
        benchmark_config = self._config["benchmark"]
        language_pairs = benchmark_config["language_pairs"]
        num_repeats = benchmark_config.get("num_repeats", 3)
        num_samples = benchmark_config.get("num_samples", 100)

        results: dict[str, Any] = {}
        for lang_pair in language_pairs:
            logger.info("Benchmarking language pair: %s", lang_pair)
            pair_results = self._benchmark_language_pair(
                model=model,
                tokenizer=tokenizer,
                lang_pair=lang_pair,
                num_samples=num_samples,
                num_repeats=num_repeats,
            )
            results[lang_pair] = pair_results

        # Assemble final output
        output = self._build_output(results)

        # Save results
        self._save_results(output)

        logger.info("Benchmark complete: %s", self._config["experiment"]["name"])
        return output

    def _benchmark_language_pair(
        self,
        model: Any,
        tokenizer: Any,
        lang_pair: str,
        num_samples: int,
        num_repeats: int,
    ) -> dict[str, Any]:
        """Benchmark a single language pair.

        Args:
            model: The loaded model.
            tokenizer: The loaded tokenizer.
            lang_pair: Language pair string (e.g., "kor_Hang-eng_Latn").
            num_samples: Number of test samples.
            num_repeats: Number of measurement repetitions.

        Returns:
            Dictionary of metrics for this language pair.
        """
        src_lang, tgt_lang = lang_pair.split("-")

        # Generate sample sentences for benchmarking
        # In production, these come from FLORES-200 or other test sets
        sample_sources = self._load_test_data(src_lang, num_samples)
        sample_references = self._load_test_data(tgt_lang, num_samples)

        framework = self._config["model"].get("framework", "transformers")

        # Build translate function based on framework
        if framework == "transformers":
            translate_fn = self._make_transformers_translate_fn(
                model, tokenizer, src_lang, tgt_lang
            )
        elif framework == "ctranslate2":
            translate_fn = self._make_ctranslate2_translate_fn(
                model, tokenizer, src_lang, tgt_lang
            )
        else:
            raise ValueError(f"Unsupported framework for benchmarking: {framework}")

        # Measure BLEU across repeats
        bleu_scores: list[float] = []
        for i in range(num_repeats):
            logger.info("BLEU measurement run %d/%d", i + 1, num_repeats)
            hypotheses = [translate_fn(s) for s in sample_sources]
            if sample_references:
                score = compute_bleu(hypotheses, sample_references)
                bleu_scores.append(score)

        # Measure latency
        latency_stats = measure_latency(
            translate_fn, sample_sources, num_repeats=1, warmup_count=5
        )

        # Measure memory
        memory_stats = measure_peak_memory(
            lambda: [translate_fn(s) for s in sample_sources[:10]]
        )

        pair_results: dict[str, Any] = {
            "latency_p50_ms": latency_stats["latency_p50_ms"],
            "latency_p99_ms": latency_stats["latency_p99_ms"],
            "tokens_per_sec": latency_stats["tokens_per_sec"],
            "peak_memory_mb": memory_stats["peak_memory_mb"],
        }

        if bleu_scores:
            pair_results["bleu"] = aggregate_metric_runs(bleu_scores)

        return pair_results

    def _load_test_data(self, lang_code: str, num_samples: int) -> list[str]:
        """Load test sentences for a language.

        Falls back to placeholder sentences if test data is not available.

        Args:
            lang_code: FLORES-200 language code (e.g., "kor_Hang").
            num_samples: Number of sentences to load.

        Returns:
            List of test sentences.
        """
        # Try loading from FLORES-200 test set
        flores_dir = Path("data/test_sets/flores200")
        test_file = flores_dir / f"{lang_code}.dev"

        if test_file.exists():
            with open(test_file) as f:
                sentences = [line.strip() for line in f if line.strip()]
            return sentences[:num_samples]

        logger.warning(
            "Test data for '%s' not found at %s. "
            "Using placeholder data. Run scripts/download_model.py to get FLORES-200.",
            lang_code,
            test_file,
        )
        # Return placeholder sentences for development/testing
        return [f"Sample sentence {i} for {lang_code}." for i in range(num_samples)]

    def _make_transformers_translate_fn(
        self,
        model: Any,
        tokenizer: Any,
        src_lang: str,
        tgt_lang: str,
    ) -> Any:
        """Create a translation function for HuggingFace Transformers models.

        Args:
            model: HuggingFace model.
            tokenizer: HuggingFace tokenizer.
            src_lang: Source language code.
            tgt_lang: Target language code.

        Returns:
            Callable that translates a single sentence.
        """
        import torch

        tokenizer.src_lang = src_lang

        def translate(text: str) -> str:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                    max_new_tokens=256,
                )

            result = tokenizer.batch_decode(generated, skip_special_tokens=True)
            return result[0]

        return translate

    def _make_ctranslate2_translate_fn(
        self,
        translator: Any,
        tokenizer: Any,
        src_lang: str,
        tgt_lang: str,
    ) -> Any:
        """Create a translation function for CTranslate2 models.

        Args:
            translator: CTranslate2 translator.
            tokenizer: HuggingFace tokenizer.
            src_lang: Source language code.
            tgt_lang: Target language code.

        Returns:
            Callable that translates a single sentence.
        """
        tokenizer.src_lang = src_lang

        def translate(text: str) -> str:
            source_tokens = tokenizer.convert_ids_to_tokens(
                tokenizer.encode(text)
            )
            target_prefix = [tokenizer.eos_token, tgt_lang]

            results = translator.translate_batch(
                [source_tokens],
                target_prefix=[target_prefix],
            )

            output_tokens = results[0].hypotheses[0]
            output_text = tokenizer.decode(
                tokenizer.convert_tokens_to_ids(output_tokens),
                skip_special_tokens=True,
            )
            return output_text

        return translate

    def _build_output(self, results: dict[str, Any]) -> dict[str, Any]:
        """Build the final output dictionary.

        Args:
            results: Per-language-pair benchmark results.

        Returns:
            Complete benchmark output dictionary.
        """
        config_str = json.dumps(self._config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()

        return {
            "experiment": self._config["experiment"]["name"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_hash": f"sha256:{config_hash}",
            "config": self._config,
            "hardware": {
                "cpu": platform.processor() or platform.machine(),
                "ram_gb": _get_system_ram_gb(),
                "device": self._config["hardware"]["device"],
                "python_version": platform.python_version(),
            },
            "results": results,
        }

    def _save_results(self, output: dict[str, Any]) -> None:
        """Save benchmark results to JSON and copy the config.

        Args:
            output: Complete benchmark output dictionary.
        """
        # Save results JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = self._config["experiment"]["name"]
        results_file = self._results_dir / f"{experiment_name}_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", results_file)

        # Copy config file
        import shutil

        config_copy = self._results_dir / self._config_path.name
        shutil.copy2(self._config_path, config_copy)
        logger.info("Config copied to %s", config_copy)


def _get_system_ram_gb() -> float:
    """Get total system RAM in GB."""
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # Fallback: read from /proc/meminfo on Linux
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return kb / (1024**2)
        except (FileNotFoundError, ValueError):
            pass
    return 0.0
