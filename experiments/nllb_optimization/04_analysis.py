"""
Experiment: NLLB-200 post-quantization analysis.
Date: 2025-03-20
Config: configs/nllb_int8.yaml
Depends on: 02_quantize.py and 03_benchmark.py completed

Analyzes attention pattern changes, per-language quantization sensitivity,
and generates quality vs. size Pareto frontier visualization.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import yaml

from src.models.loader import ModelLoader
from src.quantization.methods.dynamic_int8 import DynamicInt8Quantizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    """Run post-quantization analysis.

    Args:
        config_path: Path to the YAML configuration file.
    """
    logger.info("=" * 60)
    logger.info("NLLB-200 Post-Quantization Analysis")
    logger.info("=" * 60)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    experiment_name = config["experiment"]["name"]
    results_dir = Path(f"data/results/{experiment_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load models (original and quantized)
    loader = ModelLoader(config)
    model, tokenizer = loader.load()

    quantizer = DynamicInt8Quantizer()
    quantized_model = quantizer.quantize(model, config.get("quantization", {}))

    # Analysis 1: Attention pattern comparison
    logger.info("--- Attention Pattern Analysis ---")
    attention_results = _analyze_attention_patterns(
        model, quantized_model, tokenizer
    )

    # Analysis 2: Per-language sensitivity
    logger.info("--- Per-Language Sensitivity Analysis ---")
    sensitivity_results = _analyze_language_sensitivity(
        model, quantized_model, tokenizer, config
    )

    # Analysis 3: Generate Pareto frontier data
    logger.info("--- Pareto Frontier Data ---")
    pareto_data = _generate_pareto_data(model, quantized_model, quantizer)

    # Save all analysis results
    analysis_output = {
        "experiment": f"{experiment_name}-analysis",
        "attention_analysis": attention_results,
        "language_sensitivity": sensitivity_results,
        "pareto_frontier": pareto_data,
    }

    output_file = results_dir / "analysis_results.json"
    with open(output_file, "w") as f:
        json.dump(analysis_output, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Analysis results saved to %s", output_file)

    # Generate visualizations
    _generate_visualizations(analysis_output, results_dir)


def _analyze_attention_patterns(
    original_model: object,
    quantized_model: object,
    tokenizer: object,
) -> dict:
    """Compare attention patterns between original and quantized models.

    Args:
        original_model: Original FP32 model.
        quantized_model: INT8 quantized model.
        tokenizer: Model tokenizer.

    Returns:
        Dictionary with attention comparison metrics.
    """
    import torch

    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine translation has improved significantly in recent years.",
        "Quantization reduces model size while maintaining quality.",
    ]

    results: dict = {"sentences": [], "mean_attention_divergence": 0.0}
    divergences: list[float] = []

    for sentence in test_sentences:
        tokenizer.src_lang = "eng_Latn"
        inputs = tokenizer(sentence, return_tensors="pt", padding=True)

        with torch.no_grad():
            try:
                orig_outputs = original_model(**inputs, output_attentions=True)
                orig_attentions = orig_outputs.encoder_attentions

                quant_outputs = quantized_model(**inputs, output_attentions=True)
                quant_attentions = quant_outputs.encoder_attentions

                # Compute KL divergence between attention distributions
                layer_divs: list[float] = []
                for orig_attn, quant_attn in zip(orig_attentions, quant_attentions):
                    # Flatten and compute cosine similarity
                    orig_flat = orig_attn.flatten().float()
                    quant_flat = quant_attn.flatten().float()

                    cos_sim = torch.nn.functional.cosine_similarity(
                        orig_flat.unsqueeze(0), quant_flat.unsqueeze(0)
                    ).item()
                    layer_divs.append(1.0 - cos_sim)  # divergence

                avg_div = float(np.mean(layer_divs))
                divergences.append(avg_div)

                results["sentences"].append({
                    "text": sentence,
                    "num_layers_compared": len(layer_divs),
                    "mean_divergence": avg_div,
                    "per_layer_divergence": layer_divs,
                })
            except Exception as e:
                logger.warning("Attention analysis failed for '%s': %s", sentence, e)
                results["sentences"].append({
                    "text": sentence,
                    "error": str(e),
                })

    results["mean_attention_divergence"] = float(np.mean(divergences)) if divergences else 0.0
    logger.info("Mean attention divergence: %.6f", results["mean_attention_divergence"])
    return results


def _analyze_language_sensitivity(
    original_model: object,
    quantized_model: object,
    tokenizer: object,
    config: dict,
) -> dict:
    """Analyze which languages are most affected by quantization.

    Args:
        original_model: Original FP32 model.
        quantized_model: INT8 quantized model.
        tokenizer: Model tokenizer.
        config: Experiment configuration.

    Returns:
        Dictionary with per-language sensitivity metrics.
    """
    import torch

    language_pairs = config["benchmark"]["language_pairs"]
    results: dict = {"per_pair": {}, "most_sensitive": "", "least_sensitive": ""}

    test_sentence = "This is a test sentence for measuring quantization sensitivity."

    for lang_pair in language_pairs:
        src_lang, tgt_lang = lang_pair.split("-")
        tokenizer.src_lang = src_lang

        inputs = tokenizer(test_sentence, return_tensors="pt", padding=True)

        try:
            with torch.no_grad():
                orig_logits = original_model(**inputs).logits.float()
                quant_logits = quantized_model(**inputs).logits.float()

                # Compute logit divergence
                mse = torch.nn.functional.mse_loss(orig_logits, quant_logits).item()
                cos_sim = torch.nn.functional.cosine_similarity(
                    orig_logits.flatten().unsqueeze(0),
                    quant_logits.flatten().unsqueeze(0),
                ).item()

            results["per_pair"][lang_pair] = {
                "logit_mse": mse,
                "logit_cosine_similarity": cos_sim,
                "sensitivity_score": mse,  # Higher = more sensitive
            }
        except Exception as e:
            logger.warning("Sensitivity analysis failed for %s: %s", lang_pair, e)
            results["per_pair"][lang_pair] = {"error": str(e)}

    # Find most/least sensitive
    valid_pairs = {
        k: v for k, v in results["per_pair"].items()
        if "sensitivity_score" in v
    }
    if valid_pairs:
        results["most_sensitive"] = max(valid_pairs, key=lambda k: valid_pairs[k]["sensitivity_score"])
        results["least_sensitive"] = min(valid_pairs, key=lambda k: valid_pairs[k]["sensitivity_score"])

    return results


def _generate_pareto_data(
    original_model: object,
    quantized_model: object,
    quantizer: DynamicInt8Quantizer,
) -> dict:
    """Generate data points for a quality vs. size Pareto frontier graph.

    Args:
        original_model: Original FP32 model.
        quantized_model: INT8 quantized model.
        quantizer: The quantizer instance.

    Returns:
        Dictionary with Pareto frontier data points.
    """
    original_size = quantizer._estimate_model_size(original_model)
    quantized_size = quantizer.get_model_size_mb(quantized_model)

    # In a full experiment, BLEU scores would come from actual benchmark runs.
    # Here we provide the data structure for the Pareto plot.
    data_points = [
        {
            "method": "FP32 (original)",
            "bits": 32,
            "size_mb": original_size,
            "bleu_placeholder": "Run 01_baseline.py for actual value",
        },
        {
            "method": "Dynamic INT8",
            "bits": 8,
            "size_mb": quantized_size,
            "bleu_placeholder": "Run 02_quantize.py for actual value",
        },
        # Placeholder for future quantization methods
        {
            "method": "GPTQ 4-bit (Phase 2)",
            "bits": 4,
            "size_mb": original_size * 0.125,  # Rough estimate
            "bleu_placeholder": "Phase 2",
        },
    ]

    return {
        "data_points": data_points,
        "note": "Populate BLEU values from actual experiment results for the final Pareto graph.",
    }


def _generate_visualizations(analysis: dict, output_dir: Path) -> None:
    """Generate visualization plots from analysis results.

    Args:
        analysis: Analysis results dictionary.
        output_dir: Directory to save plot images.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Plot 1: Attention divergence per layer
        attention_data = analysis.get("attention_analysis", {})
        sentences = attention_data.get("sentences", [])

        if sentences and "per_layer_divergence" in sentences[0]:
            fig, ax = plt.subplots(figsize=(10, 6))
            for sent_data in sentences:
                if "per_layer_divergence" in sent_data:
                    divs = sent_data["per_layer_divergence"]
                    ax.plot(range(len(divs)), divs, marker="o", label=sent_data["text"][:30] + "...")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Attention Divergence (1 - cosine similarity)")
            ax.set_title("Attention Pattern Divergence: FP32 vs INT8")
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(output_dir / "attention_divergence.png", dpi=150)
            plt.close()
            logger.info("Saved attention_divergence.png")

        # Plot 2: Language sensitivity
        sensitivity = analysis.get("language_sensitivity", {}).get("per_pair", {})
        if sensitivity:
            pairs = []
            scores = []
            for pair, data in sensitivity.items():
                if "sensitivity_score" in data:
                    pairs.append(pair)
                    scores.append(data["sensitivity_score"])

            if pairs:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(pairs, scores, color="steelblue")
                ax.set_xlabel("Sensitivity Score (logit MSE)")
                ax.set_title("Per-Language Quantization Sensitivity")
                plt.tight_layout()
                plt.savefig(output_dir / "language_sensitivity.png", dpi=150)
                plt.close()
                logger.info("Saved language_sensitivity.png")

        # Plot 3: Pareto frontier
        pareto = analysis.get("pareto_frontier", {}).get("data_points", [])
        if pareto:
            fig, ax = plt.subplots(figsize=(8, 6))
            sizes = [p["size_mb"] for p in pareto]
            labels = [p["method"] for p in pareto]
            bits = [p["bits"] for p in pareto]

            ax.scatter(sizes, bits, s=100, zorder=5)
            for i, label in enumerate(labels):
                ax.annotate(label, (sizes[i], bits[i]), textcoords="offset points",
                            xytext=(10, 5), fontsize=9)
            ax.set_xlabel("Model Size (MB)")
            ax.set_ylabel("Bit Width")
            ax.set_title("Model Size vs. Bit Width (Pareto Frontier)")
            plt.tight_layout()
            plt.savefig(output_dir / "pareto_frontier.png", dpi=150)
            plt.close()
            logger.info("Saved pareto_frontier.png")

    except ImportError:
        logger.warning("matplotlib not installed. Skipping visualization generation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NLLB-200 post-quantization analysis."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g., configs/nllb_int8.yaml)",
    )
    args = parser.parse_args()
    main(args.config)
