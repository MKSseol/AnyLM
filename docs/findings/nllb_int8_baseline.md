# NLLB-200 INT8 Baseline

## Key Finding (one-line summary)
> (To be filled after running Phase 1 experiments)

## Background and Hypothesis

NLLB-200 Distilled 600M uses ~2.4 GB in FP32. We hypothesize that Dynamic INT8
quantization can reduce this to ~800 MB with BLEU loss under 1.0 point, making
the model feasible for devices with 4-8 GB RAM.

## Experiment Design

- **Model**: facebook/nllb-200-distilled-600M
- **Quantization**: PyTorch Dynamic INT8 (torch.quantization.quantize_dynamic)
- **Language pairs**: ko↔en, ko↔ja
- **Test set**: FLORES-200 dev (100 sentences)
- **Metrics**: BLEU, latency p50/p99, peak memory
- **Repeats**: 3 per measurement

## Results

(To be populated after running experiments)

| Metric | FP32 Baseline | INT8 Quantized | Delta |
|--------|--------------|----------------|-------|
| Model size (MB) | - | - | - |
| BLEU (ko→en) | - | - | - |
| BLEU (en→ko) | - | - | - |
| Latency p50 (ms) | - | - | - |
| Peak memory (MB) | - | - | - |

## Interpretation and Implications

(To be filled after analysis)

## Next Steps

- If INT8 meets criteria: proceed to 4-bit quantization (GPTQ, AWQ)
- If INT8 quality loss is too high: investigate static INT8 with calibration
- Analyze per-language sensitivity to guide future optimization

## Related Config Files / Result Data Paths

- Config: `configs/nllb_baseline.yaml`, `configs/nllb_int8.yaml`
- Results: `data/results/nllb-baseline/`, `data/results/nllb-int8/`
