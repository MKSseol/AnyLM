# ALM — Research Methodology

## Approach

We follow a progressive optimization strategy, starting with the simplest techniques
and incrementally applying more advanced methods:

1. **Baseline measurement** — Establish FP32 performance as the reference point
2. **Post-training quantization** — Apply quantization without retraining (INT8 → 4-bit)
3. **Knowledge distillation** — Train smaller models from larger teachers
4. **Format conversion** — Export to on-device runtime formats

## Why This Order?

- **Low-effort wins first**: Dynamic INT8 quantization requires zero training data and
  typically achieves 40-50% size reduction with minimal quality loss.
- **Measure before optimizing**: Every optimization is preceded by systematic measurement.
- **Diminishing returns awareness**: More aggressive compression (4-bit, 2-bit) requires
  more effort. We only pursue these if simpler methods are insufficient.

## Evaluation Protocol

- Minimum 3 repeated measurements for statistical significance
- Warm-up period (10 sentences) excluded from timing
- Metrics: BLEU (quality), latency p50/p99 (speed), peak RSS (memory)
- Hardware info recorded with every result for comparability

## Target: NLLB-200 Distilled 600M

Selected as the first optimization target because:
- Open-source (Meta, CC-BY-NC-4.0)
- 600M parameters — large enough to benefit from optimization, small enough for research iteration
- 200 languages — rich evaluation surface for sensitivity analysis
- Translation task — objective quality metrics (BLEU) available
