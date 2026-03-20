# experiments/nllb_optimization/ — NLLB Optimization Experiment

## Goal
Optimize Meta's NLLB-200 Distilled 600M model to run on smartphone-grade
hardware (4-8GB RAM).

## Experiment Sequence

### 01_baseline.py — Baseline Measurement
- Measure original NLLB-200 Distilled 600M model performance
- Measure BLEU, speed, and memory in FP32/FP16 modes
- Target language pairs: ko↔en, ko↔ja, en↔zh, en↔de (varying language distances)
- These results serve as the comparison baseline for all subsequent experiments

### 02_quantize.py — Quantization Experiment
- Apply Dynamic INT8, Static INT8, GPTQ 4-bit, AWQ 4-bit sequentially
- Record model size, BLEU loss, and speed change for each quantization level
- Generate "quality vs. size" Pareto frontier graph

### 03_benchmark.py — Device Simulation
- Verify feasibility under memory-constrained environments (4GB, 6GB, 8GB)
- Measure performance changes with CPU thread limits (2, 4, 8)
- Measure throughput by batch size

### 04_analysis.py — Model Analysis
- Analyze attention pattern changes before/after quantization
- Analyze per-language quantization sensitivity (which languages lose the most quality?)
- Analyze embedding space distortion

## Success Criteria
- INT8 quantization: BLEU loss < 1.0 (vs. original)
- 4-bit quantization: BLEU loss < 3.0
- Runnable in 4GB RAM environment
- Inference speed > 10 tokens/sec (CPU, batch=1)
