# src/benchmark/ — Benchmark Module

## Role
Systematically measure model quality, speed, and memory usage.
The output of this module serves as the basis for all research decisions.

## Core Files

### runner.py — Benchmark Runner
- Takes YAML config and runs the full benchmark pipeline
- Saves structured JSON results to `data/results/`
- Auto-compares with previous results when re-running the same config

### metrics.py — Metric Computation
Supported metrics:
| Metric | Library | Description |
|--------|---------|-------------|
| BLEU | sacrebleu | Translation quality (n-gram based) |
| COMET | comet-ml | Translation quality (neural, more accurate) |
| chrF++ | sacrebleu | Character n-gram based (better for morphologically rich languages) |
| tokens/sec | custom | Inference speed |
| latency_p50/p99 | custom | Latency percentiles |
| peak_memory_mb | memory_profiler | Peak memory usage |
| model_size_mb | custom | Model size on disk |

### profiler.py — Hardware Profiling
- Collect CPU/GPU utilization time-series data
- Track memory usage patterns
- Thermal management (mobile scenario simulation)

## Result File Format
```json
{
  "experiment": "nllb-int8-baseline",
  "timestamp": "2025-XX-XXTXX:XX:XXZ",
  "config_hash": "sha256:...",
  "hardware": {
    "cpu": "Apple M1", "ram_gb": 16, "device": "mps"
  },
  "results": {
    "kor_Hang-eng_Latn": {
      "bleu": {"mean": 32.1, "std": 0.3, "runs": [31.8, 32.3, 32.2]},
      "comet": {"mean": 0.851, "std": 0.002},
      "latency_p50_ms": {"mean": 45.2, "std": 1.1},
      "peak_memory_mb": 1842
    }
  }
}
```

## Rules
- All measurements must be repeated at least 3 times, recording mean/std
- Include warm-up runs (first 10 sentences excluded from measurement)
- Result JSON must include hardware info and config hash
- Auto-verify same hardware when comparing with previous results
