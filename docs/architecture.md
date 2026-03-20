# ALM — Architecture

## Overview

ALM follows a modular architecture where each component has a single responsibility
and communicates through well-defined interfaces.

## Module Dependency Graph

```
configs/ (YAML)
    ↓
src/models/          ← Model loading (Transformers, CTranslate2, ONNX)
    ↓
src/quantization/    ← Quantization methods (INT8, GPTQ, AWQ, GGUF)
    ↓
src/benchmark/       ← Performance measurement (BLEU, latency, memory)
    ↓
src/export/          ← On-device format conversion
    ↓
experiments/         ← Experiment scripts combining the above
    ↓
data/results/        ← Structured JSON results
```

**Dependency rule**: arrows point in the direction of dependency.
Reverse imports are prohibited. `src/models/` must not import from `src/benchmark/`.

## Key Design Decisions

1. **Config-driven experiments**: All parameters come from YAML files, never hardcoded.
2. **ABC interfaces**: Each module family (quantizers, exporters) uses abstract base classes
   so new methods can be added without modifying existing code.
3. **Framework-agnostic ModelLoader**: A single `ModelLoader` class handles all frameworks
   (Transformers, CTranslate2, ONNX) through a registry pattern.
4. **Reproducible results**: Every result JSON includes the config hash and hardware info.
