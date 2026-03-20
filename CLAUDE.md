# MAGA (Make AI Great Again) — Project Guide for Claude Code

## Project Overview

A research project for compressing commercial AI models to on-device scale.
The ultimate goal is to enable researchers to conveniently use AI models
on any environment — mobile, laptop, edge devices, etc.

## Core Principles

### 1. Reproducibility
- All experiments must be fully reproducible via YAML config files in `configs/`
- Always record random seeds, model versions, and hyperparameters
- Store experiment results as structured JSON/CSV in `data/results/`

### 2. Modularity
- Each quantization method, benchmark metric, and export format is an independent module
- Minimize changes to existing code when adding new models/techniques
- All modules in `src/` must define clear interfaces (ABC)

### 3. Measure First
- Always measure baseline performance before optimization
- Compare pre/post quantization with at least 3 metrics: quality (BLEU/COMET), speed (tokens/sec), memory (peak RSS)
- Minimum 3 repeated measurements for statistical significance

### 4. Progressive Complexity
- Phase 1: Quantization & benchmarking with existing tools (llama.cpp, CTranslate2)
- Phase 2: Translation model-specific optimization (NLLB)
- Phase 3: Custom compression research (pruning, distillation)
- Phase 4: Extend to general-purpose models (LLM, multimodal)

## Tech Stack
- **Language**: Python 3.11+
- **Core Libraries**: PyTorch, Transformers, CTranslate2, ONNX Runtime, llama.cpp (Python bindings)
- **Benchmarking**: sacrebleu, comet, py-spy, memory_profiler
- **Experiment Management**: YAML configs + structured result logging (MLflow considered from Phase 2)
- **Testing**: pytest
- **Documentation**: Markdown + Jupyter Notebooks

## Coding Conventions
- Type hints required
- Docstrings: Google style
- Functions follow single responsibility principle
- Error handling: specific exceptions, never bare except
- Logging: use Python logging module (no print)
- Path handling: use pathlib.Path

## Branch Strategy
- `main`: stable version
- `dev`: development branch
- `exp/{experiment_name}`: experiment branches (e.g., `exp/nllb-int4-pruning`)

## Commit Message Format
```
[category] description

Categories:
- feat: new feature
- fix: bug fix
- exp: experiment-related
- docs: documentation
- refactor: refactoring
- bench: benchmark-related
- chore: build/config
```

## Current Progress
- [ ] Phase 1: NLLB-200 Distilled 600M quantization benchmark
- [ ] Phase 2: Translation-specific optimization
- [ ] Phase 3: Custom compression research
- [ ] Phase 4: General-purpose model expansion
