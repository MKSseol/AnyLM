# ALM — AnyLM

> Any model, any device, anywhere.

## Vision

ALM is an open-source research project focused on compressing commercial-grade AI models
to run on resource-constrained devices — smartphones, laptops, and edge hardware
with as little as 4 GB of RAM.

The first sub-project targets **Meta's NLLB-200 Distilled 600M** translation model,
optimizing it to run efficiently on smartphone-grade hardware.

## Current Progress

- [x] Project structure design
- [ ] Phase 1: NLLB-200 quantization benchmark (INT8)
- [ ] Phase 2: Translation-specific optimization (GPTQ, AWQ)
- [ ] Phase 3: Custom compression (pruning, distillation)
- [ ] Phase 4: General-purpose model expansion

## Quick Start

### Prerequisites
- Python 3.11+
- ~4 GB disk space for models

### Installation

```bash
# Clone the repository
git clone https://github.com/MKSseol/ALM.git
cd ALM

# Setup environment (CPU only)
bash scripts/setup_env.sh

# Or with GPU support
bash scripts/setup_env.sh --gpu
```

### Download the model

```bash
source .venv/bin/activate
python scripts/download_model.py --model facebook/nllb-200-distilled-600M
```

### Run baseline benchmark

```bash
python experiments/nllb_optimization/01_baseline.py --config configs/nllb_baseline.yaml
```

### Run INT8 quantization experiment

```bash
python experiments/nllb_optimization/02_quantize.py --config configs/nllb_int8.yaml
```

### Run full pipeline

```bash
python scripts/run_experiment.py --config configs/nllb_int8.yaml --steps all
```

## Project Structure

```
ALM/
├── configs/           # Experiment YAML configurations
├── src/
│   ├── models/        # Unified model loading (Transformers, CTranslate2, ONNX)
│   ├── quantization/  # Quantization methods (INT8, GPTQ, AWQ, GGUF)
│   ├── benchmark/     # Performance measurement (BLEU, latency, memory)
│   ├── distillation/  # Knowledge Distillation pipeline
│   ├── analysis/      # Model analysis (MIA, embedding, data profiling)
│   └── export/        # On-device format conversion (CTranslate2, ONNX, TFLite)
├── experiments/       # Experiment scripts (run in numbered order)
├── data/              # Test sets and experiment results
├── notebooks/         # Exploratory Jupyter notebooks
├── scripts/           # Utility scripts (setup, download, run)
├── docs/              # Research documentation and findings
└── tests/             # Unit and integration tests
```

## Research Results

Phase 1 results will be documented in `docs/findings/` after experiments are complete.

## Testing

```bash
# Run all tests (excluding slow tests that require model downloads)
pytest -m "not slow"

# Run all tests including slow ones
pytest
```

## Contributing

1. Create an experiment branch: `git checkout -b exp/your-experiment-name`
2. Follow the coding conventions in `CLAUDE.md`
3. Ensure all tests pass: `pytest`
4. Document findings in `docs/findings/`

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

Minkyu Seol ([@MKSseol](https://github.com/MKSseol))
