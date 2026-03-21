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
- [x] Phase 1 code: quantization, benchmark, metrics pipeline
- [x] Unit tests (36/36 passing) + integration tests
- [x] FLORES-200 download tooling + sample test data
- [ ] Phase 1 experiment: NLLB-200 quantization benchmark (INT8)
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

### Download data and model

```bash
source .venv/bin/activate

# Download FLORES-200 test data (ko, en, ja)
python scripts/download_flores.py

# Download the NLLB-200 model (~1.2 GB)
python scripts/download_model.py --model facebook/nllb-200-distilled-600M
```

### Run experiments (in order)

```bash
# 1. Baseline FP32 benchmark
python experiments/nllb_optimization/01_baseline.py --config configs/nllb_baseline.yaml

# 2. INT8 quantization + comparison
python experiments/nllb_optimization/02_quantize.py --config configs/nllb_int8.yaml

# 3. Device simulation (memory/thread constraints)
python experiments/nllb_optimization/03_benchmark.py --config configs/nllb_int8.yaml

# 4. Post-quantization analysis
python experiments/nllb_optimization/04_analysis.py --config configs/nllb_int8.yaml
```

### Run full pipeline (all steps)

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
source .venv/bin/activate

# Run all tests (36 unit + integration tests)
pytest tests/ -v

# Run only fast tests (no model downloads)
pytest -m "not slow"
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
