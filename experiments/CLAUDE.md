# experiments/ — Experiment Scripts

## Role
Scripts that combine `src/` modules to run specific experiments.
Each subdirectory is an independent experiment series.

## Naming Convention
- Directory: `{model_name}_{purpose}/` (e.g., `nllb_optimization/`, `llama_distillation/`)
- Files: `{order}_{step}.py` (e.g., `01_baseline.py`, `02_quantize.py`)
- Running files in order should reproduce the entire experiment

## Common Script Structure
```python
"""
Experiment: NLLB-200 INT8 quantization benchmark
Date: YYYY-MM-DD
Config: configs/nllb_int8.yaml
Depends on: 01_baseline.py completed
"""
import argparse
from src.models import ModelLoader
from src.benchmark import BenchmarkRunner

def main(config_path: str):
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
```

## Rules
- All experiment scripts accept a `--config` argument for the YAML config file
- Experiment results are auto-saved to `data/results/{experiment_name}/`
- Script docstring must specify dependencies
- Long-running experiments must implement checkpointing
