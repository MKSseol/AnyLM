# scripts/ — Utility Scripts

## Role
Environment setup, data download, batch experiment execution, and other utilities.

## Core Files

### setup_env.sh
- Create Python virtual environment + install dependencies
- Check required system packages (cmake, etc.)
- Install CUDA-related packages when GPU is available
- Usage: `bash scripts/setup_env.sh [--gpu]`

### download_model.py
- Download models from HuggingFace + verify checksum
- Usage: `python scripts/download_model.py --model facebook/nllb-200-distilled-600M`

### run_experiment.py
- Batch execution of experiment pipeline
- Usage: `python scripts/run_experiment.py --config configs/nllb_int8.yaml --steps all`

## Rules
- All scripts must support `--help`
- Check required dependencies before execution; if missing, show installation instructions
- Long-running tasks must show progress (tqdm)
