# data/ — Data Directory

## Role
Manages test datasets and experiment results.
Training data is NOT stored in this directory (size concerns).

## Structure
```
data/
├── test_sets/          → Evaluation datasets
│   ├── flores200/      → FLORES-200 test set
│   └── custom/         → Custom test sets
├── results/            → Experiment results
│   ├── nllb_optimization/
│   │   ├── config.yaml        (experiment config copy)
│   │   ├── baseline.json      (baseline measurement)
│   │   └── int8_results.json  (quantization results)
│   └── ...
└── models/             → Converted/quantized models (excluded from git)
    └── nllb-200-distilled-600M-int8/
```

## .gitignore Rules
- `data/models/` → excluded from git (large files)
- `data/test_sets/flores200/` → excluded from git (replaced by download script)
- `data/results/` → included in git (experiment reproducibility)
- `data/test_sets/custom/` → included in git

## Rules
- Large data is not included in git; replaced by download scripts in `scripts/`
- Experiment result JSONs must be included in git (research history tracking)
- Result files must never be manually modified
