# configs/ — Experiment Configuration Directory

## Role
Manages all experiment hyperparameters and settings as YAML files.
Hardcoded settings within code are not permitted.

## YAML File Naming Convention
`{model_name}_{quantization_method}.yaml`
Examples: `nllb_int8.yaml`, `nllb_gptq_4bit.yaml`, `llama_awq_4bit.yaml`

## Required Fields
```yaml
# Experiment metadata
experiment:
  name: "nllb-int8-baseline"
  description: "NLLB-200 600M INT8 quantization benchmark"
  date: "2025-XX-XX"
  author: "minkyu"

# Model settings
model:
  name: "facebook/nllb-200-distilled-600M"
  revision: "main"  # Must be a specific commit hash or tag
  framework: "transformers"  # transformers | ctranslate2 | onnx

# Quantization settings
quantization:
  method: "dynamic_int8"  # none | dynamic_int8 | static_int8 | gptq_4bit | awq_4bit | gguf_q4_k_m
  calibration_samples: 512  # For static quantization
  group_size: 128  # For GPTQ/AWQ

# Benchmark settings
benchmark:
  language_pairs: ["kor_Hang-eng_Latn", "eng_Latn-kor_Hang"]
  test_dataset: "flores200"
  num_samples: 100
  num_repeats: 3  # For statistical significance
  metrics: ["bleu", "comet", "latency_p50", "latency_p99", "peak_memory_mb"]
  batch_sizes: [1, 4, 16]

# Hardware environment
hardware:
  device: "cpu"  # cpu | cuda | mps
  max_memory_mb: 4096
  num_threads: 4

# Seed
seed: 42
```

## Rules
- Experiment scripts must always accept the config file path as an argument
- Config files are copied alongside results in `data/results/{experiment_name}/`
- When changing settings, create a new file (never modify existing files, for history tracking)
