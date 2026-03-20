# src/ — Core Source Code

## Role
All reusable code for the MAGA project lives here.
Experiment scripts (`experiments/`) import and use this package.

## Module Structure
```
src/
├── models/        → Model loading/management
├── quantization/  → Quantization method implementations
├── benchmark/     → Performance measurement
├── distillation/  → Knowledge Distillation
├── analysis/      → Model analysis (MIA, embedding, etc.)
└── export/        → On-device format conversion
```

## Common Rules
1. **Type hints + docstrings required for all public functions**
2. **Define interfaces with ABC** — new methods must conform to existing interfaces
3. **Dependency direction**: models ← quantization ← benchmark (no reverse imports)
4. **Settings are injected externally** — no hardcoded paths, model names, or hyperparameters inside functions/classes
5. **Raise specific exceptions on errors** — `raise ValueError("Expected INT8, got {method}")` pattern
6. **Logging**: use `logger = logging.getLogger(__name__)` pattern
