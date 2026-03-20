# src/quantization/ — Quantization Module

## Role
Provides various quantization methods through a unified interface.
This is one of the core research areas of the project.

## Architecture

### Abstract Base Class (base.py)
```python
from abc import ABC, abstractmethod

class BaseQuantizer(ABC):
    """Base class for all quantization methods."""

    @abstractmethod
    def quantize(self, model, config: dict) -> Any:
        """Quantize the model."""
        ...

    @abstractmethod
    def get_model_size_mb(self, quantized_model) -> float:
        """Return the disk size (MB) of the quantized model."""
        ...

    @abstractmethod
    def supported_bits(self) -> list[int]:
        """Return list of supported bit widths. e.g., [4, 8]"""
        ...
```

### Implementations (methods/)
| File | Method | Description |
|------|--------|-------------|
| `gptq.py` | GPTQ | Post-training 4-bit, requires calibration data |
| `awq.py` | AWQ | Activation-aware 4-bit, preserves important weights |
| `gguf.py` | GGUF | llama.cpp format, various quantization levels (Q2_K ~ Q8_0) |

## Implementation Rules
- Adding a new quantization method: inherit `BaseQuantizer` → create file in `methods/` → register in `__init__.py`
- Always log model size before and after quantization
- Methods requiring calibration data must accept data path and sample count from `configs/`
- Quantized outputs are saved to `data/models/{model_name}_{method}_{bits}/`

## Phase Priorities
- Phase 1: Dynamic INT8 (PyTorch built-in), CTranslate2 INT8
- Phase 2: GPTQ 4-bit, AWQ 4-bit
- Phase 3: GGUF conversion (Q4_K_M, Q5_K_M), 2-bit experiments
- Phase 4: Custom quantization research (mixed-precision, per-layer optimal bit search)
