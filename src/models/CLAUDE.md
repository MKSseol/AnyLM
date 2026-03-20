# src/models/ — Model Loading and Management

## Role
Load and manage models from various frameworks (Transformers, CTranslate2, ONNX, etc.)
through a unified interface.

## Core Class

### ModelLoader (loader.py)
```python
class ModelLoader:
    """Unified model loader.

    Supported frameworks:
    - transformers: HuggingFace models
    - ctranslate2: CTranslate2 converted models
    - onnx: ONNX Runtime models

    Usage:
        loader = ModelLoader(config)
        model, tokenizer = loader.load()
    """
```

## Implementation Rules
- Model download path: `~/.cache/maga/models/` (overridable via `MAGA_MODEL_CACHE` env var)
- Verify SHA256 checksum on model load
- Provide memory estimation method: `estimate_memory(model_name, quantization) -> int` (in MB)
- Auto-detect GPU/CPU, overridable via config

## Adding a New Framework
1. Register in `_FRAMEWORK_REGISTRY` in `loader.py`
2. Implement `load_{framework}()` method
3. Add tests in `tests/test_models.py`
