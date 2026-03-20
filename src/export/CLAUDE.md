# src/export/ — On-Device Export Module

## Role
Convert optimized models to various on-device runtime formats.
The final step of turning "works in the lab" into "runs on actual devices."

## Supported Formats
| File | Format | Target Environment | Priority |
|------|--------|-------------------|----------|
| `ctranslate2_export.py` | CTranslate2 | CPU servers, desktop | Phase 1 |
| `onnx_export.py` | ONNX | General (ONNX Runtime Mobile) | Phase 2 |
| `tflite_export.py` | TensorFlow Lite | Android, edge devices | Phase 3 |

## Planned Additions
- CoreML (iOS) — Phase 3
- ExecuTorch (Meta) — Phase 3
- WebAssembly/WebGPU — Phase 4

## Conversion Pipeline
```
Original model (HuggingFace)
  ↓ quantize via quantization/ module
Quantized model
  ↓ convert via export/ module
Target format model
  ↓ verify via benchmark/ module
Verified model → saved to data/models/{name}/
```

## Rules
- After conversion, always verify output consistency with the original on identical inputs (tolerance: BLEU difference < 0.5)
- Converted model files must include metadata: source model, quantization method, conversion date, BLEU score
- On conversion failure, provide specific error messages (which layer/operation is unsupported)
