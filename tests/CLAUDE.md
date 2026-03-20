# tests/ — Tests

## Role
Unit tests and integration tests for all modules in `src/`.

## Rules
- Test filenames: `test_{module_name}.py`
- Function names: `test_{feature}_{scenario}()` (e.g., `test_quantize_int8_nllb()`)
- All tests must pass (`pytest tests/`) before any PR
- Tests requiring model downloads: use `@pytest.mark.slow` decorator
- CI excludes slow tests: `pytest -m "not slow"`

## Test Categories
- **Unit tests**: Individual function/class tests (using mocks)
- **Integration tests**: Full pipeline flow tests (using small dummy models)
- **Regression tests**: Verify pre/post quantization BLEU difference is within threshold

## conftest.py Fixtures
- `tiny_model`: Small model for testing (not actual NLLB, but a small seq2seq)
- `sample_config`: Test configuration dictionary
- `sample_sentences`: 10 ko↔en sentence pairs
- `tmp_results_dir`: Temporary results storage directory
