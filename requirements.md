# Requirements

The automation agent assumes the following toolchain is available on the execution host:

- **Python 3.10+** with `pip`, `venv`, and GPU-aware packages (`torch`, `accelerate`).
- **[Unsloth](https://github.com/unslothai/unsloth)** for efficient fine-tuning of Gemma models.
- **bitsandbytes** when training with 4-bit or 8-bit quantization.
- **gguf-convert** (or the conversion utilities bundled with `llama.cpp`) to export Unsloth checkpoints to GGUF.
- **Evaluation suites** such as `lm-eval` or custom scripts for downstream benchmarking.

The Python package declared in `pyproject.toml` depends on `PyYAML` for configuration parsing and optionally `pytest` for local testing.
