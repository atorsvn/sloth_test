# Technical Implementation

## Tool Commands

The agent maps high-level actions to concrete shell commands:

| Tool | Command Template |
|------|------------------|
| `prepare_dataset` | `unsloth prepare-dataset --source {source} --output-dir {output_dir} --split {split}` |
| `finetune_unsloth` | `unsloth finetune --base-model {base_model} --dataset {dataset} --epochs {epochs} --learning-rate {learning_rate} --lora-r {lora_r} [--quant {quant}]` |
| `export_gguf` | `convert-gguf --checkpoint {checkpoint_dir} --quantization {quantization} --output {output_file}` |
| `run_eval` | `unsloth evaluate --model {gguf_file} --dataset {eval_set}` |

The actual command names can be customized through environment variables if different binaries are preferred (e.g., `llama.cpp`'s conversion tools).

## Python Modules

- `gemma_agent.pipeline`
  - Defines dataclasses (`PrepareDatasetConfig`, `FinetuneConfig`, `ExportConfig`, `EvalConfig`, `PipelineConfig`).
  - Implements `GemmaAgent`, which sequences the tools and delegates execution to `CommandRunner`.
- `gemma_agent.cli`
  - Provides an `argparse`-based interface mirroring the four tool commands plus a `run` command for the full pipeline.
  - Supports `--dry-run` for safe validation.

## Command Execution

The `CommandRunner` performs the following steps:

1. Validate the existence of the target binary using `shutil.which` unless `dry_run=True`.
2. Log the command via the standard `logging` module.
3. Execute the command using `subprocess.run(..., check=True)`.

To accommodate GPU execution, callers can pass custom environment variables via the Python API.

## Configuration Files

`PipelineConfig` can be instantiated from a Python dictionary. Future extensions can add YAML/JSON loaders using the packaged `PyYAML` dependency.

## Testing Strategy

Unit tests mock the runner to capture generated commands without launching external processes. This keeps the test suite lightweight and suitable for CI environments without GPUs.
