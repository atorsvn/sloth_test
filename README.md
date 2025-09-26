# Gemma 3 (270M) Fine-Tuning Agent

This project implements the automation agent described in [`agents.md`](agents.md). It provides a Python-based orchestration layer that can:

1. Prepare and shard training datasets.
2. Launch Unsloth fine-tuning runs for the Gemma 3 270M base model with LoRA/QLoRA.
3. Export the resulting checkpoints to the GGUF format for efficient inference runtimes.
4. Run optional benchmark suites against the exported weights.

The implementation focuses on transparency and reproducibility. All actionable steps are exposed as both Python APIs and a command-line interface, enabling integration with workflow tools such as Airflow, LangChain, or CrewAI.

## Getting Started

1. Create a virtual environment and install the package in editable mode:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

2. Inspect the available commands:

   ```bash
   gemma-agent --help
   ```

   Each sub-command includes `--dry-run` support so you can verify the generated shell commands before executing them on a GPU-enabled machine.

3. Run a full pipeline in dry-run mode:

   ```bash
   gemma-agent run \
     --dataset-source huggingface://unsloth/ultrachat_200k \
     --dataset-output data/ultrachat \
     --dataset-split train \
     --base-model google/gemma-3-27b \
     --epochs 3 \
     --learning-rate 2e-5 \
     --lora-r 16 \
     --quant q8_0 \
     --export-quant q4_k_m \
     --export-output models/gemma3-270m-ultrachat.gguf \
     --dry-run
   ```

   Remove `--dry-run` to execute the commands once all required dependencies (Unsloth, gguf converters, evaluation suites) are available on your machine.

## Documentation

Additional planning and reference material lives in the `docs/` directory:

- [`docs/design_doc.md`](docs/design_doc.md) – workflow overview and execution prerequisites.
- [`docs/tech_implementation.md`](docs/tech_implementation.md) – tool-specific configuration and command templates.
- [`docs/tests.md`](docs/tests.md) – recommended validation checks.
- [`requirements.md`](requirements.md) – high-level dependency notes.

## Testing

Run the unit tests with:

```bash
pytest
```

The included tests exercise the dry-run mode to ensure commands are constructed correctly without requiring the heavyweight training stack.
