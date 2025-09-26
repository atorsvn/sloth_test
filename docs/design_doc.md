# Design Document

## Goals

- Provide a reproducible automation layer that mirrors the workflow described in `agents.md`.
- Expose the tooling as both Python APIs and a CLI for easy integration with orchestration frameworks.
- Keep the implementation lightweight so it can run in dry-run mode on CPU-only systems while still mapping to the real GPU workflow.

## Architecture Overview

```
CLI / Python API
        |
        v
 GemmaAgent (pipeline orchestrator)
        |
        +--> CommandRunner executes shell commands
        |
        +--> Configuration dataclasses hold tool parameters
```

1. **Command Preparation:** Each agent tool creates a command list compatible with shell execution. Parameters are validated and defaulted according to the expected behavior of Unsloth and the GGUF conversion utilities.
2. **Execution:** The `CommandRunner` dispatches commands. When `dry_run=True`, the runner simply logs the command for auditing purposes.
3. **Sequencing:** The `run_pipeline` method executes the `prepare_dataset → finetune_unsloth → export_gguf → run_eval` sequence, providing a single entry-point for end-to-end runs.

## Configuration Strategy

- Command options follow snake_case naming to align with CLI conventions used by Unsloth.
- Configuration dataclasses make the API explicit and allow future serialization to YAML or JSON.
- Optional values (e.g., evaluation dataset) are only appended when provided to avoid invalid command invocations.

## Error Handling

- Before executing a command, the runner checks whether the binary exists on the `PATH` unless `dry_run` is active.
- Exceptions are allowed to propagate; downstream orchestrators can catch and retry as needed.

## Extensibility

- New tools can be added by creating additional dataclasses and methods on `GemmaAgent`.
- The CLI uses sub-parsers, so registering more commands requires minimal changes.
- Configuration parsing can be extended to load YAML files using `PyYAML` if desired.
