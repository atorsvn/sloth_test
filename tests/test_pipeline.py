from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gemma_agent.pipeline import (
    CommandRunner,
    EvalConfig,
    ExportConfig,
    FinetuneConfig,
    GemmaAgent,
    PipelineConfig,
    PrepareDatasetConfig,
)


class RecordingRunner(CommandRunner):
    def __init__(self) -> None:
        super().__init__(check_path=False)
        self.commands: List[Sequence[str]] = []

    def run(self, command: Sequence[str], *, dry_run: bool = False, env=None, cwd=None):  # type: ignore[override]
        self.commands.append(tuple(command))
        return super().run(command, dry_run=True, env=env, cwd=cwd)


def test_prepare_dataset_command():
    runner = RecordingRunner()
    agent = GemmaAgent(runner)
    config = PrepareDatasetConfig(source="huggingface://dataset", output_dir="data/out", split="train")

    agent.prepare_dataset(config, dry_run=True)

    assert runner.commands[0][0:2] == ("unsloth", "prepare-dataset")


def test_finetune_command_includes_quant():
    runner = RecordingRunner()
    agent = GemmaAgent(runner)
    config = FinetuneConfig(
        base_model="google/gemma-3-27b",
        dataset="data/out",
        epochs=3,
        learning_rate=2e-5,
        lora_r=16,
        quant="q8_0",
    )

    agent.finetune_unsloth(config, dry_run=True)

    command = runner.commands[0]
    assert "--quant" in command
    assert command[command.index("--quant") + 1] == "q8_0"


def test_export_command_uses_converter_env(monkeypatch):
    monkeypatch.setenv("GGUF_CONVERTER_BIN", "custom-convert")

    config = ExportConfig(checkpoint_dir="runs/latest", quantization="q4_k_m", output_file="model.gguf")
    command = config.build_command()

    assert command[0] == "custom-convert"


def test_pipeline_executes_all_steps():
    runner = RecordingRunner()
    agent = GemmaAgent(runner)

    pipeline = PipelineConfig(
        dataset=PrepareDatasetConfig(source="src", output_dir="out"),
        finetune=FinetuneConfig(
            base_model="base",
            dataset="out",
            epochs=1,
            learning_rate=1e-5,
            lora_r=4,
        ),
        export=ExportConfig(checkpoint_dir="out", quantization="q4_k_m", output_file="model.gguf"),
        evaluation=EvalConfig(gguf_file="model.gguf", eval_set="hellaswag"),
    )

    list(agent.run_pipeline(pipeline, dry_run=True))

    recorded = [cmd[1] for cmd in runner.commands]
    assert recorded == [
        "prepare-dataset",
        "finetune",
        "--checkpoint",
        "evaluate",
    ]


def test_pipeline_config_from_dict():
    data = {
        "dataset": {"source": "src", "output_dir": "out"},
        "finetune": {
            "base_model": "base",
            "dataset": "out",
            "epochs": 1,
            "learning_rate": 1e-5,
            "lora_r": 8,
        },
        "export": {
            "checkpoint_dir": "out",
            "quantization": "q4_k_m",
            "output_file": "model.gguf",
        },
        "evaluation": {"gguf_file": "model.gguf", "eval_set": "mmlu"},
    }

    config = PipelineConfig.from_dict(data)

    assert config.dataset.source == "src"
    assert config.finetune.lora_r == 8
    assert config.export.quantization == "q4_k_m"
    assert config.evaluation is not None
