"""Pipeline implementation for the Gemma fine-tuning automation agent."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence

logger = logging.getLogger(__name__)


class CommandNotFoundError(RuntimeError):
    """Raised when a required binary is missing from the system path."""


class CommandRunner:
    """Execute shell commands with optional dry-run support."""

    def __init__(self, *, check_path: bool = True) -> None:
        self.check_path = check_path

    def run(
        self,
        command: Sequence[str],
        *,
        dry_run: bool = False,
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        """Execute *command* using :func:`subprocess.run`.

        Parameters
        ----------
        command:
            Iterable of command arguments. The first argument must be the binary
            name. The runner checks that the binary exists on ``PATH`` unless
            ``dry_run`` is set to ``True``.
        dry_run:
            When ``True`` the command is logged but not executed.
        env:
            Optional environment overrides forwarded to ``subprocess.run``.
        cwd:
            Working directory for the command.
        """

        if not command:
            raise ValueError("command must contain at least one element")

        binary = command[0]
        if not dry_run and self.check_path and shutil.which(binary) is None:
            raise CommandNotFoundError(f"Required command '{binary}' is not available on PATH")

        formatted = " ".join(shlex.quote(arg) for arg in command)
        logger.info("Executing command: %s", formatted)

        if dry_run:
            return subprocess.CompletedProcess(args=command, returncode=0)  # type: ignore[arg-type]

        completed = subprocess.run(
            command,
            check=True,
            env=_merge_environments(env),
            cwd=cwd,
            text=True,
            capture_output=False,
        )
        return completed  # type: ignore[return-value]


def _merge_environments(overrides: Optional[Mapping[str, str]]) -> Optional[MutableMapping[str, str]]:
    if overrides is None:
        return None
    merged: MutableMapping[str, str] = dict(os.environ)
    merged.update(overrides)
    return merged


@dataclass(slots=True)
class PrepareDatasetConfig:
    source: str
    output_dir: str
    split: str = "train"
    extra_args: Sequence[str] = field(default_factory=list)

    def build_command(self) -> List[str]:
        command = [
            os.environ.get("UNSLOTH_BIN", "unsloth"),
            "prepare-dataset",
            "--source",
            self.source,
            "--output-dir",
            self.output_dir,
            "--split",
            self.split,
        ]
        command.extend(self.extra_args)
        return command


@dataclass(slots=True)
class FinetuneConfig:
    base_model: str
    dataset: str
    epochs: int
    learning_rate: float
    lora_r: int
    quant: Optional[str] = None
    gradient_accumulation_steps: Optional[int] = None
    extra_args: Sequence[str] = field(default_factory=list)

    def build_command(self) -> List[str]:
        command = [
            os.environ.get("UNSLOTH_BIN", "unsloth"),
            "finetune",
            "--base-model",
            self.base_model,
            "--dataset",
            self.dataset,
            "--epochs",
            str(self.epochs),
            "--learning-rate",
            str(self.learning_rate),
            "--lora-r",
            str(self.lora_r),
        ]
        if self.quant:
            command.extend(["--quant", self.quant])
        if self.gradient_accumulation_steps:
            command.extend(["--gradient-accumulation-steps", str(self.gradient_accumulation_steps)])
        command.extend(self.extra_args)
        return command


@dataclass(slots=True)
class ExportConfig:
    checkpoint_dir: str
    quantization: str
    output_file: str
    converter_bin: str = field(default_factory=lambda: os.environ.get("GGUF_CONVERTER_BIN", "convert-gguf"))
    extra_args: Sequence[str] = field(default_factory=list)

    def build_command(self) -> List[str]:
        command = [
            self.converter_bin,
            "--checkpoint",
            self.checkpoint_dir,
            "--quantization",
            self.quantization,
            "--output",
            self.output_file,
        ]
        command.extend(self.extra_args)
        return command


@dataclass(slots=True)
class EvalConfig:
    gguf_file: str
    eval_set: str
    metrics: Sequence[str] = field(default_factory=list)
    extra_args: Sequence[str] = field(default_factory=list)

    def build_command(self) -> List[str]:
        command = [
            os.environ.get("UNSLOTH_BIN", "unsloth"),
            "evaluate",
            "--model",
            self.gguf_file,
            "--dataset",
            self.eval_set,
        ]
        for metric in self.metrics:
            command.extend(["--metric", metric])
        command.extend(self.extra_args)
        return command


@dataclass(slots=True)
class PipelineConfig:
    dataset: PrepareDatasetConfig
    finetune: FinetuneConfig
    export: ExportConfig
    evaluation: Optional[EvalConfig] = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PipelineConfig":
        """Create a :class:`PipelineConfig` from a nested mapping."""

        def _pop_section(key: str) -> Mapping[str, object]:
            section = payload.get(key)
            if not isinstance(section, Mapping):
                raise ValueError(f"Missing or invalid '{key}' section in pipeline configuration")
            return section

        dataset_cfg = PrepareDatasetConfig(**_pop_section("dataset"))
        finetune_cfg = FinetuneConfig(**_pop_section("finetune"))
        export_cfg = ExportConfig(**_pop_section("export"))

        evaluation_cfg: Optional[EvalConfig] = None
        if "evaluation" in payload and isinstance(payload["evaluation"], Mapping):
            evaluation_cfg = EvalConfig(**payload["evaluation"])  # type: ignore[arg-type]

        return cls(dataset=dataset_cfg, finetune=finetune_cfg, export=export_cfg, evaluation=evaluation_cfg)


class GemmaAgent:
    """High-level orchestrator that wires the agent's tools together."""

    def __init__(self, runner: Optional[CommandRunner] = None) -> None:
        self.runner = runner or CommandRunner()

    def prepare_dataset(
        self,
        config: PrepareDatasetConfig,
        *,
        dry_run: bool = False,
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        command = config.build_command()
        return self.runner.run(command, dry_run=dry_run, env=env, cwd=cwd)

    def finetune_unsloth(
        self,
        config: FinetuneConfig,
        *,
        dry_run: bool = False,
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        command = config.build_command()
        return self.runner.run(command, dry_run=dry_run, env=env, cwd=cwd)

    def export_gguf(
        self,
        config: ExportConfig,
        *,
        dry_run: bool = False,
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        command = config.build_command()
        return self.runner.run(command, dry_run=dry_run, env=env, cwd=cwd)

    def run_eval(
        self,
        config: EvalConfig,
        *,
        dry_run: bool = False,
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        command = config.build_command()
        return self.runner.run(command, dry_run=dry_run, env=env, cwd=cwd)

    def run_pipeline(
        self,
        config: PipelineConfig,
        *,
        dry_run: bool = False,
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> Iterable[subprocess.CompletedProcess[str]]:
        """Execute the full pipeline in the order described by ``agents.md``."""

        yield self.prepare_dataset(config.dataset, dry_run=dry_run, env=env, cwd=cwd)
        yield self.finetune_unsloth(config.finetune, dry_run=dry_run, env=env, cwd=cwd)
        yield self.export_gguf(config.export, dry_run=dry_run, env=env, cwd=cwd)
        if config.evaluation is not None:
            yield self.run_eval(config.evaluation, dry_run=dry_run, env=env, cwd=cwd)


__all__ = [
    "CommandNotFoundError",
    "CommandRunner",
    "EvalConfig",
    "ExportConfig",
    "FinetuneConfig",
    "GemmaAgent",
    "PipelineConfig",
    "PrepareDatasetConfig",
]
