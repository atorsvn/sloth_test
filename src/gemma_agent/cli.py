"""Command-line interface for the Gemma fine-tuning automation agent."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .pipeline import (
    CommandRunner,
    EvalConfig,
    ExportConfig,
    FinetuneConfig,
    GemmaAgent,
    PipelineConfig,
    PrepareDatasetConfig,
)

logging.basicConfig(level=logging.INFO)


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return ivalue


def _positive_float(value: str) -> float:
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return fvalue


def _load_pipeline_from_file(path: Path) -> PipelineConfig:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Pipeline configuration file must contain a JSON object")
    return PipelineConfig.from_dict(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Log commands without executing them")
    parser.add_argument("--cwd", type=Path, default=None, help="Working directory for commands")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare-dataset
    prepare_parser = subparsers.add_parser("prepare", help="Prepare and shard a dataset")
    prepare_parser.add_argument("--source", required=True, help="Dataset source (local path or huggingface:// URI)")
    prepare_parser.add_argument("--output-dir", required=True, help="Directory to place processed shards")
    prepare_parser.add_argument("--split", default="train", help="Dataset split to use")

    # finetune
    finetune_parser = subparsers.add_parser("finetune", help="Run Unsloth fine-tuning")
    finetune_parser.add_argument("--base-model", required=True, help="Base model identifier")
    finetune_parser.add_argument("--dataset", required=True, help="Prepared dataset directory")
    finetune_parser.add_argument("--epochs", required=True, type=_positive_int, help="Number of epochs")
    finetune_parser.add_argument("--learning-rate", required=True, type=_positive_float, help="Learning rate")
    finetune_parser.add_argument("--lora-r", required=True, type=_positive_int, help="LoRA rank")
    finetune_parser.add_argument("--quant", help="Quantization preset (e.g., q8_0)")
    finetune_parser.add_argument("--gradient-accumulation-steps", type=_positive_int, help="Gradient accumulation steps")

    # export
    export_parser = subparsers.add_parser("export", help="Export a checkpoint to GGUF")
    export_parser.add_argument("--checkpoint-dir", required=True, help="Unsloth checkpoint directory")
    export_parser.add_argument("--quantization", required=True, help="GGUF quantization target (e.g., q4_k_m)")
    export_parser.add_argument("--output", required=True, help="Destination GGUF file")
    export_parser.add_argument("--converter-bin", default=None, help="Custom converter binary")

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation on a GGUF model")
    eval_parser.add_argument("--model", required=True, help="Path to the GGUF model")
    eval_parser.add_argument("--dataset", required=True, help="Evaluation dataset identifier")
    eval_parser.add_argument("--metric", action="append", default=[], help="Additional metrics to report")

    # run pipeline
    run_parser = subparsers.add_parser("run", help="Execute the full pipeline")
    run_parser.add_argument("--config", type=Path, help="Path to a JSON pipeline configuration file")
    run_parser.add_argument("--dataset-source", help="Dataset source")
    run_parser.add_argument("--dataset-output", help="Dataset output directory")
    run_parser.add_argument("--dataset-split", default="train", help="Dataset split")
    run_parser.add_argument("--base-model", help="Base model identifier")
    run_parser.add_argument("--dataset", dest="dataset_path", help="Prepared dataset directory")
    run_parser.add_argument("--epochs", type=_positive_int, help="Epochs")
    run_parser.add_argument("--learning-rate", type=_positive_float, help="Learning rate")
    run_parser.add_argument("--lora-r", type=_positive_int, help="LoRA rank")
    run_parser.add_argument("--quant", help="Fine-tuning quantization mode")
    run_parser.add_argument("--export-quant", help="GGUF quantization preset")
    run_parser.add_argument("--export-output", help="GGUF output file")
    run_parser.add_argument("--checkpoint-dir", help="Checkpoint directory (defaults to dataset output)")
    run_parser.add_argument("--eval-dataset", help="Evaluation dataset identifier")
    run_parser.add_argument("--eval-metric", action="append", default=[], help="Evaluation metrics")

    return parser


def _prepare_agent(args: argparse.Namespace) -> GemmaAgent:
    runner = CommandRunner()
    agent = GemmaAgent(runner)
    return agent


def _prepare_config(args: argparse.Namespace) -> PipelineConfig:
    if args.config:
        return _load_pipeline_from_file(args.config)

    required_fields = {
        "dataset_source": args.dataset_source,
        "dataset_output": args.dataset_output,
        "base_model": args.base_model,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "export_quant": args.export_quant,
        "export_output": args.export_output,
    }
    missing = [key for key, value in required_fields.items() if value is None]
    if missing:
        raise ValueError(f"Missing required arguments for pipeline run: {', '.join(missing)}")

    dataset_config = PrepareDatasetConfig(
        source=args.dataset_source,
        output_dir=args.dataset_output,
        split=args.dataset_split,
    )

    finetune_dataset = args.dataset_path or args.dataset_output
    finetune_config = FinetuneConfig(
        base_model=args.base_model,
        dataset=finetune_dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        quant=args.quant,
    )

    checkpoint_dir = args.checkpoint_dir or finetune_dataset
    export_config = ExportConfig(
        checkpoint_dir=checkpoint_dir,
        quantization=args.export_quant,
        output_file=args.export_output,
    )

    eval_config: Optional[EvalConfig] = None
    if args.eval_dataset:
        eval_config = EvalConfig(
            gguf_file=args.export_output,
            eval_set=args.eval_dataset,
            metrics=args.eval_metric,
        )

    return PipelineConfig(dataset=dataset_config, finetune=finetune_config, export=export_config, evaluation=eval_config)


def main(argv: Optional[Any] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    agent = _prepare_agent(args)
    dry_run = bool(args.dry_run)
    cwd = args.cwd
    env: Optional[Dict[str, str]] = None
    cwd_str = str(cwd) if cwd else None

    if args.command == "prepare":
        config = PrepareDatasetConfig(source=args.source, output_dir=args.output_dir, split=args.split)
        agent.prepare_dataset(config, dry_run=dry_run, env=env, cwd=cwd_str)
    elif args.command == "finetune":
        config = FinetuneConfig(
            base_model=args.base_model,
            dataset=args.dataset,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            quant=args.quant,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        agent.finetune_unsloth(config, dry_run=dry_run, env=env, cwd=cwd_str)
    elif args.command == "export":
        converter = args.converter_bin or ExportConfig(checkpoint_dir="", quantization="", output_file="").converter_bin
        config = ExportConfig(
            checkpoint_dir=args.checkpoint_dir,
            quantization=args.quantization,
            output_file=args.output,
            converter_bin=converter,
        )
        agent.export_gguf(config, dry_run=dry_run, env=env, cwd=cwd_str)
    elif args.command == "evaluate":
        config = EvalConfig(gguf_file=args.model, eval_set=args.dataset, metrics=args.metric)
        agent.run_eval(config, dry_run=dry_run, env=env, cwd=cwd_str)
    elif args.command == "run":
        pipeline_config = _prepare_config(args)
        for _ in agent.run_pipeline(pipeline_config, dry_run=dry_run, env=env, cwd=cwd_str):
            pass
    else:
        parser.error(f"Unknown command: {args.command}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
