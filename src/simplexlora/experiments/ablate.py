"""Run SimplexLoRA calibration and measure held-out layer ablations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import TrainingArguments

from simplexlora.calibration import (
    StopReason,
    build_run_metadata,
    create_calibration_session,
    validate_calibration_arguments,
)
from simplexlora.calibration.ablation import (
    LayerAblationEvaluator,
    render_ablation_markdown,
    write_ablation_artifacts,
)
from simplexlora.calibration.recorder import build_registry_for_model
from simplexlora.calibration.trainer import SimplexCalibrationTrainer
from simplexlora.experiments.decompile import (
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    CausalLMCollator,
    build_optimizer,
    find_cached_snapshot,
    load_model_and_tokenizer,
    select_dataset,
    tokenize_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output-dir", default="outputs/qwen-simplex-ablation")
    parser.add_argument("--ablation-output-dir", default=None)
    parser.add_argument("--max-train-samples", type=int, default=1_000)
    parser.add_argument("--max-validation-samples", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--weight-lr", type=float, default=5e-3)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--fat-step", type=int, default=10)
    parser.add_argument("--max-fat-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--ablation-max-samples",
        type=int,
        default=64,
        help="maximum held-out examples used for each layer ablation",
    )
    parser.add_argument("--ablation-batch-size", type=int, default=1)
    parser.add_argument("--ablation-top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.eval_steps = 0
    args.save_steps = 999_999
    args.num_train_epochs = 1.0
    args.calibration_only = True
    args.calibration_no_artifacts = False
    validate_calibration_arguments(args)

    if args.ablation_max_samples < 1:
        raise ValueError(f"--ablation-max-samples must be positive, got {args.ablation_max_samples}")
    if args.ablation_batch_size < 1:
        raise ValueError(f"--ablation-batch-size must be positive, got {args.ablation_batch_size}")
    if args.ablation_top_k < 1:
        raise ValueError(f"--ablation-top-k must be positive, got {args.ablation_top_k}")
    if args.max_validation_samples < 1:
        raise ValueError("--max-validation-samples must be positive for layer ablation")

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset_path) if args.dataset_path else find_cached_snapshot(DEFAULT_DATASET)
    train_raw, validation_raw = select_dataset(
        dataset_path,
        args.max_train_samples,
        args.max_validation_samples,
        args.seed,
    )
    model, tokenizer = load_model_and_tokenizer(args)
    train_dataset = tokenize_dataset(train_raw, tokenizer, args.max_seq_length)
    validation_dataset = tokenize_dataset(validation_raw, tokenizer, args.max_seq_length)
    if len(train_dataset) == 0:
        raise RuntimeError("No training examples survived tokenization")
    if len(validation_dataset) == 0:
        raise RuntimeError("No validation examples survived tokenization")

    print(f"Train examples after filtering: {len(train_dataset)}")
    print(f"Ablation examples after filtering: {len(validation_dataset)}")

    optimizer = build_optimizer(model, args)
    total_steps = args.fat_step * args.max_fat_steps
    warmup_steps = min(max(1, int(total_steps * 0.03)), max(1, total_steps // 10))
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=total_steps,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        eval_strategy="no",
        save_strategy="no",
        train_sampling_strategy="group_by_length",
        length_column_name="length",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=[],
        run_name="qwen-simplex-layer-ablation",
        seed=args.seed,
        data_seed=args.seed,
    )

    data_collator = CausalLMCollator(tokenizer.pad_token_id)
    registry = build_registry_for_model(model, "default")
    evaluator = LayerAblationEvaluator(
        model=model,
        dataset=validation_dataset,
        data_collator=data_collator,
        registry=registry,
        adapter_name="default",
        max_samples=args.ablation_max_samples,
        batch_size=args.ablation_batch_size,
        top_k=args.ablation_top_k,
    )
    run_metadata = build_run_metadata(
        model,
        adapter_name="default",
        args=args,
        data={
            "dataset_path": str(dataset_path),
            "train_examples": len(train_dataset),
            "validation_examples": len(validation_dataset),
            "max_seq_length": args.max_seq_length,
            "ablation_max_samples": args.ablation_max_samples,
        },
        device=next(model.parameters()).device,
    )
    trainer = SimplexCalibrationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        processing_class=tokenizer,
        calibration_optimizer=optimizer,
    )
    session = create_calibration_session(
        model=model,
        optimizer=optimizer,
        args=args,
        run_metadata=run_metadata,
        pre_final_rank_update=evaluator.evaluate,
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        session.close(interrupted=True)
        raise
    except BaseException:
        session.close(failed=True)
        raise

    session.finalize()
    session.close(stop_reason=StopReason.CALIBRATION_ONLY)
    if evaluator.result is None:
        raise RuntimeError("training completed without producing a layer ablation result")

    output_dir = Path(args.output_dir)
    ablation_output_dir = Path(args.ablation_output_dir) if args.ablation_output_dir else output_dir / "ablation"
    json_path, markdown_path = write_ablation_artifacts(evaluator.result, ablation_output_dir)
    result = evaluator.result
    summary = {
        "train_examples": len(train_dataset),
        "ablation_examples_available": len(validation_dataset),
        "ablation_examples_used": result.sample_count,
        "baseline_loss": result.baseline_loss,
        "importance_delta_spearman": result.importance_delta_spearman,
        "gradient_delta_spearman": result.gradient_delta_spearman,
        "top_k": result.top_k,
        "top_k_overlap": result.top_k_overlap,
        "ablation_json": str(json_path),
        "ablation_report": str(markdown_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(render_ablation_markdown(result))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
