#!/usr/bin/env python
"""SimplexLoRA training on LLM4Binary/decompile-bench with Qwen2.5-Coder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from peft import TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    TrainingArguments,
)

from simplexlora import optimizers
from simplexlora.calibration import (
    StopReason,
    add_calibration_arguments,
    build_run_metadata,
    create_calibration_session,
    render_console_summary,
    validate_calibration_arguments,
)
from simplexlora.calibration.trainer import SimplexCalibrationTrainer
from simplexlora.peft_plugin import SimplexLoraLinear, get_simplex_peft_model, make_simplex_lora_config
from simplexlora.tools.optimization_clusters import (
    OPTIMIZATION_LEVELS,
    default_cluster_dir,
    load_optimization_indices,
)
from .arguments import as_tokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_DATASET = "datasets--LLM4Binary--decompile-bench"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument(
        "--optimization-level",
        choices=(*OPTIMIZATION_LEVELS, "all"),
        default="O0",
        help="optimization-level cluster from the derived Decompile-Bench mapping",
    )
    parser.add_argument(
        "--optimization-clusters",
        default=None,
        help="directory containing the optimization cluster manifest and parquet mapping",
    )
    parser.add_argument(
        "--optimization-min-confidence",
        type=float,
        default=0.98,
        help="minimum confidence for classifier-derived optimization cluster rows",
    )
    parser.add_argument("--output-dir", default="outputs/qwen2.5-coder-decompile-simplex")
    parser.add_argument("--max-train-samples", type=int, default=20_000)
    parser.add_argument("--max-validation-samples", type=int, default=1_000)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="explicit scheduler warmup; default is 3%% of planned steps",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--weight-lr", type=float, default=5e-3)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--fat-step", type=int, default=10)
    parser.add_argument("--max-fat-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true")
    add_calibration_arguments(parser)
    return parser.parse_args()


def find_cached_snapshot(repo_dirname: str) -> Path:
    cache_root = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub",
        )
    )
    repo_dir = cache_root / repo_dirname
    ref_file = repo_dir / "refs" / "main"
    if not ref_file.is_file():
        raise FileNotFoundError(
            f"HF cache snapshot not found for {repo_dirname}. "
            "Download it with `hf download LLM4Binary/decompile-bench --type dataset`."
        )

    revision = ref_file.read_text(encoding="utf-8").strip()
    snapshot = repo_dir / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Missing HF snapshot directory: {snapshot}")
    return snapshot


def file_bucket(path: str) -> int:
    digest = hashlib.blake2b(path.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 20


def select_dataset(
    path: Path,
    max_train: int,
    max_validation: int,
    seed: int,
    *,
    optimization_level: str = "O0",
    optimization_clusters: Path | None = None,
    optimization_min_confidence: float = 0.98,
) -> tuple[Dataset, Dataset]:
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, Dataset):
        raise TypeError("Expected a single dataset split, got DatasetDict")
    if optimization_level != "all":
        cluster_dir = optimization_clusters or default_cluster_dir()
        indices = load_optimization_indices(
            cluster_dir,
            level=optimization_level,
            min_confidence=optimization_min_confidence,
            dataset_rows=len(dataset),
        )
        dataset = dataset.select(indices)
        print(f"Optimization cluster {optimization_level}: {len(dataset)} source rows")
    sample_count = min(len(dataset), max_train + max_validation)
    sampled = dataset.shuffle(seed=seed).select(range(sample_count))

    if max_validation <= 0:
        train = sampled
        validation = sampled.select([])
    else:
        validation = sampled.filter(lambda row: file_bucket(row["file"]) == 0)
        train = sampled.filter(lambda row: file_bucket(row["file"]) != 0)
        if len(validation) == 0:
            split = train.train_test_split(test_size=min(1000, sample_count // 20), seed=seed)
            train, validation = split["train"], split["test"]

    train = train.select(range(min(len(train), max_train)))
    validation = validation.select(range(min(len(validation), max_validation)))
    return train, validation


def build_messages(row: dict) -> list[dict[str, str]]:
    assembly = row["asm"].strip()
    source = row["code"].strip()
    return [
        {
            "role": "system",
            "content": (
                "You are an expert C/C++ reverse engineer. "
                "Reconstruct the original function from Linux x86-64 assembly. "
                "Return only source code, without explanations or Markdown fences."
            ),
        },
        {
            "role": "user",
            "content": (f"# This is the assembly code:\n{row['name']}:\n{assembly}\n# What is the source code?\n"),
        },
        {
            "role": "assistant",
            "content": source,
        },
    ]


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> Dataset:
    def encode(row: dict) -> dict:
        messages = build_messages(row)
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Chat template prompt is not a prefix of the full conversation")
        if len(full_ids) > max_seq_length:
            return {"input_ids": [], "labels": [], "length": 0}

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        return {
            "input_ids": full_ids,
            "labels": labels,
            "length": len(full_ids),
        }

    tokenized = dataset.map(
        encode,
        remove_columns=dataset.column_names,
        desc="Tokenizing decompile examples",
    )
    return tokenized.filter(lambda row: row["length"] > 0)


@dataclass
class CausalLMCollator:
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        labels = []
        attention_mask = []

        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            labels.append(feature["labels"] + [-100] * padding)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = as_tokenizer(
        AutoTokenizer.from_pretrained(
            args.model_id,
            local_files_only=args.offline,
            trust_remote_code=False,
        )
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=args.offline,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    peft_config = make_simplex_lora_config(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=TARGET_MODULES,
    )
    model = get_simplex_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


def build_optimizer(model, args: argparse.Namespace):
    lora_layers = [module for _, module in model.named_modules() if isinstance(module, SimplexLoraLinear)]
    lora_ab_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ("lora_A" in name or "lora_B" in name)
    ]
    weight_params = [
        parameter for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_weight" in name
    ]

    if not lora_layers or not lora_ab_params or not weight_params:
        raise RuntimeError("No SimplexLoRA parameters were created")

    return optimizers.SimplexAdamW(
        [
            {"params": lora_ab_params, "name": "loraAB"},
            {"params": [], "name": "other_params"},
            {
                "params": weight_params,
                "proj": optimizers.proj_simplex_euclidean,
                "lr": args.weight_lr,
                "wd": args.weight_decay,
                "name": "weight_params",
            },
        ],
        lora_layers=lora_layers,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        num_adapters=len(weight_params),
        fat_step=args.fat_step,
        max_fat_steps=args.max_fat_steps,
        lora_extention="smart",
        default_lora_rank=args.lora_r,
    )


def main() -> None:
    args = parse_args()
    validate_calibration_arguments(args)
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
        optimization_level=args.optimization_level,
        optimization_clusters=Path(args.optimization_clusters) if args.optimization_clusters else None,
        optimization_min_confidence=args.optimization_min_confidence,
    )
    model, tokenizer = load_model_and_tokenizer(args)
    train_dataset = tokenize_dataset(train_raw, tokenizer, args.max_seq_length)
    validation_dataset = tokenize_dataset(validation_raw, tokenizer, args.max_seq_length)

    if len(train_dataset) == 0:
        raise RuntimeError("No training examples survived tokenization")

    print(f"Train examples after filtering: {len(train_dataset)}")
    print(f"Validation examples after filtering: {len(validation_dataset)}")

    optimizer = build_optimizer(model, args)
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.num_train_epochs)
    if args.warmup_steps is not None:
        if not 0 <= args.warmup_steps <= total_steps:
            raise ValueError(f"--warmup-steps must be within [0, {total_steps}], got {args.warmup_steps}")
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = max(1, int(total_steps * 0.03))

    calibration_only = bool(args.calibration_only)
    if calibration_only:
        # Stop exactly after the final projection cycle.
        total_steps = args.fat_step * args.max_fat_steps
        if args.warmup_steps is not None:
            if args.warmup_steps > total_steps:
                raise ValueError(f"--warmup-steps={args.warmup_steps} exceeds calibration steps={total_steps}")
        else:
            warmup_steps = min(warmup_steps, max(1, total_steps // 10))

    run_metadata = build_run_metadata(
        model,
        adapter_name="default",
        args=args,
        data={
            "dataset_path": str(dataset_path),
            "train_examples": len(train_dataset),
            "validation_examples": len(validation_dataset),
            "max_seq_length": args.max_seq_length,
            "optimization_level": args.optimization_level,
            "optimization_min_confidence": args.optimization_min_confidence,
            "optimization_clusters": str(
                Path(args.optimization_clusters) if args.optimization_clusters else default_cluster_dir()
            ),
        },
        device=next(model.parameters()).device,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=total_steps if calibration_only else args.max_steps,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        eval_strategy="no" if calibration_only else ("steps" if args.eval_steps > 0 else "no"),
        eval_steps=args.eval_steps if (args.eval_steps > 0 and not calibration_only) else None,
        save_strategy="no" if calibration_only else "steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        save_only_model=True,
        train_sampling_strategy="group_by_length",
        length_column_name="length",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=[],
        run_name="qwen2.5-coder-decompile-simplex",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = SimplexCalibrationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset if len(validation_dataset) else None,
        data_collator=CausalLMCollator(tokenizer.pad_token_id),
        optimizers=(optimizer, None),
        processing_class=tokenizer,
        calibration_optimizer=optimizer,
    )

    session = create_calibration_session(
        model=model,
        optimizer=optimizer,
        args=args,
        run_metadata=run_metadata,
    )

    result = None
    try:
        result = trainer.train()
    except KeyboardInterrupt:
        session.close(interrupted=True)
        raise
    except BaseException:
        session.close(failed=True)
        raise
    session.finalize()
    session.close(stop_reason=StopReason.CALIBRATION_ONLY if calibration_only else StopReason.TRAINING_COMPLETE)

    if not calibration_only:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    else:
        # A calibration-only run must not present its reset adapter as a trained model.
        print("Calibration-only run finished: the adapter was intentionally not saved as a trained model.")

    calibration_summary: dict[str, object] = {}
    if session.recorder is not None and session.recorder.layer_plan is not None:
        plan = session.recorder.layer_plan
        final_snapshot = session.recorder.snapshots[-1] if session.recorder.snapshots else None
        artifact_paths = session.recorder.artifact_paths
        calibration_summary = {
            "calibration_report": str(artifact_paths.report),
            "layer_plan": str(artifact_paths.layer_plan),
            "unsloth_handoff": str(artifact_paths.unsloth_handoff)
            if artifact_paths.unsloth_handoff is not None
            else None,
            "calibration_completed": session.recorder.calibration_completed,
            "selected_module_count": plan.adapter.selected_module_count,
            "total_rank": plan.adapter.total_rank,
            "concentration": final_snapshot.metrics.to_dict() if final_snapshot is not None else None,
        }

    peak_allocated_gib = torch.cuda.max_memory_allocated(0) / 2**30
    peak_reserved_gib = torch.cuda.max_memory_reserved(0) / 2**30
    metrics = {
        **(result.metrics if result is not None else {}),
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "target_modules": TARGET_MODULES,
        "optimization_level": args.optimization_level,
        "optimization_min_confidence": args.optimization_min_confidence,
        **calibration_summary,
    }
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "training_summary.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    if session.recorder is not None:
        print(
            render_console_summary(
                session.recorder.layer_plan,
                top_k=10,
                artifact_paths=session.recorder.artifact_paths,
            )
        )


if __name__ == "__main__":
    main()
