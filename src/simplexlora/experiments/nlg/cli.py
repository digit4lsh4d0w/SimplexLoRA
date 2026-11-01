import torch
import gc
import os
import wandb
import peft
import json
from pathlib import Path
from typing import cast
from peft.tuners.lora.layer import LoraLayer
from transformers import get_scheduler, Seq2SeqTrainer

from .data import nlg_prepocess
from ..arguments import (
    as_tokenizer,
    as_trainer_optimizers,
    parse_training_arguments,
    uses_wandb,
)
from simplexlora import optimizers, utils
import warnings

warnings.filterwarnings("ignore")


def main():
    for i in range(torch.cuda.device_count()):
        print("We will use the GPU:", torch.cuda.get_device_name(i))
    model_args, data_args, training_args = parse_training_arguments()
    utils.set_seed(training_args.seed)
    ################# Model, Tokenizer and Dataset Downloading #################
    (train_dataset, eval_dataset, test_dataset, data_collator, compute_metrics, model, tokenizer) = nlg_prepocess(
        data_args, training_args, model_args
    )
    tokenizer = as_tokenizer(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    ############################### PEFT Adapters ##############################
    all_params_before_peft, _ = utils.print_trainable_parameters(model, verbose=False)
    training_args.model_name = model_args.model_name_or_path  # for wandb
    peft_args = utils.get_peft_arguments(training_args)
    if peft_args is not None:
        model = peft.get_peft_model(model, peft_args)
    num_peft_adapters = utils.count_atapters(model, training_args.ft_strategy)
    if training_args.ft_strategy == "WeightLoRA":
        if training_args.use_rand:
            training_args.ft_strategy = "RandLoRA"
            utils.apply_rand_weight_lora(model, num_peft_adapters, training_args.k)
        if training_args.use_fat:
            training_args.ft_strategy = "SimplexLoRA"
    training_args.label_names = ["labels"]  # peft and compute_metrics() problem
    ######################### Optimizer and Scheduler ##########################
    optimizer, scheduler = None, None
    if training_args.use_tuned_learning_rate:
        f_name = Path(__file__).with_name("tuned_params.json")
        with open(f_name) as f:
            tuned_params = json.load(f)
        lr = tuned_params[model_args.model_name_or_path][data_args.dataset_name][training_args.ft_strategy]["lr"]
        training_args.learning_rate = lr
    else:
        training_args.learning_rate = float(training_args.learning_rate)

    if training_args.ft_strategy == "SimplexLoRA":
        weight_params = []
        loraAB_params = []
        lora_layers: list[LoraLayer] = []
        other_params = []

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                lora_layers.append(module)

        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                loraAB_params.append(param)
            if "lora_weight" in name:
                weight_params.append(param)

        optimizer = optimizers.SimplexAdamW(
            [
                {"params": loraAB_params, "name": "loraAB"},
                {"params": other_params, "name": "other_params"},
                {
                    "params": weight_params,
                    "proj": optimizers.proj_simplex_euclidean,
                    "lr": training_args.learning_rate_w,
                    "wd": training_args.weight_decay_w,
                    "name": "weight_params",
                },
            ],
            lora_layers=lora_layers,
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            num_adapters=len(weight_params),
            fat_step=training_args.fat_step,
            max_fat_steps=training_args.max_fat_steps,
            lora_extention=training_args.lora_extention,
            default_lora_rank=training_args.lora_r,
        )
        ######################################################

        import math

        len_dataloader = math.ceil(len(train_dataset) / training_args.per_device_train_batch_size)
        num_update_steps_per_epoch = len_dataloader // training_args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if training_args.max_steps > 0:
            max_steps = training_args.max_steps
        else:
            max_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)

        ######################################################
        calib_iters = training_args.max_fat_steps * training_args.fat_step
        scheduler_kwargs = (
            training_args.lr_scheduler_kwargs if isinstance(training_args.lr_scheduler_kwargs, dict) else None
        )
        if calib_iters > 0:
            scheduler1 = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.001,
                total_iters=calib_iters,
            )
            scheduler2 = get_scheduler(
                training_args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=round(training_args.warmup_steps),
                num_training_steps=max_steps - calib_iters,
                scheduler_specific_kwargs=scheduler_kwargs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[scheduler1, scheduler2], milestones=[calib_iters]
            )
        else:
            scheduler = get_scheduler(
                training_args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=round(training_args.warmup_steps),
                num_training_steps=max_steps - calib_iters,
                scheduler_specific_kwargs=scheduler_kwargs,
            )
    elif training_args.ft_strategy == "WeightLoRA":
        weight_params, other_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_weight" in name:
                weight_params.append(param)
            elif "lora_A" in name or "lora_B" in name:
                other_params.append(param)
        optimizer = optimizers.WeightAdamW(
            [
                {"params": other_params, "name": "other_params"},
                {
                    "params": weight_params,
                    "k": training_args.k,
                    "proj": optimizers.proj_0,
                    "lr": training_args.learning_rate_w,
                    "max_fat_steps": training_args.max_fat_steps,
                    "name": "weight_params",
                },
            ],
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay
        )
    ############################### Wandb Saves ################################
    training_args.all_params, training_args.trainable_params = utils.print_trainable_parameters(model)
    training_args.num_peft_adapters = num_peft_adapters
    training_args.peft_params = training_args.all_params - all_params_before_peft
    training_args.train_proportion = training_args.trainable_params / training_args.all_params * 100
    training_args.peft_proportion = training_args.peft_params / training_args.all_params * 100
    os.environ["WANDB_PROJECT"] = "S1mplexLoRA"
    if training_args.ft_strategy in [
        "SimplexLoRA",
    ]:
        run_name = f"[{training_args.ft_strategy} fs={training_args.fat_step}, mfs={training_args.max_fat_steps}, r={training_args.lora_r}, wd={training_args.weight_decay}]"
    else:
        run_name = f"[{training_args.ft_strategy} r={training_args.lora_r}]"
    run_name += f" T5-3B {data_args.dataset_name}, lr={training_args.learning_rate}"
    training_args.run_name = run_name
    training_args.output_dir = f"./outputs/nlg/{training_args.output_dir}/{run_name}"
    os.environ["WANDB_TAGS"] = f"T5 {data_args.dataset_name}"
    if optimizer is not None:
        training_args.optimizer = optimizer.__class__.__name__
    else:
        training_args.optimizer = training_args.optim
    training_args.benchmark_name = data_args.dataset_name
    training_args.tsk_name = data_args.task_name
    ############################# Training #####################################
    print("$" * 30, f" {run_name} ", "$" * 30)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        optimizers=as_trainer_optimizers(optimizer, scheduler),
    )

    if training_args.do_train:
        train_result = trainer.train()
        train_metrics: dict[str, object] = dict(train_result.metrics)
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        train_metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        train_metrics["train_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
        train_metrics["train_runtime"] = float(cast(float, train_metrics["train_runtime"])) / 60
        if training_args.ft_strategy in ["WeightLoRA", "RandLoRA", "SimplexLoRA"]:
            i = 0
            j = 0
            for name, param in model.named_parameters():
                if "lora_weight" in name:
                    if param.sum().item() > 0 and param.requires_grad:
                        i += 1
                        if model_args.model_name_or_path == "microsoft/deberta-v3-base":
                            continue
                        else:
                            load_name = name
                        train_metrics[f"active_adapters_{i}"] = load_name
                elif "lora_B" in name and param.data.norm().item() == 0:
                    j += 1
            print(f"There are {j} null B LoRA matrices")

        trainer.save_model()

        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

        if uses_wandb(training_args.report_to):
            wandb.config.update(train_metrics, allow_val_change=True)
    ################################ Evaluation ################################
    if training_args.do_eval:
        max_length = (
            training_args.generation_max_length
            if training_args.generation_max_length is not None
            else data_args.val_max_target_length
        )
        num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
        eval_metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        eval_metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        eval_metrics["eval_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
        if "eval_runtime" in eval_metrics.keys():
            eval_metrics["eval_runtime"] /= 60
        if uses_wandb(training_args.report_to):
            wandb.config.update(eval_metrics, allow_val_change=True)
    ############################################################################

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
