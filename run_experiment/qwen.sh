#!/bin/bash

clear

# LLM Qwen LoRA Fine-tuning Script

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

LRS=(8e-5 1e-4)
DATASETS=(boolq boolq)
RANKS=(1 1)
FT=(SimplexLoRA LoRA)

for i in "${!LRS[@]}";
do
    lr="${LRS[i]}"
    dataset="${DATASETS[i]}"
    rank="${RANKS[i]}"
    ft="${FT[i]}"

    for seed in 10 57 108
    do
            python ./run_experiment/run_experiment.py \
                --dataset $dataset \
                --seed $seed \
                --model Qwen/Qwen3-8B \
                --padding_side left \
                --optimizer adamw \
                --batch_size 4 \
                --gradient_accumulation_steps 1 \
                --learning_rate $lr \
                --weight_decay 1e-4 \
                --weight_decay_w 1e-4 \
                --learning_rate_w 5e0 \
                --lr_scheduler_type linear \
                --warmup_ratio 0.1 \
                --max_train_steps 3500 \
                --max_seq_length 512 \
                --logging_steps 1 \
                --ft_strategy $ft \
                --lora_r $rank \
                --lora_alpha 32 \
                --lora_dropout 0.05 \
                --quantization_bit 4 \
                --dtype bfloat16 \
                --use_fast_tokenizer \
                --max_eval_samples 1000 \
                --max_fat_steps 1 \
                --fat_step 5 \
                --eval_steps 250 \
                --wandb
    done
done