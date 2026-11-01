clear
for r in 1 2 4 8
do
    for strat in SimplexLoRA LoRA
    do
        for dataset in xsum cnn_dailymail
        do
            if [ "$dataset" = "cnn_dailymail" ];
            then
                extra_ds_args="--dataset_config 3.0.0"
            else
                extra_ds_args=""
            fi

            for lr in 8e-5 5e-5 1e-4 3e-4 5e-4 8e-4 1e-3
            do
                for seed in 18 57 100
                do
                    CUDA_VISIBLE_DEVICES=2 uv run simplexlora-nlg \
                        --dataset_name $dataset \
                        $extra_ds_args \
                        --model_name_or_path facebook/bart-large \
                        --per_device_train_batch_size 8 \
                        --per_device_eval_batch_size 32  \
                        --gradient_accumulation_steps 16 \
                        --learning_rate $lr \
                        --lr_scheduler_type linear \
                        --warmup_steps 50 \
                        --weight_decay 1e-1 \
                        --weight_decay_w 1e-4 \
                        --learning_rate_w 5e0 \
                        --num_train_epochs 15 \
                        --eval_strategy epoch \
                        --save_strategy no \
                        --ft_strategy $strat \
                        --lora_r $r \
                        --lora_dropout 0.05 \
                        --lora_alpha 32 \
                        --use_fat true \
                        --fat_step 5 \
                        --max_fat_steps 1 \
                        --lora_extention smart \
                        --seed $seed \
                        --predict_with_generate true \
                        --max_source_length 256 \
                        --max_target_length 160 \
                        --val_max_target_length 256 \
                        --max_train_samples 5000 \
                        --max_val_samples 1000 \
                        --num_beams 8 \
                        --do_eval true \
                        --do_predict false \
                        --report_to wandb # none or wandb
                done
            done
        done
    done
done