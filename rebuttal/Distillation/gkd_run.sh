export WANDB_DISABLED=true
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --config_file ./ds_config/ds_zero3.yaml ./train_gkd.py \
    --model_name_or_path /mnt/publiccache/huggingface/Qwen2.5-1.5B-Instruct \
    --teacher_model_name_or_path /mnt/publiccache/huggingface/Qwen2.5-3B-Instruct  \
    --dataset_name ../data/qwen2.5-32b-gsm-kd-prompt.jsonl \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --report_to none \
    --output_dir gkd-qwen1.5b-gsm-qwen3b-gkd-0328-5ep-1e-5-fix_aiplatform \
    --logging_steps 2 \
    --max_new_tokens 32 \
    --max_seq_length 64 \
    --num_train_epochs 5 \
    --gradient_checkpointing \
    --lmbda 1 \
    --beta 0 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 16 \
    --trust_remote_code \
    --fp16 \
    --save_strategy "steps" \
    --save_total_limit 5 \
    --warmup_steps 10 \
    --lr_scheduler_type "cosine" \
    --torch_dtype float16 \
    --report_to none \
    > logs/gkd-qwen1.5b-gsm-qwen3b-gkd-0328-5ep-1e-5-fix_aiplatform.log 2>&1 &