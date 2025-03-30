export CUDA_VISIBLE_DEVICES=0
# Task name
task="gsm"
# Transfer rate for knowledge transfer
transfer_rates="[0.1]"
steps="[7]"
data_path="./data/${task}/${task}_train_split.jsonl"
echo "Starting inference LaTen for task: $task"

# Run inference with LaTen model
python inference_LaTen_qwen_single.py \
    --source_model "/mnt/usercache/huggingface/Qwen2.5-3B-Instruct" \
    --target_model "/mnt/usercache/huggingface/Qwen2.5-1.5B-Instruct" \
    --source_model_size "3b" \
    --target_model_size "1.5b" \
    --data_path "$data_path" \
    --translator_checkpoints "./knowledge_translator/qwen-gsm-transfer_rate0.1-lr3e-5-fix" \
    --transfer_rates "$transfer_rates" \
    --steps "$steps" \
    --seed 42 \

echo "Task completed."