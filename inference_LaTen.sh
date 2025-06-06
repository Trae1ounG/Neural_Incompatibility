# Task name
task="gsm"
# Transfer rate for knowledge transfer
transfer_rates="[0.1]"
steps="[0]"
data_path="./data/${task}/${task}_train_split.jsonl"
echo "Starting inference LaTen for task: $task"

# Run inference with LaTen model
python inference_LaTen.py \
    --source_model "meta-llama/Llama-2-13b-chat-hf" \
    --target_model "meta-llama/Llama-2-7b-chat-hf" \
    --source_model_size "13b" \
    --target_model_size "7b" \
    --data_path "$data_path" \
    --translator_checkpoints "./knowledge_translator/gsm-transfer_rate0.1-lr3e-4" \
    --transfer_rates "$transfer_rates" \
    --steps "$steps" \
    --seed 42 \

echo "Task completed."