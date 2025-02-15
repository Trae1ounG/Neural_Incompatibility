task="gsm"
lr=3e-4
transfer_rate=0.1
echo "Starting training LaTen for task: $task"
extract_data_path="./data/${task}/${task}_extract_split.jsonl"
align_data_path="./data/${task}/${task}_align_split.jsonl"

echo "Task $task with lr=$lr and transfer_rate=$transfer_rate is running."

python -u train_LaTen.py \
    --source_model "/mnt/usercache/huggingface/Llama-2-13b-chat-hf" \
    --target_model "/mnt/usercache/huggingface/llama-2-7b-chat-hf" \
    --learning_rate "$lr" \
    --transfer_rate "$transfer_rate" \
    --extract_data_path "$extract_data_path" \
    --align_data_path "$align_data_path" \
    --translator_save_name "$task-transfer_rate${transfer_rate}-lr${lr}" \
    --cutoff_len 1024 \
    --seed 42 \

echo "Task completed."