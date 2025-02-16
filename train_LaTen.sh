task="gsm"
lr=3e-4
transfer_rate=0.1
extract_data_path="./data/${task}/${task}_extract_split.jsonl"
align_data_path="./data/${task}/${task}_align_split.jsonl"

echo "Starting training LaTen for task: $task"


echo "Task $task with lr=$lr and transfer_rate=$transfer_rate is running."

python train_LaTen.py \
    --source_model "/mnt/publiccache/huggingface/Llama-2-7b-chat-hf" \
    --target_model "/mnt/publiccache/huggingface/llama-2-7b-chat-hf" \
    --source_model_size "7b" \
    --target_model_size "7b" \
    --learning_rate "$lr" \
    --transfer_rate "$transfer_rate" \
    --extract_data_path "$extract_data_path" \
    --align_data_path "$align_data_path" \
    --translator_save_name "$task-transfer_rate${transfer_rate}-lr${lr}" \
    --cutoff_len 768 \
    --seed 42 \

echo "Task completed."