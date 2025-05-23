task="gsm"
lr=3e-4
transfer_rate=0.1
extract_data_path="./data/${task}/${task}_extract_split.jsonl"
align_data_path="./data/${task}/${task}_align_split.jsonl"

echo "Starting training LaTen for task: $task"


echo "Task $task with lr=$lr and transfer_rate=$transfer_rate is running."

python train_LaTen_qwen_single.py \
    --source_model "/mnt/usercache/huggingface/Qwen2.5-3B-Instruct" \
    --target_model "/mnt/usercache/huggingface/Qwen2.5-1.5B-Instruct" \
    --source_model_size "3b" \
    --target_model_size "1.5b" \
    --learning_rate "$lr" \
    --transfer_rate "$transfer_rate" \
    --extract_data_path "$extract_data_path" \
    --align_data_path "$align_data_path" \
    --translator_save_name "qwen-${task}-transfer_rate${transfer_rate}-lr${lr}-fix" \
    --cutoff_len 768 \
    --seed 42  | tee logs.log
echo "Task completed."