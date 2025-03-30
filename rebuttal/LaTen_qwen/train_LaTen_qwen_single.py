"""
LaTen: Locate-Then-Align Training Code
"""
import torch
import transformers
from transformers import Qwen2Tokenizer
import logging
import os
import gc
import sys
from typing import List
import fire
import random
from datasets import load_dataset
from utils.utils import *
from knowledge_translator import Knowledge_translator
from modeling_qwen2 import Qwen2ForCausalLM as QwenForCausalLMEdit
from transformers import Qwen2ForCausalLM
# from modeling_llama import LlamaForCausalLM as LlamaForCausalLMEdit
from knowledge_neurons import KnowledgeNeurons
from tqdm import tqdm

# Model architecture parameters
LORA_PARA = {
    '1.5b': {'dim': 1536, 'ffn': 8960, 'n_layer': 28},
    '3b':  {'dim': 2048, 'ffn': 11008, 'n_layer': 36},
    '32b': {'dim': 5120, 'ffn': 27648, 'n_layer': 64},
}

SELF_ATTENTION_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
FFN_MODULES = ['gate_proj', 'down_proj', 'up_proj']

logger = logging.getLogger(__name__)

def train(
    source_model: str = "/mnt/usercache/huggingface/Qwen2.5-3B-Instruct",
    target_model: str = "/mnt/usercache/huggingface/Qwen2.5-1.5B-Instruct",
    source_model_size: str = "3b",
    target_model_size: str = "1.5b",
    extract_data_path: str = "/mnt/userdata/LaTen/data/gsm/gsm_extract_split.jsonl",
    align_data_path: str = "/mnt/userdata/LaTen/data/gsm/gsm_align_split.jsonl",
    translator_save_name: str = "gsm",
    micro_batch_size: int = 1,
    learning_rate: float = 3e-5,
    cutoff_len: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "up_proj",
        "down_proj",
        "v_proj",
        "o_proj"
    ],
    train_on_inputs: bool = False,
    seed: int = 42,
    transfer_rate: float = 0.1,
    print_pre_loss: bool = True
):
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel('INFO')
    transformers.utils.logging.set_verbosity('INFO')
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(
            f"Training LaTen with params:\n"
            f"source_model: {source_model}\n"
            f"target_model: {target_model}\n"
            f"extract_data_path: {extract_data_path}\n"
            f"align_data_path: {align_data_path}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"translator_save_name: {translator_save_name}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"seed: {seed}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"transfer_rate: {transfer_rate}\n"
            f"print_pre_loss: {print_pre_loss}\n"
        )
    assert (
        source_model
    ), "Please specify a --source_model, e.g. --source_model='meta-llama/Llama-2-13b-chat-hf'"
    assert (
        target_model
    ), "Please specify a --target_model, e.g. --target_model='meta-llama/Llama-2-7b-chat-hf'"
    
    set_seed(seed)
    #Load models    
    teacher_model = QwenForCausalLMEdit.from_pretrained(
        source_model,
        torch_dtype=torch.float32,
        device_map="cuda:0",
        output_hidden_states=True,
    )
    logger.info(teacher_model.config)
    student_model = QwenForCausalLMEdit.from_pretrained(
        target_model,
        torch_dtype=torch.float32,
        device_map="cuda:0",
        output_hidden_states=True,
        
    )
    logger.info(f"Teacher model:\n{teacher_model}")
    logger.info(f"Student model:\n{student_model}")
    
    #Load tokenizers
    teacher_tokenizer = Qwen2Tokenizer.from_pretrained(
        source_model
    )
    student_tokenizer = Qwen2Tokenizer.from_pretrained(
        target_model
    )
    
    teacher_tokenizer.add_bos_token = False
    teacher_tokenizer.add_eos_token = False
    teacher_tokenizer.padding_side = "left"  # Allow batched inference
    student_tokenizer.add_bos_token = False
    student_tokenizer.add_eos_token = False
    student_tokenizer.padding_side = "left"  # Allow batched inference
    student_tokenizer.pad_token = student_tokenizer.eos_token
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    
    # Load and process data
    if extract_data_path.endswith(".json") or extract_data_path.endswith(".jsonl"):
        extract_data = load_dataset("json", data_files=extract_data_path)
    else:
        extract_data = load_dataset(extract_data_path)
    if align_data_path.endswith(".json") or align_data_path.endswith(".jsonl"):
        align_data = load_dataset("json", data_files=align_data_path)
    else:
        align_data = load_dataset(align_data_path)
        
    extract_data = extract_data["train"].shuffle().map(
        lambda x: generate_and_tokenize_prompt(x, student_tokenizer, cutoff_len, train_on_inputs)
    )
    align_data = align_data["train"].shuffle().map(
        lambda x: generate_and_tokenize_prompt(x, student_tokenizer, cutoff_len, train_on_inputs)
    )
    # Layer number and dimension of teacher and student model
    teacher_n_layers = LORA_PARA[source_model_size]['n_layer']
    student_n_layers = LORA_PARA[target_model_size]['n_layer']
    source_llm_hidden_dim = LORA_PARA[source_model_size]['dim']
    target_llm_hidden_dim = LORA_PARA[target_model_size]['dim']
    
    # Number of neurons to transfer
    FFN_CNT = int(LORA_PARA[target_model_size]['ffn'] * transfer_rate)
    ATTN_CNT = int(LORA_PARA[target_model_size]['dim'] * transfer_rate)
    logger.info(f"FFN_CNT:{FFN_CNT}, ATTN_CNT:{ATTN_CNT}")

    # Modules to extract
    N_LAYERS = LORA_PARA[target_model_size]['n_layer']
    EXTRACT_FFN_MODULE = ['down_proj', "up_proj" ]
    EXTRACT_ATTN_MODULE = ['o_proj']
    EXTRACT_MODULE = []
    if EXTRACT_FFN_MODULE is not None:
        EXTRACT_MODULE += EXTRACT_FFN_MODULE
    if EXTRACT_ATTN_MODULE is not None:
        EXTRACT_MODULE += EXTRACT_ATTN_MODULE
    logger.info(f"EXTRACT_MODULE:{EXTRACT_MODULE}")
    
    # Initialize knowledge translator for parameter alignment
    knowledge_translator = Knowledge_translator(EXTRACT_MODULE, student_n_layers, source_llm_hidden_dim, target_llm_hidden_dim).to(torch.float32).to("cuda:0")
    knowledge_translator.train()
    
    teacher_kn = KnowledgeNeurons(
        teacher_model, 
        teacher_tokenizer, 
        model_type="qwen", 
        is_teacher=True, 
        FFN_MODUELS=EXTRACT_MODULE, 
        device="cuda:0")
    student_kn = KnowledgeNeurons(
        student_model, 
        student_tokenizer, 
        model_type="qwen", 
        is_teacher=False, 
        FFN_MODUELS=EXTRACT_MODULE, 
        device="cuda:0")
    
    # MSE for constraint    
    mse_loss_fn = nn.MSELoss(reduction='mean')
    # Optimizer
    optimizer = torch.optim.AdamW(knowledge_translator.parameters(), lr=learning_rate, weight_decay=0.05, betas=(0.9, 0.95))
    total_step = len(extract_data)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_step,
        pct_start=0.1, 
        final_div_factor=1, 
        anneal_strategy='cos'
    )
    
    # Start training
    for idx in tqdm(range(total_step)):
        logger.info(f"Processing example {idx} of {total_step}")
        data_point = extract_data[idx]
        prompt, ground_truth = [data_point['user_prompt']], [data_point['response']]
        
        # Get top attribute layers
        top_attn_layer_idx, top_ffn_layer_idx = teacher_kn.get_top_attribute_layers(prompt, ground_truth, top_cnt=N_LAYERS)
        top_student_attn_layer_idx, top_student_ffn_layer_idx = student_kn.get_top_attribute_layers(prompt, ground_truth, top_cnt=N_LAYERS)
        top_attn_layer_idx = sorted(top_attn_layer_idx)
        top_ffn_layer_idx = sorted(top_ffn_layer_idx)
        top_student_attn_layer_idx = sorted(top_student_attn_layer_idx)
        top_student_ffn_layer_idx = sorted(top_student_ffn_layer_idx)
        
        # # Get top attribute neurons
        student_ffn_neurons = student_kn.get_top_attribute_neurons_ffn(prompt, ground_truth, top_student_ffn_layer_idx, student_dim = FFN_CNT, top=True)
        logger.info("get student ffn neurons done!")
        student_attn_neurons = student_kn.get_top_attribute_neurons_attn(prompt, ground_truth, top_student_attn_layer_idx, student_dim = ATTN_CNT, top=True)
        logger.info("get student attn neurons done!")
        student_knowledge = student_kn.extract_ffn_attn_knowledge(student_ffn_neurons, student_attn_neurons, extract_ffn_module=EXTRACT_FFN_MODULE, extract_attn_module=EXTRACT_ATTN_MODULE)
        logger.info("get student knowledge done!")
        teacher_ffn_neurons = teacher_kn.get_top_attribute_neurons_ffn(prompt, ground_truth, top_ffn_layer_idx, student_dim = FFN_CNT)
        logger.info("get teacher ffn neurons done!")
        teacher_attn_neurons = teacher_kn.get_top_attribute_neurons_attn(prompt, ground_truth, top_attn_layer_idx, student_dim = ATTN_CNT)
        logger.info("get teacher attn neurons done!")

        
        # # Knowledge Extraction        
        teacher_knowledge = teacher_kn.extract_ffn_attn_knowledge(teacher_ffn_neurons, teacher_attn_neurons, extract_ffn_module=EXTRACT_FFN_MODULE, extract_attn_module=EXTRACT_ATTN_MODULE)
        for k, v in teacher_knowledge.items():
            teacher_knowledge[k] = v.to("cuda:0").to(torch.float32)
            del v
            torch.cuda.empty_cache()
            
        total_loss = 0
        
        # # Sample P-1 data from align dataset for alignment
        random.seed(idx)
        random_indices = random.sample(range(len(align_data)),15)
        batch_items = [item for item in align_data.select(random_indices)]
        # # Append current locating example for alignment
        batch_items.append(data_point)
        
        # k = EXTRACT_FFN_MODULE + EXTRACT_ATTN_MODULE
        k = EXTRACT_MODULE
        # Prepare knowledge for alignment
        for name, param in knowledge_translator.named_parameters():
            if any([k_ in name for k_ in k]):
                param.requires_grad = True
            else:
                param.requires_grad = False
        knowledges = {}
        for key in k:
            knowledges[key] = teacher_knowledge[key].detach().to("cuda:0")
        
        # Initialize losses
        total_lm_loss = 0
        total_pre_loss = 0
        num_batches = 0
        max_length = cutoff_len
        
        # # Gradient accumulation
        for i in range(0, len(batch_items), micro_batch_size):
            micro_batch = batch_items[i:i + micro_batch_size]
            batch_inputs = {
                'input_ids': torch.stack([
                    torch.tensor(pad_sequence(item['input_ids'], max_length, student_tokenizer.pad_token_id))
                    for item in micro_batch
                ]).to(student_model.device),
                'attention_mask': torch.stack([
                    torch.tensor(pad_sequence(item['attention_mask'], max_length, 0))
                    for item in micro_batch
                ]).to(student_model.device),
                'labels': torch.stack([
                    torch.tensor(pad_sequence(item['labels'], max_length, -100))
                    for item in micro_batch
                ]).to(student_model.device)
            }
            logger.info("Forward start!")
            current_translated_knowledge = knowledge_translator(knowledges, top_ffn_layer_idx)
            student_knowledges = {}
            for key in k:
                student_knowledges[key] = student_knowledge[key].detach().requires_grad_(True)
            current_mse_loss = 0
            for key in k:
                current_mse_loss += mse_loss_fn(
                    current_translated_knowledge[key], 
                    torch.zeros_like(student_knowledges[key], dtype=torch.float32)
                )
            logger.info(f"Modify lora {k} weights start!")
            delta_weights = student_kn.knowledge_injection(
                current_translated_knowledge,  
                student_knowledge,
                ffn_layers=top_student_ffn_layer_idx,
                attn_layers=top_student_attn_layer_idx,
                ffn_neurons=student_ffn_neurons,
                ffn_module=EXTRACT_FFN_MODULE,
                attn_neurons=student_attn_neurons,
                attn_module=EXTRACT_ATTN_MODULE,
            )
            outputs = student_kn.model(**batch_inputs, return_dict=True)
            lm_loss = outputs.loss
            logger.info("Forward done!")
            logger.info(f"lm_loss: {lm_loss.item()}, mse_loss: {current_mse_loss.item()}")
            total_loss = lm_loss + current_mse_loss * 1000
            
            if type(total_loss) == torch.Tensor and str(total_loss.item()) != "nan":
                total_lm_loss += lm_loss.item()
                
            # Delete delta weights
            for k_, v in delta_weights.items():
                k_ = ".".join(k_.split(".")[:-1])
                delattr(get_attributes(student_kn.model, k_), "delta")
            
            # Backward
            if type(total_loss) == torch.Tensor and str(total_loss.item()) != "nan":
                total_loss.backward()
            num_batches += 1
            del delta_weights, current_translated_knowledge, outputs, student_knowledges, total_loss
            gc.collect()
            torch.cuda.empty_cache()
        logger.info("Backward done!")

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Calculate Pre-loss
        if print_pre_loss:
            with torch.no_grad():
                total_pre_loss = 0
                num_pre_batches = 0
                for i in range(0, len(batch_items), micro_batch_size):
                    micro_batch = batch_items[i:i + micro_batch_size]
                    batch_inputs = {
                        'input_ids': torch.stack([
                            torch.tensor(pad_sequence(item['input_ids'], max_length, student_tokenizer.pad_token_id))
                            for item in micro_batch
                        ]).to(student_model.device),
                        'attention_mask': torch.stack([
                            torch.tensor(pad_sequence(item['attention_mask'], max_length, 0))
                            for item in micro_batch
                        ]).to(student_model.device),
                        'labels': torch.stack([
                            torch.tensor(pad_sequence(item['labels'], max_length, -100))
                            for item in micro_batch
                        ]).to(student_model.device)
                    }
                    outputs = student_kn.model(**batch_inputs, return_dict=True)
                    total_pre_loss += outputs.loss.item()
                    num_pre_batches += 1
                avg_pre_loss = total_pre_loss / num_pre_batches
        avg_lm_loss = total_lm_loss / num_batches
        print(avg_pre_loss)
        if print_pre_loss:
            logger.info(f"key:{k}, edited lm_loss:{avg_lm_loss}, pre_loss:{avg_pre_loss}, mse_loss:{current_mse_loss.item()}, total_loss: {avg_lm_loss}")
        else:
            logger.info(f"key:{k}, edited lm_loss:{avg_lm_loss}, total_loss: {avg_lm_loss}")

        del teacher_knowledge, teacher_ffn_neurons, teacher_attn_neurons
        gc.collect()
        torch.cuda.empty_cache()
        
        translator_save_dir = os.path.join('./knowledge_translator/', translator_save_name, f'checkpoint_{idx}')
        os.makedirs(translator_save_dir, exist_ok=True)
        torch.save(knowledge_translator.state_dict(), 
                os.path.join(translator_save_dir, 'translator.pt'))
        logger.info(f"{idx} step knowledge translator saved!")
                
        logger.info("finish whole proccess!")


if __name__ == "__main__":
    
    fire.Fire(train)
