"""
Knowledge translation between teacher and student models using neuron-level static localization
"""
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
import logging
import os
import gc
import sys
from typing import List
import fire
import random
import collections
from datasets import load_dataset
from utils.utils import *
from knowledge_translator import Knowledge_translator
from modeling_llama import LlamaForCausalLM as LlamaForCausalLMEdit
from knowledge_neurons import KnowledgeNeurons
from tqdm import tqdm

# Model architecture parameters
LORA_PARA = {
    '1b': {'dim': 2048, 'ffn': 5632, 'n_layer': 22},
    '7b':  {'dim': 4096, 'ffn': 11008, 'n_layer': 32},
    '13b': {'dim': 5120, 'ffn': 13824, 'n_layer': 40},
    '30b': {'dim': 6656, 'ffn': 17920, 'n_layer': 60},
    '65b': {'dim': 8192, 'ffn': 22016, 'n_layer': 80},
}

SELF_ATTENTION_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
FFN_MODULES = ['gate_proj', 'down_proj', 'up_proj']

logger = logging.getLogger(__name__)

def train(
    source_model: str = "meta-llama/Llama-2-13b-chat-hf",
    target_model: str = "meta-llama/Llama-2-7b-chat-hf",
    source_model_size: str = "13b",
    target_model_size: str = "7b",
    extract_data_path: str = "./data/gsm/gsm_extract_split.jsonl",
    align_data_path: str = "./data/gsm/gsm_align_split.jsonl",
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
    use_chat_prompt: bool = True,
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
    config = locals()
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
            f"use_chat_prompt: {use_chat_prompt}\n"
            f"transfer_rate: {transfer_rate}\n"
            f"print_pre_loss: {print_pre_loss}\n"
        )
    assert (
        source_model
    ), "Please specify a --source_model, e.g. --source_model='meta-llama/Llama-2-7b-hf'"
    assert (
        target_model
    ), "Please specify a --target_model, e.g. --target_model='TinyLlama/TinyLlama-1.1B-Chat-v1.0'"
    
    set_seed(seed)
    
    def generate_and_tokenize_prompt(data_point, tokenizer):
        def tokenize(prompt, add_eos_token=True):
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
            ):
                result["input_ids"].append(tokenizer.eos_token_id)
                result["attention_mask"].append(1)
            result["labels"] = result["input_ids"].copy()
            return result
        if use_chat_prompt:
            assert "messages" in data_point
            full_prompt = generate_chat_prompt(
                data_point["messages"]
            )
        else:
            full_prompt = generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )
            
        tokenized_full_prompt = tokenize(full_prompt)
        
        if not train_on_inputs:
            if use_chat_prompt:
                user_prompt = generate_chat_prompt(
                    data_point["messages"][:-1]
                )
                user_prompt += "<|assistant|>\n"
            else:
                user_prompt = generate_prompt(
                    data_point["instruction"], data_point["input"]
                )
            tokenized_user_prompt = tokenize(
                user_prompt
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
            
        tokenized_full_prompt['user_prompt'] = user_prompt
        tokenized_full_prompt['response'] = full_prompt[len(user_prompt):]
        tokenized_full_prompt['full_prompt'] = full_prompt
        return tokenized_full_prompt
   
    
    MODEL_NAMES = {"7b": "meta-llama/Llama-2-7b-chat-hf",
                   "13b": "meta-llama/Llama-2-13b-chat-hf"}
    
    teacher_model = LlamaForCausalLMEdit.from_pretrained(
        source_model,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        output_hidden_states=True
    )
    logger.info(teacher_model.config)
    student_model = LlamaForCausalLMEdit.from_pretrained(
        target_model,
        torch_dtype=torch.float16,
        device_map="cuda:1",
        output_hidden_states=True
    )
    logger.info(f"Teacher model:\n{teacher_model}")
    logger.info(f"Student model:\n{student_model}")
    teacher_tokenizer = LlamaTokenizer.from_pretrained(
        source_model
    )
    student_tokenizer = LlamaTokenizer.from_pretrained(
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
    N_LAYERS = 32
    EXTRACT_FFN_MODULE = ['down_proj', "up_proj" ]
    EXTRACT_ATTN_MODULE = ['o_proj', "v_proj"]
    EXTRACT_MODULE = []
    if EXTRACT_FFN_MODULE is not None:
        EXTRACT_MODULE += EXTRACT_FFN_MODULE
    if EXTRACT_ATTN_MODULE is not None:
        EXTRACT_MODULE += EXTRACT_ATTN_MODULE
    logger.info(f"EXTRACT_MODULE:{EXTRACT_MODULE}")
    
    # Load data
    if extract_data_path.endswith(".json") or extract_data_path.endswith(".jsonl"):
        extract_data = load_dataset("json", data_files=extract_data_path)
    else:
        extract_data = load_dataset(extract_data_path)
    if align_data_path.endswith(".json") or align_data_path.endswith(".jsonl"):
        align_data = load_dataset("json", data_files=align_data_path)
    else:
        align_data = load_dataset(align_data_path)
    extract_data = extract_data["train"].shuffle().map(
        lambda x: generate_and_tokenize_prompt(x, student_tokenizer)
    )
    align_data = align_data["train"].shuffle().map(
        lambda x: generate_and_tokenize_prompt(x, student_tokenizer)
    )
    
    # Initialize knowledge translator for parameter alignment
    knowledge_translator = Knowledge_translator(EXTRACT_MODULE, student_n_layers, source_llm_hidden_dim, target_llm_hidden_dim).to("cuda:1")
    knowledge_translator.train()
    
    teacher_kn = KnowledgeNeurons(
        teacher_model, 
        teacher_tokenizer, 
        model_type="llama", 
        is_teacher=True, 
        FFN_MODUELS=EXTRACT_MODULE, 
        device="cuda:0")
    student_kn = KnowledgeNeurons(
        student_model, 
        student_tokenizer, 
        model_type="llama", 
        is_teacher=False, 
        FFN_MODUELS=EXTRACT_MODULE, 
        device="cuda:1")
    
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
    
    # Collect training losses    
    key_losses = collections.defaultdict(list)
    key_pre_losses = collections.defaultdict(list)
    
    for idx in tqdm(range(total_step)):
        logger.info(f"Processing example {idx} of {total_step}")
        item = extract_data[idx]
        prompt, ground_truth = [item['user_prompt']], [item['response']]
        
        # Get top attribute layers
        top_attn_layer_idx, top_ffn_layer_idx = teacher_kn.get_top_attribute_layers(prompt, ground_truth, top_cnt=N_LAYERS)
        top_student_attn_layer_idx, top_student_ffn_layer_idx = student_kn.get_top_attribute_layers(prompt, ground_truth, top_cnt=N_LAYERS)
        top_attn_layer_idx = sorted(top_attn_layer_idx)
        top_ffn_layer_idx = sorted(top_ffn_layer_idx)
        top_student_attn_layer_idx = sorted(top_student_attn_layer_idx)
        top_student_ffn_layer_idx = sorted(top_student_ffn_layer_idx)
        
        # Get top attribute neurons
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

        
        # Knowledge Extraction        
        teacher_knowledge = teacher_kn.extract_ffn_attn_knowledge(teacher_ffn_neurons, teacher_attn_neurons, extract_ffn_module=EXTRACT_FFN_MODULE, extract_attn_module=EXTRACT_ATTN_MODULE)
        for k, v in teacher_knowledge.items():
            teacher_knowledge[k] = v.to("cuda:1").to(torch.float16)
            del v
            torch.cuda.empty_cache()
            
        total_loss = 0
        
        # Sample P-1 data from align dataset for alignment
        random.seed(idx)
        random_indices = random.sample(range(len(align_data)),15)
        batch_items = [item for item in align_data.select(random_indices)]
        # Append current locating example for alignment
        batch_items.append(item)
        
        k = EXTRACT_FFN_MODULE + EXTRACT_ATTN_MODULE
        # Prepare knowledge for alignment
        for name, param in knowledge_translator.named_parameters():
            if any([k_ in name for k_ in k]):
                param.requires_grad = True
            else:
                param.requires_grad = False
        knowledges = {}
        for key in k:
            knowledges[key] = teacher_knowledge[key].detach().to("cuda:1")
        
        # Initialize losses
        total_lm_loss = 0
        total_pre_loss = 0
        num_batches = 0
        
        for i in range(0, len(batch_items), micro_batch_size):
            micro_batch = batch_items[i:i + micro_batch_size]
            max_length = cutoff_len
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
                    torch.zeros_like(student_knowledges[key], dtype=torch.float16)
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
                requires_grad=True
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
        
        if print_pre_loss:
            logger.info(f"key:{k}, edited lm_loss:{avg_lm_loss}, pre_loss:{avg_pre_loss}, mse_loss:{current_mse_loss.item()}, total_loss: {avg_lm_loss}")
        else:
            logger.info(f"key:{k}, edited lm_loss:{avg_lm_loss}, total_loss: {avg_lm_loss}")

        del teacher_knowledge, teacher_ffn_neurons, teacher_attn_neurons, cur_step_loss
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
