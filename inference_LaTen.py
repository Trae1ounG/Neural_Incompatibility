"""
Model Fusion Module for LaTen
This module implements knowledge transfer between teacher and student models
"""

import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from knowledge_translator import Knowledge_translator
from knowledge_neurons import KnowledgeNeurons
import logging
import os
import gc
import fire
import random
import json
from utils.utils import *


logger = logging.getLogger(__name__)

# Constants
LORA_PARA = {
    '1b': {'dim': 2048, 'ffn': 5632, 'n_layer': 22},
    '7b': {'dim': 4096, 'ffn': 11008, 'n_layer': 32},
    '13b': {'dim': 5120, 'ffn': 13824, 'n_layer': 40},
}

def fuse_models(
    # model/data params
    source_model: str = "meta-llama/Llama-2-7b-hf",
    target_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    source_model_size: str = "13b",
    target_model_size: str = "7b",
    target_model_path: str = "./target_model/model_name",
    translator_checkpoints: str = "./trained_translator/translator_name",
    data_path: str = "./mbpp/mbpp_train_split.jsonl",
    transfer_rates: list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    steps: list = [8],
    seed: int = 42,
):
    """Main function for model fusion"""
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
    task = data_path.split('/')[-2]
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logger.info(
            f"Inference LaTen with params:\n"
            f"source_model: {source_model}\n"
            f"target_model: {target_model}\n"
            f"data_path: {data_path}\n"
            f"task: {task}\n"
            f"transfer_rates: {transfer_rates}\n"
            f"steps: {steps}\n"
            f"source_model_size: {source_model_size}\n"
            f"target_model_size: {target_model_size}\n"
            f"target_model_path: {target_model_path}\n"
            f"translator_checkpoints: {translator_checkpoints}\n"
        )
        
    set_seed(seed)
     
    # Load models
    teacher_model = LlamaForCausalLM.from_pretrained(
        source_model,
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )
    teacher_tokenizer = LlamaTokenizer.from_pretrained(source_model)
    
    student_model = LlamaForCausalLM.from_pretrained(
        target_model,
        torch_dtype=torch.float16,
        device_map="cuda:1"
    )
    student_tokenizer = LlamaTokenizer.from_pretrained(target_model)

    # Initialize knowledge neurons、
    N_LAYERS = 32
    EXTRACT_FFN_MODULE = ["up_proj", "down_proj"]
    EXTRACT_ATTN_MODULE = ["v_proj", "o_proj"]
    
    EXTRACT_MODULE = EXTRACT_FFN_MODULE + EXTRACT_ATTN_MODULE
    
    teacher_kn = KnowledgeNeurons(
        teacher_model,
        teacher_tokenizer,
        model_type="llama",
        is_teacher=True,
        FFN_MODUELS=EXTRACT_FFN_MODULE,
        ATTN_MODUELS=EXTRACT_ATTN_MODULE,
        device="cuda:0"
    )
    
    student_kn = KnowledgeNeurons(
        student_model,
        student_tokenizer,
        model_type="llama",
        is_teacher=False,
        FFN_MODUELS=EXTRACT_FFN_MODULE,
        ATTN_MODUELS=EXTRACT_ATTN_MODULE,
        device="cuda:1"
    )
    

    # Load data
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Sample one seed example
    idxs = random.sample(range(len(data)), 1)
    prompts = [data[idx]['prompt'] for idx in idxs]
    ground_truths = [data[idx]['completion'] for idx in idxs]

    # Get top attribution layers
    
    top_attn_layer_idx, top_ffn_layer_idx = teacher_kn.get_top_attribute_layers_list(
        prompts, 
        ground_truths, 
        top_cnt=N_LAYERS
    )
    top_student_attn_layer_idx, top_student_ffn_layer_idx = student_kn.get_top_attribute_layers_list(
        prompts, 
        ground_truths, 
        top_cnt=N_LAYERS
    )

    # Sort layer indices
    top_attn_layer_idx = sorted(top_attn_layer_idx)
    top_ffn_layer_idx = sorted(top_ffn_layer_idx)
    top_student_attn_layer_idx = sorted(top_student_attn_layer_idx)
    top_student_ffn_layer_idx = sorted(top_student_ffn_layer_idx)

    

    # Process each step
    for step in steps:
        # Load translator
        knowledge_translator = Knowledge_translator(
            EXTRACT_MODULE,
            student_model.config.num_hidden_layers,
            LORA_PARA[source_model_size]['dim'],
            LORA_PARA[target_model_size]['dim']
        ).to("cuda:1")
        
        translator_path = f'{translator_checkpoints}/checkpoint_{step}/translator.pt'
        knowledge_translator.load_state_dict(torch.load(translator_path))

        # Process each transfer rate
        for transfer_rate in transfer_rates:
            print(f"\nTesting with transfer_rate: {transfer_rate}")
            
            # Calculate neuron counts
            FFN_CNT = int(LORA_PARA[target_model_size]['ffn'] * transfer_rate)
            ATTN_CNT = int(LORA_PARA[target_model_size]['dim'] * transfer_rate)
            
            # Extract student neurons
            student_ffn_neurons = student_kn.get_top_attribute_neurons_ffn_list(
                prompts, 
                ground_truths,
                top_student_ffn_layer_idx,
                student_dim=FFN_CNT,
                top=True
            )
            print(f"get student ffn neurons done with {FFN_CNT} neurons!")
            
            student_attn_neurons = student_kn.get_top_attribute_neurons_attn_list(
                prompts,
                ground_truths,
                top_student_attn_layer_idx,
                student_dim=ATTN_CNT,
                top=True
            )
            print(f"get student attn neurons done with {ATTN_CNT} neurons!")
            
            # Extract student knowledge
            student_knowledge = student_kn.extract_ffn_attn_knowledge(
                student_ffn_neurons,
                student_attn_neurons,
                extract_ffn_module=EXTRACT_FFN_MODULE,
                extract_attn_module=EXTRACT_ATTN_MODULE
            )
            print("get student knowledge done!")

            # Extract teacher neurons
            teacher_ffn_neurons = teacher_kn.get_top_attribute_neurons_ffn(
                prompts,
                ground_truths,
                top_ffn_layer_idx,
                student_dim=FFN_CNT
            )
            print("get teacher ffn neurons done!")
            
            teacher_attn_neurons = teacher_kn.get_top_attribute_neurons_attn(
                prompts,
                ground_truths,
                top_attn_layer_idx,
                student_dim=ATTN_CNT
            )
            print("get teacher attn neurons done!")
            
            # Extract teacher knowledge
            teacher_knowledge = teacher_kn.extract_ffn_attn_knowledge(
                teacher_ffn_neurons,
                teacher_attn_neurons,
                extract_ffn_module=EXTRACT_FFN_MODULE,
                extract_attn_module=EXTRACT_ATTN_MODULE
            )
            for k, v in teacher_knowledge.items():
                v = v.to("cuda:1")
                teacher_knowledge[k] = v

            # Align and inject knowledge
            with torch.no_grad():
                for module_type in [EXTRACT_FFN_MODULE, EXTRACT_ATTN_MODULE]:
                    current_knowledge = {k: teacher_knowledge[k].detach().to("cuda:1") 
                                       for k in module_type}
                    translated_knowledge = knowledge_translator(current_knowledge, top_ffn_layer_idx)
                    
                    if module_type == EXTRACT_FFN_MODULE:
                        print(f"modify lora {module_type} weights start!")
                        student_kn.knowledge_injection(
                            translated_knowledge,
                            student_knowledge,
                            ffn_layers=top_student_ffn_layer_idx,
                            ffn_neurons=student_ffn_neurons,
                            ffn_module=module_type,
                            requires_grad=True
                        )
                    else:
                        print(f"modify lora {module_type} weights start!")
                        student_kn.knowledge_injection(
                            translated_knowledge,
                            student_knowledge,
                            attn_layers=top_student_attn_layer_idx,
                            attn_neurons=student_attn_neurons,
                            attn_module=module_type,
                            requires_grad=True
                        )

            # Save fused model
            student_model.zero_grad(set_to_none=True)
            output_path = f"{target_model_path}-transfer_rate{transfer_rate * 100}%-step{step}"
            student_model.save_pretrained(output_path)
            student_tokenizer.save_pretrained(output_path)

            gc.collect()
            torch.cuda.empty_cache()

    print("finish whole proccess!")

if __name__ == "__main__":
    fire.Fire(fuse_models)