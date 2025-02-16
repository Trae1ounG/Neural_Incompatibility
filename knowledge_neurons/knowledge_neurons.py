"""
Based on https://github.com/EleutherAI/knowledge-neurons
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import collections
from typing import List, Optional, Tuple, Callable
import gc
import torch
import torch.nn.functional as F
import collections
import math
import os
from functools import partial
from transformers import PreTrainedTokenizerBase
from .patch import *
from .utils import *

import torch
import gc
import functools
from typing import Optional
import logging

class KnowledgeNeurons:
    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizerBase,
        model_type: str = "llama",
        device: str = "cuda:0",
        is_teacher: bool = False,
        FFN_MODUELS: list = None,
        ATTN_MODUELS: list = None
    ):
        self.model = model
        self.model_type = model_type
        self.device = device
        self.tokenizer = tokenizer
        self.is_teacher = is_teacher
        self.baseline_activations = None
        
        if self.model_type == "bert":
            self.transformer_layers_attr = "bert.encoder.layer"
            self.input_ff_attr = "intermediate"
            self.output_ff_attr = "output.dense.weight"
            self.word_embeddings_attr = "bert.embeddings.word_embeddings.weight"
            self.unk_token = getattr(self.tokenizer, "unk_token_id", None)
        elif "gpt" in model_type:
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.c_fc"
            self.output_ff_attr = "mlp.c_proj.weight"
            self.word_embeddings_attr = "transformer.wte.weight"
            self.n_layers = self.model.config.n_layer
            self.n_head = self.model.config.n_head
            self.n_embd = self.model.config.n_embd
            self.head_dim = self.n_embd // self.n_head
        elif model_type == "llama":
            self.transformer_layers_attr = "model.layers"
            self.act_ff_attr = "mlp.act_fn"
            self.input_ff_attr = "mlp.up_proj"
            self.output_ff_attr = "mlp.down_proj.weight"
            self.ff_attr_format =  "mlp.{}"
            self.attn_attr_format = "self_attn.{}"
            
            self.word_embeddings_attr = "model.embed_tokens.weight"
            self.input_ff_attr_loraA = "mlp.up_proj.lora_A.default.weight"
            self.input_ff_attr_loraB = "mlp.up_proj.lora_B.default.weight"
            self.output_ff_attr_loraA = "mlp.down_proj.lora_A.default.weight"
            self.output_ff_attr_loraB = "mlp.down_proj.lora_B.default.weight"
            self.ff_attr_lora_A = "mlp.{}.lora_A.default.weight"
            self.ff_attr_lora_B = "mlp.{}.lora_B.default.weight"
            self.attn_attr_loraA = "self_attn.{}.lora_A.default.weight"
            self.SELF_ATTENTION_MODULES = ATTN_MODUELS
            self.FFN_MODULES = FFN_MODUELS
            self.n_layers = self.model.config.num_hidden_layers
            self.n_head = self.model.config.num_attention_heads
            self.n_embd = self.model.config.hidden_size
            self.head_dim = self.n_embd // self.n_head
            self.intermediate_size = self.model.config.intermediate_size
        else:
            raise NotImplementedError
        print(self.model.config)
    
    @cuda_memory_cleanup(debug=True)
    @torch.no_grad()
    def get_top_attribute_layers(
        self,
        prompts: list,
        ground_truths: list,
        top_cnt: int = 2,
    ):
        total_attn_log_increase = [0] * self.n_layers
        total_ffn_log_increase = [0] * self.n_layers
        
        for prompt, ground_truth in tqdm(zip(prompts, ground_truths), desc="Calculating top attribute layers"):
            # Choose the meaningful last token.
            while len(ground_truth) > 0 and ground_truth[-1] in [".", "\"", " ", ")", "\n", ",", "]"]:
                ground_truth = ground_truth[:-1]
            text_sentence = prompt + ground_truth
            indexed_tokens = self.tokenizer.encode(text_sentence)[:-1]
            tokens = [self.tokenizer.decode(x) for x in indexed_tokens]
            predict_index = self.tokenizer.encode(text_sentence)[-1]
            tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
            with torch.no_grad():
                outputs = self.model(tokens_tensor,  use_cache=True)
                predictions = outputs[0]
            all_pos_layer_input, all_pos_attn_output, all_pos_residual_output, all_pos_ffn_output, all_pos_layer_output, \
            all_last_attn_subvalues, all_pos_coefficient_scores, all_attn_scores = transfer_output(outputs[2], self.n_layers)
            final_var = ((torch.var(torch.tensor(all_pos_layer_output[-1][-1]), -1, unbiased=False)+1e-5)**0.5).to(self.device)
            
            # Calculate Layer-wise Increase for MHSA
            for layer_i in range(self.n_layers):
                cur_attn_vector = torch.tensor(all_pos_attn_output[layer_i][-1]).to(self.device)
                cur_layer_input = torch.tensor(all_pos_layer_input[layer_i][-1]).to(self.device)
                origin_prob_log = torch.log(get_prob(get_bsvalues_llama(cur_layer_input, self.model, final_var))[predict_index])
                cur_attn_vector_plus = cur_attn_vector + cur_layer_input
                cur_attn_vector_bsvalues = get_bsvalues_llama(cur_attn_vector_plus, self.model, final_var)
                cur_attn_vector_probs = get_prob(cur_attn_vector_bsvalues)
                cur_attn_vector_probs = cur_attn_vector_probs[predict_index]
                cur_attn_vector_probs_log = torch.log(cur_attn_vector_probs)
                cur_attn_vector_probs_log_increase = cur_attn_vector_probs_log - origin_prob_log
                total_attn_log_increase[layer_i] += cur_attn_vector_probs_log_increase.item()

            # Calculate Layer-wise Increase for MLP
            for layer_i in range(self.n_layers):
                cur_ffn_vector = torch.tensor(all_pos_ffn_output[layer_i][-1]).to(self.device)
                cur_residual = torch.tensor(all_pos_residual_output[layer_i][-1]).to(self.device)
                origin_prob_log = torch.log(get_prob(get_bsvalues_llama(cur_residual, self.model, final_var))[predict_index])
                cur_ffn_vector_plus = cur_ffn_vector + cur_residual
                cur_ffn_vector_bsvalues = get_bsvalues_llama(cur_ffn_vector_plus, self.model, final_var)
                cur_ffn_vector_probs = get_prob(cur_ffn_vector_bsvalues)
                cur_ffn_vector_probs = cur_ffn_vector_probs[predict_index]
                cur_ffn_vector_probs_log = torch.log(cur_ffn_vector_probs)
                cur_ffn_vector_probs_log_increase = cur_ffn_vector_probs_log - origin_prob_log
                total_ffn_log_increase[layer_i] += cur_ffn_vector_probs_log_increase.tolist()

        attn_list = [[str(i), total_attn_log_increase[i]] for i in range(self.n_layers)]
        ffn_list = [[str(i), total_ffn_log_increase[i]] for i in range(self.n_layers)]
        attn_list_sort = sorted(attn_list, key=lambda x: x[-1])[::-1]
        ffn_list_sort = sorted(ffn_list, key=lambda x: x[-1])[::-1]
        attn_increase_compute = [(indx, round(increase, 3)) for indx, increase in attn_list_sort]
        ffn_increase_compute = [(indx, round(increase, 3)) for indx, increase in ffn_list_sort]
        print("attn sum: ", sum([x[1] for x in attn_increase_compute]), 
              "ffn sum: ", sum([x[1] for x in ffn_increase_compute]))
        print("attn: ", attn_increase_compute)
        print("ffn: ", ffn_increase_compute)
        top_attn_layer_idx = [int(x[0]) for x in attn_list_sort[:top_cnt]]
        top_ffn_layer_idx = [int(x[0]) for x in ffn_list_sort[:top_cnt]]
        return top_attn_layer_idx, top_ffn_layer_idx
    
    @torch.no_grad()
    @cuda_memory_cleanup(debug=True)
    def get_top_attribute_neurons_ffn(
        self, 
        prompts,
        ground_truths, 
        layer_idxs : list,
        student_dim : int,
        top=True
    ):
        ffn_scores = torch.zeros((self.n_layers, self.intermediate_size))
        for prompt, ground_truth in tqdm(zip(prompts, ground_truths), desc="get top ffn neurons by list"):
            while ground_truth[-1] in [".", "\"", " ", ")", "\n", ",", "]"]:
                ground_truth = ground_truth[:-1]
            text_sentence = prompt + ground_truth
            indexed_tokens = self.tokenizer.encode(text_sentence)[:-1]
            tokens = [self.tokenizer.decode(x) for x in indexed_tokens]
            predict_index = self.tokenizer.encode(text_sentence)[-1]
            tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
            with torch.no_grad():
                outputs = self.model(tokens_tensor, use_cache=True)
                predictions = outputs[0]
            all_pos_layer_input, all_pos_attn_output, all_pos_residual_output, all_pos_ffn_output, all_pos_layer_output, \
            all_last_attn_subvalues, all_pos_coefficient_scores, all_attn_scores = transfer_output(outputs[2], self.n_layers)
            final_var = ((torch.var(torch.tensor(all_pos_layer_output[-1][-1]), -1, unbiased=False)+1e-5)**0.5).to(self.device)
            all_ffn_subvalues = []
            for layer_i in range(self.n_layers):
                coefficient_scores = torch.tensor(all_pos_coefficient_scores[layer_i][-1]).to(self.device)
                fc2_vectors = get_fc2_params_llama(self.model, layer_i)
                ffn_subvalues = (coefficient_scores * fc2_vectors).T
                all_ffn_subvalues.append(ffn_subvalues)
            ffn_subvalue_list = []
            for layer_i in range(self.n_layers):
                cur_ffn_subvalues = all_ffn_subvalues[layer_i]
                cur_residual = torch.tensor(all_pos_residual_output[layer_i][-1]).to(self.device)
                cur_residual_subvalues = get_bsvalues_llama(cur_residual, self.model, final_var)
                origin_prob_log = torch.log(get_prob(cur_residual_subvalues)[predict_index])
                cur_ffn_subvalues_plus = cur_ffn_subvalues + cur_residual
                cur_ffn_subvalues_bsvalues = get_bsvalues_llama(cur_ffn_subvalues_plus, self.model, final_var)
                cur_ffn_subvalues_probs = get_prob(cur_ffn_subvalues_bsvalues)
                cur_ffn_subvalues_probs = cur_ffn_subvalues_probs[:, predict_index]
                cur_ffn_subvalues_probs_log = torch.log(cur_ffn_subvalues_probs)
                cur_ffn_subvalues_probs_log_increase = cur_ffn_subvalues_probs_log - origin_prob_log
                for index, ffn_increase in enumerate(cur_ffn_subvalues_probs_log_increase):
                    ffn_subvalue_list.append([str(layer_i)+"_"+str(index),  ffn_increase.item(), cur_residual_subvalues.cpu()])
            ffn_subvalue_list_sort = sorted(ffn_subvalue_list, key=lambda x: x[1])[::-1]
            for x in ffn_subvalue_list_sort:
                layer = int(x[0].split("_")[0])
                neuron = int(x[0].split("_")[1])
                score = round(x[1], 4)
                ffn_scores[layer, neuron] += score
        if top:
            ffn_neuron_idxs =  torch.argsort(ffn_scores[layer_idxs], descending=True)[:, :student_dim]
        else:
            ffn_neuron_idxs = torch.argsort(ffn_scores[layer_idxs], descending=False)[:, :student_dim]
        ffn_neuron_idxs = ffn_neuron_idxs.tolist()
        ffn_neurons = []
        for layer_idx, neuron_idx in zip(layer_idxs, ffn_neuron_idxs):
            for neuron in neuron_idx:
                ffn_neurons.append([layer_idx, neuron])
        del all_ffn_subvalues, ffn_subvalue_list, ffn_subvalue_list_sort, ffn_scores, ffn_neuron_idxs, fc2_vectors, cur_ffn_subvalues, cur_residual, cur_residual_subvalues, origin_prob_log, cur_ffn_subvalues_plus, cur_ffn_subvalues_bsvalues, cur_ffn_subvalues_probs, cur_ffn_subvalues_probs_log, cur_ffn_subvalues_probs_log_increase
        return ffn_neurons
    
    @torch.no_grad()
    @cuda_memory_cleanup(debug=True)
    def get_top_attribute_neurons_attn(
        self, 
        prompts,
        ground_truths, 
        layer_idxs : list,
        student_dim : int,
        top=True
    ):
        attn_scores = torch.zeros((self.n_layers, self.head_dim * self.n_head))
        for prompt, ground_truth in tqdm(zip(prompts, ground_truths), desc="Getting top attn neurons by list"):
            while ground_truth[-1] in [".", "\"", " ", ")", "\n", ",", "]"]:
                ground_truth = ground_truth[:-1]
            text_sentence = prompt + ground_truth
            indexed_tokens = self.tokenizer.encode(text_sentence)[:-1]
            tokens = [self.tokenizer.decode(x) for x in indexed_tokens]
            predict_index = self.tokenizer.encode(text_sentence)[-1]
            tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
            with torch.no_grad():
                outputs = self.model(tokens_tensor, use_cache=True)
                predictions = outputs[0]
            all_pos_layer_input, all_pos_attn_output, all_pos_residual_output, all_pos_ffn_output, all_pos_layer_output, \
            all_last_attn_subvalues, all_pos_coefficient_scores, all_attn_scores = transfer_output(outputs[2], self.n_layers)
            final_var = ((torch.var(torch.tensor(all_pos_layer_output[-1][-1]), -1, unbiased=False)+1e-5)**0.5).to(self.device)
            cur_file_attn_neuron_list = []
            
            for test_layer in range(self.n_layers):
                tmp_increase = []
                cur_layer_input = torch.tensor(all_pos_layer_input[test_layer]).to(self.device) 
                cur_v_heads_recompute = torch.tensor(all_last_attn_subvalues[test_layer]).permute(1, 0, 2)[-1].to(self.device)
                cur_attn_o_split = self.model.model.layers[test_layer].self_attn.o_proj.weight.data.T.view(self.n_head, self.head_dim, -1) 
                cur_attn_o_recompute = cur_attn_o_split * cur_v_heads_recompute.unsqueeze(-1)
                cur_layer_input_last = cur_layer_input[-1] 
                origin_prob_bs_values = get_bsvalues_llama(cur_layer_input_last, self.model, final_var)
                origin_prob = torch.log(get_prob(origin_prob_bs_values)[predict_index])
                del origin_prob_bs_values, cur_attn_o_split, cur_layer_input, cur_v_heads_recompute
                torch.cuda.empty_cache()
                cur_attn_o_head_plus = cur_attn_o_recompute + cur_layer_input_last
                cur_attn_o_head_plus_bs_values = get_bsvalues_llama(
                    cur_attn_o_head_plus, self.model, final_var)
                cur_attn_plus_probs = torch.log(get_prob(cur_attn_o_head_plus_bs_values)[:,  :, predict_index])
                cur_attn_plus_probs_increase = cur_attn_plus_probs - origin_prob
                for head_index in range(cur_attn_plus_probs_increase.size(0)):
                    for attn_neuron_index in range(cur_attn_plus_probs_increase.size(1)):
                            cur_file_attn_neuron_list.append((str(test_layer)+"_"+str(head_index)+"_"+str(
                                attn_neuron_index), 
                                cur_attn_plus_probs_increase[head_index][attn_neuron_index].item(),
                            ))
                cur_file_attn_neuron_list_sort = sorted(cur_file_attn_neuron_list, key=lambda x: x[1])[::-1]
                for x in cur_file_attn_neuron_list_sort:
                    layer = int(x[0].split("_")[0])
                    head = int(x[0].split("_")[1])
                    neuron = int(x[0].split("_")[2])
                    neuron_idx = head * self.head_dim + neuron
                    score = round(x[1], 4)
                    attn_scores[layer, neuron_idx] = score
        if top:
            attn_neuron_idxs =  torch.argsort(attn_scores[layer_idxs], descending=True)[:, :student_dim]
        else:
            attn_neuron_idxs = torch.argsort(attn_scores[layer_idxs], descending=False)[:, :student_dim]
        attn_neuron_idxs = attn_neuron_idxs.tolist()
        attn_neurons = []
        for layer_idx, neuron_idx in zip(layer_idxs, attn_neuron_idxs):
            for neuron in neuron_idx:
                attn_neurons.append([layer_idx, neuron])
        del cur_attn_o_head_plus, cur_attn_o_head_plus_bs_values, cur_attn_plus_probs, cur_attn_plus_probs_increase, cur_file_attn_neuron_list, cur_file_attn_neuron_list_sort, attn_scores, attn_neuron_idxs
        return attn_neurons
    
    
    @torch.no_grad()
    def extract_ffn_attn_knowledge(
        self, 
        ffn_neurons: List[List[int]],
        attn_neurons: List[List[int]],
        extract_ffn_module: List[str] = ['down_proj', 'up_proj'],
        extract_attn_module: List[str] = ['v_proj', 'o_proj']
    ):
        """Extract Neurons in MHSA and FFN"""
        knowledge_matrixs = {}
        if extract_ffn_module is not None:
            ffn_neurons_dict = collections.defaultdict(list)
            for neuron in ffn_neurons:
                layer_idx, pos = neuron
                ffn_neurons_dict[layer_idx].append(pos)
            for module_name in extract_ffn_module:
                knowledge_matrixs[module_name] = []
                for layer_idx in ffn_neurons_dict.keys():
                    neurons = ffn_neurons_dict[layer_idx]
                    column_idx = torch.tensor(neurons).to(self.device)
                    module = getattr(self.model.model.layers[layer_idx].mlp, module_name)
                    if module_name == 'down_proj':
                        knowledge_matrixs[module_name].append(torch.index_select(module.weight, 1, column_idx))
                    else:
                        knowledge_matrixs[module_name].append(torch.index_select(module.weight, 0, column_idx))
        if extract_attn_module is not None:
            attn_neurons_dict = collections.defaultdict(list)
            for neuron in attn_neurons:
                layer_idx, pos = neuron
                attn_neurons_dict[layer_idx].append(pos)
            for module_name in extract_attn_module:
                knowledge_matrixs[module_name] = []
                for layer_idx in attn_neurons_dict.keys():
                    neurons = attn_neurons_dict[layer_idx]
                    column_idx = torch.tensor(neurons).to(self.device)
                    module = getattr(self.model.model.layers[layer_idx].self_attn, module_name)
                    if module_name == 'o_proj':
                        knowledge_matrixs[module_name].append(torch.index_select(module.weight, 1, column_idx))
                    else:
                        knowledge_matrixs[module_name].append(torch.index_select(module.weight, 0, column_idx))
        for k, v in knowledge_matrixs.items():
            knowledge_matrixs[k] = torch.stack(v).to(torch.float16)
        return knowledge_matrixs
    
    
    def knowledge_injection(
        self,
        teacher_weights: dict,
        student_weights: dict, 
        ffn_layers: list = None,
        attn_layers: list = None,
        ffn_neurons: List[List[int]] = None,
        ffn_module: List[str] = None,
        attn_neurons: List[List[int]] = None,
        attn_module: List[str] = None,
    ) -> dict:
        transformer_layers = get_attributes(self.model, self.transformer_layers_attr)
        delta_weights = {}
        
        if ffn_neurons is not None:
            neurons_dict = collections.defaultdict(list)
            for neuron in ffn_neurons:
                layer_idx, pos = neuron
                neurons_dict[layer_idx].append(pos)
                
            for layer_idx, stu_idx in enumerate(ffn_layers):
                for module in ffn_module:
                    print(f"Insert Module: {module}")
                    teacher_weight = teacher_weights[module][layer_idx]
                    student_weight = student_weights[module][layer_idx]
                    task_vector = teacher_weight.to(torch.float16)
                    param = get_attributes(transformer_layers[stu_idx],
                                        self.ff_attr_format.format(module)).weight
                    
                    key = f'model.layers.{stu_idx}.mlp.{module}.delta'
                    delta = torch.zeros_like(param.data, dtype=task_vector.dtype, requires_grad=True)
                    delta_clone = delta.clone()
                    
                    if module != 'down_proj':
                        delta_clone[neurons_dict[stu_idx], :] += task_vector.to(param.dtype)
                    else:
                        delta_clone[:, neurons_dict[stu_idx]] += task_vector.to(param.dtype)
                    print(f"module: {module}, layer:{stu_idx}, delta_mean: {torch.mean(delta_clone)} max: {torch.max(delta_clone)}")
                    
                    delta_weights[key] = delta_clone
                    setattr(get_attributes(transformer_layers[stu_idx],
                                        self.ff_attr_format.format(module)), "delta", delta_clone)
        if attn_neurons is not None:
            neurons_dict = collections.defaultdict(list)
            for neuron in attn_neurons:
                layer_idx, pos = neuron
                neurons_dict[layer_idx].append(pos)
                
            for layer_idx, stu_idx in enumerate(attn_layers):
                for module in attn_module:
                    print(f"Insert Module: {module}")
                    teacher_weight = teacher_weights[module][layer_idx]
                    student_weight = student_weights[module][layer_idx]
                    task_vector = teacher_weight.to(torch.float16)
                    param = get_attributes(transformer_layers[stu_idx],
                                        self.attn_attr_format.format(module)).weight
                    
                    key = f'model.layers.{stu_idx}.self_attn.{module}.delta'
                    delta = torch.zeros_like(param.data, dtype=task_vector.dtype, requires_grad=True)
                    delta_clone = delta.clone()
                    
                    if module != 'o_proj':
                        delta_clone[neurons_dict[stu_idx], :] += task_vector.to(param.dtype)
                    else:
                        delta_clone[:, neurons_dict[stu_idx]] += task_vector.to(param.dtype)
                    print(f"module: {module}, layer:{stu_idx}, delta_mean: {torch.mean(delta_clone)} max: {torch.max(delta_clone)}")
                    
                    delta_weights[key] = delta_clone
                    setattr(get_attributes(transformer_layers[stu_idx],
                                        self.attn_attr_format.format(module)), "delta", delta_clone)
        return delta_weights