"""
Based on https://github.com/zepingyu0512/neuron-attribution
"""
import os
import json
import torch
import pickle
import json
import gc
import uuid
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
from collections import Counter
import functools
from typing import Optional

def get_fc2_params_llama(model, layer_num):
    return model.model.layers[layer_num].mlp.down_proj.weight.data

def get_bsvalues_llama(vector, model, final_var):
    vector = vector * torch.rsqrt(final_var + 1e-6)
    vector_rmsn = vector * model.model.norm.weight.data
    vector_rmsn = vector_rmsn.to(torch.float16)
    vector_bsvalues = model.lm_head(vector_rmsn).data
    return vector_bsvalues

def transfer_output(model_output, LAYER_NUM):
    all_pos_layer_input = []
    all_pos_attn_output = []
    all_pos_residual_output = []
    all_pos_ffn_output = []
    all_pos_layer_output = []
    all_last_attn_subvalues = []
    all_pos_coefficient_scores = []
    all_attn_scores = []
    for layer_i in range(LAYER_NUM):
        cur_layer_input = model_output[layer_i][0]
        cur_attn_output = model_output[layer_i][1]
        cur_residual_output = model_output[layer_i][2]
        cur_ffn_output = model_output[layer_i][3]
        cur_layer_output = model_output[layer_i][4]
        cur_last_attn_subvalues = model_output[layer_i][5]
        cur_coefficient_scores = model_output[layer_i][6]
        cur_attn_weights = model_output[layer_i][7]
        # print(cur_layer_input.shape)
        all_pos_layer_input.append(cur_layer_input[0].tolist())
        all_pos_attn_output.append(cur_attn_output[0].tolist())
        all_pos_residual_output.append(cur_residual_output[0].tolist())
        all_pos_ffn_output.append(cur_ffn_output[0].tolist())
        all_pos_layer_output.append(cur_layer_output[0].tolist())
        all_last_attn_subvalues.append(cur_last_attn_subvalues[0].tolist())
        all_pos_coefficient_scores.append(cur_coefficient_scores[0].tolist())
        all_attn_scores.append(cur_attn_weights)
    return all_pos_layer_input, all_pos_attn_output, all_pos_residual_output, all_pos_ffn_output, \
           all_pos_layer_output, all_last_attn_subvalues, all_pos_coefficient_scores, all_attn_scores
           
def get_bsvalues(vector, model, final_var):
    E = torch.mean(vector, -1)
    vector_ln = (vector-E.unsqueeze(-1))/final_var * model.transformer.ln_f.weight.data
    vector_bsvalues = model.lm_head(vector_ln).data
    return vector_bsvalues

def get_prob(vector):
    prob = torch.nn.Softmax(-1)(vector)
    return prob

def get_fc2_params(model, layer_num):
    return model.transformer.h[layer_num].mlp.c_proj.weight.data.T

def transfer_l(l):
    new_x, new_y = [], []
    for x in l:
        new_x.append(x[0])
        new_y.append(x[1])
    return new_x, new_y
    
def cuda_memory_cleanup(threshold_mb: Optional[float] = None, debug: bool = False):
    """
    A decorator that automatically cleans up CUDA memory after function execution.
    
    Args:
        threshold_mb (float, optional): If set, only clean memory if allocated memory exceeds this threshold (in MB)
        debug (bool): If True, print memory stats before and after cleanup
        
    Example:
        @cuda_memory_cleanup(threshold_mb=1000, debug=True)
        def my_gpu_intensive_function():
            # Your GPU operations here
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not torch.cuda.is_available():
                return func(*args, **kwargs)
            initial_allocated = torch.cuda.memory_allocated()
            initial_reserved = torch.cuda.memory_reserved()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                current_allocated = torch.cuda.memory_allocated()
                current_reserved = torch.cuda.memory_reserved()
                current_allocated_mb = current_allocated / (1024 * 1024)
                should_clean = True if threshold_mb is None else current_allocated_mb > threshold_mb
                if should_clean:
                    if debug:
                        print(f"Before cleanup - Allocated: {current_allocated_mb:.2f}MB, "
                                   f"Reserved: {current_reserved/(1024*1024):.2f}MB")
                    torch.cuda.empty_cache()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.ipc_collect()    
                    if debug:
                        final_allocated = torch.cuda.memory_allocated()
                        final_reserved = torch.cuda.memory_reserved()
                        cleaned_allocated = (current_allocated - final_allocated) / (1024 * 1024)
                        cleaned_reserved = (current_reserved - final_reserved) / (1024 * 1024)
                        print(f"After cleanup - Allocated: {final_allocated/(1024*1024):.2f}MB, "
                                   f"Reserved: {final_reserved/(1024*1024):.2f}MB")
                        print(f"Cleaned - Allocated: {cleaned_allocated:.2f}MB, "
                                   f"Reserved: {cleaned_reserved:.2f}MB")
        
        return wrapper
    return decorator