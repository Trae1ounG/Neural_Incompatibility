import os
import json
import random 
import torch
import numpy as np
from typing import List
import re
import os
from typing import Dict
import random
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import Trainer, TrainerState

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def pad_sequence(seq, max_len, num):
                if len(seq) > max_len:
                    return seq[:max_len]
                return [num] * (max_len - len(seq)) + seq
            
def generate_chat_prompt(
        messages: List[Dict[str, str]] = None,
    ) -> str:
        # returns the full prompt for a conversation history
        if len(messages) == 0:
            raise ValueError('Messages field is empty.')
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + "\n\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
    
def get_attributes(x: nn.Module, attributes: str):
        """
        gets a list of period-separated attributes
        i.e get_attributes(model, 'transformer.encoder.layer')
            should return the same as model.transformer.encoder.layer
        """
        for attr in attributes.split("."):
            x = getattr(x, attr)
        return x