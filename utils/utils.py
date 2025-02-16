import random 
import torch
import numpy as np
from typing import List
from typing import Dict
import random
import numpy as np
import torch
import torch.nn as nn

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

def generate_and_tokenize_prompt(data_point, tokenizer, cutoff_len=1024, train_on_inputs=False):
    """
    Tokenize and process chat prompts for model training/inference
    
    Args:
        data_point: Dictionary containing message data
        tokenizer: Tokenizer to use for processing
        cutoff_len: Maximum sequence length
        train_on_inputs: Whether to train on input sequences
    
    Returns:
        Dictionary containing tokenized prompts and metadata
    """
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

    assert "messages" in data_point
    full_prompt = generate_chat_prompt(
        data_point["messages"]
    )
        
    tokenized_full_prompt = tokenize(full_prompt)
    
    if not train_on_inputs:
        user_prompt = generate_chat_prompt(
            data_point["messages"][:-1]
        )
        user_prompt += "<|assistant|>\n"
       
        tokenized_user_prompt = tokenize(
            user_prompt
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]
        
    tokenized_full_prompt['user_prompt'] = user_prompt
    tokenized_full_prompt['response'] = full_prompt[len(user_prompt):]
    tokenized_full_prompt['full_prompt'] = full_prompt
    return tokenized_full_prompt