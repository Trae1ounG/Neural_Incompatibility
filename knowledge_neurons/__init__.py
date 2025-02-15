from transformers import (
    BertTokenizer,
    BertLMHeadModel,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM
)
from .knowledge_neurons import KnowledgeNeurons

BERT_MODELS = ["bert-base-uncased", "bert-base-multilingual-uncased"]
GPT2_MODELS = ["gpt2"]
GPT_NEO_MODELS = [
    "EleutherAI/gpt-neo-125m",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
]
LLAMA_MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf"
]
ALL_MODELS = BERT_MODELS + GPT2_MODELS + GPT_NEO_MODELS + LLAMA_MODELS


def initialize_model_and_tokenizer(model_name: str):
    if model_name in BERT_MODELS:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertLMHeadModel.from_pretrained(model_name)
    elif model_name in GPT2_MODELS:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    elif model_name in GPT_NEO_MODELS:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name)
    elif model_name in LLAMA_MODELS:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(model_name)
    else:
        # raise ValueError("Model {model_name} not supported")
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cuda:0")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0")
    model.eval()

    return model, tokenizer


def model_type(model_name: str):
    # return "gpt_neo"
    if model_name in BERT_MODELS:
        return "bert"
    elif model_name in GPT2_MODELS:
        return "gpt2"
    elif model_name in GPT_NEO_MODELS:
        return "gpt_neo"
    elif model_name in LLAMA_MODELS:
        return "llama"
    else:
        raise ValueError("Model {model_name} not supported")

