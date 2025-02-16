# 🤯Neural Incompatibility: The Unbridgeable Gap of Cross-Scale Parametric Knowledge Transfer in Large Language Models


This repository is built for the paper "**Neural Incompatibility: The Unbridgeable Gap of Cross-Scale Parametric Knowledge Transfer in Large Language Models**".
> Now we only provide the code for **LaTen** (Pre-Align PKT). We will release the full code for systematically evaluate both Pre-Align and Post-Align PKT soon.
<div  align="center">  
<img src="figures/icon.jpg" width="30%"> 
</div>

## 🧠 Overview

In this work, we investigate the infeasibility of cross-scale **Parametric Knowledge Transfer (PKT)** in Large Language Models (LLMs). Through in-depth analysis, we identify **Neural Incompatibility** as the ethological and parametric structural differences between LLMs of varying scales, presenting fundamental challenges to achieving effective PKT.

1. We are the first to comprehensively define and explore parametric knowledge transfer between cross-scale LLMs. The meticulous categorization of PKT is seperate into Post-Align PKT and Pre-Align PKT based on the **Alignment** time.
2. We propose a novel method **Locate-Then-Align (LaTen)** to first try to solve **Pre-Align challenge** (also first proposed in our paper), which leverages neuron attribution and hypernetwork techniques to execute alignment with minimal training data achieves promissing performance.
  

The comparison between LaTen and language-based transfer methods is shown as follows,
<div  align="center">  
<img src="figures/Intro.jpg" width="80%"> 
</div>

By conducting extensive experiments, we find that: 

* **Neural Incompatibility** which are similar to the cross-species neural mechanism as a key challenge arising from low-similarity of both ethological and parametric structural in cross-scale LLMs. 



## 🤖 Language Models and Datasets 
We conduct experiments on decoder-based LLM: Llama-2-7b-chat-hf and Llama-2-13b-chat-hf.
* We evaluate three tasks on four datasets: MMLU (Professional Knowledge), GSM8K (mathematical reasoning), and HumanEval and MBPP (code-generating).
Note that we provide our split MMLU, GSM8K, HumenEval, and MBPP datasets in `data/` folder.

## 🚀 Setting Up Environment

```
conda create --name Laten python=3.10
conda activate Laten
pip install -r requirements.txt
```

## 👨‍💻 Training **LaTen**
During the training process, LaTen contains three steps:
* **Knowledge Extraction**: We implement it based on neuron-level attribution method.
* **Parameter Alignment**: We use MLP hypernetwork to align the parameters of two scales.
* **Knowledge Injection**: We modify the source code of Llama-2 to make gradient flow backpropagate to the hypernetwork.

Note that the process can't parallelize, so we use 2 A100 GPUs for training (also for inference).

You can use the following command to train LaTen on GSM8K dataset. We also provide the training script in `train_LaTen.sh`. 
> For MMLU, we need to set `train_on_inputs` to `True` because the original answer part is only 1 character which is hard to calculate stable gradient.
```
python train_LaTen.py \
    --source_model "meta-llama/Llama-2-13b-chat-hf" \
    --target_model "meta-llama/Llama-2-7b-chat-hf" \
    --learning_rate 3e-5 \
    --transfer_rate 0.1 \
    --extract_data_path "data/gsm/gsm_extract_split.jsonl" \
    --align_data_path "data/gsm/gsm_align_split.jsonl" \
    --translator_save_name "gsm-transfer_rate0.1-lr3e-5" \
    --cutoff_len 1024 \
    --seed 42 \
    --train_on_inputs False\
```

## 📝 Inference **LaTen**
After training, we obtain the checkpoint file of hypernetwork (in code called `translator`). You can use the following command to inference with it. We also provide the inference script in `inference_LaTen.sh`.
```
python inference_LaTen.py \
    --source_model "meta-llama/Llama-2-13b-chat-hf" \
    --target_model "meta-llama/Llama-2-7b-chat-hf" \
    --data_path "data/gsm/gsm_train_split.jsonl" \
    --translator_checkpoints "./knowledge_translator/gsm-transfer_rate0.1-lr3e-5" \
    --transfer_rates "[0.1]" \
    --steps "[0]" \
    --seed 42 \
```
## 📊 Evaluation
For the evaluation, we employ [Open-Instruct](https://github.com/allenai/open-instruct) to evaluate the model across various benchmarks. 

## Acknowledgments

We are grateful to the authors of [ParaKnowTransfer](https://github.com/maszhongming/ParaKnowTransfer/), [Knowledge Neurons](https://github.com/EleutherAI/knowledge-neurons), [neuron-attribution](https://github.com/zepingyu0512/neuron-attribution) and [open-instruct](https://github.com/allenai/open-instruct) for making their project codes publicly available. We build our project based on these great works.

