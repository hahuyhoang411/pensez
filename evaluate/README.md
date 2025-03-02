# Quick Start Guide

This guide will help you evaluate Pensez using two benchmarks: Frenchbench and the French Leaderboard. Follow the steps below to set up your environment and run evaluations.

## Installation Dependencies

To get started, clone the repository and install the necessary dependencies:

1. Clone the repository:
   ```bash
   git clone https://github.com/hahuyhoang411/lm-evaluation-harness-multilingual.git
   cd lm-evaluation-harness-multilingual
   ```

2. Install the required Python packages:

    ```bash
    pip install -e .
    pip install antlr4-python3-runtime==4.11 antlr4-tools langdetect immutabledict
    ```

## Login to Hugging Face

Authenticate with Hugging Face to access models:

    ```bash
    huggingface-cli login
    ```


# # Frenchbench
The original [Frenchbench benchmark]((https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/french_bench)) is used for evaluation.

## Hugging Face Backend
Run the following command to evaluate using the Hugging Face backend:

    ```bash
    accelerate launch -m lm_eval --model hf --model_args pretrained=HoangHa/Pensez-v0.1-e1,dtype="bfloat16" --tasks french_bench --num_fewshot 5 --batch_size auto --output_path data/french_bench/pensez-v0.1-e1/results_french_bench_5shot.json --trust_remote_code --apply_chat_template --fewshot_as_multiturn --log_samples --write_out
    ```

## Vllm backend

    ```bash

    ```
    
# French Leaderboard

The original [French Leaderboard benchmark](https://github.com/mohamedalhajjar/lm-evaluation-harness-multilingual/blob/main/lm_eval/tasks/leaderboard-french/README.md) has been modified to change the `output_type` from `multiple_choice` to `generate_until` to better evaluate reasoning models.

## HuggingFace backend 

Use the following command to evaluate using the Hugging Face backend:

    ```bash
    accelerate launch -m lm_eval --model hf --model_args pretrained=HoangHa/Pensez-v0.1-e5,dtype="bfloat16" --tasks=leaderboard-fr --batch_size auto:2 --output_path data/pensez-e5 --trust_remote_code --apply_chat_template --log_samples --write_out
    ```

## Vllm backend

To evaluate using the Vllm backend, run:

    ```bash
    lm_eval --model vllm --model_args="pretrained=HoangHa/Pensez-v0.1-e5,dtype=bfloat16,gpu_memory_utilization=0.8,max_model_len=16384,data_parallel_size=2" --tasks=leaderboard-fr --batch_size=8 --apply_chat_template --log_samples --write_out --output_path data/pensez-e5
    ```