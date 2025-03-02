<div align="center">

# Pensez: Less Data, Better Reasoning – Rethinking French LLM

[**About**](#about) | [**How to Run Locally**](#run-locally) | [**Models and Datasets**](#models-and-datasets) | [**Benchmarks**](#benchmarks) | [**Training Details**](#training-details)  

![image/png](https://cdn-uploads.huggingface.co/production/uploads/630a5ef0e81e1dea2cedcec0/lbFwSuyLkixvcLWcMs7ZV.png)
</div>

## About

Pensez is a bilingual (French-English) reasoning model designed to maximize efficiency with significantly reduced training data. The model leverages a curated dataset focusing on daily reasoning tasks and scientific questions to enhance performance.

Key strategies for improved reasoning:
- **Concise reasoning** for simple tasks to prevent overthinking.
- **Extended reasoning** for complex domains like mathematics, coding, and science.
- **Special tokens (`<think>...</think>`)** to explicitly guide the model’s reasoning process.

These optimizations result in superior reasoning capabilities while maintaining robust general understanding compared to models like [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B).

## Models and Datasets

### Model Versions

Pensez is built upon [Qwen 2.5 Instruct 7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) and trained over five epochs.

| Model          | Backbone                                 | Size | Download Link |
|---------------|----------------------------------------|------|---------------|
| Pensez-v0.1-e1 | Qwen2.5-7B-Instruct | 7B  | [🤗 Pensez-v0.1-e1](https://huggingface.co/HoangHa/Pensez-v0.1-e1) |
| Pensez-v0.1-e2 | Qwen2.5-7B-Instruct | 7B  | [🤗 Pensez-v0.1-e2](https://huggingface.co/HoangHa/Pensez-v0.1-e2) |
| Pensez-v0.1-e3 | Qwen2.5-7B-Instruct | 7B  | [🤗 Pensez-v0.1-e3](https://huggingface.co/HoangHa/Pensez-v0.1-e3) |
| Pensez-v0.1-e4 | Qwen2.5-7B-Instruct | 7B  | [🤗 Pensez-v0.1-e4](https://huggingface.co/HoangHa/Pensez-v0.1-e4) |
| Pensez-v0.1-e5 | Qwen2.5-7B-Instruct | 7B  | [🤗 Pensez-v0.1-e5](https://huggingface.co/HoangHa/Pensez-v0.1-e5) |

### Dataset

Pensez was trained on the hand-curated [Pensez v0.1](https://huggingface.co/datasets/HoangHa/Pensez-v0.1) dataset containing 2,000 samples (1,000 French, 1,000 English).

| Dataset       | Description          | Size  | Link  |
|--------------|----------------------|-------|-------|
| Pensez v0.1 | SFT Training Dataset | 2K samples | [🤗 Pensez v0.1](https://huggingface.co/datasets/HoangHa/Pensez-v0.1) |

## Benchmarks

Pensez was evaluated on French-specific benchmarks, demonstrating strong reasoning ability and improved task-specific performance:

| Benchmark | Pensez-v0.1-e5 | DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-7B-Instruct |
|-----------|---------------|-----------------------------|----------------------|
| Math-hard (fr)      | 0.3458        | 0.3403   | 0.2253           |
| MMLU (fr)           | 0.5766        | 0.4961   | 0.6612           |
| Boolqa (fr)         | 0.9157        | 0.7079   | 0.9382           |
| GPQA diamond (fr)   | 0.2893        | 0.2792   | 0.3452           |
| BBH (fr)           | 0.5886        | 0.5941   | 0.6039           |
| Trivia (en)        | 0.4421        | 0.2711   | 0.5316           |
| Hellaswag (en)     | 0.5050        | 0.3540   | 0.5258           |
| AIME25 (en)        | 0.2333        | 0.3000   | 0.0333           |

**Key Observations:**
- Pensez outperforms Qwen2.5-7B-Instruct in reasoning tasks.
- Comparable to DeepSeek-R1-Distill-Qwen-7B in reasoning while maintaining strong understanding.
- Reduced degradation in knowledge-based tasks.

<details>
<summary>Click for detailed benchmark results</summary>

| Tasks                                          | Pensez v0.1 e1 | Pensez v0.1 e2 | Pensez v0.1 e3 | Pensez v0.1 e4 | Pensez v0.1 e5 | Qwen 7B instruct | R1 distil |
|------------------------------------------------|---------------|---------------|---------------|---------------|---------------|-----------------|-----------|
| leaderboard_math_hard_fr                       | 0.0918        | 0.2547        | 0.2783        | 0.3035        | 0.3458        | 0.2253          | 0.3403    |
| leaderboard_math_algebra_hard_fr               | 0.1029        | 0.3914        | 0.3971        | 0.5114        | 0.5000        | 0.4229          | 0.4771    |
| leaderboard_math_counting_and_prob_hard_fr     | 0.0765        | 0.1378        | 0.1939        | 0.2041        | 0.2398        | 0.1224          | 0.2347    |
| leaderboard_math_geometry_hard_fr              | 0.0388        | 0.1019        | 0.1408        | 0.1359        | 0.1748        | 0.1019          | 0.2330    |
| leaderboard_math_num_theory_hard_fr            | 0.1198        | 0.2581        | 0.3502        | 0.3548        | 0.4332        | 0.3180          | 0.3963    |
| leaderboard_math_prealgebra_hard_fr            | 0.1681        | 0.4425        | 0.4690        | 0.4956        | 0.5841        | 0.3274          | 0.4867    |
| leaderboard_math_precalculus_hard_fr           | 0.0357        | 0.0714        | 0.1190        | 0.1190        | 0.1429        | 0.0595          | 0.2143    |
| leaderboard_mmlu_fr                            | 0.3806        | 0.3329        |    -          |      -        | 0.5766        | 0.6612          | 0.4961    |
| french_bench_arc_challenge                     | 0.5047        | 0.5021        | 0.4919        | 0.4859        | 0.4842        | 0.5518          | 0.3447    |
| french_bench_boolqa                            | 0.9326        | 0.9326        | 0.9326        | 0.9270        | 0.9157        | 0.9382          | 0.7079    |
| french_bench_fquadv2                           | 0.4325        | 0.4400        | 0.4412        | 0.4375        | 0.4387        | 0.4800          | 0.2988    |
| french_bench_hellaswag                         | 0.4970        | 0.5055        | 0.5092        | 0.5058        | 0.5050        | 0.5258          | 0.3540    |
| french_bench_trivia                            | 0.4763        | 0.4763        | 0.4553        | 0.4395        | 0.4421        | 0.5316          | 0.2711    |

</details>

## Run Locally

You can run Pensez using Hugging Face’s `transformers` library:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "HoangHa/Pensez-v0.1-e5"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto"
)

# Example input
messages = [{"role": "user", "content": "Bonjour!"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to("cuda")

generated_ids = model.generate(input_ids, max_new_tokens=2500, temperature=0.8, repetition_penalty=1.1, do_sample=True, eos_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_space=True)
print(f"Réponse: {response}")
```

## Training Details

Pensez was trained with:
- **Packing Inputs Without Cross-Contamination Attention** ([Reference](https://github.com/MeetKai/functionary/tree/main/functionary/train/packing))
- **Liger Kernel** ([Reference](https://github.com/linkedin/Liger-Kernel))
- **DeepSpeed 3** ([Reference](https://github.com/deepspeedai/DeepSpeed))
- **NEFTune Noise** ([Reference](https://arxiv.org/abs/2310.05914)) for robustness.

| **Parameter** | **Value** |
|--------------|----------|
| Epochs | 5 |
| Global Batch Size | 200 |
| Learning Rate | 1e-5 |
| Scheduler | Cosine |
| Optimizer | AdamW |
| Warmup Ratio | 0.05 |
| Weight Decay | 0.01 |
| Max Sequence Length | 16,384 |

More details: [Training Config](https://huggingface.co/HoangHa/Pensez-v0.1-e5/blob/main/fr_full_sft.yaml) | Loss curves: [Wandb](https://wandb.ai/hahuyhoanghhh41/llamafactory?nw=nwuserhahuyhoanghhh41)

## Citation

```bibtex
@misc{ha2025pensezreasoningfrenchllm,
      title={Pensez: Less Data, Better Reasoning – Rethinking French LLM},
      author={Ha Huy Hoang},
      year={2025},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={},
}
```


## Acknowledgement

- [llama-factory](https://github.com/hiyouga/LLaMA-Factory)
- [Deepseek R1](https://github.com/deepseek-ai/DeepSeek-R1)
- [Qwen 2.5](https://github.com/QwenLM/Qwen2.5)
- [NEFTune Noise](https://arxiv.org/abs/2310.05914)
- [Packing Inputs Without Cross-Contamination Attention](https://github.com/MeetKai/functionary/tree/main/functionary/train/packing)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
- [Deepspeed](https://github.com/deepspeedai/DeepSpeed)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Hyperbolic](https://hyperbolic.xyz/)
- [Modal](https://modal.com/)