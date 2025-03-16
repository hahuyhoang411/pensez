# !pip install wandb unsloth vllm
# !pip install --upgrade pillow

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from unsloth import is_bfloat16_supported
import torch

from datasets import load_dataset
max_seq_length = 16384 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "HoangHa/Pensez-v0.1-e5", #
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.4 # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank*2,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)


dataset = load_dataset("HoangHa/Pensez-GRPO-formatted", split='train')

# Helper
from collections import defaultdict
from itertools import islice, zip_longest
import re

def check_reflection_pattern(response: str) -> dict[str, int]:
    reflection_base_words = [
        "wait",
        "recheck",
        "retry",
        "alternatively",
        "however",
        "verify",
        "actually",
        "let me think",
    ]
    
    reflection_base_words.sort(key=len, reverse=True)
    
    res = defaultdict(int)
    for word in reflection_base_words:
        pattern = r'(?:\b|^)' + re.escape(word) + r'(?:[,\s.!?]|$)'
        res[word] = len(re.findall(pattern, response.lower()))
    
    return res

def extract_boxed_answer(text: str) -> str:
    """Extract the content within the last \\boxed{} in the text."""
    # Pattern to match \boxed{...}, capturing the content inside
    pattern = r"\\boxed{(.*?)}"
    matches = re.findall(pattern, text)
    if matches:
        # Return the last match, stripped of whitespace
        return matches[-1].strip()
    # Return empty string if no \boxed{} is found
    return ""

def repeatness(s: str) -> float:
    """Calculate repetition score using suffix arrays and LCP (from your original code)."""
    def ranks(l):
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    def suffixArray(s):
        line = ranks(s)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        n, ans, k = len(arr), [0] * len(arr), 0
        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue
            j = suffixArr[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1
            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1
        return ans

    arr = [ord(i) for i in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)
    cnt = sum(lcp(arr, sa, c))
    return cnt * 2 / (n * (n + 1))

# Reward functions

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_boxed_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def repetition_penalty_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Penalty for repetitive content in completions."""
    responses = [completion[0]["content"] for completion in completions]
    repeat_penalty = 1
    rep_scores = [repeatness(r) for r in responses]  # Normalized 0 to <1
    return [-repeat_penalty * score for score in rep_scores]
    
def reflection_bonus_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Bonus for reflection patterns: 0.05 per point up to 20, then fixed at 1.0 for 20 to 50, and 0 beyond 50."""
    responses = [completion[0]["content"] for completion in completions]
    reflection_scores = []
    for r in responses:
        reflection_dict = check_reflection_pattern(r)
        reflection_count = sum(reflection_dict.values())
        if reflection_count <= 50:
            score = min(reflection_count, 20) * 0.05
        else:
            score = 0.0
        reflection_scores.append(score)
    return reflection_scores

def length_bonus_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Bonus for responses within a target token length range."""
    responses = [completion[0]["content"] for completion in completions]
    length_bonus = 0.5
    if not tokenizer:
        raise ValueError("Tokenizer required for length_bonus_reward_func")
    
    token_counts = [len(tokenizer.encode(r)) for r in responses]
    length_scores = []
    min_edge, low_target, high_target, max_edge = 4096, 4096*2, 4096*3, 4096*4
    
    for count in token_counts:
        if count < min_edge:
            score = 0.0
        elif min_edge <= count < low_target:
            score = (count - min_edge) / (low_target - min_edge)
        elif low_target <= count <= high_target:
            score = 1.0
        elif high_target < count <= max_edge:
            score = (max_edge - count) / (max_edge - high_target)
        else:
            score = 0.0
        length_scores.append(score)
    return [length_bonus * score for score in length_scores]

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 1e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 16, # Decrease if out of memory
    max_prompt_length = 2048,
    max_completion_length = 14336,
    num_train_epochs = 5, # Set to 1 for a full training run
    # max_steps = 250,
    save_strategy = "epoch",
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases wandb
    output_dir = "outputs",
)
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        correctness_reward_func,
        repetition_penalty_reward_func,
        reflection_bonus_reward_func,
        length_bonus_reward_func
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()