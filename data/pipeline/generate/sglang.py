import argparse
import asyncio
import hashlib
import json
import os
import random
from asyncio import Lock
from typing import Set, Dict, Tuple
from dataclasses import dataclass

from datasets import load_dataset
from tqdm.asyncio import tqdm

import aiofiles
import aiohttp
import uvloop


# Import the prompts from the new module
from prompts import (
    TRANSLATION_PROMPT,
    THINKING_PROMPT,
    CLASSIFICATION_PROMPT,
    CONCLUSION_PROMPT,
    DEFAULT_USER_PROMPT_TEMPLATE,
    USE_SYSTEM_PROMPT
)

file_lock = Lock()

# Default system prompt - can be changed via command line arguments
DEFAULT_SYSTEM_PROMPT = CLASSIFICATION_PROMPT


@dataclass
class ProcessedExample:
    uuid: str
    num_generations: int
    existing_generations: list
    existing_finish_reasons: list
    existing_api_metadata: list


async def generate_completion(session, prompt, args):
    retry_budget = 10
    while retry_budget > 0:
        try:
            await asyncio.sleep(random.uniform(0.0, 0.1))
            
            # Build messages based on whether to use system prompt
            messages = []
            if args.use_system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            async with session.post(
                f"http://{args.api_addr}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": messages,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
                headers={"Authorization": "Bearer EMPTY"},
            ) as response:
                return await response.json(content_type=None)
        except Exception as e:
            print(f"API error (will retry): {e}")
            retry_budget -= 1
            await asyncio.sleep(10)
    return None


async def generate_with_retry(session, prompt, args):
    """Generate a completion with validation and retry logic."""
    for attempt in range(args.max_retries):
        try:
            completion = await generate_completion(session, prompt, args)
            
            if completion is None:
                print(f"API returned None (attempt {attempt+1}/{args.max_retries})")
                continue
                
            generation = completion["choices"][0]["message"]["content"]
            finish_reason = completion["choices"][0]["finish_reason"]
            
            # Check for error conditions
            has_boxed_format = "\\boxed{" in generation
            is_length_truncated = finish_reason == "length"
            
            # Apply validation based on args
            format_valid = not args.require_boxed_format or has_boxed_format
            not_truncated = not args.reject_truncated or finish_reason != "length"
            
            if format_valid and not_truncated:
                # Success - valid generation
                return {
                    "content": generation,
                    "finish_reason": finish_reason,
                    "api_metadata": completion["usage"]
                }
            else:
                # Log the error
                if not has_boxed_format and args.require_boxed_format:
                    print(f"Missing boxed format (attempt {attempt+1}/{args.max_retries})")
                if is_length_truncated and args.reject_truncated:
                    print(f"Response truncated (attempt {attempt+1}/{args.max_retries})")
                
                # Add a short delay before retry
                await asyncio.sleep(random.uniform(0.5, 2.0))
        
        except Exception as e:
            print(f"Error during generation (attempt {attempt+1}/{args.max_retries}): {e}")
            await asyncio.sleep(random.uniform(1.0, 5.0))
    
    # If we get here, all retries failed
    return None


async def process_example(
    example, 
    session, 
    args, 
    output_file, 
    pbar,
    processed_info: ProcessedExample = None
):
    # Check if the prompt column value is None
    if example[args.prompt_column] is None:
        # Create a result with None values for generations
        result = {
            **example,
            "generations": None,
            "finish_reasons": None,
            "api_metadata": None,
        }
        
        # Write to file with lock
        async with file_lock:
            async with aiofiles.open(output_file, mode="a") as f:
                await f.write(json.dumps(result) + "\n")
                await f.flush()
        
        pbar.update(1)
        return result
    
    prompt = args.prompt_template.format(prompt=example[args.prompt_column])
    
    # Initialize with existing data if continuing
    generations = processed_info.existing_generations if processed_info else []
    finish_reasons = processed_info.existing_finish_reasons if processed_info else []
    api_metadata = processed_info.existing_api_metadata if processed_info else []
    
    # Calculate remaining generations needed
    remaining_generations = args.num_generations - len(generations)
    
    if remaining_generations > 0:
        try:
            # Create a task for each remaining generation
            raw_generation_results = []
            for _ in range(remaining_generations):
                gen_result = await generate_with_retry(session, prompt, args)
                if gen_result:
                    raw_generation_results.append(gen_result)
                else:
                    print(f"Failed to generate after {args.max_retries} retries")
                    # If we completely failed to generate, update with what we have
                    if generations:
                        pbar.update(1)
                        return {
                            **example,
                            "generations": generations,
                            "finish_reasons": finish_reasons,
                            "api_metadata": api_metadata,
                        }
                    else:
                        pbar.update(1)
                        return None

            # Add all successful generations
            for result in raw_generation_results:
                generations.append(result["content"])
                finish_reasons.append(result["finish_reason"])
                api_metadata.append(result["api_metadata"])

            # Combine original dataset fields with generations
            result = {
                **example,  # Preserve all original dataset fields
                "generations": generations,
                "finish_reasons": finish_reasons,
                "api_metadata": api_metadata,
            }

            # Write to file with lock
            async with file_lock:
                # If continuing, we need to update the existing line
                if processed_info and args.continue_incomplete:
                    await update_existing_line(output_file, example[args.uuid_column], result)
                else:
                    async with aiofiles.open(output_file, mode="a") as f:
                        await f.write(json.dumps(result) + "\n")
                        await f.flush()

            pbar.set_postfix(active=len(pbar.active_tasks), refresh=False)
            pbar.update(1)

            return result
        except Exception as e:
            print(f"Error processing example: {e}")
            pbar.update(1)
            return None
    else:
        pbar.update(1)
        return None


async def update_existing_line(output_file, uuid, new_data):
    """Update an existing line in the output file."""
    temp_file = f"{output_file}.temp"
    async with aiofiles.open(output_file, mode="r") as f_in:
        async with aiofiles.open(temp_file, mode="w") as f_out:
            async for line in f_in:
                data = json.loads(line)
                if str(data[args.uuid_column]) == str(uuid):
                    await f_out.write(json.dumps(new_data) + "\n")
                else:
                    await f_out.write(line)
    
    os.replace(temp_file, output_file)


async def load_processed_examples(output_file, uuid_column) -> Dict[str, ProcessedExample]:
    processed_examples = {}
    if os.path.exists(output_file):
        async with aiofiles.open(output_file, mode="r") as f:
            async for line in f:
                try:
                    data = json.loads(line)
                    uuid = hashlib.md5(str(data[uuid_column]).encode()).hexdigest()
                    processed_examples[uuid] = ProcessedExample(
                        uuid=uuid,
                        num_generations=len(data.get("generations", [])),
                        existing_generations=data.get("generations", []),
                        existing_finish_reasons=data.get("finish_reasons", []),
                        existing_api_metadata=data.get("api_metadata", [])
                    )
                except json.JSONDecodeError:
                    continue
    return processed_examples


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-sub", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--prompt-column", type=str, required=True)
    parser.add_argument("--uuid-column", type=str, required=True)
    parser.add_argument("--api-addr", type=str, default="localhost:39876")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--continue-incomplete", action="store_true", 
                       help="Continue processing examples with fewer generations than requested")
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=DEFAULT_USER_PROMPT_TEMPLATE,
        help="Template for formatting user prompts"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to use for chat completions"
    )
    parser.add_argument(
        "--use-system-prompt",
        action="store_true",
        default=USE_SYSTEM_PROMPT,
        help="Whether to use the system prompt (true) or user-only messages (false)"
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-concurrent", type=int, default=1000)
    
    # Add retry-related arguments
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed or invalid generations"
    )
    parser.add_argument(
        "--require-boxed-format",
        action="store_true",
        default=False,
        help="Require \\boxed{} format in the response"
    )
    parser.add_argument(
        "--reject-truncated",
        action="store_true",
        default=True,
        help="Reject and retry generations with 'length' finish reason"
    )
    
    # Add prompt selection argument
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["translation", "thinking", "classification", "conclusion"],
        default="classification",
        help="Type of system prompt to use"
    )
    
    args = parser.parse_args()
    
    # Select the appropriate system prompt based on the prompt-type argument
    if args.prompt_type == "translation":
        args.system_prompt = TRANSLATION_PROMPT
    elif args.prompt_type == "thinking":
        args.system_prompt = THINKING_PROMPT
    elif args.prompt_type == "classification":
        args.system_prompt = CLASSIFICATION_PROMPT
    elif args.prompt_type == "conclusion":
        args.system_prompt = CONCLUSION_PROMPT

    # Load the dataset
    dataset = load_dataset(args.dataset_name, args.dataset_sub, split="train")
    processed_examples = await load_processed_examples(args.output_file, args.uuid_column)
    
    # Calculate total work needed
    total_remaining = 0
    for example in dataset:
        uuid = hashlib.md5(str(example[args.uuid_column]).encode()).hexdigest()
        if uuid in processed_examples:
            if args.continue_incomplete:
                remaining = args.num_generations - processed_examples[uuid].num_generations
                if remaining > 0:
                    total_remaining += 1
        else:
            total_remaining += 1
    
    print(f"Found {len(processed_examples)} processed examples")
    if args.continue_incomplete:
        incomplete = sum(1 for info in processed_examples.values() 
                       if info.num_generations < args.num_generations)
        print(f"Found {incomplete} examples with incomplete generations")
    
    if not os.path.exists(args.output_file):
        async with aiofiles.open(args.output_file, mode="w") as f:
            await f.write("")

    active_tasks: Set[asyncio.Task] = set()

    pbar = tqdm(
        total=total_remaining,
        desc="Generating responses",
        unit="row",
        mininterval=2,
        smoothing=0.0001,
    )
    pbar.active_tasks = active_tasks

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=60 * 60),
        connector=aiohttp.TCPConnector(limit=args.max_concurrent, ttl_dns_cache=300, keepalive_timeout=60 * 60),
    ) as session:
        for example in dataset:
            uuid = hashlib.md5(str(example[args.uuid_column]).encode()).hexdigest()
            should_process = False
            processed_info = None
            
            if uuid in processed_examples:
                if args.continue_incomplete:
                    info = processed_examples[uuid]
                    if info.num_generations < args.num_generations:
                        should_process = True
                        processed_info = info
            else:
                should_process = True
            
            if should_process:
                # Wait if we've hit the concurrency limit
                while len(active_tasks) >= args.max_concurrent:
                    done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        try:
                            await task
                        except Exception as e:
                            print(f"Task failed: {e}")

                task = asyncio.create_task(
                    process_example(example, session, args, args.output_file, pbar, processed_info)
                )
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)

                pbar.set_postfix(active=len(active_tasks), refresh=True)

        # Wait for remaining tasks
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

    pbar.close()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(main())
