#!/usr/bin/env python3
"""
Generate LLM (GPT-3.5) responses for test samples.
This script generates responses using OpenAI's GPT-3.5 model and records
latency, token costs, and automatic metrics.

"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# Import OpenAI - support both old and new versions
try:
    from openai import OpenAI  # New version (>=1.0.0)
    OPENAI_NEW_VERSION = True
except ImportError:
    import openai  # Old version (<1.0.0)
    OPENAI_NEW_VERSION = False

# Import shared prompt templates
from prompt_templates import format_prompt, get_system_message, format_prompt_with_speaker_context


# ============================================================================
# 配置区：在这里修改默认参数
# ============================================================================
NUM_SAMPLES = 20          # 每个数据集生成的样本数量
MODEL_NAME = "gpt-3.5-turbo"  # OpenAI模型名称
TEMPERATURE = 0.8          # 采样温度 (0-1) - 降到0.8平衡创造性和一致性
MAX_TOKENS = 50            # 最大生成token数 - 降低到50保持简短
OUTPUT_DIR = "data/llm_outputs"  # 输出目录
# ============================================================================


def load_test_samples(test_path: Path, limit: int = None, offset: int = 0) -> List[Dict]:
    """
    Load test samples from JSONL file.
    
    Args:
        test_path: Path to test.jsonl file
        limit: Maximum number of samples to load (None = all)
        offset: Number of samples to skip from the beginning
    
    Returns:
        List of test sample dictionaries
    """
    samples = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Skip samples before offset
            if i < offset:
                continue
            # Stop after loading 'limit' samples
            if limit and len(samples) >= limit:
                break
            if line.strip():
                samples.append(json.loads(line))
    return samples


def generate_gpt35_response(
    context: str,
    speaker: str = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 150,
    client = None
) -> Tuple[str, float, Dict]:
    """
    Generate a response using GPT-3.5.
    
    Args:
        context: The conversation context (with speaker labels like "A: text")
        speaker: The speaker who should respond (e.g., "A", "B")
        model: OpenAI model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        client: OpenAI client instance (for new version)
    
    Returns:
        Tuple of (response_text, latency_seconds, token_usage)
    """
    # Get system message from prompt_templates
    system_message = get_system_message(include_speaker_guidance=True)
    
    # Format the prompt using shared template with speaker awareness
    if speaker:
        prompt = format_prompt_with_speaker_context(context, speaker)
    else:
        prompt = format_prompt(context, include_user_prefix=False)
    
    # Prepare the API call
    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    # Record start time
    start_time = time.time()
    
    try:
        # Call OpenAI API - different for new vs old version
        if OPENAI_NEW_VERSION:
            # New API (>=1.0.0)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1
            )
            # Extract response and token usage (new format)
            response_text = response.choices[0].message.content.strip()
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        else:
            # Old API (<1.0.0)
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1
            )
            # Extract response and token usage (old format)
            response_text = response.choices[0].message.content.strip()
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Clean up response: remove speaker labels if present
        # Pattern: "A: ", "B: ", "Professor Clark: ", etc.
        import re
        cleaned_response = re.sub(r'^[A-Z][^:]{0,20}:\s*', '', response_text.strip())
        
        return cleaned_response, latency, token_usage
        
    except Exception as e:
        print(f"\n❌ ERROR calling OpenAI API:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   Context: {context[:100]}...")
        print()
        return "", 0.0, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def generate_llm_responses(
    dataset_name: str,
    test_path: Path,
    output_path: Path,
    num_samples: int = 20,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 150,
    api_key: str = None,
    offset: int = 0,
    append: bool = False
) -> None:
    """
    Generate LLM responses for test samples and save results.
    
    Args:
        dataset_name: Name of the dataset (dailydialog or empathetic_dialogues)
        test_path: Path to test.jsonl file
        output_path: Path to save output results
        num_samples: Number of samples to process
        model: OpenAI model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_key: OpenAI API key (if None, will use environment variable)
        offset: Number of samples to skip from the beginning (for batch processing)
        append: Whether to append to existing output file (vs overwrite)
    """
    # Set up OpenAI API key and client
    if OPENAI_NEW_VERSION:
        # New version (>=1.0.0) - use client
        if api_key:
            client = OpenAI(api_key=api_key)
        else:
            client = OpenAI()  # Reads from OPENAI_API_KEY env variable
    else:
        # Old version (<1.0.0) - use global api_key
        if api_key:
            openai.api_key = api_key
        client = None
    
    print(f"Loading test samples from {test_path}...")
    if offset > 0:
        print(f"  Skipping first {offset} samples...")
    print(f"  Loading up to {num_samples} samples...")
    test_samples = load_test_samples(test_path, limit=num_samples, offset=offset)
    print(f"Loaded {len(test_samples)} samples (offset: {offset}, range: {offset}-{offset+len(test_samples)-1})")
    
    results = []
    total_latency = 0.0
    total_tokens = 0
    
    print(f"\nGenerating responses using {model}...")
    for i, sample in enumerate(tqdm(test_samples, desc="Generating")):
        # Generate response with speaker awareness
        speaker = sample.get("speaker", None)  # Get the speaker who should respond
        generated_response, latency, token_usage = generate_gpt35_response(
            context=sample["context"],
            speaker=speaker,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            client=client
        )
        
        # Accumulate statistics
        total_latency += latency
        total_tokens += token_usage["total_tokens"]
        
        # Create result record
        result = {
            "dataset": dataset_name,
            "dialog_id": sample["dialog_id"],
            "turn_id": sample["turn_id"],
            "context": sample["context"],
            "ground_truth_response": sample["response"],
            "generated_response": generated_response,
            "emotion": sample.get("emotion", ""),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "latency_seconds": latency,
            "token_usage": token_usage
        }
        results.append(result)
        
        # Optional: Add a small delay to avoid rate limits
        time.sleep(0.1)
    
    # Calculate average statistics
    avg_latency = total_latency / len(results) if results else 0
    avg_tokens = total_tokens / len(results) if results else 0
    
    # Save results to JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = 'a' if append else 'w'
    with open(output_path, write_mode, encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    if append:
        print(f"\n✅ Results appended to existing file: {output_path}")
    else:
        print(f"\n✅ Results saved to new file: {output_path}")
    
    # Save summary statistics
    summary_path = output_path.parent / f"{output_path.stem}_summary.json"
    summary = {
        "dataset": dataset_name,
        "model": model,
        "num_samples": len(results),
        "total_latency_seconds": total_latency,
        "total_tokens": total_tokens,
        "avg_latency_seconds": avg_latency,
        "avg_tokens_per_sample": avg_tokens,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "output_file": str(output_path)
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Generation Complete!")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model}")
    print(f"Samples processed: {len(results)}")
    print(f"Total latency: {total_latency:.2f}s")
    print(f"Average latency: {avg_latency:.3f}s per sample")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens: {avg_tokens:.1f} per sample")
    print(f"\nResults saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM responses for test samples"
    )
    parser.add_argument(
        "--dataset",
        choices=["dailydialog", "empathetic_dialogues", "both"],
        default="both",
        help="Which dataset to process"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,  # 使用文件顶部的配置
        help=f"Number of test samples to process per dataset (default: {NUM_SAMPLES})"
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,  # 使用文件顶部的配置
        help=f"OpenAI model name (default: {MODEL_NAME})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,  # 使用文件顶部的配置
        help=f"Sampling temperature (default: {TEMPERATURE})"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,  # 使用文件顶部的配置
        help=f"Maximum tokens to generate (default: {MAX_TOKENS})"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (if not set in environment)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(OUTPUT_DIR),  # 使用文件顶部的配置
        help=f"Directory to save output files (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--data-source",
        choices=["processed_sample", "full_processed_sample"],
        default="full_processed_sample",
        help="Which data directory to use (default: full_processed_sample)"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of samples to skip from the beginning (for batch processing, default: 0)"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output file instead of overwriting"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Base paths
    base_dir = Path("data") / args.data_source
    
    # Process datasets
    datasets_to_process = []
    if args.dataset in ["dailydialog", "both"]:
        datasets_to_process.append({
            "name": "dailydialog",
            "path": base_dir / "dailydialog" / "test.jsonl",
            "output": args.output_dir / "dailydialog_gpt35_responses.jsonl"
        })
    
    if args.dataset in ["empathetic_dialogues", "both"]:
        datasets_to_process.append({
            "name": "empathetic_dialogues",
            "path": base_dir / "empathetic_dialogues" / "test.jsonl",
            "output": args.output_dir / "empathetic_dialogues_gpt35_responses.jsonl"
        })
    
    # Generate responses for each dataset
    for dataset in datasets_to_process:
        print(f"\n{'#'*60}")
        print(f"Processing {dataset['name'].upper()}")
        print(f"{'#'*60}\n")
        
        generate_llm_responses(
            dataset_name=dataset["name"],
            test_path=dataset["path"],
            output_path=dataset["output"],
            num_samples=args.num_samples,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_key=args.api_key,
            offset=args.offset,
            append=args.append
        )
    
    print("\n✅ All datasets processed successfully!")

