#!/usr/bin/env python3
"""
Sample datasets by percentage while maintaining train/val/test proportions.
This script creates smaller versions of the datasets for faster training.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict


def sample_jsonl(input_path: Path, output_path: Path, sample_ratio: float, seed: int = 42) -> int:
    """
    Sample a percentage of lines from a JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        sample_ratio: Ratio of samples to keep (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        Number of samples in output file
    """
    # Read all lines
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Sample
    num_samples = int(len(lines) * sample_ratio)
    random.seed(seed)
    sampled_lines = random.sample(lines, num_samples)
    
    # Write sampled lines
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in sampled_lines:
            f.write(line + '\n')
    
    return len(sampled_lines)


def sample_dataset(
    dataset_name: str,
    input_dir: Path,
    output_dir: Path,
    sample_ratio: float,
    seed: int = 42
) -> Dict[str, int]:
    """
    Sample all splits (train/validation/test) of a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'dailydialog')
        input_dir: Input directory containing JSONL files
        output_dir: Output directory for sampled files
        sample_ratio: Ratio of samples to keep
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with counts for each split
    """
    counts = {}
    
    for split in ['train', 'validation', 'test']:
        input_path = input_dir / f"{split}.jsonl"
        
        if not input_path.exists():
            print(f"Warning: {input_path} does not exist, skipping...")
            continue
        
        output_path = output_dir / dataset_name / f"{split}.jsonl"
        
        num_samples = sample_jsonl(input_path, output_path, sample_ratio, seed)
        counts[split] = num_samples
        
        print(f"  {split}: {input_path.stat().st_size / 1024 / 1024:.2f} MB -> "
              f"{output_path.stat().st_size / 1024 / 1024:.2f} MB "
              f"({num_samples:,} samples, {sample_ratio*100:.1f}%)")
    
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Sample datasets by percentage while maintaining proportions"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/full_processed_sample"),
        help="Input directory containing original datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed_sampled"),
        help="Output directory for sampled datasets"
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.3,
        help="Ratio of samples to keep (0.0 to 1.0, default: 0.3 = 30%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["dailydialog", "empathetic_dialogues"],
        help="List of datasets to sample (default: both)"
    )
    
    args = parser.parse_args()
    
    print(f"Sampling datasets with ratio: {args.sample_ratio*100:.1f}%")
    print(f"Random seed: {args.seed}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    total_counts = {}
    
    for dataset_name in args.datasets:
        input_dir = args.input_dir / dataset_name
        
        if not input_dir.exists():
            print(f"Warning: {input_dir} does not exist, skipping...")
            continue
        
        print(f"Processing {dataset_name}:")
        counts = sample_dataset(
            dataset_name,
            input_dir,
            args.output_dir,
            args.sample_ratio,
            args.seed
        )
        total_counts[dataset_name] = counts
        print()
    
    # Print summary
    print("=" * 60)
    print("Sampling Summary:")
    print("=" * 60)
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for dataset_name, counts in total_counts.items():
        train = counts.get('train', 0)
        val = counts.get('validation', 0)
        test = counts.get('test', 0)
        
        total_train += train
        total_val += val
        total_test += test
        
        print(f"\n{dataset_name}:")
        print(f"  Train: {train:,} samples")
        print(f"  Validation: {val:,} samples")
        print(f"  Test: {test:,} samples")
        print(f"  Total: {train + val + test:,} samples")
    
    print(f"\nOverall Total:")
    print(f"  Train: {total_train:,} samples")
    print(f"  Validation: {total_val:,} samples")
    print(f"  Test: {total_test:,} samples")
    print(f"  Grand Total: {total_train + total_val + total_test:,} samples")
    print()
    print(f"Sampled datasets saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

