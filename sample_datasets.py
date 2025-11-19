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


def sample_jsonl_by_dialog(
    input_path: Path, 
    output_path: Path, 
    sample_ratio: float, 
    seed: int = 42
) -> int:
    """
    Sample a percentage of dialogs from a JSONL file by dialog_id.
    All turns from selected dialogs are kept to maintain conversation integrity.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        sample_ratio: Ratio of dialogs to keep (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        Number of samples (lines) in output file
    """
    # Read all lines and group by dialog_id
    dialogs = {}  # dialog_id -> list of lines (as dicts)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            dialog_id = record.get('dialog_id')
            if not dialog_id:
                continue
            
            if dialog_id not in dialogs:
                dialogs[dialog_id] = []
            dialogs[dialog_id].append(record)
    
    # Sample dialog_ids
    all_dialog_ids = list(dialogs.keys())
    num_dialogs_to_keep = int(len(all_dialog_ids) * sample_ratio)
    random.seed(seed)
    sampled_dialog_ids = set(random.sample(all_dialog_ids, num_dialogs_to_keep))
    
    # Collect all lines from sampled dialogs
    sampled_records = []
    for dialog_id in sampled_dialog_ids:
        sampled_records.extend(dialogs[dialog_id])
    
    # Sort by dialog_id and turn_id to maintain order (optional, but good for consistency)
    sampled_records.sort(key=lambda x: (x.get('dialog_id', ''), x.get('turn_id', 0)))
    
    # Write sampled lines
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in sampled_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    return len(sampled_records)


def sample_dataset(
    dataset_name: str,
    input_dir: Path,
    output_dir: Path,
    sample_ratio: float,
    seed: int = 42
) -> Dict[str, int]:
    """
    Sample all splits (train/validation/test) of a dataset by dialog_id.
    All turns from selected dialogs are kept.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'dailydialog')
        input_dir: Input directory containing JSONL files
        output_dir: Output directory for sampled files
        sample_ratio: Ratio of dialogs to keep
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with counts for each split (number of dialogs and samples)
    """
    counts = {}
    
    for split in ['train', 'validation', 'test']:
        input_path = input_dir / f"{split}.jsonl"
        
        if not input_path.exists():
            print(f"Warning: {input_path} does not exist, skipping...")
            continue
        
        # Count dialogs in original file
        dialog_ids = set()
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    dialog_id = record.get('dialog_id')
                    if dialog_id:
                        dialog_ids.add(dialog_id)
        
        num_original_dialogs = len(dialog_ids)
        num_original_samples = sum(1 for _ in open(input_path))
        
        output_path = output_dir / dataset_name / f"{split}.jsonl"
        
        num_samples = sample_jsonl_by_dialog(input_path, output_path, sample_ratio, seed)
        
        # Count dialogs in sampled file
        sampled_dialog_ids = set()
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    dialog_id = record.get('dialog_id')
                    if dialog_id:
                        sampled_dialog_ids.add(dialog_id)
        
        num_sampled_dialogs = len(sampled_dialog_ids)
        
        counts[split] = {
            'dialogs': num_sampled_dialogs,
            'samples': num_samples,
            'original_dialogs': num_original_dialogs,
            'original_samples': num_original_samples
        }
        
        print(f"  {split}:")
        print(f"    Dialogs: {num_original_dialogs:,} -> {num_sampled_dialogs:,} "
              f"({num_sampled_dialogs/num_original_dialogs*100:.1f}%)")
        print(f"    Samples: {num_original_samples:,} -> {num_samples:,} "
              f"({num_samples/num_original_samples*100:.1f}%)")
        print(f"    Size: {input_path.stat().st_size / 1024 / 1024:.2f} MB -> "
              f"{output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
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
        default=Path("data/processed_sample"),
        help="Output directory for sampled datasets"
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.15,
        help="Ratio of dialogs to keep (0.0 to 1.0, default: 0.15 = 15%%)"
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
    
    print(f"Sampling datasets by dialog_id with ratio: {args.sample_ratio*100:.1f}%")
    print(f"Random seed: {args.seed}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Note: All turns from selected dialogs will be kept.")
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
    
    total_train_dialogs = 0
    total_train_samples = 0
    total_val_dialogs = 0
    total_val_samples = 0
    total_test_dialogs = 0
    total_test_samples = 0
    
    for dataset_name, counts in total_counts.items():
        train = counts.get('train', {})
        val = counts.get('validation', {})
        test = counts.get('test', {})
        
        train_dialogs = train.get('dialogs', 0)
        train_samples = train.get('samples', 0)
        val_dialogs = val.get('dialogs', 0)
        val_samples = val.get('samples', 0)
        test_dialogs = test.get('dialogs', 0)
        test_samples = test.get('samples', 0)
        
        total_train_dialogs += train_dialogs
        total_train_samples += train_samples
        total_val_dialogs += val_dialogs
        total_val_samples += val_samples
        total_test_dialogs += test_dialogs
        total_test_samples += test_samples
        
        print(f"\n{dataset_name}:")
        print(f"  Train: {train_dialogs:,} dialogs, {train_samples:,} samples")
        print(f"  Validation: {val_dialogs:,} dialogs, {val_samples:,} samples")
        print(f"  Test: {test_dialogs:,} dialogs, {test_samples:,} samples")
        print(f"  Total: {train_dialogs + val_dialogs + test_dialogs:,} dialogs, "
              f"{train_samples + val_samples + test_samples:,} samples")
    
    print(f"\nOverall Total:")
    print(f"  Train: {total_train_dialogs:,} dialogs, {total_train_samples:,} samples")
    print(f"  Validation: {total_val_dialogs:,} dialogs, {total_val_samples:,} samples")
    print(f"  Test: {total_test_dialogs:,} dialogs, {total_test_samples:,} samples")
    print(f"  Grand Total: {total_train_dialogs + total_val_dialogs + total_test_dialogs:,} dialogs, "
          f"{total_train_samples + total_val_samples + total_test_samples:,} samples")
    print()
    print(f"Sampled datasets saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

