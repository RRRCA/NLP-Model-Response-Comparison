#!/usr/bin/env python3
"""
Generate test sample ID list from test.jsonl files.
This ensures both small transformer and LLM models use the exact same test samples.

Since we're using all test samples, this script simply extracts all dialog_ids
from both datasets' test sets and saves them to a JSON file.
"""

import json
from pathlib import Path
from typing import List, Dict


def extract_test_sample_ids(test_path: Path) -> List[str]:
    """
    Extract all dialog_ids from a test.jsonl file.
    
    Args:
        test_path: Path to the test.jsonl file
    
    Returns:
        List of dialog_ids in the order they appear in the file
    """
    dialog_ids = []
    
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                dialog_id = record.get('dialog_id')
                if dialog_id:
                    dialog_ids.append(dialog_id)
    
    return dialog_ids


def generate_test_sample_list(
    dailydialog_path: Path,
    empathetic_path: Path,
    output_path: Path
) -> None:
    """
    Generate a unified test sample ID list from both datasets.
    
    Args:
        dailydialog_path: Path to dailydialog test.jsonl
        empathetic_path: Path to empathetic_dialogues test.jsonl
        output_path: Path to save the output JSON file
    """
    # Extract dialog IDs from both datasets
    dd_ids = extract_test_sample_ids(dailydialog_path)
    ed_ids = extract_test_sample_ids(empathetic_path)
    
    # Create the output structure
    test_samples = {
        "dailydialog": {
            "count": len(dd_ids),
            "dialog_ids": dd_ids
        },
        "empathetic_dialogues": {
            "count": len(ed_ids),
            "dialog_ids": ed_ids
        },
        "total_count": len(dd_ids) + len(ed_ids),
        "description": "All test samples from both datasets for evaluation"
    }
    
    # Save to JSON file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, indent=2, ensure_ascii=False)
    
    print(f"Generated test sample list:")
    print(f"  - DailyDialog: {len(dd_ids)} samples")
    print(f"  - EmpatheticDialogues: {len(ed_ids)} samples")
    print(f"  - Total: {len(dd_ids) + len(ed_ids)} samples")
    print(f"  - Saved to: {output_path}")


def load_test_sample_list(test_list_path: Path) -> Dict:
    """
    Load the test sample ID list from JSON file.
    
    Args:
        test_list_path: Path to the test sample list JSON file
    
    Returns:
        Dictionary containing test sample IDs
    """
    with open(test_list_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    # Paths relative to project root
    base_dir = Path("data/processed_sample")
    dd_test_path = base_dir / "dailydialog" / "test.jsonl"
    ed_test_path = base_dir / "empathetic_dialogues" / "test.jsonl"
    output_path = Path("data/test_sample_ids.json")
    
    # Generate the test sample list
    generate_test_sample_list(dd_test_path, ed_test_path, output_path)
    
    # Verify the output
    loaded = load_test_sample_list(output_path)
    print(f"\nVerification: Loaded {loaded['total_count']} total test samples")

