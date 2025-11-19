#!/usr/bin/env python3
"""
Utility script for downloading and preprocessing DailyDialog and EmpatheticDialogues
without relying on the Hugging Face datasets runner (which now blocks remote code).

Output files (JSONL) share a unified schema:
    dataset, split, dialog_id, turn_id, speaker, context, response, emotion

Run:
    python preprocess_datasets.py --output-dir data/processed --history 3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from collections import defaultdict

from huggingface_hub import hf_hub_download

HISTORY_SEPARATOR = " <eot> "
WHITESPACE_RE = re.compile(r"\s+")
ED_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"


def normalize_text(text: str) -> str:
    """Collapse whitespace, strip quotes, and keep ASCII punctuation."""
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def save_jsonl(records: Iterable[Dict], path: Path) -> None:
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dailydialog_split(split: str) -> Tuple[List[List[str]], List[List[int]]]:
    """Download and parse DailyDialog split files."""
    filename = f"{split}.zip"
    zip_path = hf_hub_download(repo_id="roskoN/dailydialog", filename=filename, repo_type="dataset")
    dialogues: List[List[str]] = []
    emotions: List[List[int]] = []

    with zipfile.ZipFile(zip_path) as zf:
        base_dir = split
        dialog_file = f"{base_dir}/dialogues_{split}.txt"
        emo_file = f"{base_dir}/dialogues_emotion_{split}.txt"

        dialog_lines = zf.read(dialog_file).decode("utf-8").strip().splitlines()
        emo_lines = zf.read(emo_file).decode("utf-8").strip().splitlines()

        for line in dialog_lines:
            turns = [normalize_text(t) for t in line.split("__eou__") if normalize_text(t)]
            dialogues.append(turns)

        for line in emo_lines:
            labels = [int(v) for v in line.strip().split() if v]
            emotions.append(labels)

    return dialogues, emotions


def build_dailydialog_examples(split: str, history: int) -> List[Dict]:
    """Turn multi-turn DailyDialog conversations into context-response pairs."""
    dialogs, emotions = load_dailydialog_split(split)
    examples: List[Dict] = []

    for dialog_id, turns in enumerate(dialogs):
        emo_seq = emotions[dialog_id] if dialog_id < len(emotions) else [-1] * len(turns)
        history_buffer: List[str] = []
        for turn_idx, utterance in enumerate(turns):
            speaker = "A" if turn_idx % 2 == 0 else "B"
            if history_buffer:
                context = HISTORY_SEPARATOR.join(history_buffer[-history:])
                examples.append(
                    {
                        "dataset": "dailydialog",
                        "split": split,
                        "dialog_id": f"dd_{split}_{dialog_id}",
                        "turn_id": turn_idx,
                        "speaker": speaker,
                        "context": context,
                        "response": utterance,
                        "emotion": emo_seq[turn_idx] if turn_idx < len(emo_seq) else -1,
                    }
                )
            history_buffer.append(utterance)
    return examples


def download_file(url: str, cache_dir: Path) -> Path:
    """Download a remote file with a simple cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    target = cache_dir / filename
    if not target.exists():
        urllib.request.urlretrieve(url, target)
    return target


def load_empathetic_split(split: str, cache_dir: Path) -> List[Dict]:
    """Download the tarball once and read the target CSV split."""
    tar_path = download_file(ED_URL, cache_dir)
    target_file = f"empatheticdialogues/{'valid' if split == 'validation' else split}.csv"
    rows: List[Dict] = []
    with tarfile.open(tar_path, "r:gz") as tar:
        member = tar.getmember(target_file)
        with tar.extractfile(member) as f:  # type: ignore[arg-type]
            reader = csv.DictReader(line.decode("utf-8") for line in f)  # type: ignore[arg-type]
            for row in reader:
                rows.append(row)
    return rows


def build_empathetic_examples(split: str, cache_dir: Path, history: int = 3) -> List[Dict]:
    """
    Convert EmpatheticDialogues rows to the shared schema.
    
    Note: In EmpatheticDialogues, the 'context' field is actually an emotion label,
    not conversation history. We need to build the conversation history from
    utterances ordered by utterance_idx, similar to DailyDialog.
    """
    records = load_empathetic_split(split, cache_dir)
    examples: List[Dict] = []
    
    # Group records by conversation ID
    conversations: Dict[str, List[Dict]] = {}
    for record in records:
        conv_id = record["conv_id"]
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append({
            "utterance_idx": int(record["utterance_idx"]),
            "speaker": int(record.get("speaker_idx", -1)),
            "utterance": normalize_text(record["utterance"]),
            "prompt": normalize_text(record.get("prompt", "")),  # Initial context/situation
            "emotion": record.get("context", ""),  # The 'context' field is actually emotion label
            "tags": record.get("tags", "")
        })
    
    # Process each conversation
    for conv_id, turns in conversations.items():
        # Sort turns by utterance_idx to maintain conversation order
        turns.sort(key=lambda x: x["utterance_idx"])
        
        if not turns:
            continue
        
        # Get the emotion label (same for all turns in a conversation)
        emotion_label = turns[0]["emotion"] if turns else ""
        
        # Get initial prompt if available (used as first context)
        initial_prompt = turns[0].get("prompt", "") if turns else ""
        
        # Build conversation history buffer
        history_buffer: List[str] = []
        if initial_prompt:
            history_buffer.append(initial_prompt)
        
        # Process each turn to create context-response pairs
        for turn_idx, turn in enumerate(turns):
            utterance = turn["utterance"]
            if not utterance:
                continue
            
            # If we have history, create a training example
            if history_buffer:
                # Use last 'history' turns as context
                context = HISTORY_SEPARATOR.join(history_buffer[-history:])
                examples.append(
                    {
                        "dataset": "empathetic_dialogues",
                        "split": split,
                        "dialog_id": f"ed_{split}_{conv_id}",
                        "turn_id": turn["utterance_idx"],
                        "speaker": turn["speaker"],
                        "context": context,
                        "response": utterance,
                        "emotion": emotion_label,  # Store emotion label
                    }
                )
            
            # Add current utterance to history buffer
            history_buffer.append(utterance)
    
    return examples


def preprocess(output_dir: Path, history: int, cache_dir: Path) -> None:
    """Download both datasets, preprocess, and write JSONL files."""
    for split in ("train", "validation", "test"):
        dd_examples = build_dailydialog_examples(split, history)
        save_jsonl(dd_examples, output_dir / "dailydialog" / f"{split}.jsonl")

        ed_examples = build_empathetic_examples(split, cache_dir, history)
        save_jsonl(ed_examples, output_dir / "empathetic_dialogues" / f"{split}.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and preprocess dialogue datasets.")
    parser.add_argument(
        "--output-dir",
        default=Path("data/processed"),
        type=Path,
        help="Directory to store processed JSONL files.",
    )
    parser.add_argument(
        "--history",
        default=3,
        type=int,
        help="Number of previous turns to include in the context window.",
    )
    parser.add_argument(
        "--cache-dir",
        default=Path("data/cache"),
        type=Path,
        help="Cache directory for raw compressed files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(args.output_dir, args.history, args.cache_dir)

