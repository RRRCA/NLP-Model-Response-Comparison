#!/usr/bin/env python3
"""
Compute automatic metrics (BERTScore, BLEU, Distinct-n) for LLM outputs.
This script evaluates the generated responses against ground truth.

Author: Junyao Liu
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import Counter
import numpy as np

# For metrics computation
try:
    from bert_score import score as bertscore
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("Warning: bert_score not installed. BERTScore will be skipped.")
    BERTSCORE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    print("Warning: nltk not installed. BLEU will be skipped.")
    NLTK_AVAILABLE = False


def load_generated_responses(input_path: Path) -> List[Dict]:
    """Load generated responses from JSONL file."""
    responses = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))
    return responses


def compute_bleu_scores(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores.
    
    Args:
        references: List of reference texts
        hypotheses: List of generated texts
    
    Returns:
        Dictionary with BLEU scores
    """
    if not NLTK_AVAILABLE:
        return {"bleu-1": 0.0, "bleu-2": 0.0, "bleu-3": 0.0, "bleu-4": 0.0}
    
    smoothing = SmoothingFunction().method1
    
    bleu_scores = {1: [], 2: [], 3: [], 4: []}
    
    for ref, hyp in zip(references, hypotheses):
        # Tokenize
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        
        # Compute BLEU for different n-grams
        for n in [1, 2, 3, 4]:
            weights = tuple([1.0/n] * n + [0.0] * (4-n))
            try:
                score = sentence_bleu(
                    [ref_tokens],
                    hyp_tokens,
                    weights=weights,
                    smoothing_function=smoothing
                )
                bleu_scores[n].append(score)
            except:
                bleu_scores[n].append(0.0)
    
    return {
        f"bleu-{n}": np.mean(scores) if scores else 0.0
        for n, scores in bleu_scores.items()
    }


def compute_distinct_n(texts: List[str], n: int) -> float:
    """
    Compute Distinct-n metric.
    
    Args:
        texts: List of generated texts
        n: N-gram size
    
    Returns:
        Distinct-n score (ratio of unique n-grams to total n-grams)
    """
    all_ngrams = []
    
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        all_ngrams.extend(ngrams)
    
    if not all_ngrams:
        return 0.0
    
    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    
    return unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0


def compute_bertscore(references: List[str], hypotheses: List[str], model: str) -> Dict[str, float]:
    """
    Compute BERTScore using the specified model.
    
    Args:
        references: List of reference texts
        hypotheses: List of generated texts
        model: BERTScore model name
    
    Returns:
        Dictionary with precision, recall, F1 scores
    """
    if not BERTSCORE_AVAILABLE:
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    
    try:
        P, R, F1 = bertscore(
            hypotheses,
            references,
            model_type=model,
            lang="en",
            verbose=False
        )
        
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item()
        }
    except Exception as e:
        print(f"Error computing BERTScore: {e}")
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}


def compute_response_length_stats(texts: List[str]) -> Dict[str, float]:
    """Compute statistics about response lengths."""
    lengths = [len(text.split()) for text in texts]
    
    return {
        "avg_response_length": np.mean(lengths),
        "min_response_length": np.min(lengths),
        "max_response_length": np.max(lengths),
        "std_response_length": np.std(lengths)
    }


def compute_all_metrics(
    responses: List[Dict],
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"
) -> Dict:
    """
    Compute all automatic metrics for generated responses.
    
    Args:
        responses: List of response dictionaries
        bertscore_model: Model to use for BERTScore
    
    Returns:
        Dictionary with all computed metrics
    """
    # Extract references and hypotheses
    references = [r["ground_truth_response"] for r in responses]
    hypotheses = [r["generated_response"] for r in responses]
    
    print("Computing BLEU scores...")
    bleu_scores = compute_bleu_scores(references, hypotheses)
    
    print("Computing Distinct-n scores...")
    distinct_1 = compute_distinct_n(hypotheses, n=1)
    distinct_2 = compute_distinct_n(hypotheses, n=2)
    
    print(f"Computing BERTScore (using {bertscore_model})...")
    bertscore_metrics = compute_bertscore(references, hypotheses, bertscore_model)
    
    print("Computing response length statistics...")
    length_stats = compute_response_length_stats(hypotheses)
    
    # Combine all metrics
    all_metrics = {
        **bleu_scores,
        "distinct-1": distinct_1,
        "distinct-2": distinct_2,
        **bertscore_metrics,
        **length_stats
    }
    
    return all_metrics


def compute_and_save_metrics(
    input_path: Path,
    output_path: Path,
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"
) -> None:
    """
    Load generated responses, compute metrics, and save results.
    
    Args:
        input_path: Path to generated responses JSONL file
        output_path: Path to save metrics results
        bertscore_model: Model to use for BERTScore
    """
    print(f"Loading responses from {input_path}...")
    responses = load_generated_responses(input_path)
    print(f"Loaded {len(responses)} responses")
    
    if not responses:
        print("No responses to evaluate!")
        return
    
    print("\nComputing metrics...")
    metrics = compute_all_metrics(responses, bertscore_model)
    
    # Calculate token and latency statistics from responses
    latencies = [r["latency_seconds"] for r in responses]
    token_usages = [r["token_usage"]["total_tokens"] for r in responses]
    
    # Add cost and latency statistics
    metrics.update({
        "total_latency_seconds": sum(latencies),
        "avg_latency_seconds": np.mean(latencies),
        "std_latency_seconds": np.std(latencies),
        "min_latency_seconds": min(latencies),
        "max_latency_seconds": max(latencies),
        "total_tokens": sum(token_usages),
        "avg_tokens_per_sample": np.mean(token_usages),
        "std_tokens_per_sample": np.std(token_usages)
    })
    
    # Get metadata
    dataset_name = responses[0].get("dataset", "unknown")
    model_name = responses[0].get("model", "unknown")
    
    # Create results dictionary
    results = {
        "dataset": dataset_name,
        "model": model_name,
        "num_samples": len(responses),
        "bertscore_model": bertscore_model,
        "metrics": metrics
    }
    
    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Samples: {len(responses)}")
    print(f"\n--- BLEU Scores ---")
    for i in range(1, 5):
        print(f"  BLEU-{i}: {metrics[f'bleu-{i}']:.4f}")
    print(f"\n--- Distinct-n ---")
    print(f"  Distinct-1: {metrics['distinct-1']:.4f}")
    print(f"  Distinct-2: {metrics['distinct-2']:.4f}")
    print(f"\n--- BERTScore ---")
    print(f"  Precision: {metrics['bertscore_precision']:.4f}")
    print(f"  Recall:    {metrics['bertscore_recall']:.4f}")
    print(f"  F1:        {metrics['bertscore_f1']:.4f}")
    print(f"\n--- Response Length ---")
    print(f"  Average: {metrics['avg_response_length']:.2f} tokens")
    print(f"  Min:     {metrics['min_response_length']:.0f} tokens")
    print(f"  Max:     {metrics['max_response_length']:.0f} tokens")
    print(f"\n--- Latency ---")
    print(f"  Average: {metrics['avg_latency_seconds']:.3f}s")
    print(f"  Total:   {metrics['total_latency_seconds']:.2f}s")
    print(f"\n--- Token Usage ---")
    print(f"  Average: {metrics['avg_tokens_per_sample']:.1f} tokens/sample")
    print(f"  Total:   {metrics['total_tokens']} tokens")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute automatic metrics for LLM outputs"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to generated responses JSONL file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save metrics results JSON file"
    )
    parser.add_argument(
        "--bertscore-model",
        default="microsoft/deberta-xlarge-mnli",
        help="Model to use for BERTScore computation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    compute_and_save_metrics(
        input_path=args.input,
        output_path=args.output,
        bertscore_model=args.bertscore_model
    )
    
    print("\n✅ Metrics computation complete!")

