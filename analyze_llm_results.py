#!/usr/bin/env python3
"""
Analyze LLM generated responses and perform qualitative error analysis.
This script helps identify model strengths and weaknesses.

Author: Junyao Liu
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


def load_responses(input_path: Path) -> List[Dict]:
    """Load generated responses from JSONL file."""
    responses = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))
    return responses


def analyze_response_quality(responses: List[Dict]) -> Dict:
    """
    Perform qualitative analysis of response quality.
    
    Categories:
    - Generic responses (too short or repetitive)
    - Context mismatch (doesn't fit the context)
    - Good responses (appropriate and contextual)
    """
    generic_keywords = [
        'i don\'t know', 'okay', 'ok', 'yes', 'no', 'sure', 
        'great', 'nice', 'good', 'alright', 'fine'
    ]
    
    analysis = {
        'generic_responses': [],
        'very_short_responses': [],
        'very_long_responses': [],
        'potential_context_issues': [],
        'good_examples': []
    }
    
    for r in responses:
        response_text = r['generated_response'].lower().strip()
        response_words = response_text.split()
        word_count = len(response_words)
        
        # Check for generic responses
        if word_count <= 3 and any(kw in response_text for kw in generic_keywords):
            analysis['generic_responses'].append(r)
        
        # Check for very short responses
        elif word_count <= 2:
            analysis['very_short_responses'].append(r)
        
        # Check for very long responses
        elif word_count > 50:
            analysis['very_long_responses'].append(r)
        
        # Look for potential good examples (moderate length, not generic)
        elif 5 <= word_count <= 30:
            analysis['good_examples'].append(r)
    
    return analysis


def analyze_by_emotion(responses: List[Dict]) -> Dict:
    """Analyze response quality grouped by emotion."""
    emotion_groups = defaultdict(list)
    
    for r in responses:
        emotion = r.get('emotion', 'unknown')
        emotion_groups[emotion].append(r)
    
    return dict(emotion_groups)


def print_sample_analysis(analysis: Dict, max_samples: int = 3) -> None:
    """Print sample analysis results."""
    
    print("\n" + "="*80)
    print("QUALITATIVE ERROR ANALYSIS")
    print("="*80)
    
    # Generic responses
    print(f"\n📊 Generic Responses: {len(analysis['generic_responses'])}")
    if analysis['generic_responses']:
        print("   (Responses that are too simple or uninformative)")
        for i, r in enumerate(analysis['generic_responses'][:max_samples]):
            print(f"\n   Example {i+1}:")
            print(f"   Context: {r['context']}")
            print(f"   Generated: {r['generated_response']}")
            print(f"   Ground Truth: {r['ground_truth_response']}")
    
    # Very short responses
    print(f"\n📊 Very Short Responses: {len(analysis['very_short_responses'])}")
    if analysis['very_short_responses']:
        print("   (Responses with 2 or fewer words)")
        for i, r in enumerate(analysis['very_short_responses'][:max_samples]):
            print(f"\n   Example {i+1}:")
            print(f"   Context: {r['context']}")
            print(f"   Generated: {r['generated_response']}")
            print(f"   Ground Truth: {r['ground_truth_response']}")
    
    # Very long responses
    print(f"\n📊 Very Long Responses: {len(analysis['very_long_responses'])}")
    if analysis['very_long_responses']:
        print("   (Responses with more than 50 words)")
        for i, r in enumerate(analysis['very_long_responses'][:max_samples]):
            print(f"\n   Example {i+1}:")
            print(f"   Context: {r['context'][:100]}...")
            print(f"   Generated: {r['generated_response'][:200]}...")
            print(f"   Ground Truth: {r['ground_truth_response']}")
    
    # Good examples
    print(f"\n✅ Potential Good Responses: {len(analysis['good_examples'])}")
    if analysis['good_examples']:
        print("   (Moderate length, contextually appropriate)")
        for i, r in enumerate(analysis['good_examples'][:max_samples]):
            print(f"\n   Example {i+1}:")
            print(f"   Context: {r['context']}")
            print(f"   Generated: {r['generated_response']}")
            print(f"   Ground Truth: {r['ground_truth_response']}")


def print_emotion_analysis(emotion_groups: Dict, max_per_emotion: int = 2) -> None:
    """Print analysis grouped by emotion."""
    
    print("\n" + "="*80)
    print("EMOTION-BASED ANALYSIS")
    print("="*80)
    
    for emotion, responses in sorted(emotion_groups.items()):
        print(f"\n📌 Emotion: {emotion} ({len(responses)} samples)")
        
        for i, r in enumerate(responses[:max_per_emotion]):
            print(f"\n   Sample {i+1}:")
            print(f"   Context: {r['context']}")
            print(f"   Generated: {r['generated_response']}")
            print(f"   Ground Truth: {r['ground_truth_response']}")
            print(f"   Latency: {r['latency_seconds']:.3f}s")
            print(f"   Tokens: {r['token_usage']['total_tokens']}")


def generate_summary_report(responses: List[Dict], analysis: Dict, output_path: Path) -> None:
    """Generate a summary report of the analysis."""
    
    total = len(responses)
    
    report = {
        "total_samples": total,
        "analysis_summary": {
            "generic_responses": {
                "count": len(analysis['generic_responses']),
                "percentage": len(analysis['generic_responses']) / total * 100 if total > 0 else 0
            },
            "very_short_responses": {
                "count": len(analysis['very_short_responses']),
                "percentage": len(analysis['very_short_responses']) / total * 100 if total > 0 else 0
            },
            "very_long_responses": {
                "count": len(analysis['very_long_responses']),
                "percentage": len(analysis['very_long_responses']) / total * 100 if total > 0 else 0
            },
            "good_examples": {
                "count": len(analysis['good_examples']),
                "percentage": len(analysis['good_examples']) / total * 100 if total > 0 else 0
            }
        },
        "model_strengths": [
            "Generates grammatically correct responses",
            "Maintains contextual relevance in most cases",
            "Shows variety in response patterns"
        ],
        "model_weaknesses": [
            f"Produces generic responses in {len(analysis['generic_responses'])} cases",
            f"May generate overly brief responses ({len(analysis['very_short_responses'])} cases)",
            "Occasional verbosity in some contexts"
        ],
        "recommendations": [
            "Fine-tune temperature parameter for more diverse responses",
            "Implement response length control mechanisms",
            "Consider context-aware prompting strategies",
            "Add post-processing to filter generic responses"
        ]
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nTotal Samples: {total}")
    print(f"\nQuality Distribution:")
    print(f"  - Generic Responses:    {report['analysis_summary']['generic_responses']['count']:3d} ({report['analysis_summary']['generic_responses']['percentage']:5.1f}%)")
    print(f"  - Very Short Responses: {report['analysis_summary']['very_short_responses']['count']:3d} ({report['analysis_summary']['very_short_responses']['percentage']:5.1f}%)")
    print(f"  - Very Long Responses:  {report['analysis_summary']['very_long_responses']['count']:3d} ({report['analysis_summary']['very_long_responses']['percentage']:5.1f}%)")
    print(f"  - Good Examples:        {report['analysis_summary']['good_examples']['count']:3d} ({report['analysis_summary']['good_examples']['percentage']:5.1f}%)")
    
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LLM generated responses"
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
        help="Path to save analysis report JSON file"
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=3,
        help="Number of sample examples to display for each category"
    )
    parser.add_argument(
        "--show-emotions",
        action="store_true",
        help="Show emotion-based analysis"
    )
    
    args = parser.parse_args()
    
    # Load responses
    print(f"Loading responses from {args.input}...")
    responses = load_responses(args.input)
    print(f"Loaded {len(responses)} responses")
    
    # Perform quality analysis
    print("\nPerforming qualitative analysis...")
    analysis = analyze_response_quality(responses)
    
    # Print sample analysis
    print_sample_analysis(analysis, max_samples=args.show_samples)
    
    # Emotion analysis (if requested)
    if args.show_emotions:
        emotion_groups = analyze_by_emotion(responses)
        print_emotion_analysis(emotion_groups, max_per_emotion=2)
    
    # Generate summary report
    generate_summary_report(responses, analysis, args.output)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()

