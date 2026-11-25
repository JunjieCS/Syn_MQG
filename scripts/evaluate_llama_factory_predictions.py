#!/usr/bin/env python
"""
Evaluation script for LLaMA-Factory JSONL predictions.
Compute similarity metrics between predicted and reference questions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import logging
from pathlib import Path
from src.evaluation.metrics import compute_bleu, compute_rouge, compute_bertscore, compute_meteor, compute_distinct_n, compute_bleurt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_jsonl_predictions(file_path: str):
    """
    Load predictions from a JSONL or JSON file.

    Args:
        file_path: Path to the JSONL/JSON file.

    Returns:
        predictions: List of predicted questions.
        references: List of reference questions.
    """
    predictions = []
    references = []

    logger.info(f"Reading prediction file: {file_path}")
    
    # if file is json
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                predict = item.get('predict', '').strip()
                label = item.get('label', '').strip()
                predictions.append(predict)
                references.append(label)
        logger.info(f"Loaded {len(predictions)} predictions")
        return predictions, references

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                predict = data.get('predict', '').strip()
                label = data.get('label', '').strip()

                predictions.append(predict)
                references.append(label)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error at line {line_num}: {e}")
                continue

    logger.info(f"Loaded {len(predictions)} predictions")
    return predictions, references


def evaluate_predictions(predictions, references):
    """
    Evaluate predictions against references.

    Args:
        predictions: List of predicted questions.
        references: List of reference questions.

    Returns:
        results: Dictionary of evaluation metrics.
    """
    results = {}

    logger.info("Computing BLEU...")
    bleu_scores = compute_bleu(predictions, references)
    results.update(bleu_scores)

    logger.info("Computing ROUGE...")
    rouge_scores = compute_rouge(predictions, references)
    results.update(rouge_scores)

    logger.info("Computing BERTScore...")
    bertscore = compute_bertscore(predictions, references)
    results.update(bertscore)

    logger.info("Computing METEOR...")
    meteor_score = compute_meteor(predictions, references)
    results["meteor"] = meteor_score

    logger.info("Computing BLEURT...")
    bleurt_score = compute_bleurt(predictions, references)
    results["bleurt"] = bleurt_score

    logger.info("Computing Distinct-n...")
    distinct_scores = compute_distinct_n(predictions, max_n=4)
    results.update(distinct_scores)

    return results


def print_results(results):
    """Print evaluation results."""
    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info("=" * 50)

    # BLEU
    bleu_keys = [k for k in results.keys() if k.startswith('bleu_')]
    if bleu_keys:
        logger.info("\nBLEU:")
        for key in sorted(bleu_keys):
            logger.info(f"  {key}: {results[key]:.4f}")

    # ROUGE
    rouge_keys = [k for k in results.keys() if k.startswith('rouge_')]
    if rouge_keys:
        logger.info("\nROUGE:")
        for key in sorted(rouge_keys):
            logger.info(f"  {key}: {results[key]:.4f}")

    # BERTScore
    bertscore_keys = [k for k in results.keys() if k.startswith('bertscore_')]
    if bertscore_keys:
        logger.info("\nBERTScore:")
        for key in sorted(bertscore_keys):
            logger.info(f"  {key}: {results[key]:.4f}")

    # METEOR
    if 'meteor' in results:
        logger.info(f"\nMETEOR: {results['meteor']:.4f}")

    # BLEURT
    if 'bleurt' in results:
        logger.info(f"\nBLEURT: {results['bleurt']:.4f}")

    # Distinct-n (Diversity)
    distinct_keys = [k for k in results.keys() if k.startswith('distinct_')]
    if distinct_keys:
        logger.info("\nDistinct-n (Diversity):")
        for key in sorted(distinct_keys):
            n = key.split('_')[1]
            logger.info(f"  Distinct-{n}: {results[key]:.4f}")

    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaMA-Factory JSONL predictions")
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to JSONL prediction file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for evaluation results (JSON format)'
    )

    args = parser.parse_args()

    predictions, references = load_jsonl_predictions(args.input)

    if len(predictions) == 0:
        logger.error("No valid predictions found")
        return

    results = evaluate_predictions(predictions, references)

    print_results(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
