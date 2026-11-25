
import json
import logging
from typing import List, Dict, Optional
import numpy as np
import evaluate

logger = logging.getLogger(__name__)


class Evaluator:

    def __init__(self):
        self.metrics = {}

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        compute_all: bool = True
    ) -> Dict[str, float]:
        results = {}

        # BLEU
        if compute_all:
            try:
                bleu_scores = compute_bleu(predictions, references)
                results.update(bleu_scores)
            except Exception as e:
                logger.warning(f"Failed to compute BLEU: {e}")

        # ROUGE
        if compute_all:
            try:
                rouge_scores = compute_rouge(predictions, references)
                results.update(rouge_scores)
            except Exception as e:
                logger.warning(f"Failed to compute ROUGE: {e}")

        # METEOR
        if compute_all:
            try:
                meteor_score = compute_meteor(predictions, references)
                results["meteor"] = meteor_score
            except Exception as e:
                logger.warning(f"Failed to compute METEOR: {e}")

        # BERTScore
        if compute_all:
            try:
                bertscore = compute_bertscore(predictions, references)
                results.update(bertscore)
            except Exception as e:
                logger.warning(f"Failed to compute BERTScore: {e}")

        # BLEURT
        if compute_all:
            try:
                bleurt_score = compute_bleurt(predictions, references)
                results["bleurt"] = bleurt_score
            except Exception as e:
                logger.warning(f"Failed to compute BLEURT: {e}")

        return results

    def evaluate_from_files(
        self,
        prediction_file: str,
        reference_file: str,
        output_file: Optional[str] = None
    ) -> Dict[str, float]:
        logger.info(f"Loading predictions from: {prediction_file}")
        with open(prediction_file, 'r', encoding='utf-8') as f:
            predictions_data = json.load(f)

        logger.info(f"Loading references from: {reference_file}")
        with open(reference_file, 'r', encoding='utf-8') as f:
            references_data = json.load(f)

        logger.info(f"Extracting text data ({len(predictions_data)} samples)...")
        predictions = [p["generated_question"] for p in predictions_data]
        references = [r["question"] for r in references_data]

        logger.info("Computing evaluation metrics...")
        results = self.evaluate(predictions, references)

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved evaluation results to {output_file}")

        return results


def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    logger.info("Computing BLEU scores...")
    bleu_metric = evaluate.load("bleu")

    references_formatted = [[ref] for ref in references]

    results = {}
    for n in range(1, 5):
        result = bleu_metric.compute(
            predictions=predictions,
            references=references_formatted,
            max_order=n
        )
        results[f"bleu_{n}"] = result['bleu']

    return results


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    logger.info("Computing ROUGE scores...")
    rouge_metric = evaluate.load("rouge")

    results = rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )

    return {
        "rouge_1_f": results['rouge1'],
        "rouge_1_p": results.get('rouge1', 0),
        "rouge_1_r": results.get('rouge1', 0),
        "rouge_2_f": results['rouge2'],
        "rouge_l_f": results['rougeL']
    }


def compute_meteor(predictions: List[str], references: List[str]) -> float:
    logger.info("Computing METEOR scores...")
    meteor_metric = evaluate.load("meteor")

    result = meteor_metric.compute(
        predictions=predictions,
        references=references
    )

    return result['meteor']


def compute_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
    logger.info("Computing BERTScore (the first run downloads a model of about 400 MB)...")
    bertscore_metric = evaluate.load("bertscore")

    results = bertscore_metric.compute(
        predictions=predictions,
        references=references,
        lang="en",
        device="cuda"
    )

    return {
        "bertscore_precision": np.mean(results['precision']),
        "bertscore_recall": np.mean(results['recall']),
        "bertscore_f1": np.mean(results['f1'])
    }


def compute_bleurt(predictions: List[str], references: List[str]) -> float:
    logger.info("Computing BLEURT scores (the first run downloads the BLEURT-20 model, about 1.6 GB)...")
    bleurt_metric = evaluate.load("bleurt", "BLEURT-20")

    result = bleurt_metric.compute(
        predictions=predictions,
        references=references
    )

    return np.mean(result['scores'])


def compute_distinct_n(predictions: List[str], max_n: int = 4) -> Dict[str, float]:
    results = {}

    for n in range(1, max_n + 1):
        ngrams = []
        for text in predictions:
            tokens = text.lower().split()
            text_ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            ngrams.extend(text_ngrams)

        if len(ngrams) == 0:
            distinct_n = 0.0
        else:
            unique_ngrams = set(ngrams)
            distinct_n = len(unique_ngrams) / len(ngrams)

        results[f'distinct_{n}'] = distinct_n

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    predictions = [
        "What is the capital of France?",
        "How does photosynthesis work?"
    ]

    references = [
        "What is the capital city of France?",
        "Explain the process of photosynthesis."
    ]

    evaluator = Evaluator()

    results = evaluator.evaluate(predictions, references)

    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
