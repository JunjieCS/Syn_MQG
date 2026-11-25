"""
Multimodal Question Synergy Pairwise Evaluation Script (GPT version, single-dimension Synergy)

Uses GPT to directly compare two models' generated questions on Synergy
(whether joint image-text reasoning is required).
Each sample is randomly assigned AB order (saving API cost).
Reports win rates and Wilson confidence intervals.

## Usage

    python scripts/evaluate_synergy_pairwise_gpt.py \
        --predictions_a outputs/model_a/test.jsonl \
        --predictions_b outputs/model_b/test.jsonl \
        --data_file data/scienceqa_stage4_sft_caption/test.json \
        --image_root dataset/ScienceQA-main/data \
        --output outputs/evaluate/pairwise_synergy_gpt/results.jsonl \
        --summary outputs/evaluate/pairwise_synergy_gpt/summary.json \
        --model gpt-4o \
        --api_key sk-xxx \
        --api_url https://api.example.com \
        --num_workers 8
"""

import argparse
import base64
import json
import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from math import sqrt
from typing import Optional

from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


_MIME_MAP = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}


def encode_image(image_path: str) -> tuple[str, str]:
    ext = os.path.splitext(image_path)[1].lower()
    mime = _MIME_MAP.get(ext, 'image/jpeg')
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8'), mime


def build_content(prompt: str, images_b64: list) -> list:
    content = [{"type": "text", "text": prompt}]
    for img_b64, mime in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{img_b64}"}
        })
    return content


def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    for match in re.findall(json_pattern, text, re.DOTALL):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    return None


def load_data(data_file: str) -> list:
    with open(data_file, 'r', encoding='utf-8') as f:
        if data_file.endswith('.jsonl'):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def normalize_id(value) -> str:
    """Normalize prediction/data IDs for order-independent matching."""
    sid = str(value)
    if sid.startswith(('identity_', 'identy_')):
        return sid.rsplit('_', 1)[-1]
    return sid


def table_id_for(record: dict, fallback_sample: Optional[dict] = None) -> str:
    """Extract a HybridQA table key from image path, falling back to sample ID."""
    paths = record.get('image_paths') or record.get('image_path')
    if isinstance(paths, str):
        paths = [paths]
    if paths:
        path = str(paths[0]).replace('\\', '/')
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem:
            return stem
    if fallback_sample:
        paths = fallback_sample.get('image_paths') or fallback_sample.get('image_path')
        if isinstance(paths, str):
            paths = [paths]
        if paths:
            path = str(paths[0]).replace('\\', '/')
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem:
                return stem
        if fallback_sample.get('id') is not None:
            return normalize_id(fallback_sample['id'])
    return normalize_id(record.get('id', ''))


def _image_paths(sample: dict, image_root: str) -> list[str]:
    paths = sample.get('image_paths') or (
        [sample['image_path']] if sample.get('image_path') else []
    )
    result = []
    for p in paths:
        full = os.path.join(image_root, p.replace('\\', '/'))
        if os.path.isdir(full):
            exts = set(_MIME_MAP.keys())
            found = sorted(
                f for f in (os.path.join(full, fn) for fn in os.listdir(full))
                if os.path.splitext(f)[1].lower() in exts
            )
            result.extend(found)
        elif os.path.exists(full):
            result.append(full)
        else:
            logger.warning(f"Image not found, skipping: {full}")
    return result


def load_images(sample: dict, image_root: str) -> list:
    return [encode_image(p) for p in _image_paths(sample, image_root)]


# ==================== Statistics ====================

def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval"""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return round(center - margin, 4), round(center + margin, 4)


# ==================== Prompt ====================

SYSTEM_PROMPT = (
    "You are a strict multimodal synergy evaluator. "
    "Given the image(s), text context, reference answer, and two candidate questions (A and B), "
    "judge which question demonstrates higher 'synergy' — meaning it REQUIRES joint reasoning "
    "across both the image and the text to answer correctly. "
    "A question with high synergy cannot be adequately answered by the image alone or the text alone; "
    "it demands integrating information from both modalities together. "
    'Output JSON only: {"winner": "A" or "B" or "tie", "reason": "<brief justification>"}. '
    "No other content."
)

SYSTEM_PROMPT_MC = (
    "You are a strict multimodal synergy evaluator for multiple-choice questions. "
    "Given the image(s), text context, correct answer option, and two candidate questions "
    "(each with answer choices), judge which question demonstrates higher 'synergy' — "
    "meaning that SELECTING THE CORRECT OPTION from the given choices REQUIRES joint reasoning "
    "across both the image and the text. "
    "A question with high synergy cannot have its correct option identified by the image alone "
    "or the text alone; correctly choosing the answer demands integrating information from both modalities. "
    'Output JSON only: {"winner": "A" or "B" or "tie", "reason": "<brief justification>"}. '
    "No other content."
)


def _choices_block(choices: Optional[list]) -> str:
    if not choices:
        return ''
    formatted = ' | '.join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
    return f"[Choices]\n{formatted}\n\n"


def _format_question_with_choices(question: str, choices: list) -> str:
    formatted = ' | '.join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
    return f"{question}\n{formatted}"


def make_pairwise_messages(q_a: str, q_b: str, text: str, answer: str,
                           images_b64: list, choices: Optional[list] = None) -> list:
    cb = _choices_block(choices)
    prompt = (
        "You are given both the image(s) and the text context below.\n\n"
        f"[Text Context]\n{text}\n\n"
        f"[Reference Answer]\n{answer}\n\n"
        f"{cb}"
        f"[Question A]\n{q_a}\n\n"
        f"[Question B]\n{q_b}\n\n"
        "Criterion — Synergy (Joint Image+Text Necessity):\n"
        "Which question has HIGHER synergy? Synergy means the question's answer cannot be "
        "adequately supported by the image alone OR the text alone, but requires joint reasoning "
        "across both modalities. Consider:\n"
        "- Would the question be answerable with ONLY the image (no text)? If yes, low synergy.\n"
        "- Would the question be answerable with ONLY the text (no image)? If yes, low synergy.\n"
        "- Does answering REQUIRE integrating specific information from the image WITH specific "
        "information from the text? If yes, high synergy.\n\n"
        "Select the question that more strongly demands cross-modal joint reasoning.\n\n"
        f'Output JSON: {{"winner": "A" or "B" or "tie", "reason": "..."}}'
    )
    content = build_content(prompt, images_b64)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def make_pairwise_messages_mc(q_a: str, q_b: str, text: str, answer_letter: str,
                               images_b64: list, choices: list) -> list:
    q_a_formatted = _format_question_with_choices(q_a, choices)
    q_b_formatted = _format_question_with_choices(q_b, choices)

    prompt = (
        "You are given both the image(s) and the text context below.\n\n"
        f"[Text Context]\n{text}\n\n"
        f"[Correct Answer]\n{answer_letter}\n\n"
        f"[Question A]\n{q_a_formatted}\n\n"
        f"[Question B]\n{q_b_formatted}\n\n"
        "Criterion — Synergy (Joint Image+Text Necessity for Option Selection):\n"
        "Which question has HIGHER synergy? Synergy means that CORRECTLY SELECTING the right "
        "answer option from the given choices REQUIRES joint reasoning across both modalities. Consider:\n"
        "- Can you eliminate wrong options and select the correct one using ONLY the image (no text)? "
        "If yes, low synergy.\n"
        "- Can you eliminate wrong options and select the correct one using ONLY the text (no image)? "
        "If yes, low synergy.\n"
        "- Does correctly choosing the right option REQUIRE integrating specific information from "
        "the image WITH specific information from the text? If yes, high synergy.\n\n"
        "Select the question that more strongly demands cross-modal joint reasoning "
        "to identify the correct answer among the choices.\n\n"
        f'Output JSON: {{"winner": "A" or "B" or "tie", "reason": "..."}}'
    )
    content = build_content(prompt, images_b64)
    return [
        {"role": "system", "content": SYSTEM_PROMPT_MC},
        {"role": "user", "content": content},
    ]


# ==================== GPT API ====================

def _response_debug_text(resp) -> str:
    """Serialize an OpenAI-compatible response for diagnostic logging."""
    try:
        return resp.model_dump_json(indent=2)
    except Exception:
        return repr(resp)


def call_gpt(client: OpenAI, model: str, messages: list, max_retries: int,
             retry_delay: float, max_tokens: int = 1024,
             request_label: str = '') -> str:
    for attempt in range(max_retries):
        try:
            label = f" [{request_label}]" if request_label else ''
            logger.info(
                "API request%s: model=%s attempt=%d/%d message_count=%d",
                label, model, attempt + 1, max_retries, len(messages),
            )
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
            )
            logger.info("API response%s:\n%s", label, _response_debug_text(resp))

            if not getattr(resp, 'choices', None):
                logger.warning("API response%s has no choices", label)
                return ''

            choice = resp.choices[0]
            message = getattr(choice, 'message', None)
            content = getattr(message, 'content', None) if message is not None else None
            finish_reason = getattr(choice, 'finish_reason', None)
            logger.info(
                "API content%s: %r (finish_reason=%r)",
                label, content, finish_reason,
            )
            if not content:
                logger.warning(
                    "API response%s returned empty content (finish_reason=%r)",
                    label, finish_reason,
                )
                return ''
            return content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            else:
                raise


def parse_winner(raw: Optional[str]) -> tuple[Optional[str], str]:
    if not raw:
        logger.warning("Winner parse skipped: raw API content is empty")
        return None, ''
    parsed = extract_json(raw)
    if not parsed:
        logger.warning("Winner parse failed: raw API content=%r", raw)
        return None, ''
    winner = parsed.get('winner', '').strip().upper()
    reason = parsed.get('reason', '')
    if winner in ('A', 'B', 'TIE'):
        logger.info("Winner parse succeeded: winner=%s reason=%r", winner, reason)
        return winner.lower() if winner == 'TIE' else winner, reason
    logger.warning("Winner parse found invalid winner=%r in raw content=%r", winner, raw)
    return None, reason



def derandomize(winner_raw: Optional[str], method_a_source: str) -> Optional[str]:
    if winner_raw is None:
        return None
    if winner_raw == 'tie':
        return 'tie'
    if method_a_source == 'predictions_a':
        return 'a' if winner_raw == 'A' else 'b'
    else:
        return 'b' if winner_raw == 'A' else 'a'


def _call_once(client: OpenAI, model: str, messages: list,
               max_retries: int, retry_delay: float,
               request_label: str = '') -> tuple[Optional[str], str]:
    raw = call_gpt(client, model, messages, max_retries, retry_delay,
                   request_label=request_label)
    winner_raw, reason = parse_winner(raw)
    if winner_raw is None:
        logger.warning("Retrying winner parse request [%s]", request_label)
        raw = call_gpt(client, model, messages, max_retries, retry_delay,
                       request_label=f"{request_label}/parse_retry")
        winner_raw, reason = parse_winner(raw)
    return winner_raw, reason


def combine(winner_fwd: Optional[str], winner_rev: Optional[str]) -> Optional[str]:
    if winner_fwd is None and winner_rev is None:
        return None
    if winner_fwd == winner_rev:
        return winner_fwd or 'tie'
    return 'tie'


def _process_one(task: dict, client: OpenAI, model: str,
                 max_retries: int, retry_delay: float) -> dict:
    sid = task['id']
    try:
        winner_fwd_raw, reason_fwd = _call_once(
            client, model, task['fwd_messages'], max_retries, retry_delay,
            request_label=f"sample={sid} direction=fwd")
        winner_rev_raw, reason_rev = _call_once(
            client, model, task['rev_messages'], max_retries, retry_delay,
            request_label=f"sample={sid} direction=rev")
    except Exception as e:
        logger.error(f"Sample {sid} failed: {e}")
        winner_fwd_raw = winner_rev_raw = None
        reason_fwd = reason_rev = ''

    winner_fwd = derandomize(winner_fwd_raw, 'predictions_a')
    winner_rev = derandomize(winner_rev_raw, 'predictions_b')
    winner_method = combine(winner_fwd, winner_rev)

    return {
        'id': sid,
        'table_id': task.get('table_id'),
        'question_a': task['orig_question_a'],
        'question_b': task['orig_question_b'],
        'winner_fwd': winner_fwd,
        'winner_rev': winner_rev,
        'reason_fwd': reason_fwd,
        'reason_rev': reason_rev,
        'winner_method': winner_method,
    }



def compute_summary(all_results: list, args) -> dict:
    valid = [r for r in all_results if r['winner_method'] is not None]
    a_pure = sum(1 for r in valid if r['winner_method'] == 'a')
    b_pure = sum(1 for r in valid if r['winner_method'] == 'b')
    ties = sum(1 for r in valid if r['winner_method'] == 'tie')
    total = len(valid)

    a_wins = a_pure + ties * 0.5
    b_wins = b_pure + ties * 0.5

    return {
        'meta': {
            'total_samples': len(all_results),
            'valid_samples': total,
            'failed_samples': len(all_results) - total,
            'model': args.model,
            'predictions_a': args.predictions_a,
            'predictions_b': args.predictions_b,
            'seed': args.seed,
            'evaluated_at': datetime.now().isoformat(),
        },
        'results': {
            'a_wins': a_wins,
            'b_wins': b_wins,
            'a_pure_wins': a_pure,
            'b_pure_wins': b_pure,
            'ties': ties,
            'total': total,
            'a_win_rate': round(a_wins / total, 4) if total else 0.0,
            'b_win_rate': round(b_wins / total, 4) if total else 0.0,
            'a_win_ci': list(wilson_ci(int(a_wins), total)),
            'b_win_ci': list(wilson_ci(int(b_wins), total)),
        },
    }



def parse_args():
    p = argparse.ArgumentParser(description='Pairwise multimodal synergy evaluation with GPT')
    p.add_argument('--predictions_a', default='outputs/inference_results/hybridqa/mllm_grpo_gauss_mi/stage_lr1e-534_e11_448_new_nounique_2.jsonl', help='Predictions from method A (JSONL)')
    p.add_argument('--predictions_b', default='outputs/inference_results/hybridqa/multiqg_gpt4o.jsonl', help='Predictions from method B (JSONL)')
    p.add_argument('--data_file', default='data/hybridqa_sft_grpo/test.json', help='Evaluation data (JSON/JSONL)')
    p.add_argument('--image_root', default='data/hybridqa/hybridqa', help='Image root directory')
    p.add_argument('--output', default='outputs/evaluate/hybridqa/pair_syn_new/gemini/3-multiqg4o/results.jsonl')
    p.add_argument('--summary', default='outputs/evaluate/hybridqa/pair_syn_new/gemini/3-multiqg4o/summary.json')
    p.add_argument('--model', default='gemini-2.5-pro')
    p.add_argument('--api_key', default='')
    p.add_argument('--api_url', default='')
    p.add_argument('--num_workers', type=int, default=16)
    p.add_argument('--max_retries', type=int, default=5)
    p.add_argument('--retry_delay', type=float, default=2.0)
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dataset', default='hybridqa', choices=['default', 'scienceqa', 'hybridqa'])
    p.add_argument('--problems_json', default='dataset/ScienceQA-main/data/scienceqa/problems.json')
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    api_url = args.api_url
    if api_url and not api_url.endswith('/v1'):
        api_url = api_url.rstrip('/') + '/v1'
    client = OpenAI(api_key=args.api_key, base_url=api_url)

    data_list = load_data(args.data_file)
    data_map = {normalize_id(s['id']): s for s in data_list if s.get('id') is not None}

    if args.dataset == 'scienceqa':
        with open(args.problems_json, 'r', encoding='utf-8') as f:
            problems = json.load(f)
        for s in data_list:
            sid = normalize_id(s['id'])
            if sid in problems and 'choices' in problems[sid]:
                s['choices'] = problems[sid]['choices']
                answer_idx = problems[sid].get('answer')
                if answer_idx is not None:
                    s['answer_letter'] = f"({chr(65 + answer_idx)})"
        data_map = {normalize_id(s['id']): s for s in data_list if s.get('id') is not None}

    preds_a = load_data(args.predictions_a)
    preds_b = load_data(args.predictions_b)
    map_b = {normalize_id(p['id']): p['predict'] for p in preds_b
             if p.get('id') is not None and p.get('predict')}
    table_for_id = {}
    if args.dataset == 'hybridqa':
        groups = {}
        for p in preds_a:
            if p.get('id') is None or not p.get('predict'):
                continue
            sid = normalize_id(p['id'])
            tid = table_id_for(p, data_map.get(sid))
            groups.setdefault(tid, []).append(p)
        rng = random.Random(args.seed)
        selected_a = []
        for tid, records in groups.items():
            chosen = rng.choice(records)
            selected_a.append(chosen)
            table_for_id[normalize_id(chosen['id'])] = tid
    else:
        selected_a = [p for p in preds_a if p.get('id') is not None and p.get('predict')]
    map_a = {normalize_id(p['id']): p['predict'] for p in selected_a}
    common_ids = [normalize_id(p['id']) for p in selected_a
                  if normalize_id(p['id']) in map_b and normalize_id(p['id']) in data_map]
    if args.max_samples:
        common_ids = common_ids[:args.max_samples]

    done_ids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    done_ids.add(normalize_id(json.loads(line)['id']))
        logger.info(f"Resume: {len(done_ids)} samples already completed")

    todo_ids = [sid for sid in common_ids if sid not in done_ids]
    logger.info(f"Pending evaluation: {len(todo_ids)} samples ({len(common_ids)} paired samples total)")

    tasks = []
    for sid in todo_ids:
        sample = data_map[sid]
        images_b64 = load_images(sample, args.image_root)
        choices = sample.get('choices')

        def build_messages(q_pos_a: str, q_pos_b: str) -> list:
            if choices and 'answer_letter' in sample:
                return make_pairwise_messages_mc(
                    q_pos_a, q_pos_b, sample.get('text', ''),
                    sample['answer_letter'], images_b64, choices,
                )
            return make_pairwise_messages(
                q_pos_a, q_pos_b, sample.get('text', ''), sample.get('answer', ''),
                images_b64, choices,
            )

        fwd_messages = build_messages(map_a[sid], map_b[sid])
        rev_messages = build_messages(map_b[sid], map_a[sid])
        tasks.append({
            'id': sid,
            'table_id': table_for_id.get(sid),
            'fwd_messages': fwd_messages,
            'rev_messages': rev_messages,
            'orig_question_a': map_a[sid],
            'orig_question_b': map_b[sid],
        })

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    mode = 'a' if args.resume else 'w'

    results = []
    with open(args.output, mode, encoding='utf-8') as out_f:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(_process_one, t, client, args.model,
                                args.max_retries, args.retry_delay): t
                for t in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc='text'):
                result = future.result()
                results.append(result)
                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                out_f.flush()

    all_results = []
    with open(args.output, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))

    summary = compute_summary(all_results, args)
    os.makedirs(os.path.dirname(args.summary) or '.', exist_ok=True)
    with open(args.summary, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    r = summary['results']
    print(f"\n{'='*50}")
    print(f"Pairwise Synergy text ({summary['meta']['valid_samples']} valid samples)")
    print(f"{'='*50}")
    print(f"A text: {r['a_win_rate']:.2%}  (text {r['a_pure_wins']} + tie*0.5 = {r['a_wins']})  CI: {r['a_win_ci']}")
    print(f"B text: {r['b_win_rate']:.2%}  (text {r['b_pure_wins']} + tie*0.5 = {r['b_wins']})  CI: {r['b_win_ci']}")
    print(f"text: {r['ties']}")
    print(f"\ntext: {args.summary}")


if __name__ == '__main__':
    main()
