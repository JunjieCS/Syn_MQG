#!/usr/bin/env python

import re
import torch
from typing import List, Dict, Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from sentence_transformers import SentenceTransformer
import spacy
import logging

logger = logging.getLogger(__name__)


class RewardCalculator:

    def __init__(
        self,
        reward_config: dict,
        visual_encoder = None,    
        clip_model_name: str = None
    ):
        self.use_v3 = reward_config.get('use_v3', False)

        self.gt_negative_weight = reward_config.get('gt_negative_weight', 0.5)

        self.attention_weight = reward_config.get(
            'attention_weight',
            reward_config.get('modality_distinction_weight', 0.5)
        )
        self.rule_weight = reward_config.get('rule_weight', 0.3)
        self.gt_similarity_weight = reward_config.get('gt_similarity_weight', 0.2)

        attention_cfg = reward_config.get('attention', {})
        self.attn_balance_weight = attention_cfg.get('balance', 0.25)
        self.attn_image_caption_weight = attention_cfg.get('image_caption', 0.25)
        self.attn_text_input_weight = attention_cfg.get('text_input', 0.25)
        self.attn_final_answer_weight = attention_cfg.get('final_answer', 0.25)

        gt_sim_cfg = reward_config.get('gt_similarity', {})
        self.bleu_weight = gt_sim_cfg.get('bleu', 0.3)
        self.rouge_weight = gt_sim_cfg.get('rouge', 0.3)
        self.semantic_weight = gt_sim_cfg.get('semantic', 0.4)

        self.rouge = Rouge()
        self.semantic_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.semantic_model = self.semantic_model.to('cpu')
        logger.info("✓ SentenceTransformer (bge-large-en-v1.5) moved to CPU")

        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.reranker_tokenizer = AutoTokenizer.from_pretrained('../../models/Qwen3-Reranker-4B')
        self.reranker_model = AutoModelForCausalLM.from_pretrained(
            '../../models/Qwen3-Reranker-4B', torch_dtype=torch.float16
        )
        self.reranker_model = self.reranker_model.to('cuda').eval()
        self._reranker_yes_id = self.reranker_tokenizer("yes", add_special_tokens=False)["input_ids"][0]
        self._reranker_no_id  = self.reranker_tokenizer("no",  add_special_tokens=False)["input_ids"][0]
        logger.info("✓ Qwen3-Reranker-4B loaded on CUDA")

        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✓ spaCy model 'en_core_web_sm' loaded successfully")
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
            raise

        logger.info(f"text: text={self.attention_weight}, text={self.rule_weight}, GTtext={self.gt_similarity_weight}")


        self.clip_model = None
        self.clip_processor = None

        if clip_model_name is not None:
            try:
                from transformers import CLIPModel, CLIPProcessor

                logger.info(f"Loading CLIPModel from {clip_model_name}...")
                self.clip_model = CLIPModel.from_pretrained(clip_model_name)
                self.clip_model.eval()

                self.clip_model = self.clip_model.to('cpu')

                for param in self.clip_model.parameters():
                    param.requires_grad = False

                self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

                logger.info("✓ CLIPModel loaded and moved to CPU")
            except Exception as e:
                logger.warning(f"Failed to load CLIPModel: {e}")
                self.clip_model = None
                self.clip_processor = None
        else:
            logger.info("CLIP model not provided, CLIP similarity will be skipped")


        copy_penalty_cfg = reward_config.get('copy_penalty', {})
        self.image_overlap_threshold = copy_penalty_cfg.get('image_overlap_threshold', 0.7)
        self.image_max_penalty = copy_penalty_cfg.get('image_max_penalty', 0.3)
        self.text_overlap_threshold = copy_penalty_cfg.get('text_overlap_threshold', 0.6)
        self.text_max_penalty = copy_penalty_cfg.get('text_max_penalty', 0.4)
        self.lcs_threshold = copy_penalty_cfg.get('lcs_threshold', 5)
        self.lcs_penalty = copy_penalty_cfg.get('lcs_penalty', 0.2)

        self.use_clip_similarity = reward_config.get('use_clip_similarity', False)
        
        self.kl_beta = reward_config.get('kl_beta', 0.05)


    def _has_correct_format(self, text: str) -> bool:
        pattern = r'Step\s+\d+:'
        return bool(re.search(pattern, text, re.IGNORECASE))

    def _has_extra_output(self, text: str) -> bool:
        pattern = r'(Step\s+\d+:\s*.*?)(?=Step\s+\d+:|$)'
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)

        if not matches:
            return True

        reconstructed = ""
        for match in matches:
            cleaned_step = match.strip()
            reconstructed += cleaned_step + "\n"

        cleaned_original = text.strip()

        len_reconstructed = len(reconstructed.strip())
        len_original = len(cleaned_original)

        if len_reconstructed == 0:
            return True

        length_ratio = len_original / len_reconstructed

        return length_ratio > 1.1

    def _count_steps(self, text: str) -> int:
        pattern = r'Step\s+(\d+):'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return len(matches) if matches else 0

    def _has_modality_tags(self, text: str) -> bool:
        return '[Image]' in text or '[Text]' in text

    def _has_logical_flow(self, text: str) -> bool:
        connectors = [
            'therefore', 'thus', 'hence', 'so', 'because', 'since',
            'as a result', 'consequently', 'accordingly',
        ]
        text_lower = text.lower()
        return any(conn in text_lower for conn in connectors)

    def _extract_keywords_with_pos(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        doc = self.nlp(text.lower())

        keywords = []
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                keywords.append(token.text)
            elif token.pos_ == 'VERB':
                keywords.append(token.text)
            elif token.pos_ == 'NUM':
                keywords.append(token.text)

        numbers = re.findall(r'\d+\.?\d*%?|\d+/\d+', text)
        keywords.extend([n.lower() for n in numbers])

        return list(set(keywords))

    def _compute_keyword_match_ratio(self, source_keywords: List[str], target_text: str) -> float:
        if not source_keywords:
            return 0.0

        if not target_text or not target_text.strip():
            return 0.0

        target_lower = target_text.lower()
        matched = sum(1 for kw in source_keywords if kw in target_lower)

        return matched / len(source_keywords)


    def _has_question_tag(self, text: str) -> bool:
        return '[Question]' in text

    def _is_question_after_steps(self, text: str) -> bool:
        step_pattern = r'Step\s+\d+:'
        step_matches = list(re.finditer(step_pattern, text, re.IGNORECASE))

        if not step_matches:
            return False

        last_step_pos = step_matches[-1].end()

        question_pos = text.find('[Question]')

        if question_pos == -1:
            return False

        return question_pos > last_step_pos

    def _extract_generated_question(self, text: str) -> str:
        match = re.search(r'\[Question\]\s*(.+?)$', text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_gt_question_from_chain(self, reasoning_chain) -> str:
        if isinstance(reasoning_chain, list):
            chain_text = " ".join(reasoning_chain)
        else:
            chain_text = reasoning_chain

        match = re.search(r'\[Question\]\s*(.+?)$', chain_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""


    def _compute_lexical_overlap(self, generated: str, source: str) -> float:
        if not generated or not generated.strip():
            return 0.0
        if not source or not source.strip():
            return 0.0

        gen_words = set(generated.lower().split())
        src_words = set(source.lower().split())

        intersection = len(gen_words & src_words)
        union = len(gen_words | src_words)

        if union == 0:
            return 0.0

        return intersection / union

    def _detect_longest_common_substring(
        self,
        generated: str,
        source: str,
        max_length: int = 100
    ) -> int:
        if not generated or not generated.strip():
            return 0
        if not source or not source.strip():
            return 0

        gen_words = generated.lower().split()[:max_length]
        src_words = source.lower().split()[:max_length]

        m, n = len(gen_words), len(src_words)

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_len = 0

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if gen_words[i-1] == src_words[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    max_len = max(max_len, dp[i][j])
                else:
                    dp[i][j] = 0

        return max_len

    def _compute_copy_penalty(
        self,
        generated: str,
        source: str,
        overlap_threshold: float,
        max_penalty: float,
        lcs_threshold: int = 5
    ) -> float:
        if not generated or not generated.strip():
            return 0.0
        if not source or not source.strip():
            return 0.0

        overlap = self._compute_lexical_overlap(generated, source)

        if overlap < overlap_threshold:
            base_penalty = 0.0
        else:
            ratio = (overlap - overlap_threshold) / (1.0 - overlap_threshold + 1e-8)
            base_penalty = ratio * max_penalty

        lcs_length = self._detect_longest_common_substring(generated, source)
        lcs_extra_penalty = self.lcs_penalty if lcs_length >= lcs_threshold else 0.0

        total_penalty = -(base_penalty + lcs_extra_penalty)

        return total_penalty


    def _compute_clip_image_text_similarity(
        self,
        image_reasoning_text: str,
        image_tensor: torch.Tensor
    ) -> float:
        if self.clip_model is None or not self.use_clip_similarity:
            return 0.0

        if not image_reasoning_text or not image_reasoning_text.strip():
            return 0.0
        if image_tensor is None or image_tensor.numel() == 0:
            return 0.0

        try:
            with torch.no_grad():
                # image_tensor: [3, H, W] → [1, 3, H, W]
                from PIL import Image
                import numpy as np

                # tensor [3, H, W] → numpy [H, W, 3]
                img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)

                pil_image = Image.fromarray(img_np)

                inputs = self.clip_processor(
                    text=[image_reasoning_text],
                    images=[pil_image],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )

                inputs = {k: v.to('cpu') for k, v in inputs.items()}

                image_features = self.clip_model.get_image_features(pixel_values=inputs['pixel_values'])  # [1, 512]
                text_features = self.clip_model.get_text_features(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )  # [1, 512]

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                similarity = (image_features @ text_features.T).item()

                return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.warning(f"CLIP similarity computation failed: {e}")
            return 0.0


    def parse_reasoning_chain(self, chain: str) -> Dict:
        steps = re.findall(
            r'Step\s+\d+:\s*\[(\w+)\]\s*(.+?)(?=Step\s+\d+:|$)',
            chain,
            re.IGNORECASE | re.DOTALL
        )

        image_steps = []
        text_steps = []

        for modality, content in steps:
            content = content.strip()
            if modality.lower() == 'image':
                image_steps.append(content)
            elif modality.lower() == 'text':
                text_steps.append(content)

        final_step = steps[-1][1].strip() if steps else ""

        return {
            'has_image': len(image_steps) > 0,
            'has_text': len(text_steps) > 0,
            'image_steps': image_steps,
            'text_steps': text_steps,
            'final_step': final_step
        }

    def compute_attention_keyword_reward(
        self,
        parsed_chain: Dict,
        caption: str,
        text: str,
        answer: str,
        image_tensor: Optional[torch.Tensor] = None
    ) -> float:
        total_score = 0.0
        count = 0

        if parsed_chain['has_image'] and parsed_chain['image_steps']:
            caption_keywords = self._extract_keywords_with_pos(caption)
            image_content = " ".join(parsed_chain['image_steps'])

            keyword_score = self._compute_keyword_match_ratio(caption_keywords, image_content)

            semantic_score = self._compute_semantic_similarity(image_content, caption)

            if self.clip_model is not None and self.use_clip_similarity and image_tensor is not None:
                clip_score = self._compute_clip_image_text_similarity(image_content, image_tensor)
            else:
                clip_score = 0.0

            copy_penalty = self._compute_copy_penalty(
                generated=image_content,
                source=caption,
                overlap_threshold=self.image_overlap_threshold,
                max_penalty=self.image_max_penalty,
                lcs_threshold=self.lcs_threshold
            )

            if self.clip_model is not None and self.use_clip_similarity and image_tensor is not None:
                image_score_base = 0.25 * keyword_score + 0.25 * semantic_score + 0.5 * clip_score
            else:
                image_score_base = 0.5 * keyword_score + 0.5 * semantic_score

            image_score = max(0.0, image_score_base + copy_penalty)

            total_score += image_score
            count += 1

        if parsed_chain['has_text'] and parsed_chain['text_steps']:
            text_keywords = self._extract_keywords_with_pos(text)
            text_content = " ".join(parsed_chain['text_steps'])

            keyword_score = self._compute_keyword_match_ratio(text_keywords, text_content)

            semantic_score = self._compute_semantic_similarity(text_content, text)

            copy_penalty = self._compute_copy_penalty(
                generated=text_content,
                source=text,
                overlap_threshold=self.text_overlap_threshold,
                max_penalty=self.text_max_penalty,
                lcs_threshold=self.lcs_threshold
            )

            text_score_base = 0.5 * keyword_score + 0.5 * semantic_score

            text_score = max(0.0, text_score_base + copy_penalty)

            total_score += text_score
            count += 1

        if parsed_chain['final_step']:
            answer_keywords = self._extract_keywords_with_pos(answer)

            keyword_score = self._compute_keyword_match_ratio(answer_keywords, parsed_chain['final_step'])

            semantic_score = self._compute_semantic_similarity(parsed_chain['final_step'], answer)

            final_score = 0.5 * keyword_score + 0.5 * semantic_score

            total_score += final_score
            count += 1

        return total_score / count if count > 0 else 0.0

    def compute_rule_reward(
        self,
        generated_chain: str,
        hop_count: int,
        caption: str = None,
        text: str = None,
        answer: str = None,
        ground_truth: str = None,
        image_tensor: Optional[torch.Tensor] = None
    ) -> float:
        reward = 0.0
        parsed = self.parse_reasoning_chain(generated_chain)

        has_format = self._has_correct_format(generated_chain)
        has_extra = self._has_extra_output(generated_chain)

        if has_format and not has_extra:
            reward += 0.15
        elif has_format and has_extra:
            reward += 0.075

        actual_steps = self._count_steps(generated_chain)
        if actual_steps == hop_count:
            reward += 0.15

        if parsed['has_image'] and parsed['has_text']:
            reward += 0.1

        if caption is not None and text is not None and answer is not None:
            attention_reward = self.compute_attention_keyword_reward(
                parsed, caption, text, answer,
                image_tensor=image_tensor
            )
            reward += 0.3 * attention_reward

        if ground_truth is not None:
            chain_sim = self._compute_reasoning_chain_similarity(
                generated_chain, ground_truth
            )
            reward += 0.3 * chain_sim

        return reward


    def _compute_bleu(self, generated: str, reference: str) -> float:
        gen_tokens = generated.split()
        ref_tokens = reference.split()

        if len(gen_tokens) == 0:
            return 0.0

        smoothing = SmoothingFunction().method1
        score = sentence_bleu(
            [ref_tokens],
            gen_tokens,
            smoothing_function=smoothing
        )
        return score

    def _compute_rouge_l(self, generated: str, reference: str) -> float:
        if not generated.strip() or not reference.strip():
            return 0.0

        try:
            scores = self.rouge.get_scores(generated, reference)
            return scores[0]['rouge-l']['f']
        except Exception as e:
            logger.warning(f"ROUGEtext: {e}")
            return 0.0

    def _preprocess_reasoning_chain(self, chain) -> str:
        if chain is None:
            return ""

        if isinstance(chain, list):
            chain = " ".join(chain)

        pattern = r'Step\s+\d+:\s*(\[(?:Image|Text)\].*?)(?=Step\s+\d+:|$)'
        matches = re.findall(pattern, chain, re.IGNORECASE | re.DOTALL)

        if not matches:
            return chain.strip()

        result = " ".join([m.strip() for m in matches])
        return result

    def _compute_semantic_similarity(self, generated: str, reference: str) -> float:
        if not generated.strip() or not reference.strip():
            return 0.0

        embeddings = self.semantic_model.encode([generated, reference], show_progress_bar=False, normalize_embeddings=True)

        gen_emb = torch.tensor(embeddings[0], dtype=torch.float32)
        ref_emb = torch.tensor(embeddings[1], dtype=torch.float32)
        similarity = torch.nn.functional.cosine_similarity(
            gen_emb.unsqueeze(0),
            ref_emb.unsqueeze(0)
        ).item()

        return (similarity + 1) / 2

    def _compute_question_rerank_score(
        self,
        generated_question: str,
        gt_question: str,
        answer: str = None,
        question_type: str = None,
    ) -> float:
        context_parts = []
        if answer:
            context_parts.append(f"answer: {answer}")
        if question_type:
            context_parts.append(f"question type: {question_type}")
        context = ", ".join(context_parts)

        if context:
            instruction = (
                f"Given that {context}, "
                "judge if the two questions are semantically equivalent and asking about the same thing."
            )
        else:
            instruction = "Judge if the two questions are semantically equivalent and asking about the same thing."

        prompt = (
            "<|im_start|>system\n"
            'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".\n'
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"<Instruct>: {instruction}\n"
            f"<Query>: {generated_question}\n"
            f"<Document>: {gt_question}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )

        inputs = self.reranker_tokenizer(
            prompt, return_tensors='pt', truncation=True, max_length=512
        ).to('cuda')
        with torch.no_grad():
            logits = self.reranker_model(**inputs).logits[0, -1, :]
            scores = torch.softmax(
                logits[[self._reranker_no_id, self._reranker_yes_id]].float(), dim=0
            )
        return scores[1].item()

    def _compute_reasoning_chain_similarity(
        self,
        generated_chain: str,
        ground_truth_chain
    ) -> float:
        gen_processed = self._preprocess_reasoning_chain(generated_chain)
        gt_processed = self._preprocess_reasoning_chain(ground_truth_chain)

        if not gen_processed or not gt_processed:
            return 0.0

        return self._compute_semantic_similarity(gen_processed, gt_processed)

    def compute_metric_reward(
        self,
        generated_chain: str,
        ground_truth: str
    ) -> float:
        bleu_score = self._compute_bleu(generated_chain, ground_truth)

        rouge_score = self._compute_rouge_l(generated_chain, ground_truth)

        semantic_score = self._compute_semantic_similarity(generated_chain, ground_truth)

        reward = (
            self.bleu_weight * bleu_score +
            self.rouge_weight * rouge_score +
            self.semantic_weight * semantic_score
        )

        return reward

    def _compute_kl_divergence(
    self,
    old_log_probs: torch.Tensor,
    current_log_probs: torch.Tensor,
    attention_mask: torch.Tensor
) -> float:
        token_kl = current_log_probs - old_log_probs  # [batch_size, seq_len]

        masked_kl = token_kl * attention_mask  # [batch_size, seq_len]

        num_valid_tokens = attention_mask.sum(dim=1).clamp(min=1)  # [batch_size]
        kl_per_sample = masked_kl.sum(dim=1) / num_valid_tokens   # [batch_size]

        return kl_per_sample.mean().item()
    
    

    def compute_reward(
        self,
        generated_chain: str,
        ground_truth: str,
        hop_count: int,
        question_type: str
    ) -> float:
        rule_reward = self.compute_rule_reward(
            generated_chain, hop_count
        )

        metric_reward = self.compute_metric_reward(
            generated_chain, ground_truth
        )

        final_reward = (
            self.rule_weight * rule_reward +
            self.gt_similarity_weight * metric_reward
        )

        return final_reward

    def compute_batch_rewards(
        self,
        generated_chains: List[List[str]],  # [batch_size, num_samples]
        ground_truths: List[str],            # [batch_size]
        hop_counts: List[int],               # [batch_size]
        question_types: List[str]            # [batch_size]
    ) -> torch.Tensor:
        batch_size = len(generated_chains)
        num_samples = len(generated_chains[0]) if batch_size > 0 else 0

        rewards = torch.zeros(batch_size, num_samples)

        for i in range(batch_size):
            for j in range(num_samples):
                reward = self.compute_reward(
                    generated_chains[i][j],
                    ground_truths[i],
                    hop_counts[i],
                    question_types[i]
                )
                rewards[i, j] = reward

        return rewards


    def compute_attention_balance(
        self,
        attention_weights: tuple,
        token_type_ids: List[torch.Tensor],
        input_length: int
    ) -> float:
        if not attention_weights or len(attention_weights) == 0:
            return 0.0

        try:
            last_step_attn = attention_weights[0]
            if isinstance(last_step_attn, tuple):
                last_layer_attn = last_step_attn[-1]
            else:
                last_layer_attn = last_step_attn

            # [batch, heads, seq, seq] -> [batch, seq, seq]
            avg_attn = last_layer_attn.mean(dim=1)

            scores = []
            for b in range(len(token_type_ids)):
                gen_to_input = avg_attn[b, input_length:, :input_length]

                type_ids = token_type_ids[b]
                soft_prompt_mask = (type_ids == 0)
                image_mask = (type_ids == 1)
                text_mask = (type_ids == 2)

                attn_to_soft = gen_to_input[:, soft_prompt_mask].mean() if soft_prompt_mask.any() else 0.0
                attn_to_image = gen_to_input[:, image_mask].mean() if image_mask.any() else 0.0
                attn_to_text = gen_to_input[:, text_mask].mean() if text_mask.any() else 0.0

                attn_values = torch.tensor([attn_to_soft, attn_to_image, attn_to_text], dtype=torch.float32)
                std = attn_values.std()
                score = 1.0 - min(std.item() / 0.5, 1.0)
                scores.append(score)

            return sum(scores) / len(scores) if scores else 0.0

        except Exception as e:
            logger.warning(f"text: {e}")
            return 0.0

    def compute_image_caption_similarity(self, parsed_chain: Dict, caption: str) -> float:
        if not parsed_chain['has_image']:
            return 0.0

        image_content = " ".join(parsed_chain['image_steps'])
        return self._compute_semantic_similarity(image_content, caption)

    def compute_text_input_similarity(self, parsed_chain: Dict, text: str) -> float:
        if not parsed_chain['has_text']:
            return 0.0

        text_content = " ".join(parsed_chain['text_steps'])
        return self._compute_semantic_similarity(text_content, text)

    def compute_final_answer_similarity(self, parsed_chain: Dict, answer: str) -> float:
        if not parsed_chain['final_step']:
            return 0.0

        return self._compute_semantic_similarity(parsed_chain['final_step'], answer)

    def compute_modality_distinction_score(
        self,
        parsed_chain: Dict,
        caption: str,
        text: str
    ) -> float:
        score = 0.0

        if parsed_chain['has_image'] and parsed_chain['image_steps']:
            image_content = " ".join(parsed_chain['image_steps'])

            sim_caption = self._compute_semantic_similarity(image_content, caption)

            sim_text = self._compute_semantic_similarity(image_content, text)
            dist_text = 1.0 - sim_text

            image_score = (sim_caption + dist_text) / 2.0
            score += 0.5 * image_score

        if parsed_chain['has_text'] and parsed_chain['text_steps']:
            text_content = " ".join(parsed_chain['text_steps'])

            sim_text = self._compute_semantic_similarity(text_content, text)

            sim_caption = self._compute_semantic_similarity(text_content, caption)
            dist_caption = 1.0 - sim_caption

            text_score = (sim_text + dist_caption) / 2.0
            score += 0.5 * text_score

        return score


    def compute_reward_with_attention(
        self,
        generated_chain: str,
        ground_truth: str,
        hop_count: int,
        caption: str,
        text: str,
        answer: str,
        attention_weights: Optional[tuple] = None,
        token_type_ids: Optional[List[torch.Tensor]] = None,
        input_length: Optional[int] = None
    ) -> float:
        parsed = self.parse_reasoning_chain(generated_chain)

        attention_reward = 0.0

        if attention_weights is not None and token_type_ids is not None and input_length is not None:
            balance_score = self.compute_attention_balance(
                attention_weights, token_type_ids, input_length
            )
            attention_reward += self.attn_balance_weight * balance_score

        img_cap_sim = self.compute_image_caption_similarity(parsed, caption)
        attention_reward += self.attn_image_caption_weight * img_cap_sim

        text_input_sim = self.compute_text_input_similarity(parsed, text)
        attention_reward += self.attn_text_input_weight * text_input_sim

        final_ans_sim = self.compute_final_answer_similarity(parsed, answer)
        attention_reward += self.attn_final_answer_weight * final_ans_sim

        rule_reward = self.compute_rule_reward(generated_chain, hop_count)

        gt_similarity = (
            self.bleu_weight * self._compute_bleu(generated_chain, ground_truth) +
            self.rouge_weight * self._compute_rouge_l(generated_chain, ground_truth) +
            self.semantic_weight * self._compute_semantic_similarity(generated_chain, ground_truth)
        )

        final_reward = (
            self.attention_weight * attention_reward +
            self.rule_weight * rule_reward +
            self.gt_similarity_weight * gt_similarity
        )

        return final_reward

    def compute_reward_v2(
        self,
        generated_chain: str,
        ground_truth: str,
        hop_count: int,
        caption: str,
        text: str,
        answer: str
    ) -> float:
        parsed = self.parse_reasoning_chain(generated_chain)


        modality_score = self.compute_modality_distinction_score(parsed, caption, text)

        final_answer_sim = self.compute_final_answer_similarity(parsed, answer)

        modality_reward = 0.8 * modality_score + 0.2 * final_answer_sim

        rule_reward = self.compute_rule_reward(generated_chain, hop_count)

        gt_reward = (
            self.bleu_weight * self._compute_bleu(generated_chain, ground_truth) +
            self.rouge_weight * self._compute_rouge_l(generated_chain, ground_truth) +
            self.semantic_weight * self._compute_semantic_similarity(generated_chain, ground_truth)
        )

        final_reward = (
            0.375 * modality_reward +
            0.375 * rule_reward +
            0.25 * gt_reward
        )

        return final_reward

    def compute_reward_v3(
        self,
        generated_chain: str,
        ground_truth: str,
        hop_count: int,
        caption: str,
        text: str,
        answer: str,
        image_tensor: Optional[torch.Tensor] = None
    ) -> float:
        rule_reward = self.compute_rule_reward(
            generated_chain=generated_chain,
            hop_count=hop_count,
            caption=caption,
            text=text,
            answer=answer,
            ground_truth=ground_truth,
            image_tensor=image_tensor
        )

        return rule_reward

    def compute_reward_v4(
        self,
        generated_chain: str,
        ground_truth: str,
        hop_count: int,
        caption: str,
        text: str,
        answer: str,
        generated_answer: str
    ) -> float:
        rule_reward = self.compute_rule_reward(
            generated_chain=generated_chain,
            hop_count=hop_count,
            caption=caption,
            text=text,
            answer=answer
        )

        gt_loss = 0.0
        if generated_answer.strip().lower() != answer.strip().lower():
            gt_loss = 1.0

        total_reward = rule_reward - self.gt_negative_weight * gt_loss

        return total_reward

    def compute_reward_v5(
        self,
        generated_chain: str,
        ground_truth_chain,
        hop_count: int = None,
        caption: str = None,
        text: str = None
    ) -> float:
        return self._compute_v5_base(generated_chain, ground_truth_chain, hop_count)

    def _compute_v5_base(
        self,
        generated_chain: str,
        ground_truth_chain,
        hop_count: int = None,
    ) -> float:
        reward = 0.0
        parsed = self.parse_reasoning_chain(generated_chain)

        has_format = self._has_correct_format(generated_chain)
        has_extra = self._has_extra_output(generated_chain)
        if has_format and not has_extra:
            reward += 0.05
        elif has_format and has_extra:
            reward += 0.025

        if hop_count is not None:
            actual_steps = self._count_steps(generated_chain)
            if actual_steps == hop_count:
                reward += 0.05

        if parsed['has_image'] and parsed['has_text']:
            reward += 0.05

        has_question = self._has_question_tag(generated_chain)
        if not has_question:
            question_similarity = 0.0
        else:
            question_after_steps = self._is_question_after_steps(generated_chain)
            if not question_after_steps:
                question_similarity = 0.0
            else:
                generated_question = self._extract_generated_question(generated_chain)
                if not generated_question or not generated_question.strip():
                    question_similarity = 0.0
                else:
                    gt_question = self._extract_gt_question_from_chain(ground_truth_chain)
                    question_similarity = self._compute_semantic_similarity(
                        generated_question, gt_question
                    )

        return reward + 0.85 * question_similarity

    def compute_reward_v6(
        self,
        generated_chain: str,
        ground_truth_chain,
        hop_count: int = None,
        caption: str = None,
        text: str = None,
        mi_scores: dict = None,
        rule_weight: float = 0.2,
        sim_weight: float = 0.5,
        mi_weight: float = 0.3,
        syn_threshold: float = 0.0,
        answer: str = None,
        question_type: str = None,
    ) -> dict:
        rule_r = 0.0
        parsed = self.parse_reasoning_chain(generated_chain)

        has_format = self._has_correct_format(generated_chain)
        has_extra = self._has_extra_output(generated_chain)
        if has_format and not has_extra:
            rule_r += 1/3
        elif has_format and has_extra:
            rule_r += 1/6

        if hop_count is not None:
            actual_steps = self._count_steps(generated_chain)
            if actual_steps == hop_count:
                rule_r += 1/3

        if parsed['has_image'] and parsed['has_text']:
            rule_r += 1/3

        has_question = self._has_question_tag(generated_chain)
        if not has_question:
            sim_r = 0.0
        else:
            question_after_steps = self._is_question_after_steps(generated_chain)
            if not question_after_steps:
                sim_r = 0.0
            else:
                generated_question = self._extract_generated_question(generated_chain)
                if not generated_question or not generated_question.strip():
                    sim_r = 0.0
                else:
                    gt_question = self._extract_gt_question_from_chain(ground_truth_chain)
                    sim_r = self._compute_question_rerank_score(
                        generated_question, gt_question, answer, question_type
                    )

        if mi_scores is None:
            syn_reward = 0.0
        else:
            syn = mi_scores['syn']
            syn_reward = 1.0 if syn > syn_threshold else -1.0

        total = rule_weight * rule_r + sim_weight * sim_r + mi_weight * syn_reward

        return {
            'total': total,
            'rule': rule_r,
            'sim': sim_r,
            'mi': syn_reward,
        }

    def get_reward_components(
        self,
        generated_chain: str,
        ground_truth: str,
        hop_count: int,
        caption: str,
        text: str,
        answer: str,
        image_tensor: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        parsed = self.parse_reasoning_chain(generated_chain)

        has_format = self._has_correct_format(generated_chain)
        has_extra = self._has_extra_output(generated_chain)

        if has_format and not has_extra:
            format_score = 0.15
        elif has_format and has_extra:
            format_score = 0.075
        else:
            format_score = 0.0

        actual_steps = self._count_steps(generated_chain)
        step_count_score = 0.15 if actual_steps == hop_count else 0.0

        modality_score = 0.1 if (parsed['has_image'] and parsed['has_text']) else 0.0

        attention_reward = self.compute_attention_keyword_reward(
            parsed, caption, text, answer,
            image_tensor=image_tensor
        )
        attention_score = 0.3 * attention_reward

        if ground_truth is not None:
            chain_sim = self._compute_reasoning_chain_similarity(
                generated_chain, ground_truth
            )
            chain_similarity_score = 0.3 * chain_sim
        else:
            chain_similarity_score = 0.0

        return {
            'format_score': format_score,
            'step_count_score': step_count_score,
            'modality_score': modality_score,
            'attention_score': attention_score,
            'chain_similarity_score': chain_similarity_score
        }
