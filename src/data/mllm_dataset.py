"""
MLLM Disentangle Dataset

Uses MLLM processor instead of CLIP preprocessing, supports Qwen2-VL and LLaVA backbones.

Each sample outputs:
    pixel_values:            Image tensor (backbone-specific format)
    image_grid_thw:          [3], Qwen2-VL dynamic resolution info (not present for LLaVA)
    visual_mask:             [1], single image, fixed to 1 in current version
    text_input_ids:          [L_text]   Text context (disentangle module text side)
    text_attention_mask:     [L_text]
    system_input_ids:        [L_sys]    System prompt
    system_attention_mask:   [L_sys]
    constraint_input_ids:    [L_c]      answer/type (generator prefix, excluded from loss)
    constraint_attention_mask:[L_c]
    target_input_ids:        [L_t]      reasoning_chain + question (generation target)
    target_attention_mask:   [L_t]
    answer:                  str
    question_type:           str
    id:                      str
"""

import threading
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from typing import Optional, Dict, List
import logging
import os

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Given a known answer, generate a multimodal reasoning chain that leads to "
    "this answer, followed by a question that matches the question type.\n"
    "OUTPUT: Generate ONLY the reasoning chain and question. "
    "Do NOT add any prefix or suffix text.\n"
    "Use this exact format:\n"
    "[Reasoning Chain]\n"
    "Step 1: [Image/Text] ...\n"
    "Step 2: [Image/Text] ...\n\n"
    "[Question]\n"
    "Your question here"
)


class MLLMDataset(Dataset):
    """
    MLLM Disentangle training/inference dataset

    backbone_type: "qwen2vl" | "llava"
    processor: from backbone.get_processor()
    tokenizer: from backbone.get_tokenizer()
    """

    def __init__(
        self,
        data_file: str,
        image_root: str,
        processor,                              # MLLM processor
        tokenizer,                              # MLLM tokenizer
        backbone_type: str = "qwen2vl",         # "qwen2vl" | "llava"
        max_constraint_length: int = 256,
        max_target_length: int = 512,
        max_samples: Optional[int] = None,
        filter_non_multimodal: bool = True,
        inference_mode: bool = False,
        preprocess_workers: int = 4,
    ):
        super().__init__()

        self.processor = processor
        self.tokenizer = tokenizer
        self.backbone_type = backbone_type
        self.image_root = image_root
        self.inference_mode = inference_mode
        self.max_constraint_length = max_constraint_length
        self.max_target_length = max_target_length
        self._tok_lock = threading.RLock()

        with open(data_file, 'r', encoding='utf-8') as f:
            if data_file.endswith('.jsonl'):
                self.data = [json.loads(line) for line in f]
            else:
                self.data = json.load(f)

        if filter_non_multimodal:
            original_len = len(self.data)
            self.data = [s for s in self.data if s.get('is_multimodal') == 'yes']
            logger.info(f"Filtered non-multimodal samples: {original_len} -> {len(self.data)}")

        if max_samples:
            self.data = self.data[:max_samples]

        mode_str = "inference" if inference_mode else "training"
        logger.info(
            f"MLLMDataset loaded: {len(self.data)} samples ({mode_str} mode, {backbone_type})"
        )

        if preprocess_workers > 0:
            self.cache = self._preprocess_all(preprocess_workers)
        else:
            self.cache = None

    def __len__(self) -> int:
        return len(self.data)

    def _load_images(self, sample: Dict) -> List[Image.Image]:
        """Load all images, return PIL list (resized to 224x224)"""
        images = []
        if 'image_paths' in sample and sample['image_paths']:
            paths = sample['image_paths']
        elif 'image_path' in sample and sample['image_path']:
            paths = [sample['image_path'].replace('\\', '/')]
        else:
            return images
        for p in paths:
            full = os.path.join(self.image_root, p)
            try:
                images.append(Image.open(full).convert('RGB').resize((224, 224), Image.BICUBIC))
            except Exception as e:
                logger.warning(f"Failed to load image {full}: {e}")
        return images

    def _process_images(self, images: List[Image.Image]) -> Dict:
        """
        Process multiple images with processor.

        Qwen2-VL:
            pixel_values:   [total_patches, C, patch_h, patch_w] (all images flattened)
            image_grid_thw: [N_images, 3] (t/h/w per image)
            num_images:     int
        LLaVA:
            pixel_values:   [N_images, C, 224, 224]
            num_images:     int
        """
        if not images:
            if self.backbone_type == "qwen2vl":
                return {
                    'pixel_values':   torch.zeros(1, 3, 14, 14),
                    'image_grid_thw': torch.tensor([[1, 1, 1]], dtype=torch.long),
                    'num_images':     1,
                }
            else:
                return {
                    'pixel_values': torch.zeros(1, 3, 224, 224),
                    'num_images':   1,
                }

        if self.backbone_type == "qwen2vl":
            # Build multi-image messages, copy to avoid multi-thread PIL race
            images = [img.copy() for img in images]
            content = [{"type": "image", "image": img} for img in images]
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            # processor calls tokenizer internally, fast tokenizer is not thread-safe, need lock
            with self._tok_lock:
                inputs = self.processor(
                    text=[text], images=images, return_tensors='pt', padding=False
                )
            return {
                'pixel_values':   inputs['pixel_values'],
                'image_grid_thw': inputs['image_grid_thw'],
                'num_images':     len(images),
            }
        else:
            # LLaVA: process each image then stack, copy to avoid multi-thread PIL race
            pv_list = []
            for img in images:
                inp = self.processor(images=img.copy(), return_tensors='pt')
                pv_list.append(inp['pixel_values'].squeeze(0))
            return {
                'pixel_values': torch.stack(pv_list),
                'num_images':   len(images),
            }

    def _tokenize(self, text: str, max_length: int) -> Dict[str, torch.Tensor]:
        with self._tok_lock:
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors='pt',
            )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }

    def _tokenize_no_truncate(self, text: str) -> Dict[str, torch.Tensor]:
        with self._tok_lock:
            enc = self.tokenizer(
                text, truncation=False, padding=False, return_tensors='pt'
            )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }

    def _build_text_context(self, sample: Dict) -> str:
        return f"Context: {sample.get('text', '')}"

    def _build_constraint(self, sample: Dict) -> str:
        answer       = sample.get('answer', '')
        question_type = sample.get('question_type', 'what')
        hop_count    = sample.get('chain_hops', 0)
        return (
            f"Answer: {answer}\n"
            f"Question Type: {question_type}\n"
            f"Steps: {hop_count}\n\n"
            f"Generate exactly {hop_count} steps followed by a question in this format:\n"
            "[Reasoning Chain]\n"
            "Step 1: [Image/Text] ...\n"
            "...\n"
            f"Step {hop_count}: [Image/Text] ...\n\n"
            "[Question]\n"
            "Your question here"
        )

    def _build_target(self, sample: Dict) -> str:
        if self.inference_mode:
            return ''
        chain = sample.get('reasoning_chain', '')
        if isinstance(chain, (list, tuple)):
            chain = '\n'.join(str(step) for step in chain)
        elif not isinstance(chain, str):
            chain = str(chain)
        eos = self.tokenizer.eos_token or ''
        return chain + eos

    def _preprocess_single(self, idx: int) -> Dict:
        sample = self.data[idx]
        images = self._load_images(sample)
        img_data = self._process_images(images)

        text_enc = self._tokenize_no_truncate(self._build_text_context(sample))
        system_enc = self._tokenize(
            getattr(self, 'system_prompt', SYSTEM_PROMPT),
            self.max_constraint_length,
        )
        constraint_enc = self._tokenize(
            self._build_constraint(sample), self.max_constraint_length
        )

        result = {
            **img_data,
            'text_input_ids':           text_enc['input_ids'],
            'text_attention_mask':      text_enc['attention_mask'],
            'system_input_ids':         system_enc['input_ids'],
            'system_attention_mask':    system_enc['attention_mask'],
            'constraint_input_ids':     constraint_enc['input_ids'],
            'constraint_attention_mask': constraint_enc['attention_mask'],
            'answer':                   sample.get('answer', ''),
            'question_type':            sample.get('question_type', 'what'),
            'id':                       sample.get('id', str(idx)),
            'reasoning_chain':          sample.get('reasoning_chain', ''),
            'hop_count':                sample.get('chain_hops', 0),
            'caption':                  sample.get('caption', [''])[0] if isinstance(sample.get('caption'), list) else sample.get('caption', ''),
            'text':                     sample.get('text', ''),
        }

        if not self.inference_mode:
            target_enc = self._tokenize(self._build_target(sample), self.max_target_length)
            result['target_input_ids']      = target_enc['input_ids']
            result['target_attention_mask'] = target_enc['attention_mask']

        return result

    def _preprocess_all(self, num_workers: int) -> List[Dict]:
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        cache = [None] * len(self.data)
        logger.info(f"Preprocessing {len(self.data)} samples, threads={num_workers}...")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._preprocess_single, i): i
                       for i in range(len(self.data))}
            for future in tqdm(futures, desc="Preprocessing"):
                i = futures[future]
                try:
                    cache[i] = future.result()
                except Exception as e:
                    logger.error(f"Sample {i} preprocessing failed: {e}")
                    cache[i] = self._preprocess_single(i)

        logger.info("Preprocessing complete")
        return cache

    def __getitem__(self, idx: int) -> Dict:
        if self.cache is not None:
            return self.cache[idx]
        return self._preprocess_single(idx)


def mllm_collate_fn(
    batch: List[Dict],
    backbone_type: str = "qwen2vl",
    pad_token_id: int = 0,
) -> Dict:
    """
    collate_fn: merge sample list into a batch.

    Images: pixel_values and image_grid_thw are cat along dim 0 (flattened across samples).
    Text sequences: padded to same length.
    target_input_ids: padded with pad_token_id (not -100); embedding lookup needs valid ids.
                      Pad positions in labels are set to -100 by train_forward.
    num_images: records image count per sample, used by backbone.encode_images for splitting.
    """
    from torch.nn.utils.rnn import pad_sequence

    def pad_seq(tensors, padding_value=0):
        return pad_sequence(tensors, batch_first=True, padding_value=padding_value)

    # Images: cat flattened (different image counts per sample, can't stack)
    result = {
        'pixel_values': torch.cat([b['pixel_values'] for b in batch], dim=0),
        'num_images':   [b['num_images'] for b in batch],
        'text_input_ids':            pad_seq([b['text_input_ids'] for b in batch]),
        'text_attention_mask':       pad_seq([b['text_attention_mask'] for b in batch]),
        'system_input_ids':          pad_seq([b['system_input_ids'] for b in batch]),
        'system_attention_mask':     pad_seq([b['system_attention_mask'] for b in batch]),
        'constraint_input_ids':      pad_seq([b['constraint_input_ids'] for b in batch]),
        'constraint_attention_mask': pad_seq([b['constraint_attention_mask'] for b in batch]),
        'id': [str(b['id']) for b in batch],
        'answer': [b.get('answer', '') for b in batch],
        'question_type': [b.get('question_type', '') for b in batch],
        'reasoning_chain': [b.get('reasoning_chain', '') for b in batch],
        'hop_count': [b.get('hop_count', 0) for b in batch],
        'caption': [b.get('caption', '') for b in batch],
        'text': [b.get('text', '') for b in batch],
        'question': [b.get('question', '') for b in batch],
    }

    if backbone_type == "qwen2vl" and 'image_grid_thw' in batch[0]:
        # image_grid_thw: [N_images, 3], cat flattened across samples
        result['image_grid_thw'] = torch.cat(
            [b['image_grid_thw'] for b in batch], dim=0
        )

    if 'target_input_ids' in batch[0]:
        result['target_input_ids']      = pad_seq([b['target_input_ids'] for b in batch], pad_token_id)
        result['target_attention_mask'] = pad_seq([b['target_attention_mask'] for b in batch])

    return result
