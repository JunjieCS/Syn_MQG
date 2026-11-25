
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

SFT_ADAPTER = "sft"
GRPO_ADAPTER = "grpo"


class MLLMBackbone(nn.Module, ABC):

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        ...

    @abstractmethod
    def encode_images(self, pixel_values: torch.Tensor, **kwargs):
        ...

    @abstractmethod
    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        ...

    @abstractmethod
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        do_sample: bool = False,
        **kwargs,
    ) -> List[str]:
        ...

    @abstractmethod
    def get_tokenizer(self):
        ...

    @abstractmethod
    def get_processor(self):
        ...

    @abstractmethod
    def get_llm_named_parameters(self):
        ...

    @abstractmethod
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_layers: int,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def forward_logits(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def backbone_generate_with_log_probs(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> dict:
        ...


class QwenVLBackbone(MLLMBackbone):

    def __init__(
        self,
        model_path: str,
        use_flash_attention: bool = False,
        use_gradient_checkpointing: bool = False,
        use_lora: bool = False,
        lora_config: Optional[dict] = None,
        torch_dtype=torch.bfloat16,
        freeze_visual_encoder: bool = True,
    ):
        super().__init__()
        from transformers import AutoProcessor, AutoConfig

        logger.info(f"text Qwen-VL backbone: {model_path}")
        attn_impl = "flash_attention_2" if use_flash_attention else "eager"

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = getattr(config, 'model_type', 'qwen2_vl')

        if model_type == "qwen3_vl":
            from transformers import Qwen3VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )
        elif model_type == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )
        else:
            from transformers import Qwen2VLForConditionalGeneration
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        if hasattr(self.model.config, 'hidden_size'):
            self._hidden_dim = self.model.config.hidden_size
        elif hasattr(self.model.config, 'text_config') and hasattr(self.model.config.text_config, 'hidden_size'):
            self._hidden_dim = self.model.config.text_config.hidden_size
        else:
            self._hidden_dim = self.model.model.embed_tokens.embedding_dim

        if hasattr(self.model.config, 'vision_config'):
            self._raw_vit_dim = getattr(
                self.model.config.vision_config, 'hidden_size', self._hidden_dim
            )
        else:
            self._raw_vit_dim = self._hidden_dim

        if freeze_visual_encoder:
            for param in self.model.visual.parameters():
                param.requires_grad = False
            logger.info("Qwen-VL visual encoder text")

        if hasattr(self.model.config, 'vision_config'):
            self.spatial_merge_size = getattr(
                self.model.config.vision_config, 'spatial_merge_size', 2
            )
        else:
            self.spatial_merge_size = 2

        if use_gradient_checkpointing:
            self.model.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

        self._use_lora = False
        self._active_adapters = []
        if use_lora and lora_config:
            self._apply_lora(lora_config)

        logger.info(f"Qwen-VL backbone text: hidden_dim={self._hidden_dim}")

    def _apply_lora(self, lora_config: dict):
        from peft import get_peft_model, LoraConfig
        for param in self.model.model.parameters():
            param.requires_grad = False
        config = LoraConfig(
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            target_modules=lora_config.get(
                'target_modules',
                ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]
            ),
            bias=lora_config.get('bias', 'none'),
            task_type=None,
        )
        self.model.model = get_peft_model(self.model.model, config)
        self._use_lora = True
        logger.info("Qwen-VL LoRA textself.model.model text")


    def _get_peft_base_model(self):
        peft_llm = self.model.model
        if PeftModel and isinstance(peft_llm, PeftModel):
            return peft_llm.base_model
        return None

    def set_active_adapters(self, adapters: list):
        self._active_adapters = list(adapters)
        base_model = self._get_peft_base_model()
        if base_model is not None:
            base_model.set_adapter(adapters)
            self._enforce_grad_policy()

    def _enforce_grad_policy(self):
        peft_llm = self.model.model
        if not (PeftModel and isinstance(peft_llm, PeftModel)):
            return
        for n, p in peft_llm.named_parameters():
            if 'lora_' not in n:
                continue
            if f'.{SFT_ADAPTER}.' in n:
                p.requires_grad_(False)
            elif f'.{GRPO_ADAPTER}.' in n:
                p.requires_grad_(True)

    @contextmanager
    def reference_mode(self, use_base_as_ref=False):
        peft_llm = self.model.model
        if not (PeftModel and isinstance(peft_llm, PeftModel)):
            yield
            return
        base_model = self._get_peft_base_model()
        try:
            if use_base_as_ref:
                with peft_llm.disable_adapter():
                    yield
            else:
                base_model.set_adapter([SFT_ADAPTER])
                for n, p in peft_llm.named_parameters():
                    if 'lora_' in n:
                        p.requires_grad_(False)
                yield
        finally:
            base_model.set_adapter(self._active_adapters)
            self._enforce_grad_policy()

    @contextmanager
    def _no_lora_ctx(self):
        if self._use_lora:
            peft_llm = self.model.model
            if PeftModel and isinstance(peft_llm, PeftModel):
                with peft_llm.disable_adapter():
                    yield
                self._enforce_grad_policy()
            else:
                yield
        else:
            yield

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def raw_vit_dim(self) -> int:
        return self._raw_vit_dim

    def encode_images_raw_vit(self, pixel_values: torch.Tensor, **kwargs):
        image_grid_thw = kwargs.get('image_grid_thw', None)
        num_images = kwargs.get('num_images', None)

        captured = {}

        def _pre_hook(module, args, kwargs_in):
            x = args[0] if len(args) > 0 else kwargs_in.get('x', None)
            captured['raw'] = x
            return None

        try:
            handle = self.model.visual.merger.register_forward_pre_hook(
                _pre_hook, with_kwargs=True
            )
        except TypeError:
            def _pre_hook_legacy(module, args):
                captured['raw'] = args[0]
                return None
            handle = self.model.visual.merger.register_forward_pre_hook(_pre_hook_legacy)

        try:
            with self._no_lora_ctx(), torch.no_grad():
                _ = self.model.visual(
                    pixel_values.to(self.model.dtype),
                    grid_thw=image_grid_thw,
                )
        finally:
            handle.remove()

        raw = captured.get('raw', None)
        if raw is None:
            raise RuntimeError("encode_images_raw_vit: text merger text")
        # raw: [total_patches, raw_vit_dim]

        if image_grid_thw is None:
            tokens = raw.unsqueeze(0)
            mask = torch.ones(1, raw.size(0), dtype=torch.long, device=raw.device)
            return tokens, mask

        patches_per_image = (
            image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
        ).tolist()
        all_chunks = torch.split(raw, [int(n) for n in patches_per_image])

        if num_images is not None:
            B = len(num_images)
            sample_tokens = []
            idx = 0
            for n in num_images:
                chunks = all_chunks[idx: idx + n]
                if chunks:
                    sample_tokens.append(torch.cat(list(chunks), dim=0))
                else:
                    sample_tokens.append(torch.zeros(
                        1, self._raw_vit_dim, dtype=raw.dtype, device=raw.device
                    ))
                idx += n
        else:
            sample_tokens = list(all_chunks)

        max_len = max(t.size(0) for t in sample_tokens)
        B = len(sample_tokens)
        padded = torch.zeros(B, max_len, self._raw_vit_dim, dtype=raw.dtype, device=raw.device)
        mask = torch.zeros(B, max_len, dtype=torch.long, device=raw.device)
        for i, t in enumerate(sample_tokens):
            padded[i, :t.size(0)] = t
            mask[i, :t.size(0)] = 1
        return padded, mask

    def encode_images(self, pixel_values: torch.Tensor, **kwargs):
        image_grid_thw = kwargs.get('image_grid_thw', None)
        num_images = kwargs.get('num_images', None)  # List[int]

        with torch.no_grad():
            visual_out = self.model.visual(
                pixel_values.to(self.model.dtype),
                grid_thw=image_grid_thw,
            )
        if isinstance(visual_out, tuple):
            visual_out = visual_out[0]
        # visual_out: [total_tokens, H]

        if image_grid_thw is not None:
            m = self.spatial_merge_size
            tokens_per_image = (
                image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
                // (m * m)
            ).tolist()
            all_chunks = torch.split(visual_out, [int(n) for n in tokens_per_image])

            if num_images is not None:
                B = len(num_images)
                sample_tokens = []
                idx = 0
                for n in num_images:
                    chunks = all_chunks[idx: idx + n]
                    sample_tokens.append(torch.cat(list(chunks), dim=0) if chunks else
                                         torch.zeros(1, self.hidden_dim, dtype=visual_out.dtype, device=visual_out.device))
                    idx += n
            else:
                sample_tokens = list(all_chunks)

            max_len = max(t.size(0) for t in sample_tokens)
            B = len(sample_tokens)
            padded = torch.zeros(B, max_len, self.hidden_dim, dtype=visual_out.dtype, device=visual_out.device)
            mask = torch.zeros(B, max_len, dtype=torch.long, device=visual_out.device)
            for i, t in enumerate(sample_tokens):
                padded[i, :t.size(0)] = t
                mask[i, :t.size(0)] = 1
            return padded, mask  # [B, max_tokens, H], [B, max_tokens]

        tokens = visual_out.unsqueeze(0)
        mask = torch.ones(1, visual_out.size(0), dtype=torch.long, device=visual_out.device)
        return tokens, mask

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {'loss': outputs.loss}

    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        do_sample: bool = False,
        **kwargs,
    ) -> List[str]:
        tokenizer = self.processor.tokenizer
        generated_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def generate_with_ids(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        **kwargs,
    ):
        tokenizer = self.processor.tokenizer
        generated_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_ids, texts

    def get_lm_logits(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return outputs.logits

    def get_tokenizer(self):
        return self.processor.tokenizer

    def get_processor(self):
        return self.processor

    def get_llm_named_parameters(self):
        return self.model.model.named_parameters()

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_layers: int,
    ) -> torch.Tensor:
        embeds = self.model.get_input_embeddings()(input_ids)
        peft_llm = self.model.model
        ctx = peft_llm.disable_adapter() if PeftModel and isinstance(peft_llm, PeftModel) else nullcontext()
        with ctx, torch.no_grad():
            out = peft_llm(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        self._enforce_grad_policy()
        return out.hidden_states[n_layers]

    def forward_logits(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return outputs.logits  # [B, L, vocab_size]

    def backbone_generate_with_log_probs(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> dict:
        tokenizer = self.processor.tokenizer
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generated_ids = outputs.sequences  # [B, gen_len]
        scores = torch.stack(outputs.scores, dim=1)  # [B, gen_len, vocab_size]
        if temperature > 0:
            log_probs_all = torch.log_softmax(scores / temperature, dim=-1)
        else:
            log_probs_all = torch.log_softmax(scores, dim=-1)
        token_log_probs = log_probs_all.gather(
            2, generated_ids.unsqueeze(-1)
        ).squeeze(-1)  # [B, gen_len]

        pad_id = tokenizer.pad_token_id or 0
        eos_id = tokenizer.eos_token_id
        attn_mask = ((generated_ids != pad_id) & (generated_ids != eos_id)).long()

        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return {
            'generated_ids':   generated_ids,
            'token_log_probs': token_log_probs,
            'attention_mask':  attn_mask,
            'generated_texts': generated_texts,
        }


class LLaVABackbone(MLLMBackbone):
    """
    LLaVA Backbone（LLaMA + vision_tower + mm_projector）

    visual encoder: vision_tower(pixel_values) → multi_modal_projector → [B, 256, H]
    text embedding: language_model.get_input_embeddings()(input_ids)
    generator:      language_model(inputs_embeds=...)
    """

    def __init__(
        self,
        model_path: str,
        use_flash_attention: bool = False,
        use_gradient_checkpointing: bool = False,
        use_lora: bool = False,
        lora_config: Optional[dict] = None,
        torch_dtype=torch.bfloat16,
        freeze_visual_encoder: bool = True,
    ):
        super().__init__()
        from transformers import LlavaForConditionalGeneration, AutoProcessor

        logger.info(f"text LLaVA backbone: {model_path}")
        attn_impl = "flash_attention_2" if use_flash_attention else None

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self._hidden_dim = self.model.config.text_config.hidden_size

        if freeze_visual_encoder:
            for param in self.model.vision_tower.parameters():
                param.requires_grad = False
            for param in self.model.multi_modal_projector.parameters():
                param.requires_grad = False
            logger.info("LLaVA vision_tower + mm_projector text")

        if use_gradient_checkpointing:
            self.model.language_model.gradient_checkpointing_enable()
            self.model.config.text_config.use_cache = False

        if use_lora and lora_config:
            self._apply_lora(lora_config)

        logger.info(f"LLaVA backbone text: hidden_dim={self._hidden_dim}")

    def _apply_lora(self, lora_config: dict):
        from peft import get_peft_model, LoraConfig
        config = LoraConfig(
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            target_modules=lora_config.get(
                'target_modules',
                ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]
            ),
            bias=lora_config.get('bias', 'none'),
            task_type=None,
        )
        self.model.language_model = get_peft_model(self.model.language_model, config)
        logger.info("LLaVA LoRA text")

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def encode_images(self, pixel_values: torch.Tensor, **kwargs):
        num_images = kwargs.get('num_images', None)

        with torch.no_grad():
            vision_out = self.model.vision_tower(
                pixel_values.to(self.model.dtype),
                output_hidden_states=True,
            )
            vision_feature_layer = self.model.config.vision_feature_layer
            selected = vision_out.hidden_states[vision_feature_layer]
            if self.model.config.vision_feature_select_strategy == "full":
                image_features = selected
            else:
                image_features = selected[:, 1:, :]
            image_features = self.model.multi_modal_projector(image_features)
        # image_features: [total_images, 256, H]

        if num_images is not None:
            B = len(num_images)
            splits = torch.split(image_features, num_images, dim=0)
            sample_tokens = [s.reshape(-1, self.hidden_dim) for s in splits]
            max_len = max(t.size(0) for t in sample_tokens) if sample_tokens else 1
            padded = torch.zeros(B, max_len, self.hidden_dim, dtype=image_features.dtype, device=image_features.device)
            mask = torch.zeros(B, max_len, dtype=torch.long, device=image_features.device)
            for i, t in enumerate(sample_tokens):
                if t.size(0) > 0:
                    padded[i, :t.size(0)] = t
                    mask[i, :t.size(0)] = 1
            return padded, mask

        tokens = image_features.reshape(1, -1, self.hidden_dim)
        mask = torch.ones(1, tokens.size(1), dtype=torch.long, device=image_features.device)
        return tokens, mask

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.language_model.get_input_embeddings()(input_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {'loss': outputs.loss}

    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        do_sample: bool = False,
        **kwargs,
    ) -> List[str]:
        tokenizer = self.processor.tokenizer
        llm_dtype = next(self.model.language_model.parameters()).dtype
        generated_ids = self.model.generate(
            inputs_embeds=inputs_embeds.to(llm_dtype),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def generate_with_ids(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        **kwargs,
    ):
        tokenizer = self.processor.tokenizer
        llm_dtype = next(self.model.language_model.parameters()).dtype
        generated_ids = self.model.generate(
            inputs_embeds=inputs_embeds.to(llm_dtype),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_ids, texts

    def get_lm_logits(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        llm_dtype = next(self.model.language_model.parameters()).dtype
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds.to(llm_dtype),
            attention_mask=attention_mask,
        )
        return self.model.lm_head(outputs.last_hidden_state)

    def get_tokenizer(self):
        return self.processor.tokenizer

    def get_processor(self):
        return self.processor

    def get_llm_named_parameters(self):
        return self.model.language_model.named_parameters()

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_layers: int,
    ) -> torch.Tensor:
        llm = self.model.language_model
        embeds = llm.get_input_embeddings()(input_ids)
        llm_dtype = next(llm.parameters()).dtype
        ctx = llm.disable_adapter() if PeftModel and isinstance(llm, PeftModel) else nullcontext()
        with ctx, torch.no_grad():
            out = llm(
                inputs_embeds=embeds.to(llm_dtype),
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        return out.hidden_states[n_layers]  # [B, L, H]

    def forward_logits(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        llm_dtype = next(self.model.language_model.parameters()).dtype
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds.to(llm_dtype),
            attention_mask=attention_mask,
        )
        return self.model.lm_head(outputs.last_hidden_state)

    def backbone_generate_with_log_probs(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> dict:
        tokenizer = self.processor.tokenizer
        llm_dtype = next(self.model.language_model.parameters()).dtype
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds.to(llm_dtype),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generated_ids = outputs.sequences  # [B, gen_len]
        scores = torch.stack(outputs.scores, dim=1)  # [B, gen_len, vocab_size]
        if temperature > 0:
            log_probs_all = torch.log_softmax(scores / temperature, dim=-1)
        else:
            log_probs_all = torch.log_softmax(scores, dim=-1)
        token_log_probs = log_probs_all.gather(
            2, generated_ids.unsqueeze(-1)
        ).squeeze(-1)  # [B, gen_len]

        pad_id = tokenizer.pad_token_id or 0
        eos_id = tokenizer.eos_token_id
        attn_mask = ((generated_ids != pad_id) & (generated_ids != eos_id)).long()
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return {
            'generated_ids':   generated_ids,
            'token_log_probs': token_log_probs,
            'attention_mask':  attn_mask,
            'generated_texts': generated_texts,
        }


class LLaVANextBackbone(MLLMBackbone):

    def __init__(
        self,
        model_path: str,
        use_flash_attention: bool = False,
        use_gradient_checkpointing: bool = False,
        use_lora: bool = False,
        lora_config: Optional[dict] = None,
        torch_dtype=torch.bfloat16,
        freeze_visual_encoder: bool = True,
    ):
        super().__init__()
        from transformers import LlavaNextForConditionalGeneration, AutoProcessor

        logger.info(f"text LLaVA-NeXT backbone: {model_path}")
        attn_impl = "flash_attention_2" if use_flash_attention else None

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self._hidden_dim = self.model.config.text_config.hidden_size

        if freeze_visual_encoder:
            for param in self.model.vision_tower.parameters():
                param.requires_grad = False
            for param in self.model.multi_modal_projector.parameters():
                param.requires_grad = False
            if hasattr(self.model, 'image_newline'):
                self.model.image_newline.requires_grad = False
            elif hasattr(self.model.model, 'image_newline'):
                self.model.model.image_newline.requires_grad = False
            logger.info("LLaVA-NeXT vision_tower + mm_projector text")

        if use_gradient_checkpointing:
            self.model.language_model.gradient_checkpointing_enable()
            self.model.config.text_config.use_cache = False

        if use_lora and lora_config:
            self._apply_lora(lora_config)

        logger.info(f"LLaVA-NeXT backbone text: hidden_dim={self._hidden_dim}")

    def _apply_lora(self, lora_config: dict):
        from peft import get_peft_model, LoraConfig
        config = LoraConfig(
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            target_modules=lora_config.get(
                'target_modules',
                ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]
            ),
            bias=lora_config.get('bias', 'none'),
            task_type=None,
        )
        self.model.language_model = get_peft_model(self.model.language_model, config)
        logger.info("LLaVA-NeXT LoRA text")

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def encode_images(self, pixel_values: torch.Tensor, **kwargs):
        image_sizes = kwargs.get('image_sizes')
        num_images = kwargs.get('num_images')

        with torch.no_grad():
            all_image_tokens = self.model.get_image_features(
                pixel_values.to(self.model.dtype),
                image_sizes=image_sizes,
            )

        if num_images is not None:
            B = len(num_images)
            sample_tokens = []
            idx = 0
            for n in num_images:
                chunks = all_image_tokens[idx: idx + n]
                if chunks:
                    sample_tokens.append(torch.cat(chunks, dim=0))
                else:
                    sample_tokens.append(torch.zeros(
                        1, self._hidden_dim, dtype=pixel_values.dtype, device=pixel_values.device
                    ))
                idx += n
        else:
            sample_tokens = all_image_tokens

        max_len = max(t.size(0) for t in sample_tokens)
        B = len(sample_tokens)
        padded = torch.zeros(B, max_len, self._hidden_dim, dtype=pixel_values.dtype, device=pixel_values.device)
        mask = torch.zeros(B, max_len, dtype=torch.long, device=pixel_values.device)
        for i, t in enumerate(sample_tokens):
            padded[i, :t.size(0)] = t
            mask[i, :t.size(0)] = 1
        return padded, mask

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.language_model.get_input_embeddings()(input_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {'loss': outputs.loss}

    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        do_sample: bool = False,
        **kwargs,
    ) -> List[str]:
        tokenizer = self.processor.tokenizer
        llm_dtype = next(self.model.language_model.parameters()).dtype
        generated_ids = self.model.generate(
            inputs_embeds=inputs_embeds.to(llm_dtype),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def generate_with_ids(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        **kwargs,
    ):
        tokenizer = self.processor.tokenizer
        llm_dtype = next(self.model.language_model.parameters()).dtype
        generated_ids = self.model.generate(
            inputs_embeds=inputs_embeds.to(llm_dtype),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_ids, texts

    def get_lm_logits(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        llm_dtype = next(self.model.language_model.parameters()).dtype
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds.to(llm_dtype),
            attention_mask=attention_mask,
        )
        return self.model.lm_head(outputs.last_hidden_state)

    def get_tokenizer(self):
        return self.processor.tokenizer

    def get_processor(self):
        return self.processor

    def get_llm_named_parameters(self):
        return self.model.language_model.named_parameters()

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_layers: int,
    ) -> torch.Tensor:
        llm = self.model.language_model
        embeds = llm.get_input_embeddings()(input_ids)
        llm_dtype = next(llm.parameters()).dtype
        ctx = llm.disable_adapter() if PeftModel and isinstance(llm, PeftModel) else nullcontext()
        with ctx, torch.no_grad():
            out = llm(
                inputs_embeds=embeds.to(llm_dtype),
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        return out.hidden_states[n_layers]

    def forward_logits(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        llm_dtype = next(self.model.language_model.parameters()).dtype
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds.to(llm_dtype),
            attention_mask=attention_mask,
        )
        return self.model.lm_head(outputs.last_hidden_state)

    def backbone_generate_with_log_probs(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> dict:
        tokenizer = self.processor.tokenizer
        llm_dtype = next(self.model.language_model.parameters()).dtype
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds.to(llm_dtype),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generated_ids = outputs.sequences
        scores = torch.stack(outputs.scores, dim=1)
        if temperature > 0:
            log_probs_all = torch.log_softmax(scores / temperature, dim=-1)
        else:
            log_probs_all = torch.log_softmax(scores, dim=-1)
        token_log_probs = log_probs_all.gather(
            2, generated_ids.unsqueeze(-1)
        ).squeeze(-1)

        pad_id = tokenizer.pad_token_id or 0
        eos_id = tokenizer.eos_token_id
        attn_mask = ((generated_ids != pad_id) & (generated_ids != eos_id)).long()
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return {
            'generated_ids':   generated_ids,
            'token_log_probs': token_log_probs,
            'attention_mask':  attn_mask,
            'generated_texts': generated_texts,
        }


class InternVLBackbone(MLLMBackbone):

    def __init__(
        self,
        model_path: str,
        use_flash_attention: bool = False,
        use_gradient_checkpointing: bool = False,
        use_lora: bool = False,
        lora_config: Optional[dict] = None,
        torch_dtype=torch.bfloat16,
        freeze_visual_encoder: bool = True,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"text InternVL backbone: {model_path}")

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if use_flash_attention else "eager",
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False,
        )

        config = self.model.config
        self._hidden_dim = config.llm_config.hidden_size

        if freeze_visual_encoder:
            self.model.vision_model.requires_grad_(False)
            self.model.mlp1.requires_grad_(False)
            logger.info("InternVL vision_model + mlp1 text")

        # gradient checkpointing
        if use_gradient_checkpointing:
            self.model.language_model.gradient_checkpointing_enable()
            if hasattr(config, 'llm_config'):
                config.llm_config.use_cache = False

        # LoRA
        if use_lora and lora_config:
            self._apply_lora(lora_config)

        logger.info(f"InternVL backbone text: hidden_dim={self._hidden_dim}")

    def _apply_lora(self, lora_config: dict):
        from peft import get_peft_model, LoraConfig
        config = LoraConfig(
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            target_modules=lora_config.get(
                'target_modules',
                ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"],
            ),
            bias=lora_config.get('bias', 'none'),
            task_type=None,
        )
        self.model.language_model = get_peft_model(
            self.model.language_model, config,
        )
        logger.info("InternVL LoRA text")

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def encode_images(self, pixel_values: torch.Tensor, **kwargs):
        num_patches_list = kwargs.get('num_patches_list', None)
        num_images = kwargs.get('num_images', None)

        with torch.no_grad():
            vit_embeds = self.model.extract_feature(
                pixel_values.to(self.model.vision_model.dtype),
            )
        # vit_embeds: [total_tiles, tokens_per_tile, H]

        if num_patches_list is None:
            tokens = vit_embeds.reshape(1, -1, self._hidden_dim)
            mask = torch.ones(1, tokens.size(1), dtype=torch.long,
                              device=tokens.device)
            return tokens, mask

        per_image = torch.split(vit_embeds, num_patches_list, dim=0)
        all_image_tokens = [t.reshape(-1, self._hidden_dim) for t in per_image]

        if num_images is not None:
            sample_tokens = []
            idx = 0
            for n in num_images:
                if n > 0:
                    sample_tokens.append(
                        torch.cat(all_image_tokens[idx:idx + n], dim=0)
                    )
                else:
                    sample_tokens.append(
                        torch.zeros(1, self._hidden_dim,
                                    dtype=vit_embeds.dtype, device=vit_embeds.device)
                    )
                idx += n
        else:
            sample_tokens = all_image_tokens

        # pad
        B = len(sample_tokens)
        max_len = max(t.size(0) for t in sample_tokens)
        padded = torch.zeros(B, max_len, self._hidden_dim,
                             dtype=vit_embeds.dtype, device=vit_embeds.device)
        mask = torch.zeros(B, max_len, dtype=torch.long, device=vit_embeds.device)
        for i, t in enumerate(sample_tokens):
            padded[i, :t.size(0)] = t
            mask[i, :t.size(0)] = 1
        return padded, mask

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.language_model.get_input_embeddings()(input_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        llm_dtype = next(self.model.language_model.parameters()).dtype
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds.to(llm_dtype),
            attention_mask=attention_mask,
            labels=labels,
        )
        return {'loss': outputs.loss}

    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        do_sample: bool = False,
        **kwargs,
    ) -> List[str]:
        llm_dtype = next(self.model.language_model.parameters()).dtype
        generated_ids = self.model.language_model.generate(
            inputs_embeds=inputs_embeds.to(llm_dtype),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def generate_with_ids(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        **kwargs,
    ):
        llm_dtype = next(self.model.language_model.parameters()).dtype
        generated_ids = self.model.language_model.generate(
            inputs_embeds=inputs_embeds.to(llm_dtype),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_ids, texts

    def get_lm_logits(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        llm_dtype = next(self.model.language_model.parameters()).dtype
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds.to(llm_dtype),
            attention_mask=attention_mask,
        )
        return outputs.logits

    def get_tokenizer(self):
        return self.tokenizer

    def get_processor(self):
        return self.tokenizer

    def get_llm_named_parameters(self):
        return self.model.language_model.named_parameters()

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_layers: int,
    ) -> torch.Tensor:
        llm = self.model.language_model
        embeds = llm.get_input_embeddings()(input_ids)
        llm_dtype = next(llm.parameters()).dtype
        ctx = llm.disable_adapter() if PeftModel and isinstance(llm, PeftModel) else nullcontext()
        with ctx, torch.no_grad():
            out = llm(
                inputs_embeds=embeds.to(llm_dtype),
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        return out.hidden_states[n_layers]

    def forward_logits(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        llm_dtype = next(self.model.language_model.parameters()).dtype
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds.to(llm_dtype),
            attention_mask=attention_mask,
        )
        return outputs.logits

    def backbone_generate_with_log_probs(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> dict:
        llm_dtype = next(self.model.language_model.parameters()).dtype
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds.to(llm_dtype),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generated_ids = outputs.sequences
        scores = torch.stack(outputs.scores, dim=1)
        if temperature > 0:
            log_probs_all = torch.log_softmax(scores / temperature, dim=-1)
        else:
            log_probs_all = torch.log_softmax(scores, dim=-1)
        token_log_probs = log_probs_all.gather(
            2, generated_ids.unsqueeze(-1),
        ).squeeze(-1)

        pad_id = self.tokenizer.pad_token_id or 0
        eos_id = self.tokenizer.eos_token_id
        attn_mask = ((generated_ids != pad_id) & (generated_ids != eos_id)).long()
        generated_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True,
        )
        return {
            'generated_ids':   generated_ids,
            'token_log_probs': token_log_probs,
            'attention_mask':  attn_mask,
            'generated_texts': generated_texts,
        }


def build_mllm_backbone(backbone_type: str, **kwargs) -> MLLMBackbone:
    backbones = {
        "qwen2vl":    QwenVLBackbone,
        "llava":      LLaVABackbone,
        "llava_next": LLaVANextBackbone,
        "internvl":   InternVLBackbone,
    }
    if backbone_type not in backbones:
        raise ValueError(f"text backbone: {backbone_type}text: {list(backbones.keys())}")
    return backbones[backbone_type](**kwargs)
