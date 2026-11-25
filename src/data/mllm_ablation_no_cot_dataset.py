"""Dataset variant for the MLLM no-CoT ablation.

The image/text preprocessing and batch schema intentionally stay compatible
with :class:`MLLMDataset`; only the generation prompt and target are changed.
"""

from .mllm_dataset import MLLMDataset


NO_COT_SYSTEM_PROMPT = (
    "Given a known answer and question type, generate one multimodal question "
    "whose answer is the known answer.\n"
    "OUTPUT: Generate ONLY the question text, without explanations, reasoning, "
    "labels, or formatting markers."
)


class MLLMNoCotDataset(MLLMDataset):
    """MLLM dataset whose supervised target is the question only."""

    system_prompt = NO_COT_SYSTEM_PROMPT

    def _build_constraint(self, sample):
        answer = sample.get("answer", "")
        question_type = sample.get("question_type", "what")
        return (
            f"Answer: {answer}\n"
            f"Question Type: {question_type}\n\n"
            "Generate a single question matching the requested type. "
            "Do not output a reasoning chain or any extra text."
        )

    def _preprocess_single(self, idx):
        result = super()._preprocess_single(idx)
        result['question'] = self.data[idx].get('question', '')
        return result

    def _build_target(self, sample):
        if self.inference_mode:
            return ""
        eos = self.tokenizer.eos_token or ""
        return sample.get("question", "") + eos
