"""OCR Engine — Optical Character Recognition for image-to-text extraction.

Supports GLM-OCR and compatible models via mlx-vlm.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TASK_PROMPTS = {
    "text": "Text Recognition:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
}


class OCREngine:
    """OCR engine using GLM-OCR models via mlx-vlm.

    Uses chat template formatting with image tokens for proper vision encoding,
    then runs generation with KV cache for token-by-token decoding.
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._running = False
        self._executor = None

    @property
    def model_name(self) -> str:
        return self._model_path.rsplit("/", 1)[-1] if self._model_path else ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the OCR model via mlx-vlm."""
        try:
            from mlx_vlm import load as vlm_load
            self._model, self._processor = vlm_load(self._model_path)
            self._tokenizer = (
                self._processor.tokenizer
                if hasattr(self._processor, "tokenizer")
                else self._processor
            )
        except Exception as e:
            logger.warning(f"OCR model load failed: {e}")
            self._model = None
            self._processor = None

    async def start(self) -> None:
        """Start the OCR engine."""
        from .mlx_executor import get_mlx_executor

        self._executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.load)
        self._running = True

    async def stop(self) -> None:
        """Stop and release model."""
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._running = False

    def _format_prompt(self, task: str) -> str:
        """Build the chat-formatted prompt with image placeholder."""
        task_prompt = _TASK_PROMPTS.get(task, _TASK_PROMPTS["text"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": task_prompt},
                ],
            }
        ]
        return self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _extract_sync(self, image_path: str, task: str) -> dict[str, Any]:
        """Synchronous OCR extraction — runs on the MLX executor thread."""
        from mlx_vlm.utils import prepare_inputs
        from mlx_lm.models.cache import make_prompt_cache
        import mlx.core as mx

        prompt = self._format_prompt(task)
        image_token_index = getattr(self._model.config, "image_token_index", None)

        inputs = prepare_inputs(
            self._processor,
            images=image_path,
            prompts=prompt,
            image_token_index=image_token_index,
        )

        # Reset RoPE state for new request
        lm = self._model.language_model
        if hasattr(lm, "_rope_deltas"):
            lm._rope_deltas = None
        if hasattr(lm, "_position_ids"):
            lm._position_ids = None

        cache = make_prompt_cache(lm)
        current_ids = inputs["input_ids"]

        tokens = []
        for step in range(4096):
            kwargs = {"cache": cache}
            if step == 0:
                kwargs["pixel_values"] = inputs["pixel_values"]
                for k in ("image_grid_thw", "attention_mask"):
                    if k in inputs:
                        kwargs[k] = inputs[k]

            out = lm(inputs=current_ids, **kwargs)
            logits = out.logits if hasattr(out, "logits") else out

            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            tok_id = next_token.item()

            if tok_id == self._tokenizer.eos_token_id:
                break

            tokens.append(tok_id)
            current_ids = mx.array([[tok_id]])

        text = self._tokenizer.decode(tokens)
        return {"text": text, "confidence": 1.0, "language": None}

    async def extract_text(
        self,
        image_path: str,
        language: str | None = None,
        task: str = "text",
    ) -> dict[str, Any]:
        """Extract text from an image file.

        Args:
            image_path: Path to the image file
            language: Optional language hint
            task: OCR task type — "text", "formula", or "table"

        Returns:
            {"text": "...", "confidence": float, "language": str}
        """
        if not self.is_loaded:
            raise RuntimeError("OCR engine not started")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor, self._extract_sync, image_path, task
        )
        if language:
            result["language"] = language
        return result

    def get_stats(self) -> dict:
        return {
            "model": self._model_path,
            "loaded": self.is_loaded,
            "running": self._running,
        }
