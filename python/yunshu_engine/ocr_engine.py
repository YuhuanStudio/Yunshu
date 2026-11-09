"""OCR Engine — Optical Character Recognition for image-to-text extraction.

Supports GLM-OCR and compatible models via mlx-vlm.
"""

import asyncio
import contextlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

_TASK_PROMPTS = {
    "text": "Text Recognition:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
    # Case-insensitive accepted forms (per GLM-OCR model card)
    "text recognition": "Text Recognition:",
    "formula recognition": "Formula Recognition:",
    "table recognition": "Table Recognition:",
    "ocr": "Text Recognition:",
    "plain text": "Text Recognition:",
}


def _resolve_task_prompt(task: str | None) -> str:
    if not task:
        return _TASK_PROMPTS["text"]
    return _TASK_PROMPTS.get(task.lower().strip(), _TASK_PROMPTS["text"])


from .active_tracking import ActiveRequestMixin, tracks_active


class OCREngine(ActiveRequestMixin):
    """OCR engine using GLM-OCR models via mlx-vlm.

    Owns a ``VLMEngine`` for the same checkpoint and generates every OCR call
    through ``VLMEngine.generate`` (the round driver). The OCR-specific pieces
    stay here: task-prompt resolution, the language hint, the GLM-OCR processor
    import, and the image-token estimate.
    """

    def __init__(self, model_path: str, shared_vlm: Any = None) -> None:
        """``shared_vlm``: an already-loaded VLMEngine for the same checkpoint (the
        model manager passes one when it serves that checkpoint as a VLM). The OCR
        engine then reuses it and never loads a second copy of the weights."""
        self._model_path = model_path
        self._shared_vlm = shared_vlm
        self._owns_vlm = False
        self._vlm = None
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
        return self._vlm is not None and self._vlm.is_loaded

    def load(self) -> None:
        """Load the OCR model via mlx-vlm."""
        try:
            # Eagerly import GLM-OCR processing so its AutoProcessor patch is
            # installed before vlm_load resolves the processor. Without this,
            # GlmOcrProcessor falls back to a plain TokenizersBackend with no
            # image_processor, and pixel_values silently never reach the model
            # (model then emits an empty "```markdown\n\n```" wrapper).
            with contextlib.suppress(Exception):
                import mlx_vlm.models.glm_ocr.processing  # noqa: F401
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

    @staticmethod
    def _import_glm_ocr_processing() -> None:
        # Eagerly import GLM-OCR processing so its AutoProcessor patch is
        # installed before the VLM processor is resolved (see load()).
        with contextlib.suppress(Exception):
            import mlx_vlm.models.glm_ocr.processing  # noqa: F401

    async def start(self) -> None:
        """Start the OCR engine: load the checkpoint into a VLMEngine."""
        from .mlx_executor import get_mlx_executor
        from .vlm_engine import VLMEngine

        self._executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._import_glm_ocr_processing)
        shared = self._shared_vlm
        if shared is not None and getattr(shared, "is_loaded", False):
            # the checkpoint is already resident in a VLMEngine: reuse it (one copy of
            # the weights in memory). stop() then leaves it running for its owner.
            self._owns_vlm = False
            vlm = shared
        else:
            vlm = VLMEngine(self._model_path)
            try:
                await vlm.start()
                self._owns_vlm = True
            except Exception as e:
                logger.warning(f"OCR model load failed: {e}")
                vlm = None
        self._vlm = vlm
        if self._vlm is not None:
            # the engine's model/processor refs point at the VLMEngine's (no second load)
            self._model = getattr(vlm, "_model", None)
            self._processor = getattr(vlm, "_processor", None)
            self._tokenizer = getattr(vlm, "_tokenizer", None)
            self._running = True
        else:
            logger.error("OCR engine failed to load model %s", self._model_path)

    async def stop(self) -> None:
        """Stop and release model.

        previously this only nulled the refs and never released the MLX
        Metal/wired buffer pool — unlike every other engine (TTS/ASR/Image/VLM/
        Video all gc + sync_and_clear_cache on the executor). A heavy GLM-OCR
        model's Metal memory therefore lingered after unload, a real 36GB OOM
        risk. Now mirror the standard release path.
        """
        import gc

        vlm, self._vlm = self._vlm, None
        if vlm is not None and self._owns_vlm:
            await vlm.stop()
        self._owns_vlm = False
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._running = False
        gc.collect()
        executor = getattr(self, "_executor", None)
        if executor is not None:
            try:
                from .mlx_executor import sync_and_clear_cache

                loop = asyncio.get_running_loop()
                await loop.run_in_executor(executor, sync_and_clear_cache)
            except Exception:
                logger.debug(
                    "OCR sync_and_clear_cache during stop failed", exc_info=True
                )

    def _model_config(self) -> Any:
        """The loaded model's config: the mlx-vlm model object's config, else the
        config.json dict the VLMEngine read."""
        cfg = getattr(self._model, "config", None)
        if cfg is None and self._vlm is not None:
            cfg = getattr(self._vlm, "_config", None)
        return cfg

    def _get_vision_config(self) -> dict[str, Any]:
        """Return the model's vision_config dict (or empty when missing)."""
        cfg = self._model_config()
        if cfg is None:
            return {}
        # mlx-vlm wraps config either as a dataclass-like object or a dict.
        vision_cfg = getattr(cfg, "vision_config", None)
        if vision_cfg is None and isinstance(cfg, dict):
            vision_cfg = cfg.get("vision_config")
        if vision_cfg is None:
            return {}
        # Some configs expose vision_config as a dataclass; coerce to a plain dict.
        if hasattr(vision_cfg, "__dict__") and not isinstance(vision_cfg, dict):
            try:
                return {
                    k: v
                    for k, v in vision_cfg.__dict__.items()
                    if not k.startswith("_")
                }
            except Exception:
                return {}
        return dict(vision_cfg) if isinstance(vision_cfg, dict) else {}

    def _estimate_image_tokens(self, image_path: str) -> int:
        """Estimate the number of vision tokens this image contributes to the
        prompt.

        Approximates ``(H/patch_size) * (W/patch_size) / (merge_size**2)``,
        falling back to a conservative default when image dimensions or
        vision_config cannot be resolved. This is used to make the returned
        ``prompt_tokens`` reflect the real model input cost rather than just
        the text portion (which would underreport by >90% for image inputs).
        """
        try:
            from PIL import Image

            with Image.open(image_path) as im:
                width, height = im.size
        except Exception:
            return 0

        vision_cfg = self._get_vision_config()
        patch_size = int(vision_cfg.get("patch_size", 14) or 14)
        merge_size = int(vision_cfg.get("spatial_merge_size", 1) or 1)
        if patch_size <= 0:
            patch_size = 14
        if merge_size <= 0:
            merge_size = 1

        # Floor division mirrors how processors tile the image into patches.
        h_patches = max(1, height // patch_size)
        w_patches = max(1, width // patch_size)
        tokens = (h_patches * w_patches) // (merge_size * merge_size)
        return max(1, tokens)

    def _extract_sync(
        self, image_path: str, task: str, language: str | None = None
    ) -> dict[str, Any]:
        """Synchronous OCR extraction — runs on the MLX executor thread.

        Uses mlx_vlm's CANONICAL generate() + prompt_utils.apply_chat_template,
        which wraps the image as ``<|begin_of_image|><|image|><|end_of_image|>``
        and lets GlmOcrProcessor expand ``<|image|>`` to the grid-sized token
        count so the vision patches align. A previous hand-rolled prompt +
        decode loop (via the generic mlx_vlm.utils.prepare_inputs, which inserts
        only ONE image token) made the model "see" no image and emit degenerate
        output ("no images or text", <|user|> spam). Validated against the
        canonical path (GLM-OCR → "INVOICE 2026"). (2nd-pass fix.)
        """
        from mlx_vlm import generate as _vlm_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        task_prompt = _resolve_task_prompt(task)
        # honor the language hint on the primary engine path (it was
        # accepted then ignored — only the VLM-fallback in routers/ocr.py applied
        # it, so the same endpoint behaved differently per engine).
        if language:
            task_prompt = f"{task_prompt}\n\nRespond in {language}."
        messages = [{"role": "user", "content": task_prompt}]
        formatted = apply_chat_template(
            self._processor, self._model.config, messages, num_images=1
        )
        result = _vlm_generate(
            self._model,
            self._processor,
            formatted,
            [image_path],
            max_tokens=4096,
            verbose=False,
        )
        text = result.text if hasattr(result, "text") else str(result)
        prompt_tokens = int(getattr(result, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(result, "generation_tokens", 0) or 0)
        image_tokens = self._estimate_image_tokens(image_path)

        return {
            "text": text,
            # (honesty): the VLM-OCR model emits no confidence score, so
            # the old hardcoded 1.0 was a fabricated "measurement". Report None —
            # confidence is genuinely unknown, not perfect.
            "confidence": None,
            "language": None,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "image_tokens": int(image_tokens),
            "total_tokens": int(prompt_tokens + completion_tokens),
        }

    @staticmethod
    def _build_messages(
        image_path: str, task: str, language: str | None = None
    ) -> list[dict[str, Any]]:
        """Chat messages for one OCR call: the image, then the task prompt.

        The task prompt is the GLM-OCR card prompt for ``task`` (see
        ``_resolve_task_prompt``); a language hint is appended to it. VLMEngine
        expands the image part through the model's own chat template.
        """
        task_prompt = _resolve_task_prompt(task)
        # honor the language hint on the primary engine path (it was
        # accepted then ignored — only the VLM-fallback in routers/ocr.py applied
        # it, so the same endpoint behaved differently per engine).
        if language:
            task_prompt = f"{task_prompt}\n\nRespond in {language}."
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                    {"type": "text", "text": task_prompt},
                ],
            }
        ]

    @tracks_active
    async def extract_text(
        self,
        image_path: str,
        language: str | None = None,
        task: str = "text",
        cancel_event: Any = None,
    ) -> dict[str, Any]:
        """Extract text from an image file.

        Generation runs through ``VLMEngine.generate`` (the round driver), greedy
        with the 4096-token cap the native OCR path always used. ``cancel_event``
        is forwarded so a client disconnect stops the decode between steps.

        Args:
            image_path: Path to the image file (must live under YUNSHU_MEDIA_DIR)
            language: Optional language hint
            task: OCR task type — "text", "formula", or "table"
            cancel_event: optional threading.Event set on client disconnect

        Returns:
            {"text": "...", "confidence": None, "language": str|None, ...}
        """
        if not self.is_loaded:
            raise RuntimeError("OCR engine not started")

        messages = self._build_messages(image_path, task, language)
        assert self._vlm is not None  # is_loaded checked it
        result = await self._vlm.generate(
            messages=messages,
            max_tokens=4096,
            temperature=0.0,
            top_p=1.0,
            # the old native path never enabled reasoning; keep it off explicitly
            enable_thinking=False,
            cancel_event=cancel_event,
        )
        prompt_tokens = int(result.get("prompt_tokens", 0) or 0)
        completion_tokens = int(result.get("completion_tokens", 0) or 0)
        image_tokens = self._estimate_image_tokens(image_path)
        out = {
            "text": result.get("text", "") or "",
            # (honesty): the VLM-OCR model emits no confidence score, so
            # the old hardcoded 1.0 was a fabricated "measurement". Report None —
            # confidence is genuinely unknown, not perfect.
            "confidence": None,
            "language": None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "image_tokens": int(image_tokens),
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if language:
            out["language"] = language
        return out

    def get_stats(self) -> dict:
        return {
            "model": self._model_path,
            "loaded": self.is_loaded,
            "running": self._running,
        }
