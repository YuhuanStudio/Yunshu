"""Yunshu VLM Engine — Vision-Language Model inference.

Two-tier model loading:
- mlx-lm for text-only LLMs (qwen3, llama, gemma, etc.)
- mlx-vlm model classes for VLM/Omni models (qwen3_omni_moe, qwen3_vl, etc.)

Generation uses mlx-lm primitives (make_sampler, generate_step) for text-only.
For VLM models with vision features, uses model.language_model directly
with mlx-vlm's cache management.

Architecture:
- Unified load via mlx_lm.utils.load_model() with get_model_classes fallback
- Text generation: mlx_lm.generate.generate_step (1D input_ids)
- VLM text-only: model.language_model + manual generate loop
- Vision features: model.get_input_embeddings() (from mlx-vlm reference)
- GPU work serialized on shared executor (mlx_executor pattern)
- Streaming via tokenizer.detokenizer (per-request, never pooled)
"""
from __future__ import annotations

import asyncio
import base64
import gc
import importlib
import logging
import os
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx

from .engine import EngineConfig, RequestOutput

logger = logging.getLogger(__name__)


def _get_model_classes_with_vlm_fallback(config: dict):
    """Resolve model classes: try mlx-lm first, fall back to mlx-vlm."""
    from mlx_lm.utils import MODEL_REMAPPING

    model_type = config.get("model_type", "")
    remapped = MODEL_REMAPPING.get(model_type, model_type)

    # Try mlx-lm first (covers LLM + VLM types that mlx-lm supports)
    try:
        arch = importlib.import_module(f"mlx_lm.models.{remapped}")
        return arch.Model, arch.ModelArgs
    except ImportError:
        pass

    # Fall back to mlx-vlm (covers VLM/Omni types)
    try:
        arch = importlib.import_module(f"mlx_vlm.models.{remapped}")
        return arch.Model, arch.ModelConfig
    except ImportError:
        pass

    raise ValueError(f"Model type {model_type} not supported by mlx-lm or mlx-vlm.")


def _is_mlx_vlm_model(model) -> bool:
    """Check if a model was loaded from mlx-vlm's model classes (vs mlx-lm)."""
    return type(model).__module__.startswith("mlx_vlm.models.")


class VLMEngine:
    """Vision-Language Model engine with dual mlx-lm/mlx-vlm support.

    For text-only LLMs (mlx-lm supported): uses mlx_lm.generate.generate_step.
    For VLM/Omni models (mlx-vlm model classes): uses model.language_model
    with manual generate loop and LanguageModelOutput.
    """

    def __init__(self, model_path: str, config: EngineConfig | None = None) -> None:
        self._model_path = model_path
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._config: dict = {}
        self._running = False
        self._active_count = 0
        self._num_requests_processed = 0
        self._start_time = 0.0
        self._has_vision = False
        self._is_vlm = False

        from .mlx_executor import get_mlx_executor
        self._executor = get_mlx_executor()

    @property
    def model_name(self) -> str:
        return self._model_path.rsplit("/", 1)[-1] if "/" in self._model_path else self._model_path

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_running(self) -> bool:
        return self._running

    def has_active_requests(self) -> bool:
        return self._active_count > 0

    @property
    def has_vision(self) -> bool:
        return self._has_vision

    def load(self) -> None:
        """Load model and tokenizer with mlx-lm/mlx-vlm fallback.

        For mlx-vlm models, MUST run on the MLX executor thread so weights
        and compute share the same GPU stream.
        """
        from mlx_lm.utils import load_config, load_model, load_tokenizer

        model_path = Path(self._model_path)

        # Download if HF repo ID (not local path)
        if not model_path.exists():
            from mlx_lm.utils import _download
            model_path = Path(_download(self._model_path))

        self._config = load_config(model_path)

        # Check vision support
        thinker_cfg = self._config.get("thinker_config", {})
        self._has_vision = bool(
            self._config.get("vision_config")
            or thinker_cfg.get("vision_config")
        )

        # Load model with mlx-vlm fallback
        model, config = load_model(
            model_path,
            get_model_classes=_get_model_classes_with_vlm_fallback,
        )
        tokenizer = load_tokenizer(model_path)

        self._model = model
        self._tokenizer = tokenizer
        self._is_vlm = _is_mlx_vlm_model(model)

        # Load processor for VLM vision input
        if self._has_vision and self._is_vlm:
            try:
                from mlx_vlm.utils import load_processor
                self._processor = load_processor(model_path)
            except Exception as e:
                logger.warning(f"Could not load VLM processor: {e}")

        logger.info(
            f"VLM engine loaded: {self._model_path} "
            f"(vision={self._has_vision}, vlm_model={self._is_vlm})"
        )

    async def start(self) -> None:
        if self._model is not None:
            self._running = True
            self._start_time = time.monotonic()
            return
        # Always load on the MLX executor thread — required for mlx-vlm models
        # whose weights must share the same GPU stream as compute
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.load)
        self._running = True
        self._start_time = time.monotonic()

    async def stop(self) -> None:
        self._cleanup_temp_files()
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._running = False
        gc.collect()
        loop = asyncio.get_running_loop()
        from .mlx_executor import sync_and_clear_cache
        await loop.run_in_executor(self._executor, sync_and_clear_cache)

    def resolve_model_id(self, model_id: str) -> bool:
        return model_id in {
            self.model_name, self._model_path,
            self.model_name.lower(), self._model_path.lower(),
        }

    # ── Generation ──

    async def generate(
        self,
        prompt: list[dict] | None = None,
        messages: list[dict] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs,
    ) -> dict[str, Any]:
        """Non-streaming generation. Supports image input for VLM models."""
        messages = prompt or messages or []
        if self._model is None:
            raise RuntimeError("Engine not started")

        t0 = time.monotonic()
        self._active_count += 1
        image_paths = await self._extract_images(messages)

        def _generate_sync():
            if image_paths and self._has_vision and self._is_vlm:
                return self._generate_vlm_vision(messages, image_paths, max_tokens, temperature, top_p)

            prompt_text = self._format_prompt(messages)
            input_ids = mx.array(self._tokenizer.encode(prompt_text))

            if self._is_vlm:
                return self._generate_vlm_text(input_ids, max_tokens, temperature, top_p)

            from mlx_lm.generate import generate_step
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=temperature, top_p=top_p)
            eos_ids = self._get_eos_ids()

            tokens = []
            for token_id, _ in generate_step(
                input_ids, self._model,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                tokens.append(token_id)
                if token_id in eos_ids:
                    break

            return self._tokenizer.decode(tokens, skip_special_tokens=True)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self._executor, _generate_sync)

        elapsed = time.monotonic() - t0
        self._active_count -= 1
        self._num_requests_processed += 1
        self._cleanup_temp_files()

        return {"text": result, "finish_reason": "stop", "elapsed": elapsed}

    async def generate_stream(
        self,
        prompt: list[dict] | None = None,
        messages: list[dict] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs,
    ) -> AsyncIterator[RequestOutput]:
        """Streaming generation: yields RequestOutput per token."""
        messages = prompt or messages or []
        if self._model is None:
            raise RuntimeError("Engine not started")

        import uuid
        req_id = f"vlm-{uuid.uuid4().hex[:8]}"

        queue: asyncio.Queue[RequestOutput | None] = asyncio.Queue(maxsize=256)

        # Extract images for VLM vision path (same as non-streaming)
        image_paths = await self._extract_images(messages)
        has_images = bool(image_paths) and self._has_vision and self._is_vlm

        def _stream_sync():
            try:
                if has_images:
                    self._stream_vlm_vision(messages, image_paths, max_tokens, temperature, top_p, req_id, queue)
                    return

                prompt_text = self._format_prompt(messages)
                input_ids = mx.array(self._tokenizer.encode(prompt_text))

                if self._is_vlm:
                    self._stream_vlm_text(input_ids, max_tokens, temperature, top_p, req_id, queue)
                    return

                from mlx_lm.generate import generate_step
                from mlx_lm.sample_utils import make_sampler

                sampler = make_sampler(temp=temperature, top_p=top_p)
                eos_ids = self._get_eos_ids()

                has_detokenizer = hasattr(self._tokenizer, 'detokenizer')
                if has_detokenizer:
                    detokenizer = self._tokenizer.detokenizer
                    detokenizer.reset()

                token_count = 0
                for token_id, _ in generate_step(
                    input_ids, self._model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ):
                    token_count += 1
                    is_eos = token_id in eos_ids
                    finish_reason = "stop" if is_eos else None

                    if not is_eos:
                        if has_detokenizer:
                            detokenizer.add_token(token_id)
                            token_text = detokenizer.last_segment
                        else:
                            token_text = self._tokenizer.decode([token_id], skip_special_tokens=True)
                    else:
                        token_text = ""

                    output = RequestOutput(
                        request_id=req_id,
                        new_text=token_text,
                        new_token_ids=[token_id],
                        finish_reason=finish_reason,
                        finished=finish_reason is not None,
                        completion_tokens=token_count,
                    )
                    queue.put_nowait(output)

                    if finish_reason:
                        return

                output = RequestOutput(
                    request_id=req_id,
                    new_text="",
                    finish_reason="length",
                    finished=True,
                    completion_tokens=token_count,
                )
                queue.put_nowait(output)

            except Exception as e:
                logger.error(f"VLM stream error: {e}")
            finally:
                queue.put_nowait(None)

        self._active_count += 1
        loop = asyncio.get_running_loop()
        stream_task = loop.run_in_executor(self._executor, _stream_sync)

        try:
            while True:
                output = await queue.get()
                if output is None:
                    break
                yield output
        finally:
            self._active_count -= 1
            if not stream_task.done():
                stream_task.cancel()

    def _generate_vlm_vision(
        self,
        messages: list[dict],
        image_paths: list[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Vision + text generation using mlx_vlm.generate()."""
        from mlx_vlm.generate import generate as vlm_generate

        # Build messages with image references for processor's chat template
        vlm_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "image_url":
                            parts.append({"type": "image"})
                        elif part.get("type") == "text":
                            parts.append({"type": "text", "text": part.get("text", "")})
                    elif isinstance(part, str):
                        parts.append({"type": "text", "text": part})
                vlm_messages.append({"role": msg.get("role", "user"), "content": parts})
            else:
                vlm_messages.append({"role": msg.get("role", "user"), "content": str(content)})

        # Use processor's chat template to insert image tokens
        prompt = self._processor.apply_chat_template(
            vlm_messages, tokenize=False, add_generation_prompt=True,
        )

        result = vlm_generate(
            self._model,
            self._processor,
            prompt=prompt,
            image=image_paths if len(image_paths) > 1 else image_paths[0],
            max_tokens=max_tokens,
            temp=temperature,
            verbose=False,
        )
        return result.text if hasattr(result, 'text') else str(result)

    # ── VLM text generation (for mlx-vlm models) ──

    def _generate_vlm_text(
        self,
        input_ids: mx.array,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Text generation for VLM models using model.language_model."""
        from mlx_vlm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.generate import generation_stream

        lm = self._model.language_model
        cache = make_prompt_cache(lm)
        sampler = make_sampler(temp=temperature, top_p=top_p)
        eos_ids = self._get_eos_ids()

        with mx.stream(generation_stream):
            # Prefill
            output = lm(input_ids[None], cache=cache)
            logits = output.logits[:, -1, :]
            current = sampler(logits)
            mx.eval(current)

            tokens = [current.item()]
            if current.item() in eos_ids:
                return self._tokenizer.decode(tokens, skip_special_tokens=True)

            for _ in range(max_tokens - 1):
                output = lm(current[None], cache=cache)
                logits = output.logits[:, -1, :]
                current = sampler(logits)
                mx.eval(current)
                tokens.append(current.item())
                if current.item() in eos_ids:
                    break

        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    def _stream_vlm_vision(
        self,
        messages: list[dict],
        image_paths: list[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        req_id: str,
        queue: asyncio.Queue,
    ) -> None:
        """Streaming vision + text generation using mlx_vlm.stream_generate()."""
        from mlx_vlm.generate import stream_generate as vlm_stream_generate
        from mlx_lm.sample_utils import make_sampler

        vlm_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "image_url":
                            parts.append({"type": "image"})
                        elif part.get("type") == "text":
                            parts.append({"type": "text", "text": part.get("text", "")})
                    elif isinstance(part, str):
                        parts.append({"type": "text", "text": part})
                vlm_messages.append({"role": msg.get("role", "user"), "content": parts})
            else:
                vlm_messages.append({"role": msg.get("role", "user"), "content": str(content)})

        prompt = self._processor.apply_chat_template(
            vlm_messages, tokenize=False, add_generation_prompt=True,
        )

        sampler = make_sampler(temp=temperature, top_p=top_p)
        token_count = 0
        try:
            for result in vlm_stream_generate(
                self._model,
                self._processor,
                prompt=prompt,
                image=image_paths if len(image_paths) > 1 else image_paths[0],
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                token_count += 1
                text = result.text if hasattr(result, 'text') else ""
                finish_reason = None
                if hasattr(result, 'finish_reason') and result.finish_reason:
                    finish_reason = result.finish_reason
                elif token_count >= max_tokens:
                    finish_reason = "length"

                queue.put_nowait(RequestOutput(
                    request_id=req_id,
                    new_text=text,
                    finish_reason=finish_reason,
                    finished=finish_reason is not None,
                    completion_tokens=token_count,
                ))
                if finish_reason:
                    return
        except Exception as e:
            queue.put_nowait(RequestOutput(
                request_id=req_id,
                new_text="",
                finish_reason="error",
                finished=True,
                completion_tokens=token_count,
                error=str(e),
            ))

    def _stream_vlm_text(
        self,
        input_ids: mx.array,
        max_tokens: int,
        temperature: float,
        top_p: float,
        req_id: str,
        queue: asyncio.Queue,
    ) -> None:
        """Streaming text generation for VLM models."""
        from mlx_vlm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        lm = self._model.language_model
        cache = make_prompt_cache(lm)
        sampler = make_sampler(temp=temperature, top_p=top_p)
        eos_ids = self._get_eos_ids()

        has_detokenizer = hasattr(self._tokenizer, 'detokenizer')
        if has_detokenizer:
            detokenizer = self._tokenizer.detokenizer
            detokenizer.reset()

        # Prefill
        output = lm(input_ids[None], cache=cache)
        logits = output.logits[:, -1, :]
        current = sampler(logits)
        mx.eval(current)
        token_count = 1

        token_id = current.item()
        is_eos = token_id in eos_ids
        finish_reason = "stop" if is_eos else None

        if not is_eos:
            if has_detokenizer:
                detokenizer.add_token(token_id)
                token_text = detokenizer.last_segment
            else:
                token_text = self._tokenizer.decode([token_id], skip_special_tokens=True)
        else:
            token_text = ""

        queue.put_nowait(RequestOutput(
            request_id=req_id,
            new_text=token_text,
            new_token_ids=[token_id],
            finish_reason=finish_reason,
            finished=finish_reason is not None,
            completion_tokens=token_count,
        ))
        if finish_reason:
            return

        for _ in range(max_tokens - 1):
            output = lm(current[None], cache=cache)
            logits = output.logits[:, -1, :]
            current = sampler(logits)
            mx.eval(current)
            token_count += 1

            token_id = current.item()
            is_eos = token_id in eos_ids
            finish_reason = "stop" if is_eos else None

            if not is_eos:
                if has_detokenizer:
                    detokenizer.add_token(token_id)
                    token_text = detokenizer.last_segment
                else:
                    token_text = self._tokenizer.decode([token_id], skip_special_tokens=True)
            else:
                token_text = ""

            queue.put_nowait(RequestOutput(
                request_id=req_id,
                new_text=token_text,
                new_token_ids=[token_id],
                finish_reason=finish_reason,
                finished=finish_reason is not None,
                completion_tokens=token_count,
            ))

            if finish_reason:
                return

        queue.put_nowait(RequestOutput(
            request_id=req_id,
            new_text="",
            finish_reason="length",
            finished=True,
            completion_tokens=token_count,
        ))

    # ── Prompt Formatting ──

    def _format_prompt(self, messages: list[dict]) -> str:
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                clean = []
                for msg in messages:
                    clean.append({
                        "role": msg.get("role", "user"),
                        "content": self._extract_text(msg.get("content", "")),
                    })
                text = self._tokenizer.apply_chat_template(
                    clean, tokenize=False, add_generation_prompt=True,
                )
                if text:
                    return text
            except Exception:
                pass

        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = self._extract_text(msg.get("content", ""))
            parts.append(f"{role.capitalize()}: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    @staticmethod
    def _extract_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif isinstance(part, str):
                    texts.append(part)
            return " ".join(texts)
        return str(content)

    # ── Image Extraction ──

    async def _extract_images(self, messages: list[dict]) -> list[str]:
        paths = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            paths.append(await self._save_base64_image(url))
                        elif os.path.exists(url):
                            paths.append(url)
        return paths

    _temp_files: list[str] | None = None

    async def _save_base64_image(self, data_url: str) -> str:
        header, data = data_url.split(",", 1)
        ext = header.split("/")[1].split(";")[0]
        ext = ext if ext in ("png", "jpg", "jpeg", "webp", "gif") else "png"
        tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
        tmp.write(base64.b64decode(data))
        tmp.close()
        if self._temp_files is None:
            self._temp_files = []
        self._temp_files.append(tmp.name)
        return tmp.name

    def _cleanup_temp_files(self) -> None:
        if self._temp_files:
            for path in self._temp_files:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            self._temp_files.clear()

    # ── Helpers ──

    def _get_eos_ids(self) -> list[int]:
        from .text_utils import get_eos_token_ids
        return get_eos_token_ids(self._tokenizer)

    # ── Stats ──

    def get_stats(self) -> dict:
        uptime = time.monotonic() - self._start_time if self._start_time else 0.0
        return {
            "model": self._model_path,
            "loaded": self.is_loaded,
            "running": self._running,
            "has_vision": self._has_vision,
            "is_vlm": self._is_vlm,
            "num_requests_processed": self._num_requests_processed,
            "uptime_seconds": uptime,
        }
