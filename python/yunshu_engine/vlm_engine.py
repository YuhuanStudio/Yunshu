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


# Models that only support a single image input
SINGLE_IMAGE_ONLY_MODELS = frozenset({
    "glm_ocr",
    "phi3_v",
    "phi3.5_v",
    "florence2",
    "moondream1",
    "moondream2",
    "minicpmv",
    "minicpmv2",
    "llava_llama3",
    "paligemma",
})


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
        self._temp_files: list[str] | None = None

        # mRoPE state (detected during load)
        self._mrope_info = None
        self._rope_delta_manager = None

        # Vision feature cache (opt-in via YUNSHU_VISION_CACHE env var)
        self._vision_cache = None
        import os
        if os.environ.get("YUNSHU_VISION_CACHE", "").strip() in ("1", "true", "yes"):
            from .vision_feature_cache import VisionFeatureCache
            cache_dir = os.environ.get("YUNSHU_VISION_CACHE_DIR", "~/.cache/yunshu/vision")
            self._vision_cache = VisionFeatureCache(cache_dir=cache_dir)
            logger.info("Vision feature cache enabled")

        # C21: Multimodal prefix cache — maps image_hash + system_prompt hash to
        # processed token IDs, enabling reuse across conversations with same image
        self._multimodal_prefix_cache: dict[str, list[int]] = {}
        self._mm_prefix_hits = 0
        self._mm_prefix_misses = 0

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

        # Detect mRoPE support
        from .mrope import detect_mrope, BatchRopeDeltaManager
        self._mrope_info = detect_mrope(self._config)
        if self._mrope_info.enabled:
            self._rope_delta_manager = BatchRopeDeltaManager()
            logger.info(
                f"mRoPE detected: sections={self._mrope_info.sections}, "
                f"source={self._mrope_info.source_key}"
            )

        logger.info(
            f"VLM engine loaded: {self._model_path} "
            f"(vision={self._has_vision}, vlm_model={self._is_vlm}, "
            f"mrope={self._mrope_info.enabled})"
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
        top_k: int = 0,
        seed: int | None = None,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
        enable_thinking: bool | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Non-streaming generation. Supports image input for VLM models."""
        messages = prompt or messages or []
        if self._model is None:
            raise RuntimeError("Engine not started")

        t0 = time.monotonic()
        self._active_count += 1
        image_paths = await self._extract_images(messages)
        audio_paths = await self._extract_audio(messages)
        video_frames = await self._extract_video_frames(messages)
        image_paths.extend(video_frames)
        self._enable_thinking = enable_thinking

        # Compute image hash for vision feature cache lookup
        image_hash = self._compute_image_hash(image_paths) if image_paths else None

        def _generate_sync():
            if seed is not None:
                mx.random.seed(seed)

            if (image_paths and self._has_vision and self._is_vlm) or (audio_paths and self._is_vlm):
                return self._generate_vlm_vision(messages, image_paths, max_tokens, temperature, top_p, top_k, stop, audio_paths=audio_paths)

            prompt_text = self._format_prompt(messages)
            input_ids = mx.array(self._tokenizer.encode(prompt_text))

            if self._is_vlm:
                freq_p = kwargs.get('frequency_penalty', 0.0)
                pres_p = kwargs.get('presence_penalty', 0.0)
                lb = kwargs.get('logit_bias', None)
                js = kwargs.get('json_schema', None)
                return self._generate_vlm_text(input_ids, max_tokens, temperature, top_p, top_k, stop, repetition_penalty, freq_p, pres_p, lb, js)

            from mlx_lm.generate import generate_step
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k if top_k > 0 else 0)
            eos_ids = self._get_eos_ids()

            # Build stop token IDs from string stop sequences
            stop_ids = set(eos_ids)
            if stop:
                for s in stop:
                    try:
                        ids = self._tokenizer.encode(s)
                        if len(ids) == 1:
                            stop_ids.add(ids[0])
                    except Exception:
                        pass

            tokens = []
            for token_id, _ in generate_step(
                input_ids, self._model,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                tokens.append(token_id)
                if token_id in stop_ids:
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
        top_k: int = 0,
        seed: int | None = None,
        stop: list[str] | None = None,
        enable_thinking: bool | None = None,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> AsyncIterator[RequestOutput]:
        """Streaming generation: yields RequestOutput per token."""
        messages = prompt or messages or []
        if self._model is None:
            raise RuntimeError("Engine not started")

        self._enable_thinking = enable_thinking

        import uuid
        req_id = f"vlm-{uuid.uuid4().hex[:8]}"

        queue: asyncio.Queue[RequestOutput | None] = asyncio.Queue(maxsize=256)

        # Extract images for VLM vision path (same as non-streaming)
        image_paths = await self._extract_images(messages)
        audio_paths = await self._extract_audio(messages)
        video_frames = await self._extract_video_frames(messages)
        image_paths.extend(video_frames)
        has_images = bool(image_paths) and self._has_vision and self._is_vlm
        has_audio = bool(audio_paths) and self._is_vlm

        # Compute image hash for vision feature cache lookup
        _image_hash = self._compute_image_hash(image_paths) if image_paths else None

        def _stream_sync():
            try:
                if seed is not None:
                    mx.random.seed(seed)

                if has_images or has_audio:
                    self._stream_vlm_vision(messages, image_paths, max_tokens, temperature, top_p, req_id, queue, top_k, stop, audio_paths=audio_paths)
                    return

                prompt_text = self._format_prompt(messages)
                input_ids = mx.array(self._tokenizer.encode(prompt_text))

                if self._is_vlm:
                    freq_p = kwargs.get('frequency_penalty', 0.0)
                    pres_p = kwargs.get('presence_penalty', 0.0)
                    lb = kwargs.get('logit_bias', None)
                    js = kwargs.get('json_schema', None)
                    self._stream_vlm_text(input_ids, max_tokens, temperature, top_p, req_id, queue, top_k, stop, repetition_penalty, freq_p, pres_p, lb, json_schema=js)
                    return

                from mlx_lm.generate import generate_step
                from mlx_lm.sample_utils import make_sampler

                sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k if top_k > 0 else 0)
                eos_ids = self._get_eos_ids()

                # Build stop token IDs
                stop_ids = set(eos_ids)
                if stop:
                    for s in stop:
                        try:
                            ids = self._tokenizer.encode(s)
                            if len(ids) == 1:
                                stop_ids.add(ids[0])
                        except Exception:
                            pass

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
                    is_eos = token_id in stop_ids
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
        top_k: int = 0,
        stop: list[str] | None = None,
        audio_paths: list[str] | None = None,
    ) -> str:
        """Vision + text generation using mlx_vlm.generate()."""
        from mlx_vlm.generate import generate as vlm_generate

        vlm_messages = self._build_vlm_messages(messages)
        tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        if getattr(self, '_enable_thinking', None) is not None:
            tpl_kwargs["enable_thinking"] = self._enable_thinking

        # Include audio count in template kwargs for Omni models
        if audio_paths:
            tpl_kwargs["num_audios"] = len(audio_paths)

        prompt = self._processor.apply_chat_template(
            vlm_messages, **tpl_kwargs,
        )

        # Check multimodal prefix cache for hit rate tracking
        try:
            system_text = ""
            for m in messages:
                if m.get("role") == "system":
                    system_text += m.get("content", "")
            cached_prefix = self._get_mm_prefix_tokens(image_paths, system_text)
            if cached_prefix is not None:
                logger.debug(f"VLM prefix cache hit: {len(cached_prefix)} tokens")
        except Exception:
            pass

        # Check vision feature cache
        cached_features = None
        if self._vision_cache is not None and len(image_paths) == 1:
            from .vision_feature_cache import compute_image_hash
            try:
                with open(image_paths[0], "rb") as f:
                    img_hash = compute_image_hash(f.read())
                cached_features = self._vision_cache.get(img_hash, self.model_name)
            except Exception:
                logger.debug("vision cache lookup failed", exc_info=True)

        gen_kwargs: dict = {
            "max_tokens": max_tokens,
            "temp": temperature,
            "verbose": False,
        }
        if image_paths:
            gen_kwargs["image"] = image_paths if len(image_paths) > 1 else image_paths[0]
        if audio_paths:
            gen_kwargs["audio"] = audio_paths if len(audio_paths) > 1 else audio_paths[0]

        result = vlm_generate(
            self._model,
            self._processor,
            prompt=prompt,
            **gen_kwargs,
        )

        # Trim output at stop sequences if provided
        if stop and isinstance(result, str):
            for s in stop:
                idx = result.find(s)
                if idx >= 0:
                    result = result[:idx]

        # Capture mRoPE deltas after vision prefill
        if self._mrope_info and self._mrope_info.enabled:
            from .mrope import capture_rope_deltas
            delta = capture_rope_deltas(self._model)
            if delta is not None:
                logger.debug(f"mRoPE delta captured: {delta:.4f}")

        # C21: Store multimodal prefix tokens for future reuse
        try:
            system_text = ""
            for m in messages:
                if m.get("role") == "system":
                    system_text += m.get("content", "")
            prompt_ids = self._tokenizer.encode(prompt)
            self._store_mm_prefix_tokens(image_paths, system_text, prompt_ids)
        except Exception:
            logger.debug("multimodal prefix cache store failed", exc_info=True)

        # Cache vision features after generation (future calls with same image)
        if self._vision_cache is not None and cached_features is None and len(image_paths) == 1:
            try:
                # Extract features from model's vision tower for caching
                if hasattr(self._model, 'vision_tower') and hasattr(self._model.vision_tower, 'features'):
                    features = self._model.vision_tower.features
                    if features is not None:
                        mx.eval(features)
                        self._vision_cache.put(img_hash, self.model_name, features)
            except Exception:
                pass

        return result.text if hasattr(result, 'text') else str(result)

    # ── VLM text generation (for mlx-vlm models) ──

    def _generate_vlm_text(
        self,
        input_ids: mx.array,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int = 0,
        stop: list[str] | None = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        json_schema: dict | None = None,
    ) -> str:
        """Text generation for VLM models using model.language_model."""
        from mlx_vlm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.generate import generation_stream

        # Clear mRoPE state to prevent contamination from prior VLM request
        if self._mrope_info and self._mrope_info.enabled:
            from .mrope import clear_rope_state
            clear_rope_state(self._model)

        lm = self._model.language_model
        cache = make_prompt_cache(lm)
        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k if top_k > 0 else 0)
        eos_ids = self._get_eos_ids()

        # JSON schema constraint
        json_constraint = None
        if json_schema is not None:
            try:
                from .json_schema import JsonSchemaConstraint
                json_constraint = JsonSchemaConstraint(json_schema, self._tokenizer)
            except Exception:
                logger.debug("JSON schema constraint init failed", exc_info=True)

        has_penalty = repetition_penalty != 1.0 or frequency_penalty != 0.0 or presence_penalty != 0.0 or logit_bias

        # Build stop IDs from string sequences
        stop_ids = set(eos_ids)
        if stop:
            for s in stop:
                try:
                    ids = self._tokenizer.encode(s)
                    if len(ids) == 1:
                        stop_ids.add(ids[0])
                except Exception:
                    pass

        with mx.stream(generation_stream):
            # Prefill
            output = lm(input_ids[None], cache=cache)
            logits = output.logits[:, -1, :]
            current = sampler(logits)
            mx.eval(current)

            tokens = [current.item()]
            if current.item() in stop_ids:
                return self._tokenizer.decode(tokens, skip_special_tokens=True)

            for _ in range(max_tokens - 1):
                output = lm(current[None], cache=cache)
                logits = output.logits[:, -1, :]

                if has_penalty:
                    if repetition_penalty != 1.0:
                        ctx = tokens[-20:]
                        sel = logits[..., ctx]
                        sel = mx.where(sel < 0, sel * repetition_penalty, sel / repetition_penalty)
                        logits[..., ctx] = sel
                    if frequency_penalty != 0.0:
                        tid = tokens[-1]
                        logits[..., tid] -= frequency_penalty
                    if presence_penalty != 0.0:
                        tid = tokens[-1]
                        logits[..., tid] -= presence_penalty
                    if logit_bias:
                        for tid, bias in logit_bias.items():
                            logits[..., tid] += bias

                # JSON schema constraint masking
                if json_constraint is not None and tokens:
                    try:
                        allowed = json_constraint.get_allowed_tokens(self._tokenizer, tokens)
                        if allowed:
                            from .json_schema import apply_json_constraint
                            logits = apply_json_constraint(logits, allowed)
                    except Exception:
                        pass

                current = sampler(logits)
                mx.eval(current)
                tokens.append(current.item())
                if current.item() in stop_ids:
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
        top_k: int = 0,
        stop: list[str] | None = None,
        audio_paths: list[str] | None = None,
    ) -> None:
        """Streaming vision + text generation using mlx_vlm.stream_generate()."""
        from mlx_vlm.generate import stream_generate as vlm_stream_generate
        from mlx_lm.sample_utils import make_sampler

        vlm_messages = self._build_vlm_messages(messages)
        tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        if getattr(self, '_enable_thinking', None) is not None:
            tpl_kwargs["enable_thinking"] = self._enable_thinking

        if audio_paths:
            tpl_kwargs["num_audios"] = len(audio_paths)

        prompt = self._processor.apply_chat_template(
            vlm_messages, **tpl_kwargs,
        )

        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k if top_k > 0 else 0)
        stop_suffix = stop or []
        token_count = 0
        try:
            stream_kwargs: dict = {
                "max_tokens": max_tokens,
                "sampler": sampler,
            }
            if image_paths:
                stream_kwargs["image"] = image_paths if len(image_paths) > 1 else image_paths[0]
            if audio_paths:
                stream_kwargs["audio"] = audio_paths if len(audio_paths) > 1 else audio_paths[0]

            for result in vlm_stream_generate(
                self._model,
                self._processor,
                prompt=prompt,
                **stream_kwargs,
            ):
                token_count += 1
                text = result.text if hasattr(result, 'text') else ""
                finish_reason = None
                if hasattr(result, 'finish_reason') and result.finish_reason:
                    finish_reason = result.finish_reason
                elif token_count >= max_tokens:
                    finish_reason = "length"
                # Check stop suffixes
                if not finish_reason and stop_suffix:
                    for s in stop_suffix:
                        if text.endswith(s):
                            finish_reason = "stop"
                            break

                queue.put_nowait(RequestOutput(
                    request_id=req_id,
                    new_text=text,
                    finish_reason=finish_reason,
                    finished=finish_reason is not None,
                    completion_tokens=token_count,
                ))
                if finish_reason:
                    return

            # Cache vision features after streaming
            if self._vision_cache is not None and len(image_paths) == 1:
                try:
                    from .vision_feature_cache import compute_image_hash
                    with open(image_paths[0], "rb") as f:
                        img_hash = compute_image_hash(f.read())
                    if hasattr(self._model, 'vision_tower') and hasattr(self._model.vision_tower, 'features'):
                        features = self._model.vision_tower.features
                        if features is not None:
                            mx.eval(features)
                            self._vision_cache.put(img_hash, self.model_name, features)
                except Exception:
                    logger.debug("vision cache store failed", exc_info=True)
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
        top_k: int = 0,
        stop: list[str] | None = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        json_schema: dict | None = None,
    ) -> None:
        """Streaming text generation for VLM models."""
        from mlx_vlm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        # Clear mRoPE state to prevent contamination from prior VLM request
        if self._mrope_info and self._mrope_info.enabled:
            from .mrope import clear_rope_state
            clear_rope_state(self._model)

        lm = self._model.language_model
        cache = make_prompt_cache(lm)
        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k if top_k > 0 else 0)
        eos_ids = self._get_eos_ids()
        has_penalty = repetition_penalty != 1.0 or frequency_penalty != 0.0 or presence_penalty != 0.0 or logit_bias

        # JSON schema constraint
        json_constraint = None
        if json_schema is not None:
            try:
                from .json_schema import JsonSchemaConstraint
                json_constraint = JsonSchemaConstraint(json_schema, self._tokenizer)
            except Exception:
                logger.debug("JSON schema constraint init failed (stream)", exc_info=True)

        # Build stop IDs
        stop_ids = set(eos_ids)
        stop_suffixes = []
        if stop:
            for s in stop:
                try:
                    ids = self._tokenizer.encode(s)
                    if len(ids) == 1:
                        stop_ids.add(ids[0])
                    else:
                        stop_suffixes.append(s)
                except Exception:
                    pass

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
        is_eos = token_id in stop_ids
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

        tokens_list = []
        for _ in range(max_tokens - 1):
            output = lm(current[None], cache=cache)
            logits = output.logits[:, -1, :]

            if has_penalty:
                tokens_list.append(current.item())
                if repetition_penalty != 1.0:
                    ctx = tokens_list[-20:]
                    sel = logits[..., ctx]
                    sel = mx.where(sel < 0, sel * repetition_penalty, sel / repetition_penalty)
                    logits[..., ctx] = sel
                if frequency_penalty != 0.0:
                    logits[..., tokens_list[-1]] -= frequency_penalty
                if presence_penalty != 0.0:
                    logits[..., tokens_list[-1]] -= presence_penalty
                if logit_bias:
                    for tid, bias in logit_bias.items():
                        logits[..., tid] += bias

            # JSON schema constraint masking
            if json_constraint is not None and tokens_list:
                try:
                    allowed = json_constraint.get_allowed_tokens(self._tokenizer, tokens_list)
                    if allowed:
                        from .json_schema import apply_json_constraint
                        logits = apply_json_constraint(logits, allowed)
                except Exception:
                    pass

            current = sampler(logits)
            mx.eval(current)
            token_count += 1

            token_id = current.item()
            is_eos = token_id in stop_ids
            suffix_hit = False
            if not is_eos and stop_suffixes and has_detokenizer:
                if any(detokenizer.text.endswith(s) for s in stop_suffixes):
                    suffix_hit = True
            finish_reason = "stop" if (is_eos or suffix_hit) else None

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

    def _build_vlm_messages(self, messages: list[dict]) -> list[dict]:
        """Build messages with image/audio references for processor's chat template."""
        vlm_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "image_url":
                            parts.append({"type": "image"})
                        elif part.get("type") == "input_audio":
                            parts.append({"type": "audio"})
                        elif part.get("type") == "audio_url":
                            parts.append({"type": "audio"})
                        elif part.get("type") == "text":
                            parts.append({"type": "text", "text": part.get("text", "")})
                    elif isinstance(part, str):
                        parts.append({"type": "text", "text": part})
                vlm_messages.append({"role": msg.get("role", "user"), "content": parts})
            else:
                vlm_messages.append({"role": msg.get("role", "user"), "content": str(content)})
        return vlm_messages

    def _format_prompt(self, messages: list[dict]) -> str:
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                clean = []
                for msg in messages:
                    clean.append({
                        "role": msg.get("role", "user"),
                        "content": self._extract_text(msg.get("content", "")),
                    })
                tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
                if getattr(self, '_enable_thinking', None) is not None:
                    tpl_kwargs["enable_thinking"] = self._enable_thinking
                text = self._tokenizer.apply_chat_template(clean, **tpl_kwargs)
                if text:
                    return text
            except Exception:
                logger.debug("chat template failed, using fallback", exc_info=True)

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
                        elif url.startswith(("http://", "https://")):
                            paths.append(await self._download_image(url))
                        elif os.path.exists(url):
                            paths.append(url)
        # Validate: some models only support single image input
        if len(paths) > 1 and self._config:
            model_type = self._config.get("model_type", "")
            if model_type in SINGLE_IMAGE_ONLY_MODELS:
                logger.warning(
                    f"Model type '{model_type}' only supports single image, "
                    f"got {len(paths)} — using first image only"
                )
                paths = paths[:1]
        return paths

    # ── Image Hash ──

    def _compute_image_hash(self, image_paths: list[str]) -> str | None:
        """Compute a content hash for image paths (vision feature cache key)."""
        if not image_paths:
            return None
        import hashlib
        h = hashlib.sha256()
        for path in image_paths:
            h.update(path.encode())
            try:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            h.update(chunk)
            except Exception:
                pass
        return h.hexdigest()[:16]

    # ── Audio Extraction ──

    async def _extract_audio(self, messages: list[dict]) -> list[str]:
        """Extract audio file paths from OpenAI-format message content parts.

        Supports:
        - {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}
        - {"type": "audio_url", "audio_url": {"url": "file://..."}}
        """
        paths = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type", "")
                    if ptype == "input_audio":
                        ia = part.get("input_audio", {})
                        data = ia.get("data")
                        fmt = ia.get("format", "wav")
                        if data:
                            paths.append(await self._save_base64_audio(data, fmt))
                    elif ptype == "audio_url":
                        url = part.get("audio_url", {}).get("url", "")
                        if url.startswith("data:audio"):
                            header, data = url.split(",", 1)
                            fmt = header.split("/")[1].split(";")[0]
                            paths.append(await self._save_base64_audio(data, fmt))
                        elif url.startswith("file://"):
                            path = url[7:]
                            if os.path.exists(path):
                                paths.append(path)
                        elif os.path.exists(url):
                            paths.append(url)
        return paths

    async def _save_base64_audio(self, data: str, fmt: str = "wav") -> str:
        ext = fmt if fmt in ("wav", "mp3", "ogg", "flac", "pcm") else "wav"
        tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
        tmp.write(base64.b64decode(data))
        tmp.close()
        if self._temp_files is None:
            self._temp_files = []
        self._temp_files.append(tmp.name)
        return tmp.name

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

    async def _download_image(self, url: str) -> str:
        """Download an image from HTTP/HTTPS URL to a temp file."""
        import urllib.request
        import ssl

        ext = url.rsplit(".", 1)[-1].lower() if "." in url.split("?")[0] else "png"
        ext = ext if ext in ("png", "jpg", "jpeg", "webp", "gif") else "png"

        tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
        try:
            ctx = ssl.create_default_context()
            req = urllib.request.Request(url, headers={"User-Agent": "Yunshu/1.0"})
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: urllib.request.urlretrieve(url, tmp.name),
            )
        except Exception:
            # Fallback: try with less strict SSL for some CDNs
            try:
                await loop.run_in_executor(
                    None,
                    lambda: urllib.request.urlretrieve(url, tmp.name, context=ctx),
                )
            except Exception as e:
                logger.warning(f"Failed to download image from {url}: {e}")
                raise ValueError(f"Cannot download image: {e}")
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

    # ── Video Frame Extraction ──

    async def _extract_video_frames(
        self,
        messages: list[dict],
        fps: float = 1.0,
        max_frames: int = 8,
    ) -> list[str]:
        """Extract video frames from message content parts and return image paths.

        Supports:
        - {"type": "video_url", "video_url": {"url": "file://..."}}  (local file)
        - {"type": "video_url", "video_url": {"url": "data:video/..."}}  (base64)
        - {"type": "video_file", "video_file": {"file_id": "/path/to/video.mp4"}}

        Frames are extracted using ffmpeg at the given fps, up to max_frames.
        """
        video_paths = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "video_url":
                        url = part.get("video_url", {}).get("url", "")
                        if url.startswith("data:video"):
                            header, data = url.split(",", 1)
                            ext = header.split("/")[1].split(";")[0]
                            ext = ext if ext in ("mp4", "webm", "avi", "mov", "mkv") else "mp4"
                            path = await self._save_base64_file(data, ext)
                            video_paths.append(path)
                        elif url.startswith("file://"):
                            path = url[7:]
                            if os.path.exists(path):
                                video_paths.append(path)
                        elif os.path.exists(url):
                            video_paths.append(url)
                    elif part.get("type") == "video_file":
                        fid = part.get("video_file", {}).get("file_id", "")
                        if fid and os.path.exists(fid):
                            video_paths.append(fid)

        if not video_paths:
            return []

        frame_paths = []
        for vp in video_paths:
            frames = await self._extract_frames_from_file(vp, fps=fps, max_frames=max_frames)
            frame_paths.extend(frames)

        return frame_paths[:max_frames]

    async def _extract_frames_from_file(
        self,
        video_path: str,
        fps: float = 1.0,
        max_frames: int = 8,
    ) -> list[str]:
        """Extract frames from a video file using ffmpeg."""
        import subprocess

        tmpdir = tempfile.mkdtemp(prefix="yunshu_video_")
        output_pattern = os.path.join(tmpdir, "frame_%04d.jpg")

        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={fps}",
            "-frames:v", str(max_frames),
            "-q:v", "2",
            "-y", output_pattern,
        ]

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True, timeout=60, check=True),
            )
        except FileNotFoundError:
            logger.warning("ffmpeg not available — cannot extract video frames")
            return []
        except subprocess.CalledProcessError as e:
            logger.warning(f"ffmpeg failed: {e.stderr.decode()[:200] if e.stderr else 'unknown'}")
            return []
        except Exception as e:
            logger.warning(f"Video frame extraction failed: {e}")
            return []

        frames = sorted(
            os.path.join(tmpdir, f) for f in os.listdir(tmpdir)
            if f.startswith("frame_") and f.endswith(".jpg")
        )

        if self._temp_files is None:
            self._temp_files = []
        self._temp_files.append(tmpdir)
        for f in frames:
            self._temp_files.append(f)

        return frames

    async def _save_base64_file(self, data: str, ext: str = "mp4") -> str:
        """Save base64-encoded data to a temp file."""
        tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
        tmp.write(base64.b64decode(data))
        tmp.close()
        if self._temp_files is None:
            self._temp_files = []
        self._temp_files.append(tmp.name)
        return tmp.name

    # ── Helpers ──

    def _get_eos_ids(self) -> list[int]:
        from .text_utils import get_eos_token_ids
        return get_eos_token_ids(self._tokenizer)

    # ── C21: Multimodal prefix cache ──

    def _mm_prefix_key(self, image_paths: list[str], system_text: str) -> str:
        """Compute cache key from image hashes + system prompt."""
        parts = [system_text]
        for path in image_paths:
            try:
                from .vision_feature_cache import compute_image_hash
                with open(path, "rb") as f:
                    parts.append(compute_image_hash(f.read()))
            except Exception:
                parts.append(path)
        return "|".join(parts)

    def _get_mm_prefix_tokens(self, image_paths: list[str], system_text: str) -> list[int] | None:
        """Get cached token IDs for a multimodal prefix (C21)."""
        key = self._mm_prefix_key(image_paths, system_text)
        result = self._multimodal_prefix_cache.get(key)
        if result is not None:
            self._mm_prefix_hits += 1
        else:
            self._mm_prefix_misses += 1
        return result

    def _store_mm_prefix_tokens(self, image_paths: list[str], system_text: str, token_ids: list[int]) -> None:
        """Store processed token IDs for a multimodal prefix (C21)."""
        if len(self._multimodal_prefix_cache) > 64:
            # Evict oldest entries
            keys = list(self._multimodal_prefix_cache.keys())
            for k in keys[:16]:
                del self._multimodal_prefix_cache[k]
        key = self._mm_prefix_key(image_paths, system_text)
        self._multimodal_prefix_cache[key] = token_ids

    # ── Stats ──

    def get_stats(self) -> dict:
        uptime = time.monotonic() - self._start_time if self._start_time else 0.0
        total = self._mm_prefix_hits + self._mm_prefix_misses
        return {
            "model": self._model_path,
            "loaded": self.is_loaded,
            "running": self._running,
            "has_vision": self._has_vision,
            "is_vlm": self._is_vlm,
            "num_requests_processed": self._num_requests_processed,
            "uptime_seconds": uptime,
            "mm_prefix_cache_entries": len(self._multimodal_prefix_cache),
            "mm_prefix_cache_hits": self._mm_prefix_hits,
            "mm_prefix_cache_misses": self._mm_prefix_misses,
            "mm_prefix_cache_hit_rate": self._mm_prefix_hits / total if total > 0 else 0.0,
            "vision_cache_enabled": self._vision_cache is not None,
        }
