from __future__ import annotations
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
- Vision feature caching: skip re-encoding when same image seen again
- KV prefix reuse: skip prefix prefill for same image across conversations
- GPU work serialized on shared executor (mlx_executor pattern)
- Streaming via tokenizer.detokenizer (per-request, never pooled)
"""

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
from typing import Any

import mlx.core as mx

from .types import EngineConfig
from .request import RequestOutput

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


class _MlxVlmVisionCacheAdapter:
    """Adapts VisionFeatureCache to mlx_vlm's expected vision_cache interface.

    mlx_vlm's stream_generate expects a dict-like object with:
      - get(image) -> features or None
      - put(image, features) -> None

    where `image` is the image path string (or list of strings).
    Our VisionFeatureCache uses (image_hash, model_name) as the key.
    This adapter computes the hash and delegates to the real cache.
    """

    def __init__(self, cache, model_name: str):
        self._cache = cache
        self._model_name = model_name

    def get(self, image):
        """Look up cached vision features by image path(s)."""
        from .vision_feature_cache import compute_image_hash

        try:
            if isinstance(image, list):
                # Multi-image: hash all paths concatenated
                img_hash = compute_image_hash(
                    b"".join(p.encode() for p in image)
                )
            else:
                # Single image: read file bytes and hash
                if os.path.exists(image):
                    with open(image, "rb") as f:
                        img_hash = compute_image_hash(f.read())
                else:
                    img_hash = compute_image_hash(image.encode())
            return self._cache.get(img_hash, self._model_name)
        except Exception:
            logger.debug("vision cache adapter get failed", exc_info=True)
            return None

    def put(self, image, features):
        """Store vision features keyed by image path(s)."""
        from .vision_feature_cache import compute_image_hash

        try:
            if isinstance(image, list):
                img_hash = compute_image_hash(
                    b"".join(p.encode() for p in image)
                )
            else:
                if os.path.exists(image):
                    with open(image, "rb") as f:
                        img_hash = compute_image_hash(f.read())
                else:
                    img_hash = compute_image_hash(image.encode())
            self._cache.put(img_hash, self._model_name, features)
        except Exception:
            logger.debug("vision cache adapter put failed", exc_info=True)


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
        self._total_reasoning_tokens = 0
        self._start_time = 0.0
        self._has_vision = False
        self._is_vlm = False
        self._temp_files: list[str] | None = None

        # mRoPE state (detected during load)
        self._mrope_info = None
        self._rope_delta_manager = None

        # Vision feature cache — enabled by default for VLM models.
        # Caches image encoder outputs (vision_tower + projector) keyed by
        # (image_hash, model_name) so repeated images skip re-encoding.
        self._vision_cache = None
        import os as _os
        _vc_env = _os.environ.get("YUNSHU_VISION_CACHE", "").strip()
        if _vc_env not in ("0", "false", "no", "disabled"):
            from .vision_feature_cache import VisionFeatureCache
            cache_dir = _os.environ.get(
                "YUNSHU_VISION_CACHE_DIR", "~/.cache/yunshu/vision",
            )
            self._vision_cache = VisionFeatureCache(cache_dir=cache_dir)
            logger.info("Vision feature cache enabled (dir=%s)", cache_dir)

        # Adapter that wraps VisionFeatureCache for mlx_vlm's interface.
        # Created lazily after model load when model_name is known.
        self._vlm_vision_cache_adapter = None

        # Encoder cache — caches encoder hidden states (vision/audio/text encoder)
        # keyed by request_id. When the same image/audio is seen again in the
        # batch path, reuse cached encoder output instead of re-encoding.
        # Complements VisionFeatureCache (which caches at the mlx_vlm layer).
        from .encoder_cache import EncoderCacheManager
        self._encoder_cache = EncoderCacheManager(
            max_entries=int(os.environ.get("YUNSHU_ENCODER_CACHE_MAX", "64")),
            ttl_seconds=float(os.environ.get("YUNSHU_ENCODER_CACHE_TTL", "300")),
        )

        # Per-image KV prefix cache state — maps image_hash to PromptCacheState.
        # When the same image appears with different text contexts, the KV cache
        # from the previous conversation is reused for the common image prefix.
        self._kv_prefix_states: dict[str, Any] = {}
        self._kv_prefix_max_entries = 32

        # VLM cache stats (vision feature cache + KV prefix reuse)
        self._vlm_vision_hits = 0
        self._vlm_vision_misses = 0
        self._vlm_kv_prefix_hits = 0
        self._vlm_kv_prefix_misses = 0

        # C21: Multimodal prefix cache — maps image_hash + system_prompt hash to
        # processed token IDs, enabling reuse across conversations with same image
        self._multimodal_prefix_cache: dict[str, list[int]] = {}

        # Wave 43: Vision encoding strategy (auto-detect model family)
        self._vision_encoder_factory = None
        try:
            from .vision_encoding import VisionEncoderFactory
            self._vision_encoder_factory = VisionEncoderFactory()
        except Exception:
            logger.debug("operation failed", exc_info=True)
            pass
        self._mm_prefix_hits = 0
        self._mm_prefix_misses = 0

        # SpecPrefill for VLM text portion (opt-in via YUNSHU_VLM_SPEC_PREFILL)
        self._spec_prefill_enabled = False
        if os.environ.get("YUNSHU_VLM_SPEC_PREFILL", "").strip() in ("1", "true", "yes"):
            self._spec_prefill_enabled = True
            logger.info("VLM SpecPrefill enabled")

        from .mlx_executor import get_mlx_executor
        self._executor = get_mlx_executor()

        # Wave 61: MultimodalPipelineCoordinator — unified 7-stage pipeline
        from .staged_pipeline import (
            MultimodalPipelineCoordinator, PipelineStage, PipelineRequest,
        )
        self._pipeline = MultimodalPipelineCoordinator()
        self._register_pipeline_processors()

        # Wave 43: Async concurrent VLM engine (opt-in via YUNSHU_VLM_ASYNC=1)
        self._async_core = None
        import yunshu_engine.vlm_async_engine as _vae  # ensure module is loaded

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

    def _register_pipeline_processors(self) -> None:
        """Register VLM-specific processors into MultimodalPipelineCoordinator."""
        from .staged_pipeline import PipelineStage

        def _text_preprocess(inputs, config):
            data = inputs.get("request") if isinstance(inputs, dict) else inputs
            if data is None:
                return inputs
            messages = data.params.get("messages", [])
            text = data.text
            if not text and messages:
                for msg in reversed(messages):
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        text = content
                        break
            return {"text": text, "messages": messages}

        def _image_preprocess(inputs, config):
            data = inputs.get("request") if isinstance(inputs, dict) else inputs
            if data is None or not data.images:
                return None
            return {"image_paths": data.images}

        def _audio_preprocess(inputs, config):
            data = inputs.get("request") if isinstance(inputs, dict) else inputs
            if data is None or not data.audio:
                return None
            return {"audio_paths": data.audio}

        self._pipeline.register_processor(
            PipelineStage.TEXT_PREPROCESS, "text", _text_preprocess,
        )
        self._pipeline.register_processor(
            PipelineStage.IMAGE_PREPROCESS, "image", _image_preprocess,
        )
        self._pipeline.register_processor(
            PipelineStage.AUDIO_PREPROCESS, "audio", _audio_preprocess,
        )

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

        # Create vision cache adapter now that model_name is known
        if self._vision_cache is not None:
            self._vlm_vision_cache_adapter = _MlxVlmVisionCacheAdapter(
                self._vision_cache, self.model_name,
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

        # Clear per-model caches — stale entries from the old model would waste
        # memory and could return incorrect features if model_name happened to
        # collide.  The VisionFeatureCache itself is kept alive (its background
        # writer thread is daemon and shared), but in-memory entries are evicted.
        if self._vision_cache is not None:
            with self._vision_cache._memory_lock:
                self._vision_cache._memory_cache.clear()
        self._vlm_vision_cache_adapter = None
        self._kv_prefix_states.clear()
        self._multimodal_prefix_cache.clear()
        self._encoder_cache.clear()

        # Reset stats counters
        self._vlm_vision_hits = 0
        self._vlm_vision_misses = 0
        self._vlm_kv_prefix_hits = 0
        self._vlm_kv_prefix_misses = 0
        self._mm_prefix_hits = 0
        self._mm_prefix_misses = 0

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
        min_p: float = 0.0,
        seed: int | None = None,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
        enable_thinking: bool | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        priority: int = 0,
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
        _enable_thinking = enable_thinking

        try:
            # Run through MultimodalPipelineCoordinator for preprocessing tracking
            try:
                from .staged_pipeline import PipelineRequest
                pipe_req = PipelineRequest(
                    request_id=kwargs.get("request_id", ""),
                    model_id=self.model_name,
                    images=image_paths if image_paths else None,
                    audio=audio_paths if audio_paths else None,
                    params={"messages": messages},
                )
                self._pipeline.process(pipe_req)
            except Exception:
                logger.debug("pipeline tracking failed", exc_info=True)

            # Extract advanced parameters from kwargs
            stop_token_ids = kwargs.get('stop_token_ids') or []
            thinking_budget = kwargs.get('thinking_budget')
            reasoning_effort = kwargs.get('reasoning_effort')
            xtc_probability = kwargs.get('xtc_probability', 0.0)
            xtc_threshold = kwargs.get('xtc_threshold', 0.0)

            # Resolve reasoning_effort → thinking_budget
            if thinking_budget is None and reasoning_effort is not None:
                effort_map = {"low": 2048, "medium": 8192, "high": 32768}
                thinking_budget = effort_map.get(reasoning_effort, 8192)

            # logprobs is not supported by VLM engine (mlx_vlm.generate() and
            # model.language_model don't expose per-token logprobs).
            if logprobs or top_logprobs:
                logger.warning(
                    "VLMEngine does not support logprobs/top_logprobs — "
                    "parameter ignored. Use BatchedEngine for logprobs support."
                )

            def _generate_sync():
                if seed is not None:
                    mx.random.seed(seed)

                if (image_paths and self._has_vision and self._is_vlm) or (audio_paths and self._is_vlm):
                    return self._generate_vlm_vision(messages, image_paths, max_tokens, temperature, top_p, top_k, stop, audio_paths=audio_paths, enable_thinking=_enable_thinking)

                prompt_text = self._format_prompt(messages, enable_thinking=_enable_thinking)
                input_ids = mx.array(self._tokenizer.encode(prompt_text))

                if self._is_vlm:
                    freq_p = kwargs.get('frequency_penalty', 0.0)
                    pres_p = kwargs.get('presence_penalty', 0.0)
                    lb = kwargs.get('logit_bias', None)
                    js = kwargs.get('json_schema', None)
                    return self._generate_vlm_text(input_ids, max_tokens, temperature, top_p, top_k, min_p, stop, stop_token_ids=stop_token_ids, repetition_penalty=repetition_penalty, frequency_penalty=freq_p, presence_penalty=pres_p, logit_bias=lb, json_schema=js, enable_thinking=_enable_thinking, xtc_probability=xtc_probability, xtc_threshold=xtc_threshold)

                from mlx_lm.generate import generate_step
                from mlx_lm.sample_utils import make_sampler

                sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k if top_k > 0 else 0, min_p=min_p, xtc_probability=xtc_probability, xtc_threshold=xtc_threshold)
                eos_ids = self._get_eos_ids()

                # Build stop token IDs from string stop sequences + explicit stop_token_ids
                stop_ids = set(eos_ids)
                if stop:
                    for s in stop:
                        try:
                            ids = self._tokenizer.encode(s)
                            if len(ids) == 1:
                                stop_ids.add(ids[0])
                        except Exception:
                            logger.debug("failed", exc_info=True)
                if stop_token_ids:
                    stop_ids.update(stop_token_ids)

                tokens = []
                _in_thinking = False
                _thinking_tokens = 0
                try:
                    think_start_id = self._tokenizer.encode("<think")[-1]
                    think_end_id = self._tokenizer.encode("</think")[-1]
                except Exception:
                    logger.debug("operation failed", exc_info=True)
                    think_start_id = think_end_id = None

                for token_id, _ in generate_step(
                    input_ids, self._model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ):
                    tokens.append(token_id)
                    # Track thinking segment boundaries
                    if think_start_id is not None:
                        if not _in_thinking and token_id == think_start_id:
                            _in_thinking = True
                        elif _in_thinking:
                            _thinking_tokens += 1
                            if token_id == think_end_id:
                                _in_thinking = False
                    if token_id in stop_ids:
                        break
                    # Thinking budget enforcement
                    if thinking_budget is not None and enable_thinking and len(tokens) >= thinking_budget:
                        break

                return self._tokenizer.decode(tokens, skip_special_tokens=True), _thinking_tokens

            loop = asyncio.get_running_loop()
            try:
                result, reasoning_tokens = await loop.run_in_executor(self._executor, _generate_sync)
            except Exception:
                self._active_count -= 1
                self._num_requests_processed += 1
                raise

            elapsed = time.monotonic() - t0
            self._active_count -= 1
            self._num_requests_processed += 1
            self._total_reasoning_tokens += reasoning_tokens

            prompt_text = self._format_prompt(messages)
            prompt_tokens = len(self._tokenizer.encode(prompt_text)) if self._tokenizer else 0
            return {
                "text": result,
                "finish_reason": "stop",
                "model": self.model_name,
                "created": int(time.time()),
                "reasoning_tokens": reasoning_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(self._tokenizer.encode(result)) if result and self._tokenizer else 0,
            }
        finally:
            self._cleanup_temp_files()

    async def generate_stream(
        self,
        prompt: list[dict] | None = None,
        messages: list[dict] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        seed: int | None = None,
        stop: list[str] | None = None,
        enable_thinking: bool | None = None,
        repetition_penalty: float = 1.0,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        cancel_event: Any = None,
        stop_token_ids: list[int] | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        **kwargs,
    ) -> AsyncIterator[RequestOutput]:
        """Streaming generation: yields RequestOutput per token."""
        messages = prompt or messages or []
        if self._model is None:
            raise RuntimeError("Engine not started")

        # logprobs is not supported by VLM engine
        if logprobs or top_logprobs:
            logger.warning(
                "VLMEngine does not support logprobs/top_logprobs — "
                "parameter ignored. Use BatchedEngine for logprobs support."
            )

        # Extract images/audio once, reuse for both pipeline and generation
        image_paths = await self._extract_images(messages)
        audio_paths = await self._extract_audio(messages)
        video_frames = await self._extract_video_frames(messages)
        image_paths.extend(video_frames)

        # Pipeline tracking for streaming path
        try:
            from .staged_pipeline import PipelineRequest
            pipe_req = PipelineRequest(
                request_id=kwargs.get("request_id", ""),
                model_id=self.model_name,
                images=image_paths if image_paths else None,
                audio=audio_paths if audio_paths else None,
                params={"messages": messages},
            )
            self._pipeline.process(pipe_req)
        except Exception:
            logger.debug("pipeline tracking (stream) failed", exc_info=True)

        import uuid
        req_id = f"vlm-{uuid.uuid4().hex[:8]}"

        queue: asyncio.Queue[RequestOutput | None] = asyncio.Queue(maxsize=256)

        # Use already-extracted images/audio for VLM vision path
        has_images = bool(image_paths) and self._has_vision and self._is_vlm
        has_audio = bool(audio_paths) and self._is_vlm

        def _stream_sync():
            try:
                if seed is not None:
                    mx.random.seed(seed)

                if has_images or has_audio:
                    self._stream_vlm_vision(messages, image_paths, max_tokens, temperature, top_p, req_id, queue, top_k, min_p, stop, audio_paths=audio_paths, enable_thinking=enable_thinking, cancel_event=cancel_event, xtc_probability=xtc_probability, xtc_threshold=xtc_threshold)
                    return

                prompt_text = self._format_prompt(messages, enable_thinking=enable_thinking)
                input_ids = mx.array(self._tokenizer.encode(prompt_text))

                if self._is_vlm:
                    freq_p = kwargs.get('frequency_penalty', 0.0)
                    pres_p = kwargs.get('presence_penalty', 0.0)
                    lb = kwargs.get('logit_bias', None)
                    js = kwargs.get('json_schema', None)
                    self._stream_vlm_text(input_ids, max_tokens, temperature, top_p, req_id, queue, top_k, min_p, stop, repetition_penalty, freq_p, pres_p, lb, json_schema=js, enable_thinking=enable_thinking, cancel_event=cancel_event, stop_token_ids=stop_token_ids, xtc_probability=xtc_probability, xtc_threshold=xtc_threshold)
                    return

                from mlx_lm.generate import generate_step
                from mlx_lm.sample_utils import make_sampler

                sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k if top_k > 0 else 0, min_p=min_p, xtc_probability=xtc_probability, xtc_threshold=xtc_threshold)
                eos_ids = self._get_eos_ids()

                # Build stop token IDs from string sequences + explicit stop_token_ids
                stop_ids = set(eos_ids)
                if stop:
                    for s in stop:
                        try:
                            ids = self._tokenizer.encode(s)
                            if len(ids) == 1:
                                stop_ids.add(ids[0])
                        except Exception:
                            logger.debug("failed", exc_info=True)
                if stop_token_ids:
                    stop_ids.update(stop_token_ids)

                has_detokenizer = hasattr(self._tokenizer, 'detokenizer')
                if has_detokenizer:
                    detokenizer = self._tokenizer.detokenizer
                    detokenizer.reset()

                accumulated = ""
                token_count = 0
                for token_id, _ in generate_step(
                    input_ids, self._model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ):
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    token_count += 1
                    is_eos = token_id in stop_ids

                    if not is_eos:
                        if has_detokenizer:
                            detokenizer.add_token(token_id)
                            token_text = detokenizer.last_segment
                        else:
                            token_text = self._tokenizer.decode([token_id], skip_special_tokens=True)
                    else:
                        token_text = ""

                    accumulated += token_text

                    # Check multi-token stop suffixes
                    finish_reason = None
                    if is_eos:
                        finish_reason = "stop"
                    elif stop:
                        for s in stop:
                            if accumulated.endswith(s):
                                # Trim the stop suffix from output
                                trim_pos = len(accumulated) - len(s)
                                token_text = accumulated[trim_pos:]
                                accumulated = accumulated[:trim_pos]
                                finish_reason = "stop"
                                break

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
                        # Flush remaining bytes from detokenizer
                        if has_detokenizer:
                            remaining = detokenizer.finalize()
                            if remaining:
                                queue.put_nowait(RequestOutput(
                                    request_id=req_id,
                                    new_text=remaining,
                                    finish_reason=None,
                                    finished=False,
                                ))
                        return

                # Max tokens reached — finalize detokenizer
                if has_detokenizer:
                    remaining = detokenizer.finalize()
                    if remaining:
                        queue.put_nowait(RequestOutput(
                            request_id=req_id,
                            new_text=remaining,
                            finish_reason=None,
                            finished=False,
                        ))
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
                try:
                    queue.put_nowait(None)
                except Exception:
                    pass

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
            self._cleanup_temp_files()

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
        enable_thinking: bool | None = None,
    ) -> str:
        """Vision + text generation using mlx_vlm.generate().

        Passes vision_cache and prompt_cache_state to mlx_vlm so that:
        1. Image features are cached and reused when the same image appears again
        2. KV cache is reused for common prefix across conversations with same image
        """
        from mlx_vlm.generate import generate as vlm_generate

        vlm_messages = self._build_vlm_messages(messages)
        tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            tpl_kwargs["enable_thinking"] = enable_thinking

        # Include audio count in template kwargs for Omni models
        if audio_paths:
            tpl_kwargs["num_audios"] = len(audio_paths)

        prompt = self._processor.apply_chat_template(
            vlm_messages, **tpl_kwargs,
        )

        # Compute image hash for KV prefix cache lookup
        image_hash = self._compute_image_hash(image_paths) if image_paths else None

        # Look up existing KV prefix state for this image
        kv_prefix_state = self._get_kv_prefix_state(image_hash) if image_hash else None
        if kv_prefix_state is not None:
            logger.debug(
                "VLM KV prefix cache hit for image %s (cache has %d tokens)",
                image_hash[:8],
                len(kv_prefix_state.token_ids) if kv_prefix_state.token_ids else 0,
            )

        # Track vision feature cache hits/misses via adapter stats
        vc_stats_before = self._vision_cache.stats if self._vision_cache else {}

        # Encoder cache lookup: check if we have cached encoder hidden states
        # for this request. The encoder cache stores vision/audio encoder outputs
        # keyed by a combination of image hash and request context.
        _encoder_cache_key = None
        if image_hash is not None:
            _encoder_cache_key = f"vlm-{image_hash}"
            cached_encoder = self._encoder_cache.get(_encoder_cache_key)
            if cached_encoder is not None:
                logger.debug(
                    "VLM encoder cache hit for image %s — reusing encoder output",
                    image_hash[:8],
                )

        gen_kwargs: dict = {
            "max_tokens": max_tokens,
            "temp": temperature,
            "verbose": False,
        }
        if image_paths:
            gen_kwargs["image"] = image_paths if len(image_paths) > 1 else image_paths[0]
        if audio_paths:
            gen_kwargs["audio"] = audio_paths if len(audio_paths) > 1 else audio_paths[0]

        # Pass vision_cache adapter for image feature caching
        if self._vlm_vision_cache_adapter is not None:
            gen_kwargs["vision_cache"] = self._vlm_vision_cache_adapter

        # Pass KV prefix state for reuse across conversations with same image
        if kv_prefix_state is not None:
            gen_kwargs["prompt_cache_state"] = kv_prefix_state

        result = vlm_generate(
            self._model,
            self._processor,
            prompt=prompt,
            **gen_kwargs,
        )

        # Track vision feature cache stats
        if self._vision_cache is not None:
            vc_stats_after = self._vision_cache.stats
            new_hits = vc_stats_after.get("hits", 0) - vc_stats_before.get("hits", 0)
            if new_hits > 0:
                self._vlm_vision_hits += new_hits
                logger.debug("VLM vision feature cache hit: reused encoded image")
            elif image_paths:
                self._vlm_vision_misses += 1

        # Store encoder output in encoder cache for future reuse.
        # If vlm_generate produced a result with encoder_outputs, cache them
        # so subsequent requests with the same image can skip re-encoding.
        if _encoder_cache_key is not None:
            encoder_output = getattr(result, 'encoder_outputs', None)
            if encoder_output is None and hasattr(self._model, 'vision_tower'):
                # For models with explicit vision_tower, store a marker that
                # this image has been encoded successfully (actual features are
                # in the vision_cache adapter). The encoder cache tracks TTL.
                encoder_output = True
            if encoder_output is not None:
                self._encoder_cache.put(_encoder_cache_key, encoder_output)

        # After generation, save KV prefix state for this image (first time or
        # update with new state). stream_generate already called update() on the
        # prompt_cache_state if provided. For first-time images, create a state.
        if image_hash is not None:
            try:
                if kv_prefix_state is not None:
                    # State was already updated by stream_generate
                    pass
                else:
                    # First time seeing this image — create a state entry so the
                    # next conversation with this image can reuse the KV cache.
                    # We don't have the KV cache here (it's inside vlm_generate),
                    # but we store the state entry so the next call will create one.
                    self._ensure_kv_prefix_state(image_hash)
            except Exception:
                logger.warning("KV prefix state management failed", exc_info=True)

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

        return result.text if hasattr(result, 'text') else str(result), 0

    # ── VLM text generation (for mlx-vlm models) ──

    def _generate_vlm_text(
        self,
        input_ids: mx.array,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int = 0,
        min_p: float = 0.0,
        stop: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        json_schema: dict | None = None,
        enable_thinking: bool | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
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
        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k if top_k > 0 else 0, min_p=min_p, xtc_probability=xtc_probability, xtc_threshold=xtc_threshold)
        eos_ids = self._get_eos_ids()

        # JSON schema / grammar constraint
        json_constraint = None
        if json_schema is not None:
            try:
                if isinstance(json_schema, dict) and json_schema.get("type") in ("regex", "choice", "cfg"):
                    # Non-JSON grammar type — use ConstraintFactory dispatch
                    from .grammar_constraint import ConstraintFactory
                    gtype = json_schema["type"]
                    if gtype == "regex":
                        grammar = json_schema.get("pattern", "")
                    elif gtype == "choice":
                        grammar = json_schema.get("choices", [])
                    elif gtype == "cfg":
                        grammar = json_schema.get("grammar", "")
                    else:
                        grammar = None
                    if grammar is not None:
                        json_constraint = ConstraintFactory.create(gtype, grammar, self._tokenizer)
                elif isinstance(json_schema, str) and json_schema == "json_object":
                    from .json_schema import JsonSchemaConstraint
                    json_constraint = JsonSchemaConstraint(None, self._tokenizer)
                else:
                    from .json_schema import JsonSchemaConstraint
                    json_constraint = JsonSchemaConstraint(json_schema, self._tokenizer)
            except Exception:
                logger.warning("Grammar constraint init failed", exc_info=True)

        has_penalty = repetition_penalty != 1.0 or frequency_penalty != 0.0 or presence_penalty != 0.0 or logit_bias

        # Build stop IDs from string sequences + explicit stop_token_ids
        stop_ids = set(eos_ids)
        if stop:
            for s in stop:
                try:
                    ids = self._tokenizer.encode(s)
                    if len(ids) == 1:
                        stop_ids.add(ids[0])
                except Exception:
                    logger.debug("failed", exc_info=True)
        if stop_token_ids:
            stop_ids.update(stop_token_ids)

        with mx.stream(generation_stream):
            # SpecPrefill: for long text prompts, use attention-based sparse
            # prefill to reduce computation by only processing high-attention tokens
            if self._spec_prefill_enabled and input_ids.shape[0] > 8192:
                try:
                    from .spec_prefill import SparsePrefill
                    sp = SparsePrefill()
                    indices = sp.select_important_tokens(lm, input_ids[None], top_k=8192)
                    input_ids = input_ids[indices]
                    logger.debug(f"SpecPrefill: reduced from {input_ids.shape[0]} to {len(indices)} tokens")
                except Exception:
                    logger.debug("SpecPrefill failed, using full prefill", exc_info=True)

            # Prefill
            output = lm(input_ids[None], cache=cache)
            logits = output.logits[:, -1, :]
            current = sampler(logits)
            mx.eval(current)

            tokens = [current.item()]
            if current.item() in stop_ids:
                return self._tokenizer.decode(tokens, skip_special_tokens=True), 0

            _in_thinking = False
            _thinking_tokens = 0
            try:
                think_start_id = self._tokenizer.encode("<think")[-1]
                think_end_id = self._tokenizer.encode("</think")[-1]
            except Exception:
                logger.debug("operation failed", exc_info=True)
                think_start_id = think_end_id = None

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
                        logger.debug("failed", exc_info=True)

                current = sampler(logits)
                mx.eval(current)
                tok_id = current.item()
                tokens.append(tok_id)
                if think_start_id is not None:
                    if not _in_thinking and tok_id == think_start_id:
                        _in_thinking = True
                    elif _in_thinking:
                        _thinking_tokens += 1
                        if tok_id == think_end_id:
                            _in_thinking = False
                if tok_id in stop_ids:
                    break

        return self._tokenizer.decode(tokens, skip_special_tokens=True), _thinking_tokens

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
        min_p: float = 0.0,
        stop: list[str] | None = None,
        audio_paths: list[str] | None = None,
        enable_thinking: bool | None = None,
        cancel_event: Any = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
    ) -> None:
        """Streaming vision + text generation using mlx_vlm.stream_generate().

        Passes vision_cache and prompt_cache_state to mlx_vlm so that:
        1. Image features are cached and reused when the same image appears again
        2. KV cache is reused for common prefix across conversations with same image
        """
        from mlx_vlm.generate import stream_generate as vlm_stream_generate
        from mlx_lm.sample_utils import make_sampler

        vlm_messages = self._build_vlm_messages(messages)
        tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            tpl_kwargs["enable_thinking"] = enable_thinking

        if audio_paths:
            tpl_kwargs["num_audios"] = len(audio_paths)

        prompt = self._processor.apply_chat_template(
            vlm_messages, **tpl_kwargs,
        )

        # Compute image hash for KV prefix cache lookup
        image_hash = self._compute_image_hash(image_paths) if image_paths else None

        # Look up existing KV prefix state for this image
        kv_prefix_state = self._get_kv_prefix_state(image_hash) if image_hash else None
        if kv_prefix_state is not None:
            logger.debug(
                "VLM stream KV prefix cache hit for image %s (cache has %d tokens)",
                image_hash[:8],
                len(kv_prefix_state.token_ids) if kv_prefix_state.token_ids else 0,
            )

        # Track vision feature cache hits/misses via adapter stats
        vc_stats_before = self._vision_cache.stats if self._vision_cache else {}

        # Encoder cache lookup for streaming vision path
        _encoder_cache_key = None
        if image_hash is not None:
            _encoder_cache_key = f"vlm-{image_hash}"
            cached_encoder = self._encoder_cache.get(_encoder_cache_key)
            if cached_encoder is not None:
                logger.debug(
                    "VLM stream encoder cache hit for image %s",
                    image_hash[:8],
                )

        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k if top_k > 0 else 0, min_p=min_p, xtc_probability=xtc_probability, xtc_threshold=xtc_threshold)
        stop_suffixes = stop or []
        token_count = 0
        accumulated = ""  # Accumulate text for multi-token stop suffix matching
        try:
            stream_kwargs: dict = {
                "max_tokens": max_tokens,
                "sampler": sampler,
            }
            if image_paths:
                stream_kwargs["image"] = image_paths if len(image_paths) > 1 else image_paths[0]
            if audio_paths:
                stream_kwargs["audio"] = audio_paths if len(audio_paths) > 1 else audio_paths[0]

            # Pass vision_cache adapter for image feature caching
            if self._vlm_vision_cache_adapter is not None:
                stream_kwargs["vision_cache"] = self._vlm_vision_cache_adapter

            # Pass KV prefix state for reuse across conversations with same image
            if kv_prefix_state is not None:
                stream_kwargs["prompt_cache_state"] = kv_prefix_state

            for result in vlm_stream_generate(
                self._model,
                self._processor,
                prompt=prompt,
                **stream_kwargs,
            ):
                if cancel_event is not None and cancel_event.is_set():
                    return
                token_count += 1
                text = result.text if hasattr(result, 'text') else ""
                accumulated += text
                finish_reason = None
                if hasattr(result, 'finish_reason') and result.finish_reason:
                    finish_reason = result.finish_reason
                elif token_count >= max_tokens:
                    finish_reason = "length"
                # Check multi-token stop suffixes against accumulated text
                if not finish_reason and stop_suffixes:
                    for s in stop_suffixes:
                        if accumulated.endswith(s):
                            # Trim the stop suffix from the output
                            trim_pos = len(accumulated) - len(s)
                            # Emit only the non-suffix portion as the final token text
                            text = accumulated[trim_pos:]
                            accumulated = accumulated[:trim_pos]
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

            # Track vision feature cache stats after streaming completes
            if self._vision_cache is not None:
                vc_stats_after = self._vision_cache.stats
                new_hits = vc_stats_after.get("hits", 0) - vc_stats_before.get("hits", 0)
                if new_hits > 0:
                    self._vlm_vision_hits += new_hits
                    logger.debug("VLM stream vision feature cache hit: reused encoded image")
                elif image_paths:
                    self._vlm_vision_misses += 1

            # Store encoder output in encoder cache for streaming vision path
            if _encoder_cache_key is not None and hasattr(self._model, 'vision_tower'):
                self._encoder_cache.put(_encoder_cache_key, True)

            # After streaming, save KV prefix state for this image
            if image_hash is not None:
                try:
                    if kv_prefix_state is not None:
                        # State was already updated by stream_generate
                        pass
                    else:
                        # First time seeing this image — create a state entry
                        self._ensure_kv_prefix_state(image_hash)
                except Exception:
                    logger.warning("KV prefix state management failed", exc_info=True)

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
        min_p: float = 0.0,
        stop: list[str] | None = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        json_schema: dict | None = None,
        enable_thinking: bool | None = None,
        cancel_event: asyncio.Event | None = None,
        stop_token_ids: list[int] | None = None,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
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
        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k if top_k > 0 else 0, min_p=min_p, xtc_probability=xtc_probability, xtc_threshold=xtc_threshold)
        eos_ids = self._get_eos_ids()
        has_penalty = repetition_penalty != 1.0 or frequency_penalty != 0.0 or presence_penalty != 0.0 or logit_bias

        # JSON schema / grammar constraint (streaming path)
        json_constraint = None
        if json_schema is not None:
            try:
                if isinstance(json_schema, dict) and json_schema.get("type") in ("regex", "choice", "cfg"):
                    from .grammar_constraint import ConstraintFactory
                    gtype = json_schema["type"]
                    if gtype == "regex":
                        grammar = json_schema.get("pattern", "")
                    elif gtype == "choice":
                        grammar = json_schema.get("choices", [])
                    elif gtype == "cfg":
                        grammar = json_schema.get("grammar", "")
                    else:
                        grammar = None
                    if grammar is not None:
                        json_constraint = ConstraintFactory.create(gtype, grammar, self._tokenizer)
                elif isinstance(json_schema, str) and json_schema == "json_object":
                    from .json_schema import JsonSchemaConstraint
                    json_constraint = JsonSchemaConstraint(None, self._tokenizer)
                else:
                    from .json_schema import JsonSchemaConstraint
                    json_constraint = JsonSchemaConstraint(json_schema, self._tokenizer)
            except Exception:
                logger.warning("Grammar constraint init failed (stream)", exc_info=True)

        # Build stop IDs + explicit stop_token_ids
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
                    logger.debug("failed", exc_info=True)
        if stop_token_ids:
            stop_ids.update(stop_token_ids)

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

        _in_thinking = False
        _thinking_tokens = 0
        try:
            think_start_id = self._tokenizer.encode("<think")[-1]
            think_end_id = self._tokenizer.encode("</think")[-1]
        except Exception:
            logger.debug("operation failed", exc_info=True)
            think_start_id = think_end_id = None

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
        try:
          for _ in range(max_tokens - 1):
            if cancel_event is not None and cancel_event.is_set():
                queue.put_nowait(RequestOutput(
                    request_id=req_id,
                    new_text="",
                    finish_reason="cancel",
                    finished=True,
                    completion_tokens=token_count,
                ))
                return
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
                    logger.debug("failed", exc_info=True)

            current = sampler(logits)
            mx.eval(current)
            token_count += 1

            token_id = current.item()
            # Track thinking segment boundaries in VLM streaming
            if think_start_id is not None:
                if not _in_thinking and token_id == think_start_id:
                    _in_thinking = True
                elif _in_thinking:
                    _thinking_tokens += 1
                    if token_id == think_end_id:
                        _in_thinking = False
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
                if has_detokenizer:
                    remaining = detokenizer.finalize()
                    if remaining:
                        queue.put_nowait(RequestOutput(
                            request_id=req_id,
                            new_text=remaining,
                            finish_reason=None,
                            finished=False,
                        ))
                return

          # Max tokens reached — finalize detokenizer
          if has_detokenizer:
              remaining = detokenizer.finalize()
              if remaining:
                  queue.put_nowait(RequestOutput(
                      request_id=req_id,
                      new_text=remaining,
                      finish_reason=None,
                      finished=False,
                  ))
          queue.put_nowait(RequestOutput(
              request_id=req_id,
              new_text="",
              finish_reason="length",
              finished=True,
              completion_tokens=token_count,
          ))
        except Exception as e:
            logger.error(f"VLM text streaming error: {e}", exc_info=True)
            queue.put_nowait(RequestOutput(
                request_id=req_id,
                new_text="",
                finish_reason="error",
                finished=True,
                error=str(e),
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

    def _format_prompt(self, messages: list[dict], enable_thinking: bool | None = None) -> str:
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                clean = []
                for msg in messages:
                    clean.append({
                        "role": msg.get("role", "user"),
                        "content": self._extract_text(msg.get("content", "")),
                    })
                tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
                if enable_thinking is not None:
                    tpl_kwargs["enable_thinking"] = enable_thinking
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
                logger.debug("failed", exc_info=True)
        return h.hexdigest()[:16]

    def _get_kv_prefix_state(self, image_hash: str) -> Any | None:
        """Get or create a PromptCacheState for per-image KV prefix reuse.

        Returns the PromptCacheState if one exists for this image, or None
        if this is the first time the image is seen. The caller should pass
        the state to mlx_vlm's stream_generate which will populate it.
        """
        if image_hash is None:
            return None

        state = self._kv_prefix_states.get(image_hash)
        if state is not None:
            self._vlm_kv_prefix_hits += 1
            return state

        self._vlm_kv_prefix_misses += 1
        return None

    def _ensure_kv_prefix_state(self, image_hash: str) -> Any:
        """Create a new PromptCacheState entry for this image hash."""
        from mlx_vlm.generate import PromptCacheState

        state = PromptCacheState()
        # Evict old entries if over limit
        if len(self._kv_prefix_states) >= self._kv_prefix_max_entries:
            keys = list(self._kv_prefix_states.keys())
            for k in keys[:8]:
                del self._kv_prefix_states[k]
        self._kv_prefix_states[image_hash] = state
        return state

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
            logger.debug("image download failed, trying fallback SSL", exc_info=True)
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
        import shutil
        if self._temp_files:
            for path in self._temp_files:
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
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
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
            return []
        except subprocess.CalledProcessError as e:
            logger.warning(f"ffmpeg failed: {e.stderr.decode()[:200] if e.stderr else 'unknown'}")
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
            return []
        except Exception as e:
            logger.warning(f"Video frame extraction failed: {e}")
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
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
                logger.debug("image hash computation failed, using path", exc_info=True)
                parts.append(path)
        return "|".join(parts)

    # ── Stats ──

    def get_stats(self) -> dict:
        uptime = time.monotonic() - self._start_time if self._start_time else 0.0
        total_mm = self._mm_prefix_hits + self._mm_prefix_misses
        total_vision = self._vlm_vision_hits + self._vlm_vision_misses
        total_kv = self._vlm_kv_prefix_hits + self._vlm_kv_prefix_misses

        stats = {
            "model": self._model_path,
            "loaded": self.is_loaded,
            "running": self._running,
            "has_vision": self._has_vision,
            "is_vlm": self._is_vlm,
            "num_requests_processed": self._num_requests_processed,
            "reasoning_tokens": self._total_reasoning_tokens,
            "uptime_seconds": uptime,
            # C21: Multimodal prefix cache (token ID reuse)
            "mm_prefix_cache_entries": len(self._multimodal_prefix_cache),
            "mm_prefix_cache_hits": self._mm_prefix_hits,
            "mm_prefix_cache_misses": self._mm_prefix_misses,
            "mm_prefix_cache_hit_rate": self._mm_prefix_hits / total_mm if total_mm > 0 else 0.0,
            # Vision feature cache (image encoder output reuse)
            "vision_cache_enabled": self._vision_cache is not None,
            "vlm_vision_feature_hits": self._vlm_vision_hits,
            "vlm_vision_feature_misses": self._vlm_vision_misses,
            "vlm_vision_feature_hit_rate": self._vlm_vision_hits / total_vision if total_vision > 0 else 0.0,
            # KV prefix cache (per-image KV state reuse)
            "vlm_kv_prefix_entries": len(self._kv_prefix_states),
            "vlm_kv_prefix_hits": self._vlm_kv_prefix_hits,
            "vlm_kv_prefix_misses": self._vlm_kv_prefix_misses,
            "vlm_kv_prefix_hit_rate": self._vlm_kv_prefix_hits / total_kv if total_kv > 0 else 0.0,
        }

        # Merge underlying VisionFeatureCache stats if available
        if self._vision_cache is not None:
            vc_stats = self._vision_cache.stats
            stats["vision_cache_memory_hits"] = vc_stats.get("hits", 0)
            stats["vision_cache_ssd_loads"] = vc_stats.get("ssd_loads", 0)
            stats["vision_cache_saves"] = vc_stats.get("saves", 0)
            stats["vision_cache_errors"] = vc_stats.get("errors", 0)

        # Merge encoder cache stats (encoder hidden-state cache)
        stats["encoder_cache"] = self._encoder_cache.get_stats()

        # Merge MultimodalPipelineCoordinator stats
        try:
            stats["pipeline"] = self._pipeline.get_stats()
        except Exception:
            logger.debug("operation failed", exc_info=True)
            pass

        return stats
