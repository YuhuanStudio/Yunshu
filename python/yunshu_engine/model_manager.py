from __future__ import annotations

"""Yunshu Model Manager — Multi-model serving with LRU eviction.

Manages multiple loaded models with memory-aware lifecycle:
- Lazy loading on first request
- LRU eviction when memory pressure exceeds threshold
- Pinned models that are never evicted
- Memory settle barrier (verify MLX actually freed memory)
- Support for multiple engine types: LLM, VLM, TTS, ASR, ImageGen

Model type detection uses dynamic library probing instead of static lists:
- Tries mlx-lm first (import mlx_lm.models.{model_type})
- Falls back to config structure heuristics (vision_config, etc.)
- This gives 0-day support for any new model mlx-lm adds
"""


import asyncio
import gc
import importlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

import mlx.core as mx

from .types import EngineConfig

logger = logging.getLogger(__name__)


class ModelType(Enum):
    LLM = auto()
    VLM = auto()
    TTS = auto()
    ASR = auto()
    IMAGE_GEN = auto()
    OCR = auto()
    STS = auto()
    VIDEO = auto()


# mlx-lm's MODEL_REMAPPING (subset we need to replicate for probing)
_MODEL_TYPE_REMAP = {
    "mistral": "llama",
    "llava": "mistral3",
    "phi-msft": "phixtral",
    "falcon_mamba": "mamba",
    "joyai_llm_flash": "deepseek_v3",
    "kimi_k2": "deepseek_v3",
    "qwen2_5_vl": "qwen2_vl",
    "minimax_m2": "minimax",
    "iquestcoder": "llama",
}


def _detect_model_type(model_path: str) -> ModelType:
    """Detect model type from config files.

    Dynamic detection strategy (0-day support):
    1. Image gen: model_index.json or diffusion directory structure
    2. Read config.json → model_type → try mlx-lm import
    3. If mlx-lm supports it, check for vision_config → VLM vs LLM
    4. If mlx-lm doesn't support it, check for ASR/TTS heuristics
    5. Default: LLM
    """
    p = Path(model_path)

    # Image gen: diffusion pipeline models have model_index.json
    if (p / "model_index.json").exists():
        return ModelType.IMAGE_GEN
    if all((p / d).is_dir() for d in ("transformer", "vae", "text_encoder")):
        return ModelType.IMAGE_GEN

    config_path = p / "config.json"
    if not config_path.exists():
        # No config — detect from directory name heuristics
        name_lower = p.name.lower()
        if "vae" in name_lower:
            return ModelType.IMAGE_GEN
        if "depth" in name_lower or "depth_anything" in name_lower:
            return ModelType.IMAGE_GEN
        if "tts" in name_lower or "voice" in name_lower:
            return ModelType.TTS
        if "asr" in name_lower or "whisper" in name_lower:
            return ModelType.ASR
        if "vision" in name_lower or "vlm" in name_lower or "omni" in name_lower:
            return ModelType.VLM
        if "image" in name_lower or "flux" in name_lower or "z-image" in name_lower:
            return ModelType.IMAGE_GEN
        if any(k in name_lower for k in ("wan", "ltx", "video")):
            return ModelType.VIDEO
        return ModelType.LLM

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError):
        return ModelType.LLM

    model_type = config.get("model_type", "").lower().replace("-", "_")
    architectures = config.get("architectures", [])
    name_lower = p.name.lower()

    # Lance dual-expert tower models (ByteDance): config.json has
    # model_type=qwen2_5_vl + vision_config because LLM_UND is based on
    # Qwen2.5-VL, but these are image/video generation models, not VLM.
    # Signal: conversion_report.json with "lance" variant + vae.safetensors.
    if (p / "conversion_report.json").exists() and (p / "vae.safetensors").exists():
        try:
            with open(p / "conversion_report.json") as _cr:
                _variant = json.load(_cr).get("variant", "")
            if "lance" in _variant:
                if "video" in _variant or "video" in name_lower:
                    return ModelType.VIDEO
                return ModelType.IMAGE_GEN
        except (json.JSONDecodeError, OSError):
            pass

    # Depth estimation models (Depth Anything 3 / DA3Nested): non-standard
    # config.json with depth_anything_3 module paths, no standard model_type.
    if not model_type or model_type == "":
        _model_name = config.get("model_name", "").lower()
        _config_str = json.dumps(config).lower()
        if any(
            k in name_lower for k in ("da3nested", "depth_anything", "depth-anything")
        ):
            return ModelType.IMAGE_GEN
        if "depth_anything" in _model_name or "depth_anything" in _config_str:
            return ModelType.IMAGE_GEN

    # Image gen: known diffusion model types
    if model_type in ("flux", "sd3", "sdxl", "z_image", "stable-diffusion", "lance"):
        return ModelType.IMAGE_GEN
    if "vae" in model_type:
        return ModelType.IMAGE_GEN
    if model_type.startswith("gemma4") and "assistant" in model_type:
        return ModelType.VLM
    for arch in architectures:
        if any(k in arch for k in ("Transformer2D", "Lance", "DepthAnything")):
            return ModelType.IMAGE_GEN

    # Check for ASR/TTS by architecture keywords
    for arch in architectures:
        arch_lower = arch.lower()
        if "speech" in arch_lower and "text" in arch_lower:
            if "tts" in arch_lower or "synthes" in arch_lower:
                return ModelType.TTS
            if (
                "asr" in arch_lower
                or "recogni" in arch_lower
                or "whisper" in arch_lower
            ):
                return ModelType.ASR
        if "tts" in arch_lower:
            return ModelType.TTS

    # Check model_type for known ASR/TTS types
    if model_type in ("whisper", "qwen3_asr", "parakeet", "qwen2_audio"):
        return ModelType.ASR
    if model_type in ("qwen3_tts", "kokoro", "chatterbox"):
        return ModelType.TTS

    # Keyword-based ASR/TTS/STS fallback
    if any(k in model_type for k in ("tts", "voice", "speech_synthes")):
        return ModelType.TTS
    if any(k in model_type for k in ("whisper", "asr", "speech_to_text")):
        return ModelType.ASR
    if any(
        k in model_type
        for k in (
            "sts",
            "speech_to_speech",
            "deepfilter",
            "mossformer",
            "voice_conversion",
        )
    ):
        return ModelType.STS
    if any(k in model_type for k in ("video", "text_to_video", "image_to_video")):
        return ModelType.VIDEO

    # OCR models (e.g. GLM-OCR) carry a vision_config, so they MUST be detected
    # as OCR before the has_vision→VLM branch below — otherwise they load as a
    # generic VLM and /v1/ocr falls back to VLM-chat, producing garbage output
    # (the model needs OCREngine's task-prompt + vision handling). Found in live
    # testing: GLM-OCR returned empty markdown fences.
    if (
        "ocr" in model_type
        or "ocr" in name_lower
        or any("ocr" in arch.lower() for arch in architectures)
    ):
        return ModelType.OCR

    # Try mlx-lm — if it can import the model_type, it's supported
    remapped = _MODEL_TYPE_REMAP.get(model_type, model_type)
    mlx_lm_supported = False
    try:
        importlib.import_module(f"mlx_lm.models.{remapped}")
        mlx_lm_supported = True
    except ImportError:
        pass

    # Also check mlx-vlm for VLM/Omni model types
    mlx_vlm_supported = False
    try:
        importlib.import_module(f"mlx_vlm.models.{remapped}")
        mlx_vlm_supported = True
    except ImportError:
        pass

    # Check for VLM indicators (works for both mlx-lm and mlx-vlm)
    has_vision = (
        "vision_config" in config
        or config.get("thinker_config", {}).get("vision_config") is not None
    )
    for arch in architectures:
        if any(
            k in arch.lower() for k in ("vl", "vision", "omni", "florence", "pixtral")
        ):
            has_vision = True
    if remapped in ("qwen2_vl", "qwen3_vl", "qwen3_vl_moe", "mistral3", "pixtral"):
        has_vision = True

    if has_vision:
        return ModelType.VLM

    if not mlx_lm_supported and not mlx_vlm_supported:
        # Neither mlx-lm nor mlx-vlm supports this model_type
        # Check heuristics for ASR/TTS from directory name
        name_lower = p.name.lower()
        if "vae" in name_lower:
            return ModelType.IMAGE_GEN
        if "depth" in name_lower or "depth_anything" in name_lower:
            return ModelType.IMAGE_GEN
        if "tts" in name_lower or "voice" in name_lower:
            return ModelType.TTS
        if "asr" in name_lower or "whisper" in name_lower:
            return ModelType.ASR
        if "ocr" in name_lower:
            return ModelType.OCR
        if "sts" in name_lower or "speech_to_speech" in name_lower:
            return ModelType.STS
        if any(k in name_lower for k in ("video", "text_to_video", "image_to_video")):
            return ModelType.VIDEO
        return ModelType.LLM

    # Supported by mlx-lm or mlx-vlm, no vision indicators → standard LLM
    return ModelType.LLM


@dataclass
class ModelEntry:
    """Tracks a model's lifecycle in the pool."""

    model_id: str
    model_path: str
    model_type: ModelType = ModelType.LLM
    estimated_bytes: int = 0
    engine: Any = None  # Engine | TTSEngine | ASREngine | VLMEngine | ImageGenEngine
    last_access: float = 0.0
    is_loading: bool = False
    is_pinned: bool = False
    is_loaded: bool = False
    load_error: str | None = None
    settings: Any = None  # ModelSettings — loaded lazily
    load_time: float = (
        0.0  # monotonic timestamp when model was loaded (for /models endpoint)
    )


class ModelManager:
    """Manages multiple inference engines with memory-aware loading.

    Design decisions:
    - Uses asyncio.Lock (our engine is async-native)
    - Memory settle barrier with tighter polling
    - Per-model engine isolation
    - Per-model loading events so concurrent requests wait instead of failing
    - Post-load memory pressure check triggers LRU eviction
    """

    # Default threshold for post-load memory pressure eviction.
    # When MLX active memory exceeds this fraction of the max working set,
    # the manager will proactively evict the least-recently-used model.
    _MEMORY_PRESSURE_THRESHOLD = 0.90

    def __init__(
        self,
        max_memory_bytes: int | None = None,
        kv_reserve_ratio: float = 0.25,
        settle_timeout_s: float = 5.0,
        ttl_seconds: float | None = None,
        max_models: int = 0,
        memory_pressure_threshold: float = 0.90,
    ) -> None:
        self.max_memory_bytes = max_memory_bytes  # None = unlimited
        self.kv_reserve_ratio = kv_reserve_ratio
        self.settle_timeout_s = settle_timeout_s
        self.ttl_seconds = ttl_seconds
        self.max_models = max_models  # 0 = unlimited
        self.memory_pressure_threshold = memory_pressure_threshold

        self._entries: dict[str, ModelEntry] = {}
        self._current_memory_bytes: int = 0
        self._lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
        # Per-model loading events: concurrent requests for the same model
        # wait on this event instead of raising RuntimeError
        self._loading_events: dict[str, asyncio.Event] = {}
        self._eviction_stats: dict[str, int] = {
            "pressure_evictions": 0,
            "budget_evictions": 0,
            "slot_evictions": 0,
        }
        self._shutting_down = False

    @staticmethod
    def _get_mlx_executor():
        from .mlx_executor import get_mlx_executor

        return get_mlx_executor()

    def register_model(
        self,
        model_id: str,
        model_path: str,
        estimated_bytes: int = 0,
        pinned: bool = False,
        model_type: ModelType | None = None,
    ) -> None:
        """Register a model (does not load it).

        Auto-detects model type from config files if not specified.
        Logs a warning if the model's estimated size exceeds the memory budget,
        but still registers it (it may be loadable after evicting others).
        """
        with self._sync_lock:
            if model_type is None:
                model_type = _detect_model_type(model_path)

            # Load per-model settings from model_settings.json + env vars
            from .model_settings import load_model_settings

            settings = load_model_settings(model_path, model_id)

            # Memory budget warning: if this single model exceeds the budget,
            # log early so operators know it will never fit
            if self.max_memory_bytes and estimated_bytes > self.max_memory_bytes:
                logger.warning(
                    "Model '%s' estimated size (%.1fGB) exceeds memory budget (%.1fGB) "
                    "— loading will fail unless budget is increased or other models evicted",
                    model_id,
                    estimated_bytes / 1e9,
                    self.max_memory_bytes / 1e9,
                )

            # Guard: never overwrite a loaded entry — would leak the engine's GPU memory
            # Also never overwrite a loading entry — would orphan the loading event
            # and cause concurrent get_engine() waiters to hang forever.
            existing = self._entries.get(model_id)
            if existing is not None and (existing.is_loaded or existing.is_loading):
                logger.warning(
                    "register_model('%s'): %s, skipping re-registration",
                    model_id,
                    "already loaded" if existing.is_loaded else "currently loading",
                )
                return

            # Preserve pinned status when re-registering an existing entry
            preserve_pinned = existing.is_pinned if existing is not None else False

            self._entries[model_id] = ModelEntry(
                model_id=model_id,
                model_path=model_path,
                estimated_bytes=estimated_bytes,
                is_pinned=pinned or preserve_pinned,
                model_type=model_type,
                settings=settings,
            )
            logger.info(f"Registered model: {model_id} (type={model_type.name})")

    async def get_engine(
        self,
        model_id: str,
        engine_config: EngineConfig | None = None,
    ) -> Any:
        """Get an engine for the model, loading it if necessary.

        Handles:
        - LRU eviction of other models if memory is tight
        - Concurrent load protection via per-model Event (waiters block
          until the first loader completes, instead of raising RuntimeError)
        - Memory settle barrier after unload
        - Different engine types (LLM, VLM, TTS, ASR, ImageGen)
        """
        if self._shutting_down:
            raise RuntimeError("ModelManager is shutting down, cannot get engine")
        entry = self._entries.get(model_id)
        if entry is None:
            raise KeyError(f"Model not registered: {model_id}")

        # Already loaded — snapshot under lock to prevent TOCTOU with unload
        with self._sync_lock:
            if entry.is_loaded and entry.engine is not None:
                engine = entry.engine
                entry.last_access = time.monotonic()
                return engine

        # Check if another coroutine is already loading this model.
        # Wait on the per-model event instead of raising RuntimeError.
        if entry.is_loading and model_id in self._loading_events:
            load_event = self._loading_events[model_id]
            logger.info(
                "Waiting for model '%s' load to complete (concurrent request)", model_id
            )
            await load_event.wait()
            # Re-check: the load may have succeeded or failed
            if entry.is_loaded and entry.engine is not None:
                entry.last_access = time.monotonic()
                return entry.engine
            if entry.load_error:
                raise RuntimeError(f"Model {model_id} load failed: {entry.load_error}")
            raise KeyError(f"Model {model_id} not available after load attempt")

        async with self._lock:
            # Double-check after acquiring lock
            if entry.is_loaded and entry.engine is not None:
                entry.last_access = time.monotonic()
                return entry.engine

            if entry.is_loading:
                # Another holder of the lock is loading — wait on the
                # per-model event instead of raising RuntimeError.
                # This can happen when two coroutines race to acquire
                # self._lock after the outer is_loading check succeeds
                # for both (before either creates the loading event).
                load_event = self._loading_events.get(model_id)
                if load_event is not None:
                    logger.info(
                        "Inner is_loading race for '%s' — waiting on load event",
                        model_id,
                    )
                    # Release self._lock while waiting to avoid deadlock
                    # with the loader coroutine that also holds self._lock.
                    self._lock.release()
                    try:
                        await load_event.wait()
                    finally:
                        await self._lock.acquire()
                    if entry.is_loaded and entry.engine is not None:
                        entry.last_access = time.monotonic()
                        return entry.engine
                    if entry.load_error:
                        raise RuntimeError(
                            f"Model {model_id} load failed: {entry.load_error}"
                        )
                    raise KeyError(f"Model {model_id} not available after load attempt")
                raise RuntimeError(f"Model {model_id} is already being loaded")

            # Check memory budget
            if self.max_memory_bytes is not None:
                required = entry.estimated_bytes
                kv_headroom = (
                    0
                    if entry.model_type in (ModelType.TTS, ModelType.ASR)
                    else int(required * self.kv_reserve_ratio)
                )
                total_needed = required + kv_headroom
                await self._ensure_memory_available(total_needed)

            # Check max_models limit
            await self._ensure_model_slot_available()

            # Create per-model loading event so waiters can block on us
            load_event = asyncio.Event()
            self._loading_events[model_id] = load_event

            # Load using the appropriate engine type
            entry.is_loading = True
            try:
                engine = await self._create_and_load_engine(entry, engine_config)

                entry.engine = engine
                entry.is_loaded = True
                entry.is_loading = False
                entry.last_access = time.monotonic()
                entry.load_time = time.time()  # wall-clock for /models endpoint
                entry.load_error = None

                # _current_memory_bytes is also mutated by
                # track_lora_memory on the MLX executor THREAD under _sync_lock.
                # The asyncio _lock held here does not exclude that thread, so a
                # bare += could lose a concurrent LoRA delta. Make _sync_lock the
                # single authority for this counter (short, non-blocking).
                with self._sync_lock:
                    self._current_memory_bytes += entry.estimated_bytes

                # Register model ownership in ModelRegistry to prevent
                # BatchKVCache conflicts when multiple engines share a model
                try:
                    from .model_registry import get_registry

                    model_obj = getattr(engine, "_model", None)
                    if model_obj is not None:
                        get_registry().acquire(
                            model_obj,
                            engine,
                            f"model_manager:{model_id}",
                        )
                except Exception:
                    logger.debug("model_registry acquire failed", exc_info=True)

                # Post-load cache clear (weight loading creates large
                # Metal buffer temporaries that stay in the buffer pool)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    self._get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )

                # Post-load memory pressure check:
                # After loading a new model, actual MLX active memory may
                # exceed safe thresholds even when no explicit budget was set.
                # This triggers LRU eviction of the oldest idle model.
                await self._check_post_load_memory_pressure(exclude_model_id=model_id)

                logger.info(
                    f"Loaded model {model_id} "
                    f"({entry.model_type.name}, "
                    f"{entry.estimated_bytes / 1e9:.1f} GB, "
                    f"total: {self._current_memory_bytes / 1e9:.1f} GB)"
                )

                return engine

            except Exception as e:
                entry.is_loading = False
                entry.is_loaded = False
                entry.engine = None
                entry.load_error = str(e)
                logger.error(f"Failed to load model {model_id}: {e}")
                # Clean up any partial Metal buffers from the failed load.
                # Weight loading may have allocated Metal buffer temporaries
                # that stay in the buffer pool even after the Python exception.
                gc.collect()
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        self._get_mlx_executor(),
                        lambda: (mx.synchronize(), mx.clear_cache()),
                    )
                except Exception:
                    logger.debug("post-failure cache clear failed", exc_info=True)
                raise

            finally:
                # Signal any waiters that loading is done (success or failure)
                load_event.set()
                self._loading_events.pop(model_id, None)

    async def _create_and_load_engine(
        self,
        entry: ModelEntry,
        config: EngineConfig | None,
    ) -> Any:
        """Create the right engine type and load it."""
        asyncio.get_running_loop()

        if entry.model_type == ModelType.TTS:
            from .audio_engine import TTSEngine

            engine = TTSEngine(entry.model_path, config)
            await engine.start()
            return engine

        elif entry.model_type == ModelType.ASR:
            from .audio_engine import ASREngine

            engine = ASREngine(entry.model_path, config)
            await engine.start()
            return engine

        elif entry.model_type == ModelType.VLM:
            from .vlm_engine import VLMEngine

            engine = VLMEngine(entry.model_path, config)
            await engine.start()
            return engine

        elif entry.model_type == ModelType.IMAGE_GEN:
            from .image_engine import ImageGenEngine

            engine = ImageGenEngine(entry.model_path, config)
            await engine.start()
            return engine

        elif entry.model_type == ModelType.STS:
            from .sts_engine import STSEngine

            engine = STSEngine(entry.model_path, config)
            engine.start()
            return engine

        elif entry.model_type == ModelType.VIDEO:
            from .video_engine import VideoEngine

            engine = VideoEngine(entry.model_path, config)
            engine.start()
            return engine

        elif entry.model_type == ModelType.OCR:
            # OCR models (GLM-OCR) load via mlx-vlm (with the GlmOcrProcessor
            # patch), NOT mlx-lm — the LLM default raises "model type glm_ocr not
            # supported". OCREngine installs the processor patch so pixel_values
            # reach the model.
            from .ocr_engine import OCREngine

            engine = OCREngine(entry.model_path)
            await engine.start()
            return engine

        else:
            # Default: LLM engine (BatchedEngine with EngineCore backend)
            from .batched_engine import BatchedEngine

            engine = BatchedEngine(
                model_name=entry.model_path,
                stream_interval=getattr(config, "stream_interval", 1) if config else 1,
            )
            await engine.start()
            return engine

    async def unload_model(self, model_id: str, force: bool = False) -> bool:
        """Unload a model and reclaim memory.

        Acquires self._lock for the critical section, then does the
        expensive GC/cache-clear outside the lock.

        Idempotent: safe to call on already-unloaded or non-existent models.
        ``force=True`` (shutdown) tears down even with active requests; the
        default skips an in-use model to avoid crashing an in-flight generation.

        Returns True if the model was actually unloaded, False if the unload was
        skipped (already unloaded / not found, or refused because the model has
        active requests). The caller relies on this to avoid reporting a
        false "unloaded" when the unload was actually a no-op/refusal.
        """
        async with self._lock:
            return await self._unload_model_locked(model_id, force=force)

    async def _unload_model_locked(self, model_id: str, force: bool = False) -> bool:
        """Internal unload — caller MUST hold self._lock. Returns True if unloaded.

        Unload sequence:
        - Stop engine, clear reference BEFORE settle barrier
        - gc.collect() + sync + clear_cache on MLX executor
        - Poll mx.get_active_memory() until Metal buffers released
        """
        entry = self._entries.get(model_id)
        if entry is None or not entry.is_loaded:
            return False

        # CRITICAL: re-check active requests at the single unload choke
        # point (under the lock), so NO caller can tear down an in-use model. The
        # callers that select by _find_lru_victim check this, but the
        # process_memory_enforcer DROPS the lock between victim selection and
        # unload (TOCTOU), and check_ttl never checked at all (last_access is only
        # updated at get_engine time, so a long stream has a stale timestamp).
        # Tearing down a model mid-generation crashes the in-flight request.
        # shutdown passes force=True.
        if not force and entry.engine is not None:
            # FAIL SAFE. Only LLM/VLM engines define has_active_requests;
            # TTS/ASR/Image/STS/Video/OCR engines don't, so the old `try/except:
            # debug` SWALLOWED the AttributeError and proceeded to unload — tearing
            # down a model mid-generation (e.g. a long image/video gen) and null-
            # dereferencing its now-None weights → crash. When we cannot PROVE the
            # engine is idle, refuse a non-forced unload (TTL / memory-pressure /
            # enforcer paths). Force unload (admin / shutdown) still proceeds.
            _has_active = getattr(entry.engine, "has_active_requests", None)
            if not callable(_has_active):
                logger.warning(
                    "Skipping non-forced unload of '%s' — engine type %s lacks "
                    "has_active_requests, cannot prove it is idle.",
                    model_id,
                    type(entry.engine).__name__,
                )
                return False
            try:
                _active = _has_active()
            except Exception:
                logger.debug(
                    "has_active_requests check failed; assuming active", exc_info=True
                )
                _active = True
            if _active:
                logger.warning(
                    "Skipping unload of '%s' — it has active requests (would "
                    "crash an in-flight generation).",
                    model_id,
                )
                return False

        # Clean up any pending loading event
        load_event = self._loading_events.pop(model_id, None)
        if load_event is not None:
            load_event.set()

        # Mark as not loaded early to prevent concurrent get_engine() from
        # seeing an inconsistent state (loading=False, is_loaded=True, engine=None)
        entry.is_loaded = False

        pre_unload_active = mx.get_active_memory()
        estimated_bytes = entry.estimated_bytes

        if entry.engine is not None:
            # Release model ownership in ModelRegistry
            try:
                from .model_registry import get_registry

                model_obj = getattr(entry.engine, "_model", None)
                if model_obj is not None:
                    get_registry().release(
                        model_obj,
                        f"model_manager:{model_id}",
                    )
            except Exception:
                logger.debug("model_registry release failed", exc_info=True)

            # Unload any active LoRA adapters before stopping the engine.
            # Without this, the LoRA manager's state (active adapter refs,
            # memory tracking) is orphaned when the engine is destroyed,
            # and the memory callback points to a stale model_manager that
            # may have already been reused for a different model.
            try:
                lora_mgr = getattr(entry.engine, "_lora_manager", None)
                if lora_mgr is not None:
                    for adapter_id in list(lora_mgr._adapters.keys()):
                        try:
                            lora_mgr.unload_adapter(adapter_id)
                        except Exception:
                            logger.debug(
                                "LoRA unload failed during model eviction: %s",
                                adapter_id,
                                exc_info=True,
                            )
                    lora_mgr.set_memory_callback(None)
            except Exception:
                logger.debug("LoRA cleanup during model eviction failed", exc_info=True)

            try:
                _stop_result = entry.engine.stop()
                if asyncio.iscoroutine(_stop_result):
                    await _stop_result
            except Exception as e:
                logger.warning(f"Error stopping engine for {model_id}: {e}")
            entry.engine = None

        # Clear load_error so a subsequent load gets a clean slate
        entry.load_error = None
        entry.load_time = 0.0

        # Clear stale Prometheus gauge labels for this model to prevent
        # ghost metric series accumulating after model unload.
        try:
            from yunshu_gateway.middleware.prometheus_exporter import (
                get_prometheus_metrics,
            )

            get_prometheus_metrics().clear_model_labels(model_id)
        except Exception:
            pass

        # Force GC + clear cache on MLX executor
        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._get_mlx_executor(),
            lambda: (mx.synchronize(), mx.clear_cache()),
        )

        # Memory settle barrier: poll until Metal buffers actually released
        settle_tolerance = max(2 * 1024**3, int(estimated_bytes * 0.05))
        min_expected_freed = max(0, estimated_bytes - settle_tolerance)
        settled = False

        for _ in range(10):
            active_now = mx.get_active_memory()
            actual_freed = pre_unload_active - active_now
            if actual_freed >= min_expected_freed:
                settled = True
                break
            await asyncio.sleep(0.5)
            gc.collect()
            await loop.run_in_executor(
                self._get_mlx_executor(),
                lambda: (mx.synchronize(), mx.clear_cache()),
            )

        # Update memory tracker. When estimated_bytes is 0 (unknown), use
        # the actual measured delta from the settle barrier instead of
        # subtracting 0 (which would never decrease the tracker).
        # Guard under _sync_lock — same counter is mutated by
        # track_lora_memory on the MLX executor thread.
        with self._sync_lock:
            if estimated_bytes > 0:
                self._current_memory_bytes = max(
                    0, self._current_memory_bytes - estimated_bytes
                )
            else:
                actual_freed = max(0, pre_unload_active - mx.get_active_memory())
                self._current_memory_bytes = max(
                    0, self._current_memory_bytes - actual_freed
                )

            # Fallback: if the settle barrier did not converge within the
            # polling loop, reconcile the tracker against the sum of estimated
            # sizes of currently loaded models.  This prevents tracker drift
            # (from inaccurate estimates or MLX not releasing memory promptly)
            # from accumulating across load/unload cycles and eventually
            # blocking all future loads in _ensure_memory_available.
            if not settled:
                self._current_memory_bytes = sum(
                    e.estimated_bytes
                    for e in self._entries.values()
                    if e.is_loaded and e.estimated_bytes > 0
                )

        logger.info(
            f"Unloaded model {model_id} "
            f"(freed: {(pre_unload_active - mx.get_active_memory()) / 1e9:.1f}GB, "
            f"settled: {settled})"
        )
        return True

    def track_lora_memory(self, delta_bytes: int) -> None:
        """Track LoRA adapter memory delta in the total memory accounting."""
        with self._sync_lock:
            new_val = self._current_memory_bytes + delta_bytes
            if new_val < 0:
                logger.warning(
                    "track_lora_memory underflow: %d + %d = %d, clamping to 0",
                    self._current_memory_bytes,
                    delta_bytes,
                    new_val,
                )
                new_val = 0
            self._current_memory_bytes = new_val

    async def _ensure_memory_available(self, needed_bytes: int) -> None:
        """Evict LRU models until enough memory is free.

        Caller MUST hold self._lock. Uses _unload_model_locked to avoid
        deadlock since asyncio.Lock is not reentrant.
        """
        if self.max_memory_bytes is None:
            return

        while self._current_memory_bytes + needed_bytes > self.max_memory_bytes:
            victim = self._find_lru_victim()
            if victim is None:
                raise MemoryError(
                    f"Cannot free enough memory: need {needed_bytes / 1e9:.1f} GB, "
                    f"used {self._current_memory_bytes / 1e9:.1f} / "
                    f"{self.max_memory_bytes / 1e9:.1f} GB"
                )
            # check the result — _unload_model_locked refuses (returns False,
            # frees nothing) for any engine it cannot prove idle. Spinning on a refused
            # victim is an infinite loop holding self._lock. Selection (_find_lru_victim)
            # now excludes un-evictable engines, but guard here too so no refusal path can
            # ever livelock; count the stat only on a real eviction.
            if not await self._unload_model_locked(victim.model_id):
                raise MemoryError(
                    f"Cannot free enough memory: LRU victim '{victim.model_id}' could "
                    f"not be evicted (engine cannot be proven idle). Need "
                    f"{needed_bytes / 1e9:.1f} GB, used "
                    f"{self._current_memory_bytes / 1e9:.1f} / {self.max_memory_bytes / 1e9:.1f} GB"
                )
            self._eviction_stats["budget_evictions"] += 1

    async def _ensure_model_slot_available(self) -> None:
        """Evict LRU models until under max_models limit.

        Caller MUST hold self._lock. Recounts after each eviction to
        handle concurrent state changes correctly.
        """
        if self.max_models <= 0:
            return

        while True:
            loaded_count = sum(1 for e in self._entries.values() if e.is_loaded)
            if loaded_count < self.max_models:
                break
            victim = self._find_lru_victim()
            if victim is None:
                raise MemoryError(
                    f"Cannot free model slot: max_models={self.max_models} reached, "
                    f"all loaded models are pinned or have active requests"
                )
            # see _ensure_memory_available — guard against an unbounded loop on
            # a refused (un-evictable) victim; count only real evictions.
            if not await self._unload_model_locked(victim.model_id):
                raise MemoryError(
                    f"Cannot free model slot: LRU victim '{victim.model_id}' could not "
                    f"be evicted (engine cannot be proven idle). max_models="
                    f"{self.max_models} reached."
                )
            self._eviction_stats["slot_evictions"] += 1

    async def _check_post_load_memory_pressure(
        self, exclude_model_id: str | None = None
    ) -> None:
        """Check memory pressure after loading and evict LRU model if needed.

        After loading a new model, the combined memory
        footprint of all loaded models may push utilization above a safe
        threshold.  Unlike ``_ensure_memory_available`` (which only fires
        when an explicit ``max_memory_bytes`` budget is exceeded), this
        method looks at *actual* MLX active memory relative to the Metal
        working-set limit.  This catches cases where:
        - ``max_memory_bytes`` was not set (unlimited budget mode)
        - Estimated sizes are inaccurate
        - The new model is the only model but uses most of working-set

        Caller MUST hold ``self._lock``.
        """
        active = mx.get_active_memory()
        try:
            max_ws = mx.metal.get_memory_info()
        except AttributeError:
            try:
                max_ws = mx.metal.device_info()
            except AttributeError:
                max_ws = None
        limit = 0
        if isinstance(max_ws, dict):
            limit = int(max_ws.get("max_recommended_working_set_size", 0))
        elif hasattr(max_ws, "max_recommended_working_set_size"):
            limit = int(max_ws.max_recommended_working_set_size)
        if limit <= 0:
            from .utils.hardware import get_total_memory_bytes

            limit = int(get_total_memory_bytes() * 0.75)

        if limit <= 0:
            return

        utilization = active / limit
        if utilization < self.memory_pressure_threshold:
            return

        logger.warning(
            "Post-load memory pressure: %.1f%% active (%.1fGB / %.1fGB working-set), "
            "threshold=%.0f%% — initiating LRU eviction",
            utilization * 100,
            active / 1e9,
            limit / 1e9,
            self.memory_pressure_threshold * 100,
        )

        # Evict one LRU model to relieve pressure.  Keep evicting while
        # pressure remains high, but stop after a safety limit to avoid
        # unloading everything.
        max_evictions = sum(
            1 for e in self._entries.values() if e.is_loaded and not e.is_pinned
        )
        for _ in range(max_evictions):
            victim = self._find_lru_victim(exclude_model_id=exclude_model_id)
            if victim is None:
                break
            # stop if the victim was refused (frees nothing) — don't burn the
            # remaining iterations or over-count the stat against an un-evictable model.
            if not await self._unload_model_locked(victim.model_id):
                break
            self._eviction_stats["pressure_evictions"] += 1

            # Re-check after each eviction
            active = mx.get_active_memory()
            if active / limit < self.memory_pressure_threshold:
                break

    async def _evict_lru_model(self) -> str | None:
        """Evict the least-recently-used non-pinned, idle model.

        Public wrapper around ``_find_lru_victim`` + ``_unload_model_locked``
        so callers (e.g. periodic TTL checks, manual admin triggers) can
        request a single LRU eviction without knowing internal state.

        Returns the model_id of the evicted model, or None if nothing was
        eligible for eviction.
        """
        async with self._lock:
            victim = self._find_lru_victim()
            if victim is None:
                return None
            # report the eviction only if it actually happened (the unloader
            # can refuse). Returning victim.model_id on a refusal was a false positive.
            if not await self._unload_model_locked(victim.model_id):
                return None
            return victim.model_id

    def _find_lru_victim(
        self, exclude_model_id: str | None = None
    ) -> ModelEntry | None:
        """Find the least-recently-used non-pinned, loaded model.

        Skip models with active requests to avoid interrupting in-flight
        generation. ``exclude_model_id`` skips a model the
        caller must not evict — e.g. the one JUST loaded, which has no active
        requests yet (the caller hasn't reached generate()) so would otherwise be a
        valid victim and could evict ITSELF, returning a dead engine.
        """
        victims = [
            e
            for e in self._entries.values()
            if e.is_loaded
            and not e.is_pinned
            and not e.is_loading
            and e.model_id != exclude_model_id
        ]
        # Filter out engines with active requests.
        # A victim is only "safe" if we can PROVE it is idle —
        # i.e. its engine has a callable has_active_requests that returns False. This
        # must agree with _unload_model_locked's fail-safe, which REFUSES a
        # non-forced unload of any engine lacking has_active_requests (TTS/ASR/Image/
        # STS/Video/OCR). The old code instead APPENDED such engines as valid victims:
        # _ensure_memory_available / _ensure_model_slot_available then looped forever
        # (nominate → unload refuses, frees nothing → condition unchanged → livelock
        # holding self._lock → ModelManager-wide DoS). Exclude un-provable engines so
        # the selector never nominates a victim the unloader will refuse.
        safe_victims = []
        for v in victims:
            _has_active = (
                getattr(v.engine, "has_active_requests", None)
                if v.engine is not None
                else None
            )
            if not callable(_has_active):
                # Cannot prove idle → not a safe non-forced victim (would be refused).
                continue
            try:
                if _has_active():
                    continue
            except Exception:
                logger.debug(
                    f"has_active_requests check failed for {v.model_id}", exc_info=True
                )
                continue  # cannot prove idle → skip (matches unload's assume-active)
            safe_victims.append(v)

        if not safe_victims:
            # All loaded models have active requests — refuse to evict
            # to avoid killing in-flight generation requests.
            logger.warning(
                "No safe eviction targets — all %d loaded models have active requests",
                len(victims),
            )
            return None

        return min(safe_victims, key=lambda e: e.last_access)

    @property
    def loaded_count(self) -> int:
        """Number of currently loaded models."""
        return sum(1 for e in self._entries.values() if e.is_loaded)

    def list_models(self) -> list[dict]:
        """Return status of all registered models."""
        return [
            {
                "id": e.model_id,
                "type": e.model_type.name,
                "loaded": e.is_loaded,
                "pinned": e.is_pinned,
                "loading": e.is_loading,
                "size_gb": e.estimated_bytes / 1e9,
                "last_access": e.last_access,
                "error": e.load_error,
            }
            for e in self._entries.values()
        ]

    async def check_ttl(self) -> list[str]:
        """Unload models that exceeded their TTL. Returns unloaded model IDs."""
        if self.ttl_seconds is None:
            return []

        unloaded = []
        now = time.monotonic()
        cutoff = now - self.ttl_seconds
        async with self._lock:
            candidates = [
                e
                for e in self._entries.values()
                if e.is_loaded and not e.is_pinned and e.last_access <= cutoff
            ]
            for entry in candidates:
                # Re-check: a concurrent access may have updated last_access
                if entry.last_access > cutoff:
                    continue
                await self._unload_model_locked(entry.model_id)
                unloaded.append(entry.model_id)
        return unloaded

    @property
    def memory_usage(self) -> dict:
        return {
            "current_gb": self._current_memory_bytes / 1e9,
            "max_gb": (self.max_memory_bytes or 0) / 1e9,
            "models_loaded": sum(1 for e in self._entries.values() if e.is_loaded),
            "models_registered": len(self._entries),
        }

    def get_entry(self, model_id: str) -> ModelEntry | None:
        """Get entry for a specific model."""
        return self._entries.get(model_id)

    def list_entries(self) -> list[ModelEntry]:
        """Return all registered model entries."""
        return list(self._entries.values())

    def unregister_model(self, model_id: str) -> bool:
        """Remove a model registration (must not be loaded).

        Returns True if the model was removed, False if not found.
        Raises ValueError if the model is still loaded.
        """
        with self._sync_lock:
            entry = self._entries.get(model_id)
            if entry is None:
                return False
            if entry.is_loaded:
                raise ValueError(
                    f"Cannot unregister loaded model '{model_id}'. Unload first."
                )
            # also refuse while LOADING — symmetric with register_model's
            # is_loading guard (line ~345). Deleting the entry mid-load orphans the
            # in-flight loader's local `entry` reference: the load completes outside
            # _entries, so (1) `_current_memory_bytes += estimated_bytes` accounts a
            # model that no longer exists → permanent drift that eventually blocks all
            # future loads, (2) the loaded engine is never reachable by shutdown()/
            # eviction → engine.stop() never runs (GPU/thread leak), and (3) the
            # finally's `_loading_events.pop(model_id)` can pop a re-registered model's
            # NEW event → fresh waiters hang. Force an unload (or wait) first.
            if entry.is_loading:
                raise ValueError(
                    f"Cannot unregister loading model '{model_id}'. Wait for load or unload first."
                )
            del self._entries[model_id]
        logger.info("Unregistered model: %s", model_id)
        return True

    def discover_models(self, models_dir: str) -> int:
        """Scan a directory for model subdirectories and auto-register them.

        Auto-detect model type from config files.
        Returns the number of models discovered.
        """
        models_path = Path(models_dir)
        if not models_path.exists():
            return 0

        count = 0
        for subdir in sorted(models_path.iterdir()):
            if not subdir.is_dir():
                continue

            has_config = (subdir / "config.json").exists()
            has_weights = any(subdir.glob("*.safetensors"))
            has_nested_weights = any(subdir.rglob("*.safetensors"))
            has_model_index = (subdir / "model_index.json").exists()
            has_diffusion_dirs = all(
                (subdir / d).is_dir() for d in ("transformer", "vae", "text_encoder")
            )
            if not (
                has_config
                or has_weights
                or has_nested_weights
                or has_model_index
                or has_diffusion_dirs
            ):
                continue

            model_id = subdir.name

            # Skip if already registered and loaded/loading.
            # Loading entries must not be overwritten — would orphan loading events.
            existing = self._entries.get(model_id)
            if existing is not None and (existing.is_loaded or existing.is_loading):
                continue

            # Preserve pinned status from existing registration
            was_pinned = existing.is_pinned if existing is not None else False

            try:
                raw_bytes = sum(
                    f.stat().st_size
                    for f in subdir.rglob("*.safetensors")
                    if f.is_file()
                )
            except (OSError, PermissionError) as e:
                logger.warning("Skipping model %s: cannot read files: %s", model_id, e)
                continue
            estimated = int(raw_bytes * 1.8)

            self.register_model(
                model_id=model_id,
                model_path=str(subdir),
                estimated_bytes=estimated,
                pinned=was_pinned,
            )
            count += 1

        logger.info(f"Discovered {count} models in {models_dir}")
        return count

    def resolve_model_id(self, model_id: str) -> str | None:
        """Resolve a model ID with case-insensitive + prefix stripping fallback."""
        if model_id in self._entries:
            return model_id

        lower = model_id.lower()
        for mid in self._entries:
            if mid.lower() == lower:
                return mid

        if "/" in model_id:
            stripped = model_id.rsplit("/", 1)[-1]
            if stripped in self._entries:
                return stripped
            for mid in self._entries:
                if mid.lower() == stripped.lower():
                    return mid

        return None

    async def shutdown(self) -> None:
        """Gracefully unload all loaded models.

        Also signals any in-progress loading events so waiters don't hang.
        Idempotent: safe to call multiple times.
        """
        # Signal all loading events so any waiters unblock
        for _model_id, event in list(self._loading_events.items()):
            event.set()
        self._loading_events.clear()

        # If no models are loaded, shutdown is a no-op
        if not any(e.is_loaded for e in self._entries.values()):
            return

        # Collect loaded model IDs under the lock, then unload each
        async with self._lock:
            self._shutting_down = True
            loaded_ids = [mid for mid, e in self._entries.items() if e.is_loaded]

        for model_id in loaded_ids:
            try:
                # shutdown must tear down even models with active
                # requests (the process is going away regardless).
                await self.unload_model(model_id, force=True)
            except Exception as e:
                logger.error(
                    f"Error unloading {model_id} during shutdown: {e}", exc_info=True
                )

        logger.info(
            "ModelManager shutdown complete: %d models registered, %d loaded",
            len(self._entries),
            sum(1 for e in self._entries.values() if e.is_loaded),
        )

    def get_status(self) -> dict:
        """Return detailed pool status."""
        return {
            "max_memory_gb": (self.max_memory_bytes or 0) / 1e9,
            "current_memory_gb": self._current_memory_bytes / 1e9,
            "models_registered": len(self._entries),
            "models_loaded": sum(1 for e in self._entries.values() if e.is_loaded),
            "eviction_stats": dict(self._eviction_stats),
            "memory_pressure_threshold": self.memory_pressure_threshold,
            "models": [
                {
                    "id": e.model_id,
                    "type": e.model_type.name,
                    "loaded": e.is_loaded,
                    "loading": e.is_loading,
                    "pinned": e.is_pinned,
                    "size_gb": round(e.estimated_bytes / 1e9, 2),
                    "last_access": e.last_access if e.last_access > 0 else None,
                    "error": e.load_error,
                }
                for e in sorted(self._entries.values(), key=lambda x: x.model_id)
            ],
        }
