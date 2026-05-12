"""Yunshu Model Manager — Multi-model serving with LRU eviction.

Manages multiple loaded models with memory-aware lifecycle:
- Lazy loading on first request
- LRU eviction when memory pressure exceeds threshold
- Pinned models that are never evicted
- Memory settle barrier (inspired by oMLX — verify MLX actually freed memory)
- Support for multiple engine types: LLM, VLM, TTS, ASR, ImageGen

Model type detection uses dynamic library probing instead of static lists:
- Tries mlx-lm first (import mlx_lm.models.{model_type})
- Falls back to config structure heuristics (vision_config, etc.)
- This gives 0-day support for any new model mlx-lm adds
"""

from __future__ import annotations

import asyncio
import gc
import importlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx

from .engine import EngineConfig

logger = logging.getLogger(__name__)


class ModelType(Enum):
    LLM = auto()
    VLM = auto()
    TTS = auto()
    ASR = auto()
    IMAGE_GEN = auto()


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
        if "tts" in name_lower or "voice" in name_lower:
            return ModelType.TTS
        if "asr" in name_lower or "whisper" in name_lower:
            return ModelType.ASR
        if "vision" in name_lower or "vlm" in name_lower or "omni" in name_lower:
            return ModelType.VLM
        if "image" in name_lower or "flux" in name_lower or "z-image" in name_lower:
            return ModelType.IMAGE_GEN
        return ModelType.LLM

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError):
        return ModelType.LLM

    model_type = config.get("model_type", "").lower().replace("-", "_")
    architectures = config.get("architectures", [])

    # Image gen: known diffusion model types
    if model_type in ("flux", "sd3", "sdxl", "z_image", "stable-diffusion"):
        return ModelType.IMAGE_GEN
    for arch in architectures:
        if "Transformer2D" in arch:
            return ModelType.IMAGE_GEN

    # Check for ASR/TTS by architecture keywords
    for arch in architectures:
        arch_lower = arch.lower()
        if "speech" in arch_lower and "text" in arch_lower:
            if "tts" in arch_lower or "synthes" in arch_lower:
                return ModelType.TTS
            if "asr" in arch_lower or "recogni" in arch_lower or "whisper" in arch_lower:
                return ModelType.ASR
        if "tts" in arch_lower:
            return ModelType.TTS

    # Check model_type for known ASR/TTS types
    if model_type in ("whisper", "qwen3_asr", "parakeet", "qwen2_audio"):
        return ModelType.ASR
    if model_type in ("qwen3_tts", "kokoro", "chatterbox"):
        return ModelType.TTS

    # Keyword-based ASR/TTS fallback
    if any(k in model_type for k in ("tts", "voice", "speech_synthes")):
        return ModelType.TTS
    if any(k in model_type for k in ("whisper", "asr", "speech_to_text")):
        return ModelType.ASR

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
        if any(k in arch.lower() for k in ("vl", "vision", "omni", "florence", "pixtral")):
            has_vision = True
    if remapped in ("qwen2_vl", "qwen3_vl", "qwen3_vl_moe", "mistral3", "pixtral"):
        has_vision = True

    if has_vision:
        return ModelType.VLM

    if not mlx_lm_supported and not mlx_vlm_supported:
        # Neither mlx-lm nor mlx-vlm supports this model_type
        # Check heuristics for ASR/TTS from directory name
        name_lower = p.name.lower()
        if "tts" in name_lower or "voice" in name_lower:
            return ModelType.TTS
        if "asr" in name_lower or "whisper" in name_lower:
            return ModelType.ASR
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
    load_error: Optional[str] = None


class ModelManager:
    """Manages multiple inference engines with memory-aware loading.

    Design decisions vs oMLX:
    - Uses asyncio.Lock instead of oMLX's threading (our engine is async-native)
    - Memory settle barrier with tighter polling (learned from oMLX's 10-round approach)
    - Per-model engine isolation (oMLX shares a single ThreadPoolExecutor)
    """

    def __init__(
        self,
        max_memory_bytes: Optional[int] = None,
        kv_reserve_ratio: float = 0.25,
        settle_timeout_s: float = 5.0,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        self.max_memory_bytes = max_memory_bytes  # None = unlimited
        self.kv_reserve_ratio = kv_reserve_ratio
        self.settle_timeout_s = settle_timeout_s
        self.ttl_seconds = ttl_seconds

        self._entries: dict[str, ModelEntry] = {}
        self._current_memory_bytes: int = 0
        self._lock = asyncio.Lock()

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
        """
        if model_type is None:
            model_type = _detect_model_type(model_path)

        self._entries[model_id] = ModelEntry(
            model_id=model_id,
            model_path=model_path,
            estimated_bytes=estimated_bytes,
            is_pinned=pinned,
            model_type=model_type,
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
        - Concurrent load protection (is_loading flag)
        - Memory settle barrier after unload
        - Different engine types (LLM, VLM, TTS, ASR, ImageGen)
        """
        entry = self._entries.get(model_id)
        if entry is None:
            raise KeyError(f"Model not registered: {model_id}")

        # Already loaded — update access time
        if entry.is_loaded and entry.engine is not None:
            entry.last_access = time.monotonic()
            return entry.engine

        async with self._lock:
            # Double-check after acquiring lock
            if entry.is_loaded and entry.engine is not None:
                entry.last_access = time.monotonic()
                return entry.engine

            if entry.is_loading:
                raise RuntimeError(f"Model {model_id} is already being loaded")

            # Check memory budget
            if self.max_memory_bytes is not None:
                required = entry.estimated_bytes
                kv_headroom = 0 if entry.model_type in (ModelType.TTS, ModelType.ASR) else int(required * self.kv_reserve_ratio)
                total_needed = required + kv_headroom
                await self._ensure_memory_available(total_needed)

            # Load using the appropriate engine type
            entry.is_loading = True
            try:
                engine = await self._create_and_load_engine(entry, engine_config)

                entry.engine = engine
                entry.is_loaded = True
                entry.is_loading = False
                entry.last_access = time.monotonic()
                entry.load_error = None

                self._current_memory_bytes += entry.estimated_bytes

                # Post-load cache clear (oMLX #429: weight loading creates large
                # Metal buffer temporaries that stay in the buffer pool)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    self._get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )

                logger.info(
                    f"Loaded model {model_id} "
                    f"({entry.model_type.name}, "
                    f"{entry.estimated_bytes / 1e9:.1f} GB, "
                    f"total: {self._current_memory_bytes / 1e9:.1f} GB)"
                )

                return engine

            except Exception as e:
                entry.is_loading = False
                entry.load_error = str(e)
                logger.error(f"Failed to load model {model_id}: {e}")
                raise

    async def _create_and_load_engine(
        self,
        entry: ModelEntry,
        config: EngineConfig | None,
    ) -> Any:
        """Create the right engine type and load it."""
        loop = asyncio.get_running_loop()

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

        else:
            # Default: LLM engine (BatchedEngine with EngineCore backend)
            from .batched_engine import BatchedEngine
            engine = BatchedEngine(
                model_name=entry.model_path,
                stream_interval=getattr(config, 'stream_interval', 1) if config else 1,
            )
            await engine.start()
            return engine

    async def unload_model(self, model_id: str) -> None:
        """Unload a model and reclaim memory.

        oMLX EnginePool._unload_engine pattern:
        - Stop engine, clear reference BEFORE settle barrier
        - gc.collect() + sync + clear_cache on MLX executor
        - Poll mx.get_active_memory() until Metal buffers released
        """
        entry = self._entries.get(model_id)
        if entry is None or not entry.is_loaded:
            return

        pre_unload_active = mx.get_active_memory()

        if entry.engine is not None:
            try:
                await entry.engine.stop()
            except Exception as e:
                logger.warning(f"Error stopping engine for {model_id}: {e}")
            entry.engine = None

        entry.is_loaded = False

        # Force GC + clear cache on MLX executor (oMLX #85, #300)
        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._get_mlx_executor(),
            lambda: (mx.synchronize(), mx.clear_cache()),
        )

        # Memory settle barrier: poll until Metal buffers actually released
        settle_tolerance = max(2 * 1024**3, int(entry.estimated_bytes * 0.05))
        min_expected_freed = max(0, entry.estimated_bytes - settle_tolerance)
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

        self._current_memory_bytes = max(
            0, self._current_memory_bytes - entry.estimated_bytes
        )

        logger.info(
            f"Unloaded model {model_id} "
            f"(freed: {(pre_unload_active - mx.get_active_memory()) / 1e9:.1f}GB, "
            f"settled: {settled})"
        )

    async def _ensure_memory_available(self, needed_bytes: int) -> None:
        """Evict LRU models until enough memory is free."""
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
            await self.unload_model(victim.model_id)

    def _find_lru_victim(self) -> Optional[ModelEntry]:
        """Find the least-recently-used non-pinned, loaded model.

        oMLX pattern: skip models with active requests to avoid
        interrupting in-flight generation.
        """
        victims = [
            e
            for e in self._entries.values()
            if e.is_loaded and not e.is_pinned and not e.is_loading
        ]
        # Filter out engines with active requests (oMLX EnginePool pattern)
        safe_victims = []
        for v in victims:
            if v.engine is not None and hasattr(v.engine, "has_active_requests"):
                try:
                    if v.engine.has_active_requests():
                        continue
                except Exception:
                    pass
            safe_victims.append(v)

        if not safe_victims:
            # Fallback: allow evicting models with active requests if desperate
            if not victims:
                return None
            return min(victims, key=lambda e: e.last_access)

        return min(safe_victims, key=lambda e: e.last_access)

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
        for entry in list(self._entries.values()):
            if (
                entry.is_loaded
                and not entry.is_pinned
                and now - entry.last_access > self.ttl_seconds
            ):
                await self.unload_model(entry.model_id)
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

    def get_loaded_model_ids(self) -> list[str]:
        """Get list of currently loaded model IDs."""
        return [e.model_id for e in self._entries.values() if e.is_loaded]

    def get_entry(self, model_id: str) -> Optional[ModelEntry]:
        """Get entry for a specific model."""
        return self._entries.get(model_id)

    def list_entries(self) -> list[ModelEntry]:
        """Return all registered model entries."""
        return list(self._entries.values())

    def discover_models(self, models_dir: str) -> int:
        """Scan a directory for model subdirectories and auto-register them.

        oMLX pattern: auto-detect model type from config files.
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
            if not (has_config or has_weights or has_nested_weights or has_model_index or has_diffusion_dirs):
                continue

            model_id = subdir.name

            # Skip if already registered and loaded (oMLX pattern)
            existing = self._entries.get(model_id)
            if existing is not None and existing.is_loaded:
                continue

            raw_bytes = sum(f.stat().st_size for f in subdir.rglob("*.safetensors"))
            estimated = int(raw_bytes * 1.8)

            self.register_model(
                model_id=model_id,
                model_path=str(subdir),
                estimated_bytes=estimated,
            )
            count += 1

        logger.info(f"Discovered {count} models in {models_dir}")
        return count

    def resolve_model_id(self, model_id: str) -> Optional[str]:
        """Resolve a model ID with case-insensitive + prefix stripping fallback.

        oMLX EnginePool.resolve_model_id pattern.
        """
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
        """Gracefully unload all loaded models."""
        for model_id in list(self._entries.keys()):
            entry = self._entries.get(model_id)
            if entry and entry.is_loaded:
                try:
                    await self.unload_model(model_id)
                except Exception as e:
                    logger.error(f"Error unloading {model_id} during shutdown: {e}")

    def get_status(self) -> dict:
        """Return detailed pool status (oMLX EnginePool.get_status pattern)."""
        return {
            "max_memory_gb": (self.max_memory_bytes or 0) / 1e9,
            "current_memory_gb": self._current_memory_bytes / 1e9,
            "models_registered": len(self._entries),
            "models_loaded": sum(1 for e in self._entries.values() if e.is_loaded),
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
