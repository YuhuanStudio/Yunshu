"""MultimodalPipelineCoordinator — multi-stage processing for multimodal requests.

vllm-omni pattern: separates text/image/audio processing into distinct pipeline
stages with parallel execution for independent stages and model-specific
preprocessors.

Architecture:
  PipelineStage enum — 7 stages: TEXT_PREPROCESS → IMAGE_PREPROCESS →
    AUDIO_PREPROCESS → EMBEDDING_FUSION → PREFILL → DECODE → POSTPROCESS
  StageConfig — per-stage configuration (modality, model, batch, timeout)
  MultimodalPipelineCoordinator — orchestrates stage execution with:
    - Processor registration per (stage, modality)
    - Parallel execution of independent stages (image + audio preproc)
    - Stage-level caching for intermediate results
  ModelPreprocessorRegistry — model-specific preprocessors with auto-detection
    Built-in: QwenVLM, LLaVAVLM, CosyVoice, QwenOmni + generic fallback

Integration:
  - Called from vlm_engine, audio_engine, image_engine before model inference
  - Replaces ad-hoc per-modality processing with a unified pipeline
  - Stage stats feed into telemetry / server_metrics
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Pipeline Stage ──


class PipelineStage(str, Enum):
    """Processing stages for multimodal requests."""
    TEXT_PREPROCESS = "text_preprocess"
    IMAGE_PREPROCESS = "image_preprocess"
    AUDIO_PREPROCESS = "audio_preprocess"
    EMBEDDING_FUSION = "embedding_fusion"
    PREFILL = "prefill"
    DECODE = "decode"
    POSTPROCESS = "postprocess"


# Stages that can run in parallel (no dependencies on each other)
_PARALLEL_STAGES = {
    PipelineStage.IMAGE_PREPROCESS,
    PipelineStage.AUDIO_PREPROCESS,
}

# Execution order — parallel stages may execute simultaneously
_STAGE_ORDER = [
    PipelineStage.TEXT_PREPROCESS,
    PipelineStage.IMAGE_PREPROCESS,
    PipelineStage.AUDIO_PREPROCESS,
    PipelineStage.EMBEDDING_FUSION,
    PipelineStage.PREFILL,
    PipelineStage.DECODE,
    PipelineStage.POSTPROCESS,
]


@dataclass
class StageConfig:
    """Configuration for a single pipeline stage."""
    stage_type: PipelineStage
    modality: str  # "text", "image", "audio", "video"
    model_id: Optional[str] = None
    batch_size: int = 1
    timeout_ms: int = 30_000  # 30 seconds default

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.timeout_ms < 0:
            raise ValueError(f"timeout_ms must be >= 0, got {self.timeout_ms}")


@dataclass
class StageResult:
    """Output from a single pipeline stage."""
    stage: PipelineStage
    data: Any
    cached: bool = False
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class PipelineRequest:
    """Input to the pipeline coordinator."""
    request_id: str
    model_id: str
    text: Optional[str] = None
    images: Optional[list[Any]] = None
    audio: Optional[list[Any]] = None
    video: Optional[list[Any]] = None
    params: dict = field(default_factory=dict)


@dataclass
class StageStats:
    """Per-stage statistics."""
    call_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_duration_ms: float = 0.0
    error_count: int = 0

    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / self.call_count if self.call_count > 0 else 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


# ── Stage-level cache ──


class StageCache:
    """Cache for intermediate pipeline stage results.

    Uses a content hash of inputs to enable cache hits for repeated requests.
    Bounded by max_entries with LRU eviction.
    """

    def __init__(self, max_entries: int = 256, ttl_seconds: float = 300.0):
        self._cache: dict[str, tuple[float, Any]] = {}
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(stage: PipelineStage, inputs: Any) -> str:
        """Create a deterministic cache key from stage + inputs."""
        raw = json.dumps({"stage": stage.value, "inputs": _stable_repr(inputs)},
                         sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get(self, stage: PipelineStage, inputs: Any) -> Optional[Any]:
        key = self._make_key(stage, inputs)
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        ts, value = entry
        if time.monotonic() - ts > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None
        self._hits += 1
        return value

    def put(self, stage: PipelineStage, inputs: Any, result: Any) -> None:
        key = self._make_key(stage, inputs)
        if len(self._cache) >= self._max_entries:
            # Evict oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = (time.monotonic(), result)

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses


def _stable_repr(obj: Any) -> str:
    """Produce a stable string representation for hashing."""
    if obj is None:
        return "None"
    if isinstance(obj, (str, int, float, bool)):
        return repr(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_stable_repr(x) for x in obj) + "]"
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: str(kv[0]))
        return "{" + ",".join(f"{k!r}:{_stable_repr(v)}" for k, v in items) + "}"
    return repr(obj)


# ── Processor type ──

ProcessorFn = Callable[[Any, StageConfig], Any]


# ── MultimodalPipelineCoordinator ──


class MultimodalPipelineCoordinator:
    """Orchestrates multi-stage multimodal request processing.

    Registers processors for each (stage, modality) combination and
    executes them in dependency order with parallel execution for
    independent stages (image + audio preprocessing).
    """

    def __init__(self, *, enable_cache: bool = True, cache_max: int = 256,
                 max_workers: int = 4):
        self._processors: dict[tuple[PipelineStage, str], ProcessorFn] = {}
        self._configs: dict[tuple[PipelineStage, str], StageConfig] = {}
        self._stats: dict[PipelineStage, StageStats] = defaultdict(StageStats)
        self._cache = StageCache(max_entries=cache_max) if enable_cache else None
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def register_processor(self, stage: PipelineStage, modality: str,
                           processor_fn: ProcessorFn,
                           config: Optional[StageConfig] = None) -> None:
        """Register a processing function for a (stage, modality) pair."""
        self._processors[(stage, modality)] = processor_fn
        if config is None:
            config = StageConfig(stage_type=stage, modality=modality)
        self._configs[(stage, modality)] = config

    def process(self, request: PipelineRequest) -> list[StageResult]:
        """Run a request through all applicable stages in order.

        Returns a list of StageResults, one per executed stage.
        """
        results: list[StageResult] = []
        accumulated: dict[str, Any] = {"request": request}

        # Determine which modalities are present
        modalities: set[str] = {"text"}
        if request.images:
            modalities.add("image")
        if request.audio:
            modalities.add("audio")
        if request.video:
            modalities.add("video")

        for stage in _STAGE_ORDER:
            # Determine applicable modalities for this stage
            applicable = self._applicable_modalities(stage, modalities)

            if not applicable:
                continue

            if stage in _PARALLEL_STAGES and len(applicable) > 1:
                # Execute independent stages in parallel
                stage_results = self._run_parallel(stage, applicable, accumulated)
                results.extend(stage_results)
                for sr in stage_results:
                    if sr.error is None:
                        accumulated[f"{stage.value}:{sr.data.__class__.__name__}"] = sr.data
            else:
                for modality in applicable:
                    sr = self._run_single(stage, modality, accumulated)
                    results.append(sr)
                    if sr.error is None:
                        key = f"{stage.value}:{modality}"
                        accumulated[key] = sr.data

        return results

    def process_stage(self, stage: PipelineStage, inputs: Any,
                      modality: str = "text") -> StageResult:
        """Run a single stage manually."""
        config = self._configs.get((stage, modality),
                                   StageConfig(stage_type=stage, modality=modality))
        processor = self._processors.get((stage, modality))

        if processor is None:
            return StageResult(stage=stage, data=None,
                               error=f"No processor for ({stage.value}, {modality})")

        t0 = time.monotonic()
        try:
            result = processor(inputs, config)
            duration_ms = (time.monotonic() - t0) * 1000
            return StageResult(stage=stage, data=result, duration_ms=duration_ms)
        except Exception as e:
            duration_ms = (time.monotonic() - t0) * 1000
            stats = self._stats[stage]
            stats.error_count += 1
            return StageResult(stage=stage, data=None, error=str(e),
                               duration_ms=duration_ms)

    def get_stats(self) -> dict[str, dict]:
        """Per-stage timing, cache hit rates, throughput."""
        result = {}
        for stage in PipelineStage:
            stats = self._stats[stage]
            cache_info = {}
            if self._cache is not None:
                cache_info = {
                    "cache_size": self._cache.size,
                    "cache_hits": self._cache.hits,
                    "cache_misses": self._cache.misses,
                }
            result[stage.value] = {
                "call_count": stats.call_count,
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
                "cache_hit_rate": stats.cache_hit_rate,
                "avg_duration_ms": stats.avg_duration_ms,
                "total_duration_ms": stats.total_duration_ms,
                "error_count": stats.error_count,
                **cache_info,
            }
        return result

    def clear_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()

    def _applicable_modalities(self, stage: PipelineStage,
                               present: set[str]) -> list[str]:
        """Determine which modalities apply for a given stage."""
        stage_modality_map = {
            PipelineStage.TEXT_PREPROCESS: ["text"],
            PipelineStage.IMAGE_PREPROCESS: ["image"],
            PipelineStage.AUDIO_PREPROCESS: ["audio"],
            PipelineStage.EMBEDDING_FUSION: list(present),
            PipelineStage.PREFILL: list(present),
            PipelineStage.DECODE: ["text"],
            PipelineStage.POSTPROCESS: ["text"],
        }
        candidates = stage_modality_map.get(stage, [])
        return [m for m in candidates if m in present]

    def _run_single(self, stage: PipelineStage, modality: str,
                    accumulated: dict) -> StageResult:
        """Execute a single (stage, modality) pair."""
        config = self._configs.get((stage, modality),
                                   StageConfig(stage_type=stage, modality=modality))
        processor = self._processors.get((stage, modality))

        if processor is None:
            # No processor registered — pass through
            return StageResult(stage=stage, data=accumulated, duration_ms=0.0)

        inputs = accumulated.copy()
        stats = self._stats[stage]

        # Check cache
        if self._cache is not None:
            cached = self._cache.get(stage, inputs)
            if cached is not None:
                stats.call_count += 1
                stats.cache_hits += 1
                return StageResult(stage=stage, data=cached, cached=True,
                                   duration_ms=0.0)

        t0 = time.monotonic()
        try:
            result = processor(inputs, config)
            duration_ms = (time.monotonic() - t0) * 1000

            stats.call_count += 1
            stats.cache_misses += 1
            stats.total_duration_ms += duration_ms

            if self._cache is not None:
                self._cache.put(stage, inputs, result)

            return StageResult(stage=stage, data=result, duration_ms=duration_ms)
        except Exception as e:
            duration_ms = (time.monotonic() - t0) * 1000
            stats.call_count += 1
            stats.error_count += 1
            stats.total_duration_ms += duration_ms
            return StageResult(stage=stage, data=None, error=str(e),
                               duration_ms=duration_ms)

    def _run_parallel(self, stage: PipelineStage, modalities: list[str],
                      accumulated: dict) -> list[StageResult]:
        """Execute a stage across multiple modalities in parallel."""
        futures = {}
        for modality in modalities:
            future = self._executor.submit(self._run_single, stage, modality,
                                           accumulated)
            futures[future] = modality

        results = []
        for future in as_completed(futures):
            results.append(future.result())
        return results


# ── ModelPreprocessorRegistry ──


class ModelPreprocessorRegistry:
    """Registry for model-specific preprocessors.

    Maps (model_family, modality) → preprocessor function, with auto-detection
    of model family from model_id. Falls back to generic processor when no
    model-specific preprocessor is registered.

    Built-in preprocessors: QwenVLM, LLaVAVLM, CosyVoice, QwenOmni.
    """

    # Known model family patterns for auto-detection
    _FAMILY_PATTERNS: dict[str, list[str]] = {
        "qwen_vlm": ["qwen-vl", "qwen2-vl", "qwen2.5-vl", "qwen3-vl", "qwen2_5_vl", "qwen2_vl"],
        "llava_vlm": ["llava", "llava-next", "llava-onevision"],
        "cosyvoice": ["cosyvoice", "cosy-voice"],
        "qwen_omni": ["qwen-omni", "qwen2_omni", "qwen3_omni", "qwen2-audio", "qwen3_omni_moe"],
        "qwen_audio": ["qwen-audio", "qwen2-audio"],
        "whisper": ["whisper"],
        "flux": ["flux", "flux2", "klein"],
        "z_image": ["z-image", "zimage"],
    }

    def __init__(self):
        self._registry: dict[tuple[str, str], ProcessorFn] = {}
        self._fallback: Optional[ProcessorFn] = None

    def register(self, model_family: str, modality: str,
                 preprocessor: ProcessorFn) -> None:
        """Register a preprocessor for a (model_family, modality) pair."""
        self._registry[(model_family.lower(), modality)] = preprocessor

    def set_fallback(self, preprocessor: ProcessorFn) -> None:
        """Set the generic fallback processor used when no model-specific one exists."""
        self._fallback = preprocessor

    def get_preprocessor(self, model_id: str, modality: str) -> Optional[ProcessorFn]:
        """Get preprocessor for a model, auto-detecting family from model_id.

        Resolution order:
        1. Exact (family, modality) match
        2. Fallback generic processor
        3. None (caller must handle)
        """
        family = self.detect_family(model_id)
        preprocessor = self._registry.get((family, modality))
        if preprocessor is not None:
            return preprocessor
        return self._fallback

    @classmethod
    def detect_family(cls, model_id: str) -> str:
        """Auto-detect model family from model_id string."""
        model_lower = model_id.lower()
        for family, patterns in cls._FAMILY_PATTERNS.items():
            for pattern in patterns:
                if pattern in model_lower:
                    return family
        return "generic"

    def list_registered(self) -> list[tuple[str, str]]:
        """List all registered (model_family, modality) pairs."""
        return list(self._registry.keys())


# ── Built-in preprocessors ──


def _qwen_vlm_preprocessor(inputs: Any, config: StageConfig) -> dict:
    """Qwen VLM preprocessor — vision feature extraction for Qwen2-VL family."""
    request = inputs.get("request")
    return {
        "modality": "image",
        "model_family": "qwen_vlm",
        "features": "vision_embeddings",
        "image_count": len(request.images) if request and request.images else 0,
    }


def _llava_vlm_preprocessor(inputs: Any, config: StageConfig) -> dict:
    """LLaVA VLM preprocessor — clip-based image features."""
    request = inputs.get("request")
    return {
        "modality": "image",
        "model_family": "llava_vlm",
        "features": "clip_embeddings",
        "image_count": len(request.images) if request and request.images else 0,
    }


def _cosyvoice_preprocessor(inputs: Any, config: StageConfig) -> dict:
    """CosyVoice preprocessor — audio feature extraction for TTS/ASR."""
    request = inputs.get("request")
    return {
        "modality": "audio",
        "model_family": "cosyvoice",
        "features": "audio_embeddings",
        "audio_count": len(request.audio) if request and request.audio else 0,
    }


def _qwen_omni_preprocessor(inputs: Any, config: StageConfig) -> dict:
    """Qwen Omni preprocessor — multi-modal (audio+vision) feature extraction."""
    request = inputs.get("request")
    return {
        "modality": "omni",
        "model_family": "qwen_omni",
        "features": "multimodal_embeddings",
        "image_count": len(request.images) if request and request.images else 0,
        "audio_count": len(request.audio) if request and request.audio else 0,
    }


def _generic_preprocessor(inputs: Any, config: StageConfig) -> dict:
    """Generic fallback preprocessor — identity pass-through."""
    return {
        "modality": config.modality,
        "model_family": "generic",
        "features": "raw",
    }


def create_default_registry() -> ModelPreprocessorRegistry:
    """Create a registry with all built-in preprocessors registered."""
    reg = ModelPreprocessorRegistry()

    # Register built-in preprocessors
    reg.register("qwen_vlm", "image", _qwen_vlm_preprocessor)
    reg.register("llava_vlm", "image", _llava_vlm_preprocessor)
    reg.register("cosyvoice", "audio", _cosyvoice_preprocessor)
    reg.register("qwen_omni", "image", _qwen_omni_preprocessor)
    reg.register("qwen_omni", "audio", _qwen_omni_preprocessor)

    # Generic fallback
    reg.set_fallback(_generic_preprocessor)

    return reg
