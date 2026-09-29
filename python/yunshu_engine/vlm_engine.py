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
import concurrent.futures
import contextlib
import contextvars
import functools
import gc
import hashlib
import json
import logging
import os
import shutil
import ssl
import tempfile
import threading
import time
import urllib.request
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import mlx.core as mx

from . import settings
from .request import RequestOutput
from .types import EngineConfig

logger = logging.getLogger(__name__)

# Per-request temp-file registry. Set to a fresh list at the start of each
# generate()/generate_stream() call so cleanup deletes EXACTLY the files THIS request
# created — identity-based, not the old positional `_temp_files[offset:]` slice that
# raced (offset captured before the extraction `await`s, so two concurrent requests —
# or a single n>1 request via asyncio.gather — captured the same offset and one deleted
# the other's images mid-inference). ContextVars copy-on-task-create, so sibling tasks
# stay isolated. Registration still mirrors into the engine's global list for stop() sweep.
_request_temp_files: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "vlm_request_temp_files", default=None
)


class _RunnerCall(functools.partial):
    """A runner request prepared on the MLX thread, to be consumed off it.

    The batch runner's driver needs the (single) MLX executor, so the request
    must wait for its tokens on another thread; ``_generate_sync`` /
    ``_stream_sync`` return this instead of generating inline.
    """


def _quantize_shape_safety_patch():
    """Context manager that wraps `nn.Linear.to_quantized` so layers whose
    last-dim weight shape is not divisible by the quantization group size
    fall back to keeping bf16 weights (mixed-precision model) instead of
    raising a ValueError that aborts the whole model load.

    Why this exists
    ---------------
    mlx_vlm.utils.load_model gates which modules to quantize with this
    predicate (mlx_vlm 0.x):

        if hasattr(m, "weight") and m.weight.size % 64 != 0:
            return False

    It checks the **total element count** rather than the **last
    dimension**, but `mx.quantize` requires the *last* dim to be divisible
    by group_size. Shape (1152, 4304) has size 4,958,208 (passes the
    size-mod-64 check) but last dim 4304 % 64 == 16 (fails the real
    constraint). Result: every Qwen3-Omni-4bit and Qwen3.5-9B-MLX-4bit
    load aborted with `ValueError: [quantize] The last dimension of the
    matrix needs to be divisible by the quantization group size 64`.

    Patch strategy
    --------------
    Wrap `Linear.to_quantized` so it inspects shape[-1] before delegating
    to `QuantizedLinear.from_linear`. Bad shapes return self unchanged.

    This is a CONTEXT MANAGER (not a permanent monkey-patch) — we only
    apply it during mlx_vlm.load_model. The original method is restored
    on exit even if load raises.
    """
    import contextlib

    import mlx.nn as _nn

    @contextlib.contextmanager
    def _ctx():
        _orig = _nn.Linear.to_quantized

        def _safe_to_quantized(self, group_size=None, bits=None, mode="affine"):
            gs = group_size if group_size is not None else 64
            w = getattr(self, "weight", None)
            if w is not None and w.shape and w.shape[-1] % gs != 0:
                logger.debug(
                    "quantize-skip: Linear shape %s last-dim %d not divisible by gs=%d — keeping bf16",
                    tuple(w.shape),
                    w.shape[-1],
                    gs,
                )
                return self
            return _orig(self, group_size=group_size, bits=bits, mode=mode)

        _nn.Linear.to_quantized = _safe_to_quantized
        try:
            yield
        finally:
            _nn.Linear.to_quantized = _orig

    return _ctx()


def _VALIDATE_URL(url: str) -> None:
    """SSRF protection: reject URLs pointing to private/reserved IPs."""
    import ipaddress
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Blocked URL scheme: {parsed.scheme}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname")
    try:
        resolved = socket.getaddrinfo(
            hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
    except socket.gaierror:
        raise ValueError(f"Cannot resolve hostname: {hostname}") from None
    _PRIVATE_NETWORKS = [
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("169.254.0.0/16"),
        ipaddress.ip_network("0.0.0.0/8"),
        ipaddress.ip_network("::1/128"),
        ipaddress.ip_network("fc00::/7"),
        ipaddress.ip_network("fe80::/10"),
    ]
    for _, _, _, _, addr in resolved:
        ip = ipaddress.ip_address(addr[0])
        for net in _PRIVATE_NETWORKS:
            if ip in net:
                raise ValueError(
                    f"SSRF blocked: {hostname} resolves to private IP {ip}"
                )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """SECURITY: block SSRF-via-redirect. _VALIDATE_URL only checks the
    INITIAL host, but urllib follows 3xx by default — a permitted host could 302
    to http://169.254.169.254/… (cloud metadata) or any internal host. Returning
    None makes urllib surface the 3xx as an error instead of following it.
    (Mirrors routers/bench.py's _NoRedirect.)"""

    def redirect_request(self, *args, **kwargs):
        return None


def _VALIDATE_LOCAL_PATH(path: str) -> str:
    """SECURITY: validate that a local filesystem path used for VLM
    media input (image/audio/video) is within the configured `YUNSHU_MEDIA_DIR`.

    Without this check, an authenticated user can pass `file:///etc/passwd` or
    a bare `/etc/passwd` path and exfiltrate sensitive host files via the VLM
    output. With `YUNSHU_ALLOW_LOCAL_FILES` unset, only paths under
    `YUNSHU_MEDIA_DIR` (default: `$TMPDIR/yunshu_media`) are allowed. Set
    `YUNSHU_ALLOW_LOCAL_FILES=1` to opt back into the legacy unrestricted
    behavior (NOT recommended for multi-tenant deployments).

    Returns the resolved absolute path on success.
    Raises ValueError if the path escapes the allow-listed directory.
    """
    from pathlib import Path as _Path

    if settings.get_bool("YUNSHU_ALLOW_LOCAL_FILES"):
        return path

    from .paths import media_dir

    media_root = _Path(media_dir()).resolve()
    resolved = _Path(path).resolve()
    try:
        resolved.relative_to(media_root)
    except ValueError as e:
        raise ValueError(
            f"Local file access blocked: {path} is not under YUNSHU_MEDIA_DIR={media_root}. "
            f"Set YUNSHU_ALLOW_LOCAL_FILES=1 to bypass (NOT recommended in multi-tenant)."
        ) from e
    return str(resolved)


# Models that only support a single image input
SINGLE_IMAGE_ONLY_MODELS = frozenset(
    {
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
    }
)


class _VLMTextPromptCache:
    """Thread-safe LRU cache for VLM text prompt tokenization results.

    Caches the output of _format_prompt() + tokenizer.encode() keyed by a
    content hash of the input messages.  On cache hit, the expensive
    tokenization step is skipped entirely.

    Also caches the output of _processor.apply_chat_template() for the VLM
    vision path (VLM models with images/audio), keyed by messages + template
    kwargs hash.

    Memory-bounded with LRU eviction. Thread-safe via internal lock.
    """

    def __init__(self, max_entries: int = 256) -> None:
        from collections import OrderedDict

        self._cache: OrderedDict[str, list[int]] = OrderedDict()
        self._template_cache: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    @staticmethod
    def _compute_messages_hash(
        messages: list[dict], enable_thinking: bool | None = None
    ) -> str:
        """Stable hash of message content for cache keying."""
        import hashlib
        import json

        h = hashlib.blake2b(digest_size=16)
        h.update(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode())
        if enable_thinking is not None:
            h.update(str(enable_thinking).encode())
        return h.hexdigest()

    def get_token_ids(self, cache_key: str) -> list[int] | None:
        """Look up cached token IDs for a prompt."""
        with self._lock:
            entry = self._cache.get(cache_key)
            if entry is not None:
                self._cache.move_to_end(cache_key)
                self._stats["hits"] += 1
                return entry
            self._stats["misses"] += 1
            return None

    def put_token_ids(self, cache_key: str, token_ids: list[int]) -> None:
        """Store token IDs for a prompt."""
        with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                self._cache[cache_key] = token_ids
                return
            while self._cache and len(self._cache) >= self._max_entries:
                self._cache.popitem(last=False)
                self._stats["evictions"] += 1
            self._cache[cache_key] = token_ids

    def get_template_text(self, cache_key: str) -> str | None:
        """Look up cached chat template text for VLM vision path."""
        with self._lock:
            entry = self._template_cache.get(cache_key)
            if entry is not None:
                self._template_cache.move_to_end(cache_key)
                self._stats["hits"] += 1
                return entry
            self._stats["misses"] += 1
            return None

    def put_template_text(self, cache_key: str, template_text: str) -> None:
        """Store chat template text for VLM vision path."""
        with self._lock:
            if cache_key in self._template_cache:
                self._template_cache.move_to_end(cache_key)
                self._template_cache[cache_key] = template_text
                return
            while (
                self._template_cache and len(self._template_cache) >= self._max_entries
            ):
                self._template_cache.popitem(last=False)
                self._stats["evictions"] += 1
            self._template_cache[cache_key] = template_text

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._template_cache.clear()

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            s = dict(self._stats)
            s["tokenization_entries"] = len(self._cache)
            s["template_entries"] = len(self._template_cache)
            s["max_entries"] = self._max_entries
            return s


def _derive_vlm_quantization(config: dict) -> dict | None:
    """Resolve the effective ``quantization`` dict for a VLM checkpoint.

    _load_vision_model previously read only ``config.get("quantization")``,
    so a checkpoint that ships an HF-style ``quantization_config`` block (mxfp4 /
    compressed-tensors VLMs, possibly nested inside ``text_config``) but NO top-level
    ``quantization`` key loaded UN-quantized → model.load_weights() shape-mismatched
    against the packed/scales weights → a silent text-only mlx_lm fallback (vision lost)
    or a Metal fault. (The "mirrors mlx_vlm.utils.load_model" comment in _load_vision_model
    was only true for the affine ``quantization``-key case.) Mirror mlx_vlm's derivation;
    return None when the model is not quantized or uses an unsupported method.
    """
    q = config.get("quantization")
    if isinstance(q, dict):
        return q
    qc = config.get("quantization_config")
    if qc is None and isinstance(config.get("text_config"), dict):
        qc = config["text_config"].get("quantization_config")
    if not isinstance(qc, dict):
        return None
    qm = qc.get("quant_method")
    if qm == "compressed-tensors":
        return {"group_size": 32, "bits": 4, "mode": "affine"}
    if qm == "mxfp4":
        return {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    if qm in ("awq", "gptq", "bitnet"):
        logger.warning(
            "VLM quantization method %s is not supported by mlx; loading un-quantized",
            qm,
        )
    return None


class VLMEngine:
    """Multimodal (mlx-vlm) model engine.

    Every request is served by ``VLMBatchRunner`` (upstream mlx-vlm
    ``BatchGenerator``): shared continuous batching, APC prefix reuse, and —
    for Qwen3.5-family checkpoints — MTP/DFlash speculative decoding. Text-only
    models are served by ``BatchedEngine`` (model_manager routes them there).
    The pre-runner generation loops are recorded in
    docs/archive/legacy_vlm_loop/README.md.
    """

    def __init__(self, model_path: str, config: EngineConfig | None = None) -> None:
        self._model_path = model_path
        self._model = None
        self._tokenizer = None
        self._apc_backend = None
        self._apc_semantic_hash = None
        self._batch_runner = None
        self._processor = None
        self._config: dict = {}
        self._running = False
        self._active_count = 0
        self._active_count_lock = threading.Lock()
        self._num_requests_processed = 0
        self._total_reasoning_tokens = 0
        self._start_time = 0.0
        self._has_vision = False
        self._is_vlm = False
        self._temp_files: list[str] | None = None
        self._temp_files_lock = threading.Lock()

        # Text prompt tokenization cache — caches _format_prompt() output and
        # tokenizer.encode() results keyed by message content hash (and the
        # template extras), plus _processor.apply_chat_template() output for
        # media prompts. Avoids re-templating identical prompts.
        self._text_prompt_cache = _VLMTextPromptCache(
            max_entries=256,
        )
        # Memoized backbone serving capabilities (see model_backend.py).
        self._backend_caps: Any = None
        self._mx_large_model = False

        from .mlx_executor import get_mlx_executor

        self._executor = get_mlx_executor()

        # MultimodalPipelineCoordinator — unified 7-stage pipeline
        from .staged_pipeline import (
            MultimodalPipelineCoordinator,
        )

        self._pipeline = MultimodalPipelineCoordinator()
        self._register_pipeline_processors()

    @property
    def model_name(self) -> str:
        return (
            self._model_path.rsplit("/", 1)[-1]
            if "/" in self._model_path
            else self._model_path
        )

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
            PipelineStage.TEXT_PREPROCESS,
            "text",
            _text_preprocess,
        )
        self._pipeline.register_processor(
            PipelineStage.IMAGE_PREPROCESS,
            "image",
            _image_preprocess,
        )
        self._pipeline.register_processor(
            PipelineStage.AUDIO_PREPROCESS,
            "audio",
            _audio_preprocess,
        )

    def load(self) -> None:
        """Load the model, tokenizer and processor with mlx_vlm.

        Runs on the MLX executor thread so weights and compute share the same
        GPU stream. A model mlx_vlm cannot load is an error: text-only models
        belong to BatchedEngine, not here.
        """
        from mlx_lm.utils import load_config, load_tokenizer

        model_path = Path(self._model_path)

        # Download if HF repo ID (not local path)
        if not model_path.exists():
            from mlx_lm.utils import _download

            model_path = Path(_download(self._model_path))

        self._config = load_config(model_path)

        thinker_cfg = self._config.get("thinker_config", {})
        self._has_vision = bool(
            self._config.get("vision_config") or thinker_cfg.get("vision_config")
        )
        try:
            with _quantize_shape_safety_patch():
                self._model = self._load_vision_model(model_path)
        except Exception as e:
            err_str = str(e)
            if "needs to be divisible by the quantization group size" in err_str:
                raise RuntimeError(
                    f"VLM/Omni model {model_path.name} cannot be loaded: "
                    f"mlx_vlm quantization rejected weight shape "
                    f"({err_str.split('shape')[-1].strip().rstrip(').')} - "
                    f"last dim not divisible by group size 64). "
                    "Use a bf16 build of this model, or a 4-bit quant "
                    "produced with group_size matching the weight shapes."
                ) from e
            raise RuntimeError(
                f"mlx_vlm could not load {model_path.name}: {e}. Text-only models "
                "are served by the LLM engine (BatchedEngine), not VLMEngine."
            ) from e
        self._tokenizer = load_tokenizer(model_path)
        from .text_utils import cache_tokenizer_vocab

        cache_tokenizer_vocab(self._tokenizer)  # avoid ~98ms/req get_vocab rebuild
        self._resolve_reasoning_channel_ids()
        self._is_vlm = True
        logger.info("Loaded model via mlx_vlm (vision=%s)", self._has_vision)
        self._finish_vlm_load(model_path)

    def _load_vision_model(self, model_path):
        """Load a vision model using mlx_vlm with proper nested config handling."""
        # Apply mlx-vlm patches BEFORE load: the nemotron model_type remap must be
        # in MODEL_REMAPPING before get_model_and_args() reads the config, else
        # newer Nemotron omni variants fail "model type … not supported".
        # Idempotent — _finish_vlm_load's later call becomes a no-op.
        try:
            from .mlx_vlm_patches import apply_mlx_vlm_patches

            apply_mlx_vlm_patches()
        except Exception as e:
            logger.warning(f"Could not apply mlx-vlm patches pre-load: {e}")

        import glob

        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten as flatten_tree
        from mlx_vlm.utils import (
            get_model_and_args,
            update_module_configs,
        )
        from mlx_vlm.utils import (
            load_config as vlm_load_config,
        )

        config = vlm_load_config(model_path)

        weight_files = [
            wf
            for wf in glob.glob(str(Path(model_path) / "*.safetensors"))
            if not wf.endswith("consolidated.safetensors")
        ]
        if not weight_files:
            raise FileNotFoundError(f"No safetensors in {model_path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        model_class, _ = get_model_and_args(config=config)

        config.setdefault("text_config", config.pop("llm_config", {}))
        config.setdefault("vision_config", {})
        config.setdefault("audio_config", {})

        model_config = model_class.ModelConfig.from_dict(config)
        modules = ["text", "vision", "perceiver", "projector", "audio"]
        model_config = update_module_configs(model_config, model_class, config, modules)

        model = model_class.Model(model_config)

        # Apply quantization BEFORE filtering — the predicate needs to see
        # `.scales` keys to decide which modules to quantize. If we filter
        # first, scales/biases get dropped (they're not in the bf16 model's
        # parameters() yet) and the predicate falls through to bf16,
        # producing shape-mismatch on load_weights.
        #
        # The predicate mirrors mlx_vlm.utils.load_model so quantization
        # only touches modules whose scales exist in the saved file
        # (e.g. language_model.* on Qwen3.5-9B-4bit, while vision_tower.*
        # stays bf16 as authored).
        # derive the quantization dict (handles mxfp4 / compressed-tensors
        # quantization_config, incl. text_config nesting) instead of reading only the
        # top-level "quantization" key — see _derive_vlm_quantization.
        quantization = _derive_vlm_quantization(config)
        if quantization is not None:
            config["quantization"] = (
                quantization  # so the per-layer predicate (p in ...) holds
            )

            def _quantize_predicate(p, m):
                # honor PER-LAYER quantization overrides. A model's
                # `quantization` config can map specific module paths to their own
                # settings (a dict) or to False (don't quantize). Qwen3-Omni-30B
                # (MoE) ships these per-path entries; the previous predicate
                # ignored them and fell through to the scales heuristic, mis-
                # quantizing the MoE experts → garbage weights → a Metal GPU
                # Address Fault on the first forward (the model loads fine via
                # mlx_vlm.load, which DOES honor this). Mirror
                # mlx_vlm.utils.get_class_predicate.
                if isinstance(quantization, dict) and p in quantization:
                    return quantization[p]
                if not hasattr(m, "to_quantized"):
                    return False
                if hasattr(m, "weight") and m.weight.size % 64 != 0:
                    return False
                return f"{p}.scales" in weights

            nn.quantize(
                model,
                group_size=quantization.get("group_size", 64),
                bits=quantization.get(
                    "bits", 4
                ),  # .get — a per-layer-only dict has no top-level bits
                mode=quantization.get("mode", "affine"),
                class_predicate=_quantize_predicate,
            )

        # NOW the model's parameter set reflects QuantizedLinear's
        # (weight + scales + biases), so we can safely drop weights the
        # model does not expect (e.g. MTP heads).
        model_params = set(dict(flatten_tree(model.parameters())).keys())
        weight_params = set(dict(weights).keys())
        extra = weight_params - model_params
        if extra:
            logger.info(
                f"Filtering {len(extra)} unexpected weights: {list(extra)[:5]}..."
            )
            weights = {k: v for k, v in weights.items() if k in model_params}

        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        model.eval()
        return model

    def _finish_vlm_load(self, model_path) -> None:
        """Common post-load initialization."""

        # Apply pinned-mlx-vlm runtime patches (Omni audio path). Idempotent;
        # no-op for non-audio / non-Omni requests. See mlx_vlm_patches.py.
        try:
            from .mlx_vlm_patches import apply_mlx_vlm_patches

            apply_mlx_vlm_patches()
        except Exception as e:
            logger.warning(f"Could not apply mlx-vlm patches: {e}")

        # The processor prepares image/audio/video inputs and renders media
        # chat templates; every mlx_vlm model needs it for media requests.
        try:
            from mlx_vlm.utils import load_processor

            self._processor = load_processor(Path(model_path))
        except Exception as e:
            logger.warning(f"Could not load VLM processor: {e}")

        # Large VLMs (e.g. 30B-MoE) GPU-hang under sustained load unless the MLX
        # buffer pool is released between requests. Use the on-disk weight size
        # (an MoE's .parameters() can undercount lazily-structured experts);
        # the runner clears the pool whenever its batch drains.
        try:
            import glob as _glob

            _bytes = 0
            for _f in _glob.glob(os.path.join(self._model_path, "*.safetensors")):
                with contextlib.suppress(OSError):
                    _bytes += os.path.getsize(_f)
            _thresh = 10 * 1024**3
            self._mx_large_model = _bytes > _thresh
            logger.info(
                "VLM model ~%.1fGB on disk, large=%s (clear buffer pool on idle %s)",
                _bytes / 1e9,
                self._mx_large_model,
                "ON" if self._mx_large_model else "off",
            )
        except Exception:
            logger.debug("VLM model-size probe failed", exc_info=True)

        logger.info(
            f"VLM engine loaded: {self._model_path} (vision={self._has_vision})"
        )
        # Every request goes through the runner; a model it cannot serve fails
        # at load instead of silently taking another path.
        self._batch_runner = self._build_batch_runner(str(model_path))

    async def start(self) -> None:
        if self._model is not None:
            self._running = True
            self._start_time = time.monotonic()
            return
        # Always load on the MLX executor thread — required for mlx-vlm models
        # whose weights must share the same GPU stream as compute
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self.load)
        except Exception:
            # load() failed — do NOT set _running=True.  The engine has no
            # model and must not be used.  Re-raise so the caller knows.
            self._running = False
            raise
        self._running = True
        self._start_time = time.monotonic()

    async def stop(self) -> None:
        """Stop and release resources.

        Idempotent: safe to call multiple times.
        """
        if not self._running and self._model is None:
            return
        # Wait for active requests to complete before releasing model
        # to prevent use-after-free on _model during in-flight inference.
        import time as _time_mod

        deadline = _time_mod.monotonic() + 10.0
        while self.has_active_requests() and _time_mod.monotonic() < deadline:
            await asyncio.sleep(0.1)
        if self.has_active_requests():
            logger.warning("VLM stop(): active requests still pending after 10s wait")
        self._cleanup_temp_files()
        self._model = None
        self._tokenizer = None
        self._apc_backend = None
        self._apc_semantic_hash = None
        self._batch_runner = None
        self._processor = None
        self._running = False

        self._text_prompt_cache.clear()
        self._backend_caps = None

        gc.collect()
        loop = asyncio.get_running_loop()
        from .mlx_executor import sync_and_clear_cache

        await loop.run_in_executor(self._executor, sync_and_clear_cache)

    def resolve_model_id(self, model_id: str) -> bool:
        return model_id in {
            self.model_name,
            self._model_path,
            self.model_name.lower(),
            self._model_path.lower(),
        }

    # ── Generation ──

    def _apc_capacity_allows(self, input_ids) -> bool:
        """Avoid costly chunked prefill when this Qwen cache cannot be retained.

        Calibrated from 8K/32K/64K checkpoint resident-byte probes. Keep a
        matching existing checkpoint eligible even when the full new prompt
        exceeds budget: partial prefix reuse can still save prefill time.
        """
        manager = self._apc_backend
        if input_ids is None or manager is None:
            return True
        config = self._config.get("text_config", {})
        if not all(
            (
                config.get("model_type") == "qwen3_5_text",
                config.get("num_hidden_layers") == 64,
                config.get("full_attention_interval") == 4,
                config.get("num_key_value_heads") == 4,
                config.get("head_dim") == 256,
                config.get("hidden_size") == 5120,
            )
        ):
            return True
        budget = getattr(manager, "memory_max_bytes", None)
        if budget is None:
            return True
        # Empirical resident size: ~160 MiB + 130 KiB per token (within 0.6%
        # of three measured points). The 20% margin covers short history growth
        # and prevents a nominal fit from paying chunk overhead but not storing.
        estimated = (160 << 20) + len(input_ids) * (130 << 10)
        if estimated * 1.2 <= budget:
            return True
        try:
            tokens = tuple(input_ids.tolist())
            with manager.lock:
                return any(
                    entry.extra_hash == self._apc_semantic_hash
                    and len(entry.token_ids) >= 512
                    and len(entry.token_ids) < len(tokens)
                    and tokens[: len(entry.token_ids)] == entry.token_ids
                    for entry in manager._exact_cache.values()
                )
        except Exception:
            logger.debug("APC capacity prefix check unavailable", exc_info=True)
            return False

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
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif prompt is not None:
            messages = prompt
        else:
            messages = messages or []
        if self._model is None:
            raise RuntimeError("Engine not started")
        tpl_extra = self._request_template_extra(kwargs)
        # A timeout below must be able to stop the generation it abandons.
        if kwargs.get("cancel_event") is None:
            kwargs["cancel_event"] = threading.Event()

        with self._active_count_lock:
            self._active_count += 1
        # Per-request temp-file tracking (identity-based, race-free). _register_temp_file
        # appends every file this request creates into this list; cleanup deletes exactly
        # those — not a positional slice that collides with concurrent/n>1 requests.
        _req_temp_files: list[str] = []
        _temp_token = _request_temp_files.set(_req_temp_files)

        try:
            image_paths = await self._extract_images(messages)
            audio_paths = await self._extract_audio(messages)
            video_frames = await self._extract_video_frames(messages)
            image_paths.extend(video_frames)
            _enable_thinking = self._default_enable_thinking(
                enable_thinking, constrained=kwargs.get("json_schema") is not None
            )
            self._track_pipeline(kwargs, messages, image_paths, audio_paths)
            self._check_request_supported(image_paths, audio_paths, kwargs)

            _runner_extras: dict = {}
            runner_params = self._runner_kwargs(
                max_tokens=max(1, int(max_tokens)),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                seed=seed,
                stop=stop,
                stop_token_ids=kwargs.get("stop_token_ids") or [],
                repetition_penalty=repetition_penalty,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                enable_thinking=_enable_thinking,
                thinking_budget=kwargs.get("thinking_budget"),
                cancel_event=kwargs["cancel_event"],
                kwargs=kwargs,
            )

            def _prepare():
                # Templating + media encoding on the MLX thread; generation is
                # then consumed off it (the runner's driver needs that thread).
                ids, pkw, salt = self._runner_input(
                    messages, image_paths, audio_paths, _enable_thinking, tpl_extra
                )
                return _RunnerCall(
                    self._generate_vlm_runner_text,
                    ids,
                    extras=_runner_extras,
                    prompt_kwargs=pkw,
                    apc_semantic_hash=salt,
                    **runner_params,
                )

            loop = asyncio.get_running_loop()
            try:
                # Only a client-set timeout applies: a fixed default would also
                # count time spent waiting for a batch slot.
                _timeout_seconds = kwargs.get("timeout_seconds") or None
                call = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, _prepare),
                    timeout=_timeout_seconds,
                )
                (
                    result,
                    reasoning_tokens,
                    completion_token_count,
                    stop_hit,
                    budget_hit,
                    cached_token_count,
                ) = await asyncio.wait_for(
                    loop.run_in_executor(self._runner_consumers(), call),
                    timeout=_timeout_seconds,
                )
            except TimeoutError:
                self._num_requests_processed += 1
                kwargs["cancel_event"].set()
                logger.warning(
                    f"VLM non-streaming generate timed out after {_timeout_seconds}s"
                )
                return {
                    "text": "",
                    "finish_reason": "timeout",
                    "model": self.model_name,
                    "created": int(time.time()),
                    "reasoning_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
            except Exception:
                self._num_requests_processed += 1
                raise

            self._num_requests_processed += 1
            self._total_reasoning_tokens += reasoning_tokens
            # The runner reports the exact prompt length (incl. media tokens).
            prompt_tokens = int(_runner_extras.get("prompt_tokens") or 0)
            # Determine correct finish_reason based on exit condition
            _finish_reason = "stop" if stop_hit or budget_hit else "length"
            # record ServerMetrics for VLM NON-streaming. Only
            # generate_stream recorded (line ~1934); when the gateway stopped double-recording
            # (commit 37568f5), non-streaming VLM requests stopped being counted at all. The
            # engine is the single source of truth (like BatchedEngine._generate_fast).
            try:
                from .server_metrics import get_server_metrics

                get_server_metrics().record_request_complete(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_token_count,
                    model_id=self.model_name,
                )
            except Exception:
                pass
            return {
                "text": result,
                "finish_reason": _finish_reason,
                "model": self.model_name,
                "created": int(time.time()),
                "reasoning_tokens": reasoning_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_token_count,
                "cached_tokens": cached_token_count,
                "logprobs": _runner_extras.get("logprobs"),
            }
        finally:
            with self._active_count_lock:
                self._active_count = max(0, self._active_count - 1)
            _request_temp_files.reset(_temp_token)
            # Our files are exactly those THIS request registered (identity-based).
            _mine = list(_req_temp_files)
            if _mine:
                _mine_set = set(_mine)
                with self._temp_files_lock:
                    if self._temp_files is not None:
                        self._temp_files[:] = [
                            f for f in self._temp_files if f not in _mine_set
                        ]
            self._cleanup_temp_files(_mine)

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
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif prompt is not None:
            messages = prompt
        else:
            messages = messages or []
        if self._model is None:
            raise RuntimeError("Engine not started")

        tpl_extra = self._request_template_extra(kwargs)
        if cancel_event is None:
            cancel_event = threading.Event()

        # Extract images/audio once, reuse for both pipeline and generation.
        # Per-request identity-based temp tracking (see generate()): avoids the
        # positional-offset race that deleted concurrent/n>1 requests' files.
        _req_temp_files: list[str] = []
        _temp_token = _request_temp_files.set(_req_temp_files)
        image_paths = await self._extract_images(messages)
        audio_paths = await self._extract_audio(messages)
        video_frames = await self._extract_video_frames(messages)
        image_paths.extend(video_frames)

        enable_thinking = self._default_enable_thinking(
            enable_thinking, constrained=kwargs.get("json_schema") is not None
        )
        self._track_pipeline(kwargs, messages, image_paths, audio_paths)
        self._check_request_supported(image_paths, audio_paths, kwargs)

        import uuid

        req_id = f"vlm-{uuid.uuid4().hex[:8]}"

        queue: asyncio.Queue[RequestOutput | None] = asyncio.Queue(maxsize=256)

        # Capture event loop for thread-safe queue writes from executor thread.
        # asyncio.Queue.put_nowait() is NOT thread-safe — must schedule puts
        # via call_soon_threadsafe (same pattern as batched_engine.py).
        _loop_for_queue = asyncio.get_running_loop()

        class _ThreadSafeQueue:
            """Wraps asyncio.Queue with thread-safe put_nowait."""

            def __init__(self, async_q, ev_loop):
                self._q = async_q
                self._loop = ev_loop

            def put_nowait(self, item):
                with contextlib.suppress(
                    RuntimeError
                ):  # event loop closed during shutdown
                    self._loop.call_soon_threadsafe(self._q.put_nowait, item)

            def put_blocking(self, item, cancel):
                """Bound the runner's producer by the async consumer, including slow clients."""
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._q.put(item), self._loop
                    )
                except RuntimeError:
                    return False
                while True:
                    try:
                        future.result(timeout=0.05)
                        return True
                    except TimeoutError:
                        if cancel is not None and cancel.is_set():
                            future.cancel()
                            return False
                    except Exception:
                        return False

        _safe_queue = _ThreadSafeQueue(queue, _loop_for_queue)

        runner_params = self._runner_kwargs(
            max_tokens=max(1, int(max_tokens)),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
            stop=stop,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            enable_thinking=enable_thinking,
            thinking_budget=kwargs.get("thinking_budget"),
            cancel_event=cancel_event,
            kwargs={
                **kwargs,
                "xtc_probability": xtc_probability,
                "xtc_threshold": xtc_threshold,
            },
        )

        def _prepare():
            ids, pkw, salt = self._runner_input(
                messages, image_paths, audio_paths, enable_thinking, tpl_extra
            )
            return _RunnerCall(
                self._stream_vlm_runner_text,
                ids,
                req_id,
                _safe_queue,
                prompt_kwargs=pkw,
                apc_semantic_hash=salt,
                **runner_params,
            )

        with self._active_count_lock:
            self._active_count += 1
        loop = asyncio.get_running_loop()

        async def _stream_job():
            try:
                # Templating + media encoding on the MLX thread; tokens are then
                # consumed off it (the runner's driver needs that thread).
                call = await loop.run_in_executor(self._executor, _prepare)
            except Exception as e:
                logger.error(f"VLM stream error: {e}", exc_info=True)
                _safe_queue.put_nowait(
                    RequestOutput(
                        request_id=req_id,
                        new_text="",
                        finish_reason="error",
                        finished=True,
                        error=str(e),
                    )
                )
                _safe_queue.put_nowait(None)
                return

            def _consume():
                try:
                    call()
                finally:
                    with contextlib.suppress(Exception):
                        _safe_queue.put_nowait(None)

            await loop.run_in_executor(self._runner_consumers(), _consume)

        stream_task = asyncio.ensure_future(_stream_job())

        # The executor owns the model lease. A disconnected ASGI task can be
        # cancelled during an await in the async generator's cleanup, so its
        # finally block cannot be the sole owner of this counter. Release only
        # after the Metal worker actually exits.
        def _release_worker_lease(_future):
            with self._active_count_lock:
                self._active_count = max(0, self._active_count - 1)

        stream_task.add_done_callback(_release_worker_lease)

        # the gateway passes timeout_seconds=req.timeout (chat.py), but this
        # read the wrong key 'timeout' → a user-set per-request timeout was SILENTLY
        # ignored and the inactivity timeout was permanently hardcoded to 300s. The
        # non-streaming twin (generate, ~line 1578) correctly reads 'timeout_seconds'.
        # Only a client-set timeout applies: a fixed default also fired during
        # legitimate long prefills (a 200K-token prompt takes ~390 s before its
        # first token); a gone client is handled by the disconnect cancel.
        _timeout_seconds = (
            kwargs.get("timeout_seconds") or kwargs.get("timeout") or None
        )
        _prompt_tokens_count = 0
        _completion_tokens_count = 0
        _model_id = self.model_name

        _ended_normally = False

        try:
            while True:
                try:
                    output = await asyncio.wait_for(
                        queue.get(), timeout=_timeout_seconds
                    )
                except TimeoutError:
                    logger.warning(
                        f"VLM stream timeout: no token for {_timeout_seconds}s"
                    )
                    # Stop the runner row; it only stops when it sees the event.
                    cancel_event.set()
                    yield RequestOutput(
                        request_id=req_id,
                        new_text="",
                        finish_reason="error",
                        finished=True,
                        error=f"Streaming timeout: no token for {_timeout_seconds}s",
                    )
                    break
                if output is None:
                    _ended_normally = True
                    break
                if output.finished and output.finish_reason != "error":
                    _ended_normally = True
                if output.prompt_tokens > 0:
                    _prompt_tokens_count = output.prompt_tokens
                if output.completion_tokens > 0:
                    _completion_tokens_count = output.completion_tokens
                # Record TTFT in Prometheus on the first token
                if output.ttft_ms > 0:
                    try:
                        from yunshu_gateway.middleware.prometheus_exporter import (
                            get_prometheus_metrics,
                        )

                        get_prometheus_metrics().observe_histogram(
                            "ttft_seconds",
                            output.ttft_ms / 1000.0,
                            labels={"model_id": self.model_name},
                        )
                    except Exception:
                        pass
                yield output
        finally:
            # Record in ServerMetrics for VLM streaming
            try:
                from .server_metrics import get_server_metrics

                get_server_metrics().record_request_complete(
                    prompt_tokens=_prompt_tokens_count,
                    completion_tokens=_completion_tokens_count,
                    model_id=_model_id,
                )
            except Exception:
                pass
            if not stream_task.done():
                # Only an abnormal exit (client gone, error, timeout) cancels. The
                # cancel event is shared with the router, which reads it after the
                # stream ends: setting it on a normal finish (runner still winding
                # down) reported a spurious "cancelled".
                if not _ended_normally:
                    cancel_event.set()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.shield(stream_task)
            # Drain remaining queue items to unblock the executor thread
            # so it can observe the cancellation and exit promptly.
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            # This async generator is driven step-by-step from a separate task by
            # the SSE keepalive wrapper, so this finally can run in a different
            # contextvars.Context than the one .set() ran in — reset() would then
            # raise "Token was created in a different Context". File cleanup below
            # uses the closure list, not the ContextVar, so a failed reset is safe.
            with contextlib.suppress(ValueError):
                _request_temp_files.reset(_temp_token)
            _mine = list(_req_temp_files)
            if _mine:
                _mine_set = set(_mine)
                with self._temp_files_lock:
                    if self._temp_files is not None:
                        self._temp_files[:] = [
                            f for f in self._temp_files if f not in _mine_set
                        ]
            self._cleanup_temp_files(_mine)

    def backend_capabilities(self, lm: Any = None) -> Any:
        """This backbone's cache layout via the shared ``model_backend`` layer
        (memoized); the runner build uses it to skip APC for sliding-window
        caches."""
        if self._backend_caps is not None:
            return self._backend_caps
        from .model_backend import BackendKind, derive_capabilities

        if lm is None:
            lm = self._model.language_model
        layers = []
        try:
            from mlx_vlm.models.cache import make_prompt_cache

            layers = make_prompt_cache(lm)
        except Exception:
            logger.debug("VLM cache probe failed; assuming non-reusable", exc_info=True)
        # A get_rope_index / _rope_deltas method means position lives outside
        # the cache (mRoPE-style).
        is_mrope = hasattr(lm, "get_rope_index") or hasattr(lm, "_rope_deltas")
        caps = derive_capabilities(BackendKind.VLM, layers, is_mrope=is_mrope)
        self._backend_caps = caps
        return caps

    # ── Unified BatchGenerator text path (APC + MTP + sampling + constraints) ──

    # Checkpoints whose speculative decoding (MTP / DFlash draft, verify and
    # batch-invariant kernels) is validated. Every family uses the runner.
    _SPEC_MODEL_TYPES = ("qwen3_5", "qwen3_6", "qwen3_5_moe")

    def _apc_disk_tier(self):
        """Optional APC SSD tier (``YUNSHU_VLM_APC_DISK_DIR``), off by default.

        Holds evicted prefix checkpoints (incl. hybrid recurrent state) so a
        long document revisited after RAM eviction is read back instead of
        re-prefilled. Namespaced by model path; capped by
        ``YUNSHU_VLM_APC_DISK_GB``.
        """
        path = settings.get("YUNSHU_VLM_APC_DISK_DIR")
        if not path:
            return None
        import hashlib

        from mlx_vlm.apc import DiskBlockStore

        max_gb = settings.get("YUNSHU_VLM_APC_DISK_GB")
        namespace = hashlib.sha256(str(self._model_path).encode()).hexdigest()[:16]
        try:
            disk = DiskBlockStore(
                Path(path).expanduser(),
                namespace=namespace,
                num_workers=1,
                max_bytes=int(max_gb * (1 << 30)) if max_gb > 0 else None,
            )
        except Exception:
            logger.warning("APC disk tier unavailable at %s", path, exc_info=True)
            return None
        logger.info("APC disk tier at %s (cap %.0f GiB)", disk.dir, max_gb)
        return disk

    def _round_driver_wanted(self, lm) -> bool:
        """``YUNSHU_ROUND_DRIVER`` on a dense Qwen3.5-family text decoder with
        bf16 KV (int8 KV precision keeps the upstream path)."""
        if not settings.get_bool("YUNSHU_ROUND_DRIVER"):
            return False
        from .round_driver import forward as rd_forward

        if not rd_forward.supports(lm):
            logger.warning("YUNSHU_ROUND_DRIVER: not a dense Qwen3.5-family model")
            return False
        if settings.get("YUNSHU_KV_PRECISION") != "bf16":
            logger.warning("YUNSHU_ROUND_DRIVER: int8 KV keeps the upstream path")
            return False
        return True

    def _build_batch_runner(self, model_path: str):
        """Build the runner that serves every request of this model.

        APC (prefix cache, ``YUNSHU_VLM_APC_MEMORY_GB``; 0 disables it) for every
        family whose cache has no sliding window. Speculative decoding
        (checkpoint MTP head, or ``YUNSHU_VLM_DRAFT``) and its verify kernels are
        Qwen3.5-family only (``_SPEC_MODEL_TYPES``); ``YUNSHU_MTP=0`` disables the
        MTP draft.
        """
        from .vlm_batch_runner import VLMBatchRunner

        spec_family = self._config.get("model_type") in self._SPEC_MODEL_TYPES
        lm = self._model.language_model
        use_driver = self._round_driver_wanted(lm)
        if use_driver:
            # Row-invariant lane projections everywhere (before any verify
            # kernel install repacks them): the round driver's rows, and the
            # upstream paths that still serve image prompts, both use them.
            from .kernels import lane_linear

            lanes = lane_linear.convert(lm)
            if lm.args.tie_word_embeddings:
                lm._yunshu_lane_head = lane_linear.lane_head(lm.model.embed_tokens)
            logger.info(
                "Round driver: %d lane projections (%d skipped)",
                lanes["converted"],
                len(lanes["skipped"]),
            )
        budget = settings.get("YUNSHU_VLM_APC_MEMORY_GB")
        if self._apc_backend is None and budget > 0:
            from mlx_vlm.apc import APCManager, semantic_extra_hash

            # Sliding-window (rotating) caches cannot be checkpointed at a
            # prefix boundary, so those families decode without APC.
            if not self.backend_capabilities(lm).cache.has_sliding_window:
                self._apc_backend = APCManager(
                    num_blocks=512,
                    block_size=16,
                    disk=self._apc_disk_tier(),
                    overrides={"memory_max_gb": budget},
                )
                self._apc_semantic_hash = semantic_extra_hash(
                    image_hash=0,
                    media={"audio": None, "video": None},
                    model=lm,
                    processor=self._processor,
                )
        drafter = None
        draft_kind = "mtp"
        from . import spec_select
        from .mlxvlm_mtp import is_mtp_capable

        choice = spec_select.choose(
            self._config,
            spec_family=spec_family,
            mtp_capable=spec_family and is_mtp_capable(model_path),
        )
        external = choice.drafter
        if choice.kind != "none":
            logger.info("Speculative decoding: %s (%s)", choice.kind, choice.reason)
        if external:
            # DFlash drafter directory, e.g. incoai/Qwen3.8-27B-DFlash2.
            from mlx_vlm.speculative.drafters import (
                load_drafter,
                validate_drafter_compatibility,
            )

            try:
                drafter, draft_kind = load_drafter(external)
                validate_drafter_compatibility(self._model, drafter, draft_kind)
            except Exception:
                if not choice.automatic:
                    raise
                logger.warning(
                    "DFlash drafter %s is not usable; falling back to MTP",
                    external,
                    exc_info=True,
                )
                drafter, draft_kind = None, "mtp"
            if draft_kind == "dflash":
                # 8-bit drafter: drafts are verified, so this only trades a
                # little acceptance for half the drafter bytes per cycle
                # (27B server: faster than the shipped weights at 1K-32K).
                from .dflash_tree import quantize_drafter

                quantize_drafter(drafter, 8)
                # Project only the context window the drafter attends to.
                from .dflash_context import install as install_dflash_context

                install_dflash_context(self._model.language_model)
        if drafter is None and choice.kind in ("mtp", "dflash"):
            # "dflash" here means the automatic drafter failed to load.
            from mlx_vlm.speculative.drafters import validate_drafter_compatibility

            from .mlxvlm_mtp import _load_drafter_in_memory

            if is_mtp_capable(model_path):
                drafter = _load_drafter_in_memory(model_path)
                validate_drafter_compatibility(self._model, drafter, "mtp")
        block = settings.get("YUNSHU_MTP_BLOCK_SIZE")
        kernels = None
        if drafter is not None:
            from .kernels.omlx import apply as apply_verify_kernels

            # Experimental alternative (YUNSHU_MTP_ROW_EXACT): upstream oMLX
            # row-exact verify keeps the stock decode path and computes each
            # verify row with one-row decode arithmetic.
            row_exact = settings.get_bool("YUNSHU_MTP_ROW_EXACT")
            kernels = apply_verify_kernels(row_exact=row_exact)
            # Lossless MTP decode: decode and verify share row-invariant kernels,
            # so speculative output == this engine's plain decode. On Qwen3.8-27B
            # (M5 Max) invariant + NAX-packed at block 6 decodes code/prose/json
            # at 88.6/59.9/67.3 tok/s vs 57-67/50-53/58-62 for exact kernels at
            # block 3 and 83.8/59.9/66.7 for the non-exact fast verify, with
            # parity on every task
            # (docs/research/runs/2026-09-28-matrix/invariant-packed-mtp-sweep.jsonl).
            # DFlash verifies through the same target forward, so it gets the
            # same guarantee.
            invariant = not row_exact
            if invariant:
                from .kernels.batch_invariant import install as install_invariant
                from .kernels.batch_invariant import set_active
                from .kernels.omlx import is_nax_available

                kernels["invariant"] = install_invariant(
                    self._model.language_model,
                    model=self._model,
                    packed=is_nax_available(),
                )
                # The runner turns them on only while its speculative lane steps.
                set_active(False)
            else:
                # Exact: 5-bit layers use the fixed streamed kernel for >= 5 verify rows.
                from .kernels.verify_select import install as install_streamed5

                kernels["streamed5"] = install_streamed5()
            if draft_kind == "mtp" and invariant:
                # The lane's own MTP rounds (one host read per cycle, the head
                # run over every verify row inside the verify's graph), fused
                # residual+norm layers and a reduced draft vocabulary. Same
                # tokens as upstream's loop; Qwen3.8-27B, single requests, HTTP
                # decode tok/s 1K/8K/32K/131K: code_python 45.8/68.6/53.4/55.9
                # -> 55.2/69.5/62.5/59.7, novel_en 48.5/42.2/43.3/32.7 ->
                # 55.3/49.9/44.5/35.0 (docs/research/runs/2026-09-29-step-efficiency).
                from . import mtp_lane
                from .draft_vocab import DRAFT_VOCAB
                from .draft_vocab import install as install_draft_vocab
                from .kernels import lane_layers

                kernels["mtp_lane"] = mtp_lane.install()
                kernels["lane_layers"] = lane_layers.install()
                found = install_draft_vocab(
                    drafter, self._model.language_model, DRAFT_VOCAB
                )
                kernels["draft_vocab"] = DRAFT_VOCAB if found else None
            # Invariant kernels cost little per extra verify row and peak at 6
            # for MTP; exact kernels peak at 3: at 5 rows a cycle jumps to
            # ~108 ms. A DFlash drafter proposes a whole block in one forward
            # and upstream adapts the depth to acceptance under this ceiling,
            # so the ceiling is the block it was trained on.
            if block is None:
                if draft_kind == "dflash":
                    block = int(getattr(drafter.config, "block_size", 0)) or None
                else:
                    block = 6 if invariant else 3
        if drafter is not None and draft_kind == "dflash":
            # Cost-aware chain depth from measured cycle costs (27B server:
            # 57/48/46 vs upstream adaptive 46/38/38 tok/s at 1K/8K/32K).
            from .spec_schedule import install_chain_budget

            install_chain_budget()
        if drafter is not None:
            # Tree drafts through the tree verify (single greedy row,
            # batch-invariant kernels only; every other round keeps the loop).
            if settings.get("YUNSHU_SPEC_TREE") == "tree":
                if draft_kind == "dflash":
                    from .dflash_tree import install as install_tree
                else:
                    from .mtp_tree import install as install_tree
                install_tree()
        runner = VLMBatchRunner(
            self._model,
            self._processor,
            apc_manager=self._apc_backend,
            apc_semantic_hash=self._apc_semantic_hash,
            drafter=drafter,
            draft_block_size=int(block) if block else None,
            apc_admit=self._apc_capacity_allows,
            draft_kind=draft_kind,
            executor=self._executor,
        )
        runner.clear_on_idle = bool(getattr(self, "_mx_large_model", False))
        runner.stop_tokens = set(self._get_eos_ids())
        runner.inflight = lambda: self._active_count
        # Per-row-length (ragged) KV + ragged decode attention for the shared
        # batch and the speculative lane: the layout for every model with
        # qwen3_5 attention (lossless; docs/research/runs/2026-09-29-ragged-idle).
        # KV precision is the user's memory/quality choice.
        precision = settings.get("YUNSHU_KV_PRECISION")
        from .kernels import ragged_kv

        if ragged_kv.supports(self._model.language_model):
            ragged_kv.install()
            ragged_kv.enable(None)  # the runner sets the format while it steps
            runner.ragged_kv = precision
        elif precision != "bf16":
            logger.warning(
                "YUNSHU_KV_PRECISION=%s applies to Qwen3.5-family attention "
                "only; this model's KV stays bf16",
                precision,
            )
        if use_driver:
            from .round_driver.driver import RoundDriver

            runner.driver = RoundDriver(
                self._model,
                drafter=drafter if draft_kind == "mtp" else None,
                stop_tokens=runner.stop_tokens,
                chunk=settings.get("YUNSHU_ROUND_PREFILL_CHUNK"),
            )
        logger.info(
            "VLM batch runner: apc=%s draft=%s block=%s verify_kernels=%s",
            f"{self._apc_backend.memory_max_bytes / 2**30:.1f}GiB"
            if self._apc_backend is not None
            else "off",
            draft_kind if drafter is not None else "off",
            block or "default",
            kernels,
        )
        return runner

    # Knobs the runner does not implement; such requests take the legacy loop.
    # (spec_decode is accepted and ignored: the runner drafts on its own.)
    _RUNNER_UNSUPPORTED_KWARGS = ("logits_processors", "lora_adapter")

    def _build_text_constraint(self, json_schema):
        if json_schema is None:
            return None
        try:
            if isinstance(json_schema, dict) and json_schema.get("type") in (
                "regex",
                "choice",
                "cfg",
            ):
                from .grammar_constraint import ConstraintFactory

                gtype = json_schema["type"]
                grammar = {
                    "regex": json_schema.get("pattern", ""),
                    "choice": json_schema.get("choices", []),
                    "cfg": json_schema.get("grammar", ""),
                }[gtype]
                return ConstraintFactory.create(gtype, grammar, self._tokenizer)
            from .json_schema import JsonSchemaConstraint

            if isinstance(json_schema, str) and json_schema == "json_object":
                return JsonSchemaConstraint(None)
            return JsonSchemaConstraint(json_schema)
        except Exception as exc:
            raise ValueError("Grammar constraint initialization failed") from exc

    def _reasoning_markers(self) -> tuple[int | None, int | None, str, str, bool]:
        """Single-token reasoning open/close ids for this tokenizer.

        ``<think>`` / ``</think>`` (Qwen, most models) or Gemma-4's
        ``<|channel>`` / ``<channel|>`` (whose reasoning starts with a
        ``thought`` channel label). Returns ``(open_id, close_id, open_text,
        close_text, channel)``; ids are None when the model has neither."""
        cached = getattr(self, "_reasoning_markers_cache", None)
        if cached is None:
            try:
                ts = self._tokenizer.encode("<think>", add_special_tokens=False)
                te = self._tokenizer.encode("</think>", add_special_tokens=False)
            except Exception:
                ts = te = []
            if len(ts) == 1 and len(te) == 1:
                cached = (ts[0], te[0], "<think>", "</think>", False)
            else:
                co, cc = getattr(self, "_reasoning_channel_ids", (None, None))
                if co is not None and cc is not None:
                    cached = (co, cc, "<|channel>", "<channel|>", True)
                else:
                    cached = (None, None, "<think>", "</think>", False)
            self._reasoning_markers_cache = cached
        return cached

    def _runner_events(
        self,
        input_ids: mx.array,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        seed: int | None,
        stop: list[str] | None,
        stop_token_ids: list[int] | None,
        repetition_penalty: float,
        frequency_penalty: float,
        presence_penalty: float,
        logit_bias: dict | None,
        json_schema,
        enable_thinking: bool | None,
        thinking_budget: int | None,
        cancel_event: Any,
        stats: Any,
        prompt_kwargs: dict | None = None,
        apc_semantic_hash: int | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        min_tokens: int = 0,
        ignore_eos: bool = False,
        suppress_tokens: list[int] | None = None,
        top_n_sigma: float = 0.0,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
    ):
        """Yield ``(text, token_id, state, finish_reason, thinking_tokens, logprob)``.

        ``logprob`` is the token's {"token_id", "logprob", "top_logprobs"} entry
        when ``logprobs`` is requested, else ``None``.

        ``finish_reason`` is set only on the last event: "stop" (EOS or stop
        string), "length", "budget" (thinking budget reached) or "cancel".
        """
        from .text_utils import StopHoldbackBuffer
        from .vlm_batch_runner import (
            ConstraintProcessor,
            TokenMaskProcessor,
            build_penalty_processors,
        )

        processors = build_penalty_processors(
            repetition_penalty, frequency_penalty, presence_penalty, logit_bias
        )
        mask_kw = {
            "suppress": suppress_tokens,
            "min_tokens": min_tokens,
            "ignore_eos": ignore_eos,
            "top_n_sigma": top_n_sigma,
        }
        if TokenMaskProcessor.active(**mask_kw):
            processors.append(
                TokenMaskProcessor(eos_ids=list(self._get_eos_ids()), **mask_kw)
            )
        constraint = self._build_text_constraint(json_schema)
        if constraint is not None:
            processors.append(ConstraintProcessor(constraint, self._tokenizer))

        stop_ids = set() if ignore_eos else set(self._get_eos_ids())
        stop_ids.update(stop_token_ids or [])
        stop_strings = [s for s in (stop or []) if s]
        holdback = StopHoldbackBuffer(stop_strings)
        detok = self._tokenizer.detokenizer
        detok.reset()
        think_start, think_end, open_text, close_text, channel = (
            self._reasoning_markers()
        )
        # Qwen3.x templates open <think> at the end of the prompt when thinking
        # is on, so generation starts inside the reasoning block; streaming
        # clients rely on current_state to split reasoning from content.
        in_think = False
        # Gemma channel reasoning begins with a "thought" label line.
        label_buf: str | None = None
        if think_start is not None:
            tail = list(input_ids[-64:])
            if think_start in tail:
                last_open = len(tail) - 1 - tail[::-1].index(think_start)
                in_think = think_end is None or think_end not in tail[last_open + 1 :]
        thinking_tokens = 0
        count = 0
        for token in self._batch_runner.iter_tokens(
            input_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
            logits_processors=processors,
            prompt_kwargs=prompt_kwargs,
            apc_semantic_hash=apc_semantic_hash,
            cancel_event=cancel_event,
            stats=stats,
            logprobs=bool(logprobs),
            top_logprobs=int(top_logprobs or 0),
            thinking_budget=(
                thinking_budget
                if enable_thinking is not False and think_start is not None
                else None
            ),
            prompt_preopens_thinking=in_think,
            thinking_start_token=open_text,
            thinking_end_token=close_text,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            xtc_special_tokens=list(self._get_eos_ids()),
        ):
            count += 1
            lp = stats.last_logprob if logprobs else None
            if token in stop_ids:
                detok.finalize()
                tail = holdback.feed(detok.last_segment) + holdback.flush()
                yield tail, token, "normal", "stop", thinking_tokens, lp
                return
            if think_start is not None and token in (think_start, think_end):
                # The tags only switch state; they are never shown as text.
                in_think = token == think_start
                label_buf = "" if (channel and in_think) else None
                state = "reasoning" if in_think else "normal"
                if count >= max_tokens:
                    detok.finalize()
                    tail = holdback.feed(detok.last_segment) + holdback.flush()
                    yield tail, token, state, "length", thinking_tokens, lp
                    return
                continue
            if in_think:
                thinking_tokens += 1
            state = "reasoning" if in_think else "normal"
            detok.add_token(token)
            segment = detok.last_segment
            if stop_strings and not in_think:
                text = holdback.feed(segment)
                if holdback.contains_stop():
                    yield (
                        text + holdback.take_stopped(),
                        token,
                        state,
                        "stop",
                        thinking_tokens,
                        lp,
                    )
                    return
            else:
                text = segment
            if label_buf is not None and in_think:
                # Hold the channel label back until its line ends, then drop it.
                label_buf += text
                if "\n" not in label_buf and len(label_buf) < 32:
                    text = ""
                else:
                    head = label_buf.lstrip()
                    if head.startswith("thought"):
                        head = head[len("thought") :].lstrip("\n")
                    text, label_buf = head, None
            if count >= max_tokens:
                detok.finalize()
                tail = holdback.feed(detok.last_segment) + holdback.flush()
                yield text + tail, token, state, "length", thinking_tokens, lp
                return
            yield text, token, state, None, thinking_tokens, lp
        if stats.finish_reason == "cancel":
            yield "", None, "normal", "cancel", thinking_tokens, None
            return
        detok.finalize()
        tail = holdback.feed(detok.last_segment) + holdback.flush()
        yield (
            tail,
            None,
            "normal",
            "stop" if count < max_tokens else "length",
            thinking_tokens,
            None,
        )

    # Request knobs no VLM path implements. They are rejected (400 at the
    # gateway) instead of being accepted and silently ignored.
    _UNSUPPORTED_KWARGS = ("lora_adapter", "logits_processors")

    def _default_enable_thinking(
        self, enable_thinking: bool | None, constrained: bool = False
    ) -> bool | None:
        """Gemma-4's template enables thinking by default but emits inline
        ``thought`` tokens that don't auto-stop; default it off unless the
        caller asked. Applied to streaming and non-streaming alike.

        A JSON-schema / grammar constraint masks the output from its first token,
        so a template that opens ``<think>`` would have the reasoning forced into
        the schema and the content would come back empty. Constrained requests
        therefore default to thinking off unless the caller asked for it."""
        if enable_thinking is not None:
            return enable_thinking
        if constrained:
            return False
        model_type = str(self._config.get("model_type", "")).lower()
        if model_type.startswith("gemma4") or "gemma-4" in self.model_name.lower():
            return False
        return None

    def _track_pipeline(self, kwargs, messages, image_paths, audio_paths) -> None:
        try:
            from .staged_pipeline import PipelineRequest

            self._pipeline.process(
                PipelineRequest(
                    request_id=kwargs.get("request_id", ""),
                    model_id=self.model_name,
                    images=image_paths or None,
                    audio=audio_paths or None,
                    params={"messages": messages},
                )
            )
        except Exception:
            logger.debug("pipeline tracking failed", exc_info=True)

    def _check_request_supported(self, image_paths, audio_paths, kwargs) -> None:
        unsupported = [k for k in self._UNSUPPORTED_KWARGS if kwargs.get(k)]
        if unsupported:
            raise ValueError(
                f"{', '.join(unsupported)} is not supported for multimodal (VLM) models"
            )
        if self._batch_runner is None:
            raise RuntimeError("VLM engine has no batch runner (model not loaded)")
        if image_paths and not self._has_vision:
            raise ValueError(
                f"model {self.model_name!r} has no vision encoder; image input is "
                "not supported"
            )
        if (image_paths or audio_paths) and self._processor is None:
            raise RuntimeError(
                f"model {self.model_name!r} has no mlx_vlm processor; media input "
                "cannot be prepared (see load-time logs)"
            )

    def _runner_input(
        self,
        messages: list[dict],
        image_paths: list[str],
        audio_paths: list[str],
        enable_thinking: bool | None,
        template_extra: dict | None,
    ):
        """Template + tokenize (and encode media) for one request on the MLX
        thread. Returns ``(token_ids, prompt_kwargs, apc_semantic_hash)``."""
        if image_paths or audio_paths:
            prompt = self._apply_vlm_template_with_cache(
                messages,
                enable_thinking=enable_thinking,
                num_audios=len(audio_paths),
                max_images=len(image_paths) if image_paths else None,
                template_extra=template_extra,
            )
            ids, pkw, salt = self._batch_runner.prepare_media(
                prompt,
                image_paths=image_paths,
                audio=self._audio_arg(audio_paths) if audio_paths else None,
            )
            return ids.tolist(), pkw, salt
        ids = self._tokenize_with_cache(
            messages, enable_thinking=enable_thinking, template_extra=template_extra
        )
        return ids.tolist(), None, None

    def _runner_kwargs(self, **params) -> dict:
        kwargs = params.pop("kwargs")
        js = kwargs.get("json_schema")
        if js is None:
            js = kwargs.get("grammar")
        tb = params.pop("thinking_budget", None)
        if tb is None and kwargs.get("reasoning_effort") is not None:
            tb = {"low": 2048, "medium": 8192, "high": 32768}.get(
                kwargs["reasoning_effort"], 8192
            )
        return dict(
            params,
            frequency_penalty=kwargs.get("frequency_penalty", 0.0) or 0.0,
            presence_penalty=kwargs.get("presence_penalty", 0.0) or 0.0,
            logit_bias=kwargs.get("logit_bias"),
            json_schema=js,
            thinking_budget=tb,
            min_tokens=int(kwargs.get("min_tokens") or 0),
            ignore_eos=bool(kwargs.get("ignore_eos")),
            suppress_tokens=kwargs.get("suppress_tokens"),
            top_n_sigma=float(kwargs.get("top_n_sigma") or 0.0),
            xtc_probability=float(kwargs.get("xtc_probability") or 0.0),
            xtc_threshold=float(kwargs.get("xtc_threshold") or 0.0),
        )

    def _generate_vlm_runner_text(
        self, input_ids: mx.array, extras: dict | None = None, **params
    ):
        """Non-streaming runner path; returns the legacy 6-tuple."""
        from .vlm_batch_runner import RunStats

        stats = RunStats()
        parts: list[str] = []
        finish = None
        thinking = 0
        lps: list[dict] = []
        reasoning: list[str] = []
        for event in self._runner_events(input_ids, stats=stats, **params):
            text, _token, state, finish, thinking, lp = event
            if text:
                (reasoning if state == "reasoning" else parts).append(text)
            if lp is not None:
                lps.append(lp)
        if reasoning:
            # The router splits reasoning from content on the think tags,
            # which the event stream itself never shows.
            parts = ["<think>", *reasoning, "</think>", *parts]
        if extras is not None:
            extras["prompt_tokens"] = stats.prompt_tokens
            if params.get("logprobs"):
                extras["logprobs"] = lps
        if finish == "cancel":
            raise asyncio.CancelledError()
        return (
            "".join(parts),
            thinking,
            stats.generated,
            finish in ("stop", "budget"),
            finish == "budget",
            stats.cached_tokens,
        )

    def _stream_vlm_runner_text(
        self, input_ids: mx.array, req_id: str, queue: Any, **params
    ) -> None:
        """Streaming runner path with bounded delivery to the async consumer."""
        from .vlm_batch_runner import RunStats

        stats = RunStats()
        cancel_event = params.get("cancel_event")
        put = getattr(queue, "put_blocking", None)

        def deliver(output: RequestOutput) -> bool:
            if put is not None:
                return put(output, cancel_event)
            queue.put_nowait(output)
            return True

        prompt_tokens = len(input_ids)
        try:
            for text, token, state, finish, thinking, lp in self._runner_events(
                input_ids, stats=stats, **params
            ):
                reason = {"budget": "stop"}.get(finish, finish)
                if not (text or reason):
                    continue
                first = stats.generated == 1 and token is not None
                ok = deliver(
                    RequestOutput(
                        request_id=req_id,
                        new_text=text,
                        new_token_ids=[token] if token is not None else [],
                        finish_reason=reason,
                        finished=reason is not None,
                        completion_tokens=stats.generated,
                        prompt_tokens=prompt_tokens,
                        cached_tokens=stats.cached_tokens,
                        current_state=state,
                        reasoning_tokens=thinking,
                        ttft_ms=round(stats.first_token_s * 1000, 1) if first else 0.0,
                        logprobs=[lp] if lp is not None else None,
                    )
                )
                if not ok or reason is not None:
                    return
        except Exception as exc:
            logger.error("VLM runner streaming error: %s", exc, exc_info=True)
            queue.put_nowait(
                RequestOutput(
                    request_id=req_id,
                    finish_reason="error",
                    finished=True,
                    error=str(exc),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=stats.generated,
                )
            )
        finally:
            logger.debug(
                "VLM runner: prompt=%d cached=%d generated=%d apc=%s draft=%s finish=%s",
                prompt_tokens,
                stats.cached_tokens,
                stats.generated,
                stats.used_apc,
                stats.used_draft,
                stats.finish_reason,
            )

    # ── Prompt Formatting ──

    def _build_vlm_messages(
        self, messages: list[dict], max_images: int | None = None
    ) -> list[dict]:
        """Build messages with image/audio references for processor's chat template.

        Args:
            max_images: If set, limits the number of image placeholders to this
                count.  This must match the actual number of image paths passed
                to the model (after single-image truncation).  Without this,
                the template would contain more image placeholders than actual
                images, causing a mismatch for SINGLE_IMAGE_ONLY_MODELS.
        """
        # VIDEO : _extract_video_frames appends each video's frames to
        # image_paths and the caller passes max_images=len(image_paths). But this
        # builder previously emitted NO placeholder for video_url/video_file parts, so
        # placeholders < images → mlx_vlm masked_scatter crash ("tokens 0, features N").
        # The extra placeholders needed for frames = max_images − (image_url parts we
        # emit). Pre-count to derive it, then emit them at the FIRST video part (frames
        # are appended after real images, so this keeps the count exact + order sane).
        _num_img_parts = 0
        _has_video = False
        for _m in messages:
            _c = _m.get("content", "")
            if isinstance(_c, list):
                for _p in _c:
                    if isinstance(_p, dict):
                        if _p.get("type") == "image_url":
                            _num_img_parts += 1
                        elif _p.get("type") in ("video_url", "video_file"):
                            _has_video = True
        _emitted_img = (
            min(_num_img_parts, max_images)
            if max_images is not None
            else _num_img_parts
        )
        _video_ph_total = (
            max(0, max_images - _emitted_img)
            if (max_images is not None and _has_video)
            else 0
        )

        vlm_messages = []
        _image_count = 0
        _video_emitted = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                _emit_video_frames_here = False
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "image_url":
                            if max_images is None or _image_count < max_images:
                                parts.append({"type": "image"})
                                _image_count += 1
                            # Skip excess image placeholders beyond max_images
                        elif part.get("type") in ("video_url", "video_file"):
                            # DEFER frame placeholders to the END of this
                            # message instead of emitting them inline at the first
                            # video part. image_paths is [images]+[frames]; emitting
                            # frames mid-stream before a LATER image part put the
                            # placeholders out of order vs pixel_values → the model
                            # grounded on the wrong images. Deferring to the end
                            # matches the [images-then-frames] concat order.
                            if not _video_emitted:
                                _emit_video_frames_here = True
                        elif (
                            part.get("type") == "input_audio"
                            or part.get("type") == "audio_url"
                        ):
                            parts.append({"type": "audio"})
                        elif part.get("type") == "text":
                            parts.append({"type": "text", "text": part.get("text", "")})
                    elif isinstance(part, str):
                        parts.append({"type": "text", "text": part})
                # emit deferred video-frame placeholders at the end of this
                # message's parts, so placeholder order matches image_paths
                # ([real images in doc order] + [video frames]).
                if _emit_video_frames_here:
                    _video_emitted = True
                    for _ in range(_video_ph_total):
                        parts.append({"type": "image"})
                vlm_messages.append({"role": msg.get("role", "user"), "content": parts})
            else:
                vlm_messages.append(
                    {"role": msg.get("role", "user"), "content": str(content)}
                )
        return vlm_messages

    @staticmethod
    def _normalize_vlm_tool_calls(tool_calls: list) -> list:
        """Convert tool-call argument JSON strings to dicts so templates that
        iterate argument keys (GLM-4V etc.) don't raise on a string. Mirrors
        BatchedEngine._normalize_messages_for_chat_template's tool-call branch."""
        import json as _json

        patched = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                patched.append(tc)
                continue
            func = tc.get("function")
            if isinstance(func, dict) and isinstance(func.get("arguments"), str):
                try:
                    parsed = _json.loads(func["arguments"])
                except Exception:
                    parsed = {"value": func["arguments"]}
                tc = dict(tc)
                tc["function"] = dict(func)
                tc["function"]["arguments"] = (
                    parsed if isinstance(parsed, dict) else {"value": parsed}
                )
            patched.append(tc)
        return patched

    def _runner_consumers(self):
        """Threads that wait for batched-runner tokens (never MLX work)."""
        pool = getattr(self, "_runner_consumer_pool", None)
        if pool is None:
            pool = self._runner_consumer_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=64, thread_name_prefix="vlm-runner-consumer"
            )
        return pool

    def supports_native_tools(self) -> bool:
        """True when the chat template renders a ``tools`` variable itself
        (Qwen3.x: ``<tool_call><function=...><parameter=...>``). The gateway
        then passes tool definitions here instead of injecting a generic tool
        system prompt."""
        return any(
            isinstance(t, str) and "tools" in t
            for t in (
                getattr(self._tokenizer, "chat_template", None),
                getattr(self._processor, "chat_template", None),
            )
        )

    def _request_template_extra(self, kwargs: dict) -> dict | None:
        """Per-request chat-template variables: ``reasoning_effort`` (when the
        template supports it) and native ``tools``. Pops them from ``kwargs``;
        the result is passed explicitly to every template helper and is part
        of their cache keys."""
        extra = dict(self._template_effort_extra(kwargs) or {})
        tools = kwargs.pop("tools", None)
        if tools:
            extra["tools"] = tools
        return extra or None

    def _template_effort_extra(self, kwargs: dict) -> dict | None:
        """Move ``reasoning_effort`` into the chat template when it supports it.

        Qwen3.8's template takes ``reasoning_effort`` (low / medium / xhigh)
        and changes how the model reasons; mapping it to a thinking-token cap
        instead would keep the model at the template default and cut it off.
        Pops the key from ``kwargs`` so no budget mapping runs afterwards.
        """
        effort = kwargs.get("reasoning_effort")
        if effort is None:
            ctk = kwargs.get("chat_template_kwargs") or {}
            effort = ctk.get("reasoning_effort") if isinstance(ctk, dict) else None
        if effort is None:
            return None
        supported = getattr(self, "_template_has_effort", None)
        if supported is None:
            texts = [
                getattr(self._tokenizer, "chat_template", None),
                getattr(self._processor, "chat_template", None),
            ]
            supported = any(
                isinstance(t, str) and "reasoning_effort" in t for t in texts
            )
            self._template_has_effort = supported
            from .model_card import reasoning_levels

            self._template_effort_levels = next(
                (
                    lv
                    for t in texts
                    if isinstance(t, str) and (lv := reasoning_levels(t)[0])
                ),
                [],
            )
        if not supported:
            return None
        kwargs.pop("reasoning_effort", None)
        # The template rejects levels it does not list (Qwen3.8 has no "high"): map the
        # OpenAI ladder onto the template's own.
        from .model_card import normalize_effort

        return {
            "reasoning_effort": normalize_effort(
                str(effort), getattr(self, "_template_effort_levels", [])
            )
        }

    def _format_prompt(
        self,
        messages: list[dict],
        enable_thinking: bool | None = None,
        template_extra: dict | None = None,
    ) -> str:
        # this text-only-chat path (a VLM model serving a non-image chat
        # turn) was a stale clone missing three BatchedEngine fixes — role
        # normalization, family adapter, and the assistant-prefill gate
        # — so a `developer`/`function` role or a mid-conversation system message
        # raised in the template and collapsed to the lossy plaintext fallback, and an
        # assistant prefill restarted the answer. Mirror BatchedEngine._apply_chat_template.
        if any(m.get("role") in ("developer", "function") for m in messages):
            _remapped = []
            for _m in messages:
                _r = _m.get("role")
                if _r in ("developer", "function"):
                    _m = dict(_m)
                    _m["role"] = "system" if _r == "developer" else "tool"
                _remapped.append(_m)
            messages = _remapped
        try:
            from yunshu_engine.message_adapter import adapt_messages

            messages = adapt_messages(messages, self.model_name)
        except Exception:
            logger.debug("VLM message adapter failed", exc_info=True)

        if self._tokenizer is not None and hasattr(
            self._tokenizer, "apply_chat_template"
        ):
            try:
                clean = []
                for msg in messages:
                    _c = {
                        "role": msg.get("role", "user"),
                        "content": self._extract_text(msg.get("content", "")),
                    }
                    # preserve tool-related fields so a VLM + tools conversation
                    # (e.g. GLM-4V) keeps prior assistant tool-call turns in the prompt. The
                    # old code rebuilt {role, content} ONLY → every tool_call/tool_call_id/
                    # name was dropped (tool-use history vanished), and string-form tool args
                    # left intact would trip a GLM template's `is not mapping` raise → the
                    # whole prompt collapsing to the plaintext fallback. Mirror the
                    # BatchedEngine path: keep the fields + normalize string args to dicts.
                    _tcs = msg.get("tool_calls")
                    if isinstance(_tcs, list):
                        _c["tool_calls"] = self._normalize_vlm_tool_calls(_tcs)
                    if msg.get("tool_call_id"):
                        _c["tool_call_id"] = msg["tool_call_id"]
                    if msg.get("name"):
                        _c["name"] = msg["name"]
                    if msg.get("reasoning_content"):
                        _c["reasoning_content"] = msg["reasoning_content"]
                    clean.append(_c)
                # assistant-prefill gate — a trailing assistant
                # message with non-empty STRING content means "continue THIS turn"
                # (Anthropic/OpenAI prefill); add_generation_prompt=True would close the
                # prefill and restart the answer. Use continue_final_message instead,
                # with the retry when the template rejects it (null/tool_calls-only
                # trailing assistant → normal add_generation_prompt).
                _last = clean[-1] if clean else None
                _is_prefill = (
                    _last is not None
                    and _last.get("role") == "assistant"
                    and isinstance(_last.get("content"), str)
                    and _last["content"] != ""
                )
                tpl_kwargs: dict = {"tokenize": False}
                if _is_prefill:
                    tpl_kwargs["continue_final_message"] = True
                else:
                    tpl_kwargs["add_generation_prompt"] = True
                if enable_thinking is not None:
                    tpl_kwargs["enable_thinking"] = enable_thinking
                tpl_kwargs.update(template_extra or {})
                try:
                    text = self._tokenizer.apply_chat_template(clean, **tpl_kwargs)
                except (TypeError, ValueError) as e:
                    _es = str(e)
                    if "continue_final_message" in _es:
                        tpl_kwargs.pop("continue_final_message", None)
                        tpl_kwargs["add_generation_prompt"] = False
                        try:
                            text = self._tokenizer.apply_chat_template(
                                clean, **tpl_kwargs
                            )
                        except (TypeError, ValueError) as e2:
                            if "enable_thinking" in str(e2):
                                tpl_kwargs.pop("enable_thinking", None)
                                text = self._tokenizer.apply_chat_template(
                                    clean, **tpl_kwargs
                                )
                            else:
                                raise
                    elif "enable_thinking" in _es:
                        tpl_kwargs.pop("enable_thinking", None)
                        text = self._tokenizer.apply_chat_template(clean, **tpl_kwargs)
                    else:
                        raise
                if text:
                    return text
            except Exception as exc:
                # A model that has a chat template must be prompted with it; a
                # silent plaintext prompt gives the model a format it was never
                # trained on and the caller no hint why answers degrade.
                if getattr(self._tokenizer, "chat_template", None):
                    raise ValueError(f"Chat template failed: {exc}") from exc
                logger.debug("apply_chat_template failed", exc_info=True)

        logger.warning(
            "Model %s has no chat template; using a plain 'Role: text' prompt",
            getattr(self, "model_name", "?"),
        )
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = self._extract_text(msg.get("content", ""))
            parts.append(f"{role.capitalize()}: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _tokenize_with_cache(
        self,
        messages: list[dict],
        enable_thinking: bool | None = None,
        template_extra: dict | None = None,
    ) -> mx.array:
        """Format prompt and tokenize, using the text prompt cache to skip work.

        Caches the tokenizer.encode() result keyed by a stable hash of the
        message content. On cache hit, skips _format_prompt() + encode().
        """
        cache_key = _VLMTextPromptCache._compute_messages_hash(
            messages,
            enable_thinking,
        )
        if template_extra:
            cache_key = f"{cache_key}|{json.dumps(template_extra, sort_keys=True)}"

        cached_ids = self._text_prompt_cache.get_token_ids(cache_key)
        if cached_ids is not None:
            logger.debug("VLM text prompt cache hit: %d tokens", len(cached_ids))
            return mx.array(cached_ids)

        prompt_text = self._format_prompt(
            messages, enable_thinking=enable_thinking, template_extra=template_extra
        )
        # Avoid double-BOS: prompt_text came from apply_chat_template(tokenize=False),
        # which already injected the literal bos_token for BOS-prepending models
        # (Gemma-3-VL, Llama-3.2-Vision, Pixtral). A bare encode() defaults to
        # add_special_tokens=True → a SECOND BOS into the model input, corrupting
        # the first-token distribution. Mirror the guard in _count_text_tokens /
        # _encode_prompt (sibling-sweep: this third encode site was missed).
        bos = getattr(self._tokenizer, "bos_token", None)
        _add_special = not (
            isinstance(bos, str) and bos and prompt_text.startswith(bos)
        )
        try:
            token_ids = self._tokenizer.encode(
                prompt_text, add_special_tokens=_add_special
            )
        except TypeError:
            token_ids = self._tokenizer.encode(prompt_text)

        self._text_prompt_cache.put_token_ids(cache_key, token_ids)
        logger.debug("VLM text prompt cache miss: tokenized %d tokens", len(token_ids))
        return mx.array(token_ids)

    def _apply_vlm_template_with_cache(
        self,
        messages: list[dict],
        enable_thinking: bool | None = None,
        num_audios: int = 0,
        max_images: int | None = None,
        template_extra: dict | None = None,
    ) -> str:
        """Apply VLM processor chat template with caching.

        Caches the _processor.apply_chat_template() result for the VLM vision
        path. On cache hit, skips the template application entirely.

        Args:
            max_images: Limits image placeholders in the template to match the
                actual number of image paths that will be passed to the model.
                Critical for SINGLE_IMAGE_ONLY_MODELS where _extract_images()
                truncates to 1 image but messages may contain multiple image_url
                entries.
        """
        # Build cache key from messages + template kwargs
        key_parts = [json.dumps(messages, sort_keys=True, ensure_ascii=False)]
        if enable_thinking is not None:
            key_parts.append(f"thinking={enable_thinking}")
        if num_audios > 0:
            key_parts.append(f"audios={num_audios}")
        if max_images is not None:
            key_parts.append(f"max_images={max_images}")
        extra = template_extra or {}
        if extra:
            key_parts.append(json.dumps(extra, sort_keys=True))
        cache_key = hashlib.blake2b(
            "|".join(key_parts).encode(),
            digest_size=16,
        ).hexdigest()

        cached = self._text_prompt_cache.get_template_text(cache_key)
        if cached is not None:
            logger.debug("VLM template cache hit: %d chars", len(cached))
            return cached

        vlm_messages = self._build_vlm_messages(messages, max_images=max_images)
        tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            tpl_kwargs["enable_thinking"] = enable_thinking
        if num_audios > 0:
            tpl_kwargs["num_audios"] = num_audios
        tpl_kwargs.update(extra)

        try:
            template_text = self._processor.apply_chat_template(
                vlm_messages,
                **tpl_kwargs,
            )
        except (ValueError, AttributeError) as e:
            # Some omni processors (e.g. NVIDIA Nemotron-Omni) ship no chat
            # template on the PROCESSOR — it lives on the tokenizer instead. The
            # text path already uses the tokenizer template successfully; reuse it
            # here so the vision/audio path doesn't hard-fail. num_audios/num_images
            # kwargs may be processor-only, so drop them for the tokenizer call.
            if not (
                hasattr(self, "_tokenizer")
                and getattr(self._tokenizer, "chat_template", None)
            ):
                raise
            logger.info(
                "Processor lacks a chat template (%s); falling back to the "
                "tokenizer chat template for the VLM multimodal path.",
                str(e)[:80],
            )
            tok_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if enable_thinking is not None:
                tok_kwargs["enable_thinking"] = enable_thinking
            tok_kwargs.update(extra)
            template_text = self._tokenizer.apply_chat_template(
                vlm_messages,
                **tok_kwargs,
            )

        self._text_prompt_cache.put_template_text(cache_key, template_text)
        logger.debug(
            "VLM template cache miss: applied template (%d chars)", len(template_text)
        )
        return template_text

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
                        # defensive — image_url may be the canonical {"url":…}
                        # object OR the OpenAI bare-string variant. .get("url") on a str
                        # raised AttributeError → 500; the gateway normalizes this now, but
                        # guard here too (other callers feed the engine directly).
                        _iu = part.get("image_url", {})
                        url = (
                            _iu.get("url", "")
                            if isinstance(_iu, dict)
                            else (_iu if isinstance(_iu, str) else "")
                        )
                        if url.startswith("data:image"):
                            paths.append(await self._save_base64_image(url))
                        elif url.startswith(("http://", "https://")):
                            paths.append(await self._download_image(url))
                        elif url.startswith("file://"):
                            # A requested image that can't be loaded must FAIL the
                            # request (matching the data:/http: branches, which raise
                            # on failure) — NOT be silently dropped, which would make
                            # the VLM confidently answer about an image it never saw.
                            file_path = _VALIDATE_LOCAL_PATH(
                                url[7:]
                            )  # raises ValueError if blocked
                            if not os.path.exists(file_path):
                                raise ValueError(
                                    f"image file:// path does not exist: {file_path}"
                                )
                            paths.append(file_path)
                        elif url and os.path.exists(url):
                            # bare-path access is gated by YUNSHU_MEDIA_DIR.
                            paths.append(_VALIDATE_LOCAL_PATH(url))  # raises if blocked
                        else:
                            # an image_url routed to the VLM that resolves to
                            # NOTHING — empty url, a non-image data: mime
                            # (data:application/...), or an unknown scheme — must fail
                            # loud, not be silently dropped. A silent drop both makes the
                            # model hallucinate about an image it never saw AND shifts the
                            # surviving placeholders to the wrong positions in document
                            # order (it grounds image B's pixels at image A's text slot).
                            raise ValueError(
                                f"unsupported or unloadable image_url: {url!r}"
                            )
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

    # ── Audio Extraction ──

    def _audio_arg(self, audio_paths: list[str]):
        """Convert audio file PATHS into loaded float32 sample arrays.

        mlx_vlm's `process_inputs` passes the `audio` argument straight to the
        model processor without loading it, so handing it a path string makes
        Qwen3-Omni (and other audio VLMs) try to treat the path as samples →
        "could not convert string to float: '/…/tmp.wav'". We must pre-load the
        path into an ndarray via mlx_vlm.load_audio at the model's sample rate.
        (2nd-pass live fix.)

        Always returns a LIST of arrays — even for a single audio. mlx_vlm's
        stream_generate does ``audio = audio or None``, which raises "truth value
        of an array is ambiguous" on a bare multi-element ndarray (hit on NVIDIA
        Nemotron-Omni). A list is truthy regardless of contents, and the
        multi-audio path already returned a list, so single-element-list is the
        uniform, processor-friendly form (verified: Nemotron-Omni then perceives
        the audio instead of reporting "no auditory input").
        """
        from mlx_vlm.utils import load_audio

        sr = 16000
        fe = getattr(self._processor, "feature_extractor", None)
        if fe is not None and getattr(fe, "sampling_rate", None):
            sr = int(fe.sampling_rate)
        elif getattr(self._processor, "audio_sampling_rate", None):
            # Some omni-input processors (e.g. NVIDIA nemotron_h_nano_omni) store
            # the audio rate directly on the processor, not under feature_extractor.
            sr = int(self._processor.audio_sampling_rate)
        return [load_audio(p, sr) for p in audio_paths]

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
                        else:
                            # fail-loud (see file:// note below). An input_audio
                            # part with no data was routed here but would be silently
                            # dropped → placeholder/feature desync → masked_scatter crash.
                            raise ValueError("input_audio content part has no 'data'")
                    elif ptype == "audio_url":
                        url = part.get("audio_url", {}).get("url", "")
                        if url.startswith("data:audio"):
                            if "," not in url:
                                raise ValueError(
                                    "malformed data:audio URL (no payload separator)"
                                )
                            header, data = url.split(",", 1)
                            # a subtype-less data:audio URL (data:audio;base64,… or
                            # data:audio,…) has no '/' in the header, so header.split("/")[1]
                            # raised IndexError → opaque 500. Default the format (fixed
                            # this on the image sibling but not here). Mirror that.
                            _hp = header.split("/", 1)
                            fmt = _hp[1].split(";")[0] if len(_hp) > 1 else "wav"
                            paths.append(await self._save_base64_audio(data, fmt))
                        elif url.startswith("file://"):
                            # a requested audio that can't be loaded must
                            # FAIL the request, not be silently dropped. _build_vlm_messages
                            # emits one {"type":"audio"} placeholder PER audio part
                            # unconditionally, so dropping a part here desyncs the
                            # placeholder/feature counts → mlx_vlm masked_scatter crash
                            # ("tokens N, features N-1"). Mirror the image file:// branch.
                            path = _VALIDATE_LOCAL_PATH(
                                url[7:]
                            )  # raises ValueError if blocked
                            if not os.path.exists(path):
                                raise ValueError(
                                    f"audio file:// path does not exist: {path}"
                                )
                            paths.append(path)
                        elif os.path.exists(url):
                            paths.append(_VALIDATE_LOCAL_PATH(url))  # raises if blocked
                        else:
                            # http(s):// audio download is not supported and an
                            # empty/unknown-scheme audio_url must not be silently dropped
                            # (the desync the file:// branch above guards against —
                            # but only for file://). Fail loud for the sibling schemes.
                            raise ValueError(
                                f"unsupported or unloadable audio_url: {url!r}"
                            )
        return paths

    async def _save_base64_audio(self, data: str, fmt: str = "wav") -> str:
        # normalize the MIME subtype to a canonical file extension BEFORE the
        # allowlist. The decode path (mlx_audio → miniaudio.get_file_info) dispatches on
        # the file SUFFIX, so a data:audio/mpeg URL (fmt="mpeg") that fell through to the
        # ".wav" fallback wrote MP3 bytes into a .wav file → miniaudio DecodeError → the
        # audio was lost for the single most common compressed format. (PIL/ffmpeg sniff
        # content, so image/video are unaffected — only this path trusts the suffix.)
        _AUDIO_MIME_TO_EXT = {
            "mpeg": "mp3",
            "mpeg3": "mp3",
            "x-mpeg": "mp3",
            "mp3": "mp3",
            "wav": "wav",
            "wave": "wav",
            "x-wav": "wav",
            "vnd.wave": "wav",
            "vnd.wav": "wav",
            "flac": "flac",
            "x-flac": "flac",
            "ogg": "ogg",
            "x-ogg": "ogg",
            "vorbis": "ogg",
            "pcm": "pcm",
            "l16": "pcm",
        }
        _norm = _AUDIO_MIME_TO_EXT.get(
            (fmt or "").lower().strip(), (fmt or "").lower().strip()
        )
        ext = _norm if _norm in ("wav", "mp3", "ogg", "flac", "pcm") else "wav"
        tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)  # noqa: SIM115 — file must outlive function; cleanup via _temp_files registry

        def _sync_decode_write() -> None:
            # MB-size base64 decode + write blocks event loop;
            # move to executor.
            decoded = base64.b64decode(data)
            try:
                tmp.write(decoded)
                tmp.close()
            except Exception:
                tmp.close()
                with contextlib.suppress(OSError):
                    os.unlink(tmp.name)
                raise

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_decode_write)
        self._register_temp_file(tmp.name)
        return tmp.name

    async def _save_base64_image(self, data_url: str) -> str:
        if "," not in data_url:
            raise ValueError("malformed data: URL (no base64 payload separator)")
        header, data = data_url.split(",", 1)
        # a subtype-less data URL (e.g. `data:image;base64,...`) has no `/` in the
        # header, so header.split("/")[1] raised IndexError → a generic 500 instead of a clean
        # client error. Default the extension when the MIME subtype is absent.
        _mime = header.split("/", 1)
        ext = _mime[1].split(";")[0] if len(_mime) > 1 else "png"
        ext = ext if ext in ("png", "jpg", "jpeg", "webp", "gif") else "png"
        tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)  # noqa: SIM115 — file must outlive function; cleanup via _temp_files registry

        def _sync_decode_write() -> None:
            # offload sync I/O off the event loop.
            decoded = base64.b64decode(data)
            try:
                tmp.write(decoded)
                tmp.close()
            except Exception:
                tmp.close()
                with contextlib.suppress(OSError):
                    os.unlink(tmp.name)
                raise

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_decode_write)
        self._register_temp_file(tmp.name)
        return tmp.name

    async def _download_image(self, url: str) -> str:
        """Download an image from HTTP/HTTPS URL to a temp file."""

        # SSRF validation: block private/internal IPs
        _VALIDATE_URL(url)

        ext = url.rsplit(".", 1)[-1].lower() if "." in url.split("?")[0] else "png"
        ext = ext if ext in ("png", "jpg", "jpeg", "webp", "gif") else "png"

        tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)  # noqa: SIM115 — file must outlive function; cleanup via _temp_files registry
        tmp.close()  # Close FD immediately — urlretrieve writes by path, not handle
        # Register temp file eagerly so cleanup happens even if download fails
        self._register_temp_file(tmp.name)

        # urllib.request.urlretrieve does NOT accept a Request object (only a
        # URL string), so to send a custom User-Agent we use urlopen(Request)
        # and stream the body to disk manually.
        def _download(ssl_ctx):
            req = urllib.request.Request(url, headers={"User-Agent": "Yunshu/1.0"})
            # SSRF-via-redirect defense: _VALIDATE_URL only validated the initial
            # host; a redirect to an internal host would otherwise be followed.
            opener = urllib.request.build_opener(
                _NoRedirect, urllib.request.HTTPSHandler(context=ssl_ctx)
            )
            # enforce a max download size. SSRF (_VALIDATE_URL), redirect
            # (_NoRedirect) and a 30s timeout were all present, but there was NO size cap —
            # shutil.copyfileobj streamed the whole body, so a multi-GB (or slowly-streamed)
            # remote image_url filled the disk + then PIL loaded it whole into memory. This
            # bypasses the request-body size middleware because the bytes arrive
            # out-of-band from the model server. Check Content-Length up front AND count bytes
            # while streaming (a lying/absent header can't evade it). "size limit" in the
            # message lets the outer handler skip the insecure-SSL retry (no re-download).
            _cap = settings.get("YUNSHU_VLM_MAX_IMAGE_BYTES")
            with opener.open(req, timeout=30) as resp:
                _clen = resp.headers.get("Content-Length")
                if _clen is not None and str(_clen).isdigit() and int(_clen) > _cap:
                    raise ValueError(
                        f"remote image exceeds size limit ({_clen} > {_cap} bytes)"
                    )
                with open(tmp.name, "wb") as fh:
                    _written = 0
                    while True:
                        _chunk = resp.read(65536)
                        if not _chunk:
                            break
                        _written += len(_chunk)
                        if _written > _cap:
                            raise ValueError(
                                f"remote image exceeds size limit (> {_cap} bytes)"
                            )
                        fh.write(_chunk)

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, _download, ssl.create_default_context())
        except Exception as e:
            # a size-limit violation is final — don't fall through to the insecure-
            # SSL retry (which would re-download the oversized body). Map straight to a clean
            # error (→ 400/413 at the gateway).
            if "size limit" in str(e):
                logger.warning(f"Rejected oversized remote image from {url}: {e}")
                raise ValueError(str(e)) from e
            # Insecure SSL fallback is a MITM vector — only enable when the
            # operator explicitly opts in via YUNSHU_VLM_INSECURE_SSL=true.
            _insecure_ok = settings.get_bool("YUNSHU_VLM_INSECURE_SSL")
            if not _insecure_ok:
                logger.warning(f"Failed to download image from {url}: {e}")
                raise ValueError(f"Cannot download image: {e}") from e
            logger.warning(
                "image download SSL verification failed; retrying with verification "
                "disabled because YUNSHU_VLM_INSECURE_SSL is set (insecure)"
            )
            try:
                relaxed = ssl.create_default_context()
                relaxed.check_hostname = False
                relaxed.verify_mode = ssl.CERT_NONE
                await loop.run_in_executor(None, _download, relaxed)
            except Exception as e2:
                logger.warning(f"Failed to download image from {url}: {e2}")
                raise ValueError(f"Cannot download image: {e2}") from e2
        return tmp.name

    def _register_temp_file(self, path: str) -> None:
        """Register a temp file/dir for cleanup: into the global registry (stop() sweep)
        AND the current request's ContextVar list (identity-exact per-request cleanup)."""
        with self._temp_files_lock:
            if self._temp_files is None:
                self._temp_files = []
            self._temp_files.append(path)
        _rl = _request_temp_files.get()
        if _rl is not None:
            _rl.append(path)

    def _cleanup_temp_files(self, files: list[str] | None = None) -> None:
        if files is not None:
            targets = files
        else:
            with self._temp_files_lock:
                targets = list(self._temp_files or [])
                self._temp_files = []
        if targets:
            for path in targets:
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.unlink(path)
                except OSError:
                    pass
            targets.clear()

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
                    # a REFERENCED video that can't be loaded must FAIL LOUD, not
                    # be silently dropped. Silently dropping it (the old behaviour for http
                    # video_url, and for rejected/nonexistent file://, bare-path, and
                    # file_id) made the model answer about a video it never saw — the exact
                    # hallucinate-on-silent-media-drop class that image+audio
                    # already guard against; video was the lone un-propagated sibling.
                    if part.get("type") == "video_url":
                        url = part.get("video_url", {}).get("url", "")
                        if url.startswith("data:video"):
                            # sibling: a subtype-less data:video URL
                            # (data:video;base64,… or data:video,…) has no '/' in the
                            # header, so header.split("/")[1] raised IndexError — and
                            # unlike the image/audio paths this becomes an UNCAUGHT
                            # IndexError (not a ValueError) at the gateway → opaque 500
                            # instead of a clean 400. Guard the comma + subtype split
                            # the same way fixed the image/audio siblings.
                            if "," not in url:
                                raise ValueError(
                                    "malformed data:video URL (no payload separator)"
                                )
                            header, data = url.split(",", 1)
                            _hp = header.split("/", 1)
                            ext = _hp[1].split(";")[0] if len(_hp) > 1 else "mp4"
                            ext = (
                                ext
                                if ext in ("mp4", "webm", "avi", "mov", "mkv")
                                else "mp4"
                            )
                            path = await self._save_base64_file(data, ext)
                            video_paths.append(path)
                        elif url.startswith("file://"):
                            path = _VALIDATE_LOCAL_PATH(url[7:])  # raises on traversal
                            if not os.path.exists(path):
                                raise ValueError(f"video file not found: {url}")
                            video_paths.append(path)
                        elif url.startswith(("http://", "https://")):
                            raise ValueError(
                                "video_url http(s) fetch is not supported — provide a "
                                "data: URL or a file under YUNSHU_MEDIA_DIR"
                            )
                        elif url:
                            path = _VALIDATE_LOCAL_PATH(url)  # raises on traversal
                            if not os.path.exists(path):
                                raise ValueError(f"video file not found: {url}")
                            video_paths.append(path)
                        else:
                            raise ValueError(
                                "video_url part present but its url is empty"
                            )
                    elif part.get("type") == "video_file":
                        fid = part.get("video_file", {}).get("file_id", "")
                        # SECURITY: contain to YUNSHU_MEDIA_DIR like every
                        # sibling media branch. Without this, file_id="/etc/passwd"
                        # (an absolute host path) bypassed the containment the
                        # video_url/file:// branches enforce → arbitrary file read.
                        if not fid:
                            raise ValueError(
                                "video_file part present but file_id is empty"
                            )
                        _vp = _VALIDATE_LOCAL_PATH(fid)  # raises on traversal
                        if not os.path.exists(_vp):
                            raise ValueError(f"video file not found: {fid}")
                        video_paths.append(_vp)

        if not video_paths:
            return []

        frame_paths = []
        for vp in video_paths:
            frames = await self._extract_frames_from_file(
                vp, fps=fps, max_frames=max_frames
            )
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
        # LEAK fix: eager-register tmpdir into _temp_files BEFORE
        # the await call so asyncio.CancelledError from a client disconnect
        # mid-extract doesn't orphan a directory full of frames on disk.
        self._register_temp_file(tmpdir)
        output_pattern = os.path.join(tmpdir, "frame_%04d.jpg")

        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            f"fps={fps}",
            "-frames:v",
            str(max_frames),
            "-q:v",
            "2",
            "-y",
            output_pattern,
        ]

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd, capture_output=True, timeout=60, check=True
                ),
            )
        except FileNotFoundError:
            logger.warning("ffmpeg not available — cannot extract video frames")
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)
            return []
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg timed out extracting video frames")
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)
            return []
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"ffmpeg failed: {e.stderr.decode()[:200] if e.stderr else 'unknown'}"
            )
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)
            return []
        except Exception as e:
            logger.warning(f"Video frame extraction failed: {e}")
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)
            return []

        frames = sorted(
            os.path.join(tmpdir, f)
            for f in os.listdir(tmpdir)
            if f.startswith("frame_") and f.endswith(".jpg")
        )
        # tmpdir was already eagerly registered above; don't append it a
        # second time (was a double entry in _temp_files).
        return frames

    async def _save_base64_file(self, data: str, ext: str = "mp4") -> str:
        """Save base64-encoded data to a temp file."""
        tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)  # noqa: SIM115 — file must outlive function; cleanup via _temp_files registry

        def _sync_decode_write() -> None:
            # offload sync decode+write off the event loop.
            decoded = base64.b64decode(data)
            try:
                tmp.write(decoded)
                tmp.close()
            except Exception:
                tmp.close()
                with contextlib.suppress(OSError):
                    os.unlink(tmp.name)
                raise

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_decode_write)
        self._register_temp_file(tmp.name)
        return tmp.name

    # ── Helpers ──

    def _resolve_reasoning_channel_ids(self) -> None:
        """Resolve the single-token ids for Gemma-style reasoning channels.

        Gemma 4 emits reasoning as ``<|channel>thought\\n … <channel|>`` where
        ``<|channel>`` / ``<channel|>`` are SINGLE special tokens that decode to
        '' under skip_special_tokens — so the markers vanish before the string-
        based ``<think>`` reasoning machinery (and reasoning_parser) ever see
        them, and the channel text leaks into content. Capture the ids here so
        the decode path can segment on them at the TOKEN level instead.
        Returns (None, None) for models without these tokens (the common case)."""
        self._reasoning_channel_ids: tuple[int | None, int | None] = (None, None)
        tok = getattr(self, "_tokenizer", None)
        if tok is None:
            return

        def _single(s: str) -> int | None:
            try:
                ids = tok.encode(s, add_special_tokens=False)
            except TypeError:
                ids = tok.encode(s)
            except Exception:
                return None
            return ids[0] if ids and len(ids) == 1 else None

        open_id, close_id = _single("<|channel>"), _single("<channel|>")
        if open_id is not None and close_id is not None:
            self._reasoning_channel_ids = (open_id, close_id)
            logger.info(
                "Reasoning channel tokens resolved: open=%d close=%d (Gemma-style)",
                open_id,
                close_id,
            )

    def _get_eos_ids(self) -> list[int]:
        """Tokenizer EOS plus generation_config / config eos_token_id (Gemma-4
        ends turns with <turn|>, which only generation_config lists)."""
        cached = getattr(self, "_eos_ids_cache", None)
        if cached is not None:
            return cached
        from .text_utils import get_eos_token_ids

        ids = set(get_eos_token_ids(getattr(self, "_tokenizer", None)))
        sources = [getattr(self, "_config", None) or {}]
        with contextlib.suppress(Exception):
            path = Path(getattr(self, "_model_path", "")) / "generation_config.json"
            if path.exists():
                sources.append(json.loads(path.read_text()))
        for src in sources:
            for cfg in (src, src.get("text_config") or {}):
                eid = cfg.get("eos_token_id") if isinstance(cfg, dict) else None
                if isinstance(eid, int):
                    ids.add(eid)
                elif isinstance(eid, list):
                    ids.update(i for i in eid if isinstance(i, int))
        self._eos_ids_cache = sorted(ids)
        return self._eos_ids_cache

    # ── Stats ──

    def get_stats(self) -> dict:
        uptime = time.monotonic() - self._start_time if self._start_time else 0.0
        stats = {
            "model": self._model_path,
            "loaded": self.is_loaded,
            "running": self._running,
            "has_vision": self._has_vision,
            "is_vlm": self._is_vlm,
            "num_requests_processed": self._num_requests_processed,
            "reasoning_tokens": self._total_reasoning_tokens,
            "uptime_seconds": uptime,
            "text_prompt_cache": self._text_prompt_cache.stats,
        }
        apc = self._apc_backend
        if apc is not None:
            with contextlib.suppress(Exception):
                stats["apc"] = {
                    "memory_max_bytes": apc.memory_max_bytes,
                    "matched_tokens": apc.stats.matched_tokens,
                }
        try:
            stats["pipeline"] = self._pipeline.get_stats()
        except Exception:
            logger.debug("operation failed", exc_info=True)
        return stats
