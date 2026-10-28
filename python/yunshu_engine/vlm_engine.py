# Upstream (inspired): vllm-project/vllm (Apache-2.0) vllm/reasoning/qwen3_reasoning_parser.py @ 22a58640b
#   (a <tool_call> token ends unclosed reasoning; see vendor.json)
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

import array
import asyncio
import base64
import collections
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
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import mlx.core as mx

from . import gc_policy, netguard, next_turn, settings
from .memory_admission import (
    MEMORY_LIMIT_MARK,
    MemoryAdmissionError,
    memory_limit_error,
)
from .moe_pack import tensor_quantization_override, unsupported_pack_reason
from .paths import private_media_dir
from .request import RequestOutput
from .special_tokens import neutralize_special_text
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


def _prepare_vlm_weights(model, model_class, model_config, weights):
    """Apply the upstream checkpoint transforms before quantization/filtering."""
    from mlx_vlm.utils import sanitize_weights

    weights = sanitize_weights(model, weights)
    for component in ("Vision", "Language", "Audio"):
        config_name = (
            "text_config" if component == "Language" else f"{component.lower()}_config"
        )
        cls = getattr(model_class, f"{component}Model", None)
        cfg = getattr(model_config, config_name, None)
        if cls is not None and cfg is not None:
            weights = sanitize_weights(cls, weights, cfg)
    return weights


def _vlm_module_quantization(model, quantization, path):
    """Honor checkpoint names before and after a model's key sanitation."""
    aliases = [path]
    if path.startswith("language_model."):
        aliases.append(path.removeprefix("language_model."))
    model_aliases = getattr(model, "quantization_path_aliases", None)
    if callable(model_aliases):
        aliases.extend(model_aliases(path))
    for alias in aliases:
        value = quantization.get(alias)
        if isinstance(value, dict) or value is False:
            return value
    return None


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


async def _resolve_media_target(url: str):
    """SSRF protection for remote media: one shared network policy (see ``netguard``) resolves
    the host off the event loop and refuses private, loopback, link-local, CGNAT, multicast,
    reserved and IPv4-mapped IPv6 addresses. Raises ValueError("SSRF blocked: ...")."""
    from .netguard import UrlNotAllowedError, resolve_target

    try:
        return await resolve_target(url, allow_private=False)
    except UrlNotAllowedError as e:
        raise ValueError(f"SSRF blocked: {e}") from e


def _VALIDATE_LOCAL_PATH(path: str) -> str:
    """SECURITY: validate that a local filesystem path used for VLM
    media input (image/audio/video) is within the configured `YUNSHU_MEDIA_DIR`.

    Without this check, an authenticated user can pass `file:///etc/passwd` or
    a bare `/etc/passwd` path and exfiltrate sensitive host files via the VLM
    output. With `YUNSHU_ALLOW_LOCAL_FILES` unset, only paths under
    `YUNSHU_MEDIA_DIR` (default: `$TMPDIR/yunshu_media`) are allowed. Set
    `YUNSHU_ALLOW_LOCAL_FILES=1` to opt back into the legacy unrestricted
    behavior (NOT recommended when the API is reachable by anyone but the owner).

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
            f"Set YUNSHU_ALLOW_LOCAL_FILES=1 to bypass (NOT recommended when the API is reachable by anyone but the owner)."
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


def _completion_count(stats) -> int:
    """Tokens the reply consumed: the consumer's count, else the runner's progress."""
    return int(getattr(stats, "consumed", 0) or stats.generated)


_UNSET: Any = object()  # "argument not given" (None is a real value for an orphan key)


class _ThreadSafeQueue:
    """Wraps asyncio.Queue for the MLX/consumer threads.

    ``put_blocking`` bounds data by the async consumer. ``put_nowait`` carries the terminal
    items (end sentinel, error): it never drops one when the queue is full, it parks it
    behind the data and delivers it, in order, as soon as the consumer makes room.
    """

    def __init__(self, async_q, ev_loop):
        self._q = async_q
        self._loop = ev_loop
        self._backlog: collections.deque = collections.deque()
        self._drain: asyncio.Future | None = None
        self._closed = False

    def close(self):  # event-loop thread
        """The consumer is gone: drop parked terminal items and stop the drain task, which
        would otherwise wait forever on a queue nobody reads."""
        self._closed = True
        self._backlog.clear()
        if self._drain is not None and not self._drain.done():
            self._drain.cancel()

    def put_nowait(self, item):
        with contextlib.suppress(RuntimeError):  # event loop closed during shutdown
            self._loop.call_soon_threadsafe(self._put_terminal, item)

    def _put_terminal(self, item):  # event-loop thread
        if self._closed:
            return
        if self._backlog or self._q.full():
            self._backlog.append(item)
            if self._drain is None or self._drain.done():
                self._drain = asyncio.ensure_future(self._drain_backlog())
        else:
            self._q.put_nowait(item)

    async def _drain_backlog(self):
        while self._backlog:
            await self._q.put(self._backlog[0])
            if self._backlog:
                self._backlog.popleft()

    def put_blocking(self, item, cancel):
        """Bound the runner's producer by the async consumer, including slow clients."""
        try:
            future = asyncio.run_coroutine_threadsafe(self._q.put(item), self._loop)
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


class _VLMTextPromptCache:
    """Thread-safe LRU cache for VLM text prompt tokenization results.

    Caches the output of _format_prompt() + tokenizer.encode() keyed by a
    content hash of the input messages.  On cache hit, the expensive
    tokenization step is skipped entirely.

    Also caches the output of _processor.apply_chat_template() for the VLM
    vision path (VLM models with images/audio), keyed by messages + template
    kwargs hash.

    Memory-bounded with LRU eviction by entry count AND total size (a few long agent
    prompts must not pin hundreds of MB): token ids are kept as a compact int32 array
    (4 bytes a token, a Python list costs about 36), and the cache holds at most
    ``max_total_tokens`` tokens / ``max_total_chars`` template characters.
    Thread-safe via internal lock.
    """

    def __init__(
        self,
        max_entries: int = 256,
        max_total_tokens: int = 4_000_000,
        max_total_chars: int = 16_000_000,
    ) -> None:
        from collections import OrderedDict

        self._cache: OrderedDict[str, array.array] = OrderedDict()
        self._template_cache: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._max_total_tokens = max_total_tokens
        self._max_total_chars = max_total_chars
        self._total_tokens = 0
        self._total_chars = 0
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
        from .tokenizer_prefix import TokenizerPrefixCache

        self.prefix_tokens = TokenizerPrefixCache()

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
                return entry.tolist()
            self._stats["misses"] += 1
            return None

    def put_token_ids(self, cache_key: str, token_ids: list[int]) -> None:
        """Store token IDs for a prompt."""
        if len(token_ids) > self._max_total_tokens:
            return  # one prompt larger than the whole budget is not worth keeping
        packed = array.array("i", token_ids)
        with self._lock:
            old = self._cache.pop(cache_key, None)
            if old is not None:
                self._total_tokens -= len(old)
            while self._cache and (
                len(self._cache) >= self._max_entries
                or self._total_tokens + len(packed) > self._max_total_tokens
            ):
                _, evicted = self._cache.popitem(last=False)
                self._total_tokens -= len(evicted)
                self._stats["evictions"] += 1
            self._cache[cache_key] = packed
            self._total_tokens += len(packed)

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
        if len(template_text) > self._max_total_chars:
            return
        with self._lock:
            old = self._template_cache.pop(cache_key, None)
            if old is not None:
                self._total_chars -= len(old)
            while self._template_cache and (
                len(self._template_cache) >= self._max_entries
                or self._total_chars + len(template_text) > self._max_total_chars
            ):
                _, evicted = self._template_cache.popitem(last=False)
                self._total_chars -= len(evicted)
                self._stats["evictions"] += 1
            self._template_cache[cache_key] = template_text
            self._total_chars += len(template_text)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._template_cache.clear()
            self._total_tokens = self._total_chars = 0
        self.prefix_tokens.clear()

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            s = dict(self._stats)
            s["tokenization_entries"] = len(self._cache)
            s["template_entries"] = len(self._template_cache)
            s["max_entries"] = self._max_entries
            s["cached_tokens"] = self._total_tokens
            s["template_chars"] = self._total_chars
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


def resolve_external_ple_manifest(config: dict, model_path) -> None:
    """Make a relative ``text_config.ple_storage.manifest`` (qwen4_exp external PLE checkpoints)
    absolute against the model directory, as ``mlx_vlm.utils.load_model`` does; without it the
    model opens ``ple-store.json`` relative to the server's working directory and fails to load."""
    ple = (config.get("text_config") or {}).get("ple_storage")
    manifest = ple.get("manifest") if isinstance(ple, dict) else None
    if isinstance(ple, dict) and manifest and not Path(manifest).is_absolute():
        ple["manifest"] = str(Path(model_path) / manifest)


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
        self._rehearsal_cancel: threading.Event | None = None
        self._rehearsal_state: next_turn.RehearsalState | None = None
        self._rehearsal_task: asyncio.Future | None = None
        self._active_count_lock = threading.Lock()
        self._num_requests_processed = 0
        self._total_reasoning_tokens = 0
        self._start_time = 0.0
        self._has_vision = False
        self._is_vlm = False
        self._temp_files: list[str] | None = None
        self._temp_files_lock = threading.Lock()

        from yunshu_control.token_counter import TokenCountCache

        self._guard_token_counts = TokenCountCache()

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
        self._preflight_memory(model_path)

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
        from .tool_call_grammar import prewarm_for_engine

        prewarm_for_engine(self)  # background llguidance token tables
        self._resolve_reasoning_channel_ids()
        self._is_vlm = True
        logger.info("Loaded model via mlx_vlm (vision=%s)", self._has_vision)
        self._finish_vlm_load(model_path)

    def _load_vision_model(self, model_path):
        """Load a vision model using mlx_vlm with proper nested config handling."""
        # Apply mlx-vlm patches BEFORE load: the nemotron model_type remap must be
        # in MODEL_REMAPPING before get_model_and_args() reads the config, else
        # newer Nemotron omni variants fail "model type … not supported".
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
        if config.get("model_type") == "qwen4_exp":
            # MLX bills a command-buffer commit for every large bound buffer (expert banks exceed its
            # 50 MB default cap twice per MoE layer); raising the caps before the first Metal command
            # removes those commits (-0.6 to -1.0 ms per plain Flash-Next step, same tokens).
            # A user-set value wins.
            os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "100000")
            os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "200")

        weight_files = [
            wf
            for wf in glob.glob(str(Path(model_path) / "*.safetensors"))
            if not wf.endswith("consolidated.safetensors")
        ]
        if not weight_files:
            raise FileNotFoundError(f"No safetensors in {model_path}")

        from .ple_ssd import external_ple, is_ple_scale, is_ple_tensor

        ple_from_files = external_ple(config, model_path)
        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))
        if ple_from_files:
            # the n-gram table is read row by row from these files (ple_ssd); mx.load is lazy, so
            # dropping the entries here means the table is never read into memory
            weights = {k: v for k, v in weights.items() if not is_ple_tensor(k)}
        ple_table_scales = {}
        if config.get("model_type") == "qwen4_exp":
            from .ple_scale import collect_table_scales

            ple_table_scales = collect_table_scales(weights)
            if ple_from_files:
                # the per-table scales are applied by apply_table_scales below; left in place without
                # their shards, upstream's FP8 conversion would take them for an FP8 PLE pack (upstream
                # drops them anyway when the shards are loaded)
                weights = {k: v for k, v in weights.items() if not is_ple_scale(k)}

        reason = unsupported_pack_reason(config, weights)
        if reason:
            raise ValueError(f"unsupported checkpoint pack: {reason}")

        model_class, _ = get_model_and_args(config=config)

        config.setdefault("text_config", config.pop("llm_config", {}))
        config.setdefault("vision_config", {})
        config.setdefault("audio_config", {})
        resolve_external_ple_manifest(config, model_path)
        from . import ple_parallel

        ple_parallel.install(int(settings.get("YUNSHU_PLE_READ_THREADS")))
        if config.get("model_type") == "qwen4_exp":
            from .kernels import tf_hc_forward

            tf_hc_forward.install()

        model_config = model_class.ModelConfig.from_dict(config)
        modules = ["text", "vision", "perceiver", "projector", "audio"]
        model_config = update_module_configs(model_config, model_class, config, modules)
        if hasattr(model_config, "model_path"):
            model_config.model_path = str(model_path)

        model = model_class.Model(model_config)
        if config.get("model_type") == "qwen4_exp":
            from .model_patches import sanitize_qwen4_checkpoint

            weights, self._qwen4_mtp_weights = sanitize_qwen4_checkpoint(
                model,
                weights,
                prepare=lambda w: _prepare_vlm_weights(
                    model, model_class, model_config, w
                ),
            )
            from .qwen4_norms import infer_direct_gamma, normalize

            self._qwen4_direct_norms = infer_direct_gamma(weights)
            normalize(model, weights, direct=self._qwen4_direct_norms)
        else:
            weights = _prepare_vlm_weights(model, model_class, model_config, weights)

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
        if quantization is None and any(k.endswith(".scales") for k in weights):
            # Native packed tensors without a quantization block (e.g. mx fp4/fp8 packs):
            # the tensor shapes decide per module.
            quantization = {}
            logger.info("checkpoint has scales but no quantization config; deriving")
        if quantization is not None:
            config["quantization"] = (
                quantization  # so the per-layer predicate (p in ...) holds
            )

            _quant_overrides: dict[str, dict] = {}

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
                per_module = _vlm_module_quantization(model, quantization, p)
                derived = tensor_quantization_override(
                    weights,
                    p,
                    getattr(getattr(m, "weight", None), "shape", None),
                    quantization,
                    per_module,
                )
                if derived is not None:
                    _quant_overrides[p] = derived
                    return derived
                if per_module is not None:
                    return per_module
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
            if _quant_overrides:
                logger.warning(
                    "quantization config disagreed with %d tensors; used tensor-derived "
                    "bits/group_size (first: %s)",
                    len(_quant_overrides),
                    next(iter(_quant_overrides.items())),
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
        if ple_table_scales:
            from .ple_scale import apply_table_scales

            n_scaled = apply_table_scales(model.language_model, ple_table_scales)
            if n_scaled:
                logger.info(
                    "Qwen4 PLE: %d n-gram table(s) scaled by %s",
                    n_scaled,
                    sorted(set(ple_table_scales.values())),
                )
        from .apc_manager import total_memory_bytes
        from .weight_residency import eval_set

        to_eval, lazy_names = eval_set(
            flatten_tree(model.parameters()), total_memory_bytes()
        )
        mx.eval(to_eval)
        if lazy_names:
            logger.info(
                "weights take most of RAM: %d vision tensors stay lazy until an image request",
                len(lazy_names),
            )
        model.eval()
        return model

    def _finish_vlm_load(self, model_path) -> None:
        """Common post-load initialization."""

        # The processor prepares image/audio/video inputs and renders media
        # chat templates; every mlx_vlm model needs it for media requests.
        try:
            from mlx_vlm.utils import load_processor

            from .image_processor_fixes import install as install_image_fixes

            install_image_fixes()  # PIL images read channels-last (mlx-vlm 8dd7746)
            self._processor = load_processor(Path(model_path))
        except Exception as e:
            logger.warning(f"Could not load VLM processor: {e}")

        if self._config.get("model_type") == "deepseek_v4":
            from .deepseek_v4_chat import install

            if install(self._tokenizer, self._processor):
                logger.info("DeepSeek V4: installed official 0731 chat encoder")

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
                "VLM model ~%.1fGiB on disk, large=%s (clear buffer pool on idle %s)",
                _bytes / (1 << 30),
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
        runner = self._build_batch_runner(str(model_path))
        self._batch_runner = runner
        policy = settings.get("YUNSHU_GC_POLICY")
        self._gc_policy = policy
        runner.gc_guard = gc_policy.install(policy)
        # YUNSHU_GIL_SWITCH_MS is reload-class: re-read it on every model load.
        from . import gil_switch

        gil_switch.apply()

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
        apc = self._apc_backend
        if apc is not None:
            # Persist resident checkpoints to the SSD tier (if any) before dropping them.
            with contextlib.suppress(Exception):
                await asyncio.get_running_loop().run_in_executor(None, apc.close)
        from .kernels import tf_hc_forward

        tf_hc_forward.forget(self._model)
        self._model = None
        self._tokenizer = None
        self._apc_backend = None
        self._apc_semantic_hash = None
        self._batch_runner = None
        self._processor = None
        self._running = False
        if getattr(self, "_gc_policy", "default") != "default":
            gc_policy.unfreeze()
            self._gc_policy = "default"

        self._text_prompt_cache.clear()
        guard_counts = getattr(self, "_guard_token_counts", None)
        if guard_counts is not None:
            guard_counts.clear()
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
        if budget is None or getattr(manager, "disk", None) is not None:
            # A checkpoint too big for RAM still lands on the SSD tier when there is one.
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

    def _release_active_lease(self, _future=None) -> None:
        if _future is not None and not _future.cancelled():
            _future.exception()  # mark a worker failure retrieved (the request already timed out)
        with self._active_count_lock:
            self._active_count = max(0, self._active_count - 1)

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
        _params = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
            repetition_penalty=repetition_penalty,
            stop=stop,
            enable_thinking=enable_thinking,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            **kwargs,
        )
        _key = self._orphan_key(messages, _params)
        if _key is not None:
            # One generation core for both modes: a retry that arrives non-streaming can
            # adopt a streamed orphan (and the reverse) instead of recomputing it.
            return await self._generate_collect(messages, _params, _key)
        from .structural_tag import constrains_initial_output

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
        _consumer_fut = None
        _lease_deferred = False

        try:
            image_paths = await self._extract_images(messages)
            audio_paths = await self._extract_audio(messages)
            video_frames = await self._extract_video_frames(messages)
            image_paths.extend(video_frames)
            _enable_thinking = self._default_enable_thinking(
                enable_thinking,
                constrained=constrains_initial_output(kwargs.get("json_schema")),
            )
            self._track_pipeline(kwargs, messages, image_paths, audio_paths)
            self._check_request_supported(image_paths, audio_paths, kwargs)
            await self._settle_rehearsal_early(messages, _enable_thinking, tpl_extra)

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

            from .request_tracker import current_request_info

            _latency_info = current_request_info.get()

            def _prepare():
                # Templating + media encoding on the MLX thread; generation is
                # then consumed off it (the runner's driver needs that thread).
                _marks = getattr(_latency_info, "latency_marks", {})
                _marks["template_start"] = time.perf_counter()
                ids, pkw, salt = self._runner_input(
                    messages, image_paths, audio_paths, _enable_thinking, tpl_extra
                )
                self._settle_rehearsal(ids)
                _marks["template_end"] = time.perf_counter()
                return _RunnerCall(
                    self._generate_vlm_runner_text,
                    ids,
                    extras=_runner_extras,
                    prompt_kwargs=pkw,
                    apc_semantic_hash=salt,
                    prompt_cache_plan=self._resolve_prompt_cache_plan(
                        kwargs.get("prompt_cache_plan"),
                        messages,
                        ids,
                        pkw,
                        _enable_thinking,
                        tpl_extra,
                        kwargs.get("prompt_cache_options")
                        or {"ttl": kwargs.get("prompt_cache_retention")},
                    ),
                    **runner_params,
                )

            loop = asyncio.get_running_loop()
            try:
                # Only a client-set timeout applies: a fixed default would also
                # count time spent waiting for a batch slot.
                _timeout_seconds = kwargs.get("timeout_seconds") or None
                from .grammar_compile import prepare_constraint

                async def _prepare_call():
                    # gather waits for validation before the consumer can admit any token.
                    call, _ = await asyncio.gather(
                        loop.run_in_executor(self._executor, _prepare),
                        prepare_constraint(kwargs.get("json_schema")),
                    )
                    return call

                call = await asyncio.wait_for(
                    _prepare_call(),
                    timeout=_timeout_seconds,
                )
                # Shielded: a timeout cancels the wait, not the worker thread, which keeps its
                # runner row until it sees the cancel event.
                _consumer_fut = loop.run_in_executor(self._runner_consumers(), call)
                (
                    result,
                    reasoning_tokens,
                    completion_token_count,
                    stop_hit,
                    budget_hit,
                    cached_token_count,
                ) = await asyncio.wait_for(
                    asyncio.shield(_consumer_fut),
                    timeout=_timeout_seconds,
                )
            except TimeoutError:
                self._num_requests_processed += 1
                kwargs["cancel_event"].set()
                if _consumer_fut is not None and not _consumer_fut.done():
                    # The executor thread is still winding down: it holds the lease
                    # (runner.inflight reads this count), so release when it exits, as the
                    # stream path does, not now.
                    _lease_deferred = True
                    _consumer_fut.add_done_callback(self._release_active_lease)
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
            self._maybe_rehearse(
                messages,
                result.rsplit("</think>", 1)[-1].strip(),
                finish_reason=_finish_reason,
                completion_tokens=completion_token_count,
                prompt_tokens=prompt_tokens,
                enable_thinking=_enable_thinking,
                tpl_extra=tpl_extra,
                has_media=bool(image_paths or audio_paths),
                kwargs=kwargs,
            )
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
                "stopped_by_stop_sequence": bool(
                    _runner_extras.get("stopped_by_stop_sequence")
                ),
            }
        finally:
            if not _lease_deferred:
                self._release_active_lease()
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

    async def _generate_collect(
        self, messages, params: dict, key: Any = _UNSET
    ) -> dict[str, Any]:
        """Non-streaming result assembled from the streaming core (same shape as the
        direct path: reasoning wrapped in think tags, exact counts)."""
        parts: list[str] = []
        reasoning: list[str] = []
        lps: list[Any] = []
        thinking = generated = cached = prompt_tokens = 0
        finish = None
        stop_seq = False
        timed_out = {
            "text": "",
            "finish_reason": "timeout",
            "model": self.model_name,
            "created": int(time.time()),
            "reasoning_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        # A non-streaming ``timeout`` bounds the whole call (the streaming core reads the same
        # field as a gap between tokens): the deadline below ends a reply that keeps arriving.
        deadline = asyncio.timeout(params.get("timeout_seconds") or None)
        try:
            async with (
                deadline,
                contextlib.aclosing(  # type: ignore[type-var]  # an async generator at runtime
                    self.generate_stream(messages=messages, orphan_key=key, **params)
                ) as stream,
            ):
                async for out in stream:
                    if out.error:
                        self._num_requests_processed += 1
                        if str(out.error).startswith("Streaming timeout"):
                            return timed_out
                        refusal = memory_limit_error(str(out.error))
                        if refusal is not None:
                            raise refusal
                        raise RuntimeError(out.error)
                    if out.new_text:
                        (
                            reasoning if out.current_state == "reasoning" else parts
                        ).append(out.new_text)
                    if out.logprobs:
                        lps.extend(out.logprobs)
                    thinking = max(thinking, out.reasoning_tokens or 0)
                    generated = max(generated, out.completion_tokens or 0)
                    prompt_tokens = out.prompt_tokens or prompt_tokens
                    cached = out.cached_tokens or cached
                    if out.finished:
                        finish = out.finish_reason
                        stop_seq = bool(out.stopped_by_stop_sequence)
        except TimeoutError:
            if not deadline.expired():
                raise
            self._num_requests_processed += 1
            logger.warning(
                "VLM non-streaming generate timed out after %ss",
                params.get("timeout_seconds"),
            )
            return timed_out
        self._num_requests_processed += 1
        if finish == "cancel":
            raise asyncio.CancelledError()
        self._total_reasoning_tokens += thinking
        if reasoning:
            parts = ["<think>", *reasoning, "</think>", *parts]
        return {
            "text": "".join(parts),
            "finish_reason": "stop" if finish == "stop" else "length",
            "model": self.model_name,
            "created": int(time.time()),
            "reasoning_tokens": thinking,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generated,
            "cached_tokens": cached,
            "logprobs": lps if params.get("logprobs") else None,
            "stopped_by_stop_sequence": stop_seq,
        }

    def _orphan_key(self, messages, params: dict) -> str | None:
        from . import orphan_adoption, settings

        if settings.get("YUNSHU_ORPHAN_GRACE_SECONDS") <= 0:
            return None
        return orphan_adoption.request_key(self.model_name, messages, params)

    async def generate_stream(
        self, prompt=None, messages=None, cancel_event=None, orphan_key=_UNSET, **kwargs
    ) -> AsyncIterator[RequestOutput]:
        """Streaming generation. A client that vanishes mid-way parks the generation
        for ``YUNSHU_ORPHAN_GRACE_SECONDS``; an identical request adopts it (see
        ``orphan_adoption``)."""
        from . import orphan_adoption, settings

        if isinstance(prompt, str):
            msgs = [{"role": "user", "content": prompt}]
        else:
            msgs = prompt if prompt is not None else (messages or [])
        if cancel_event is None:
            cancel_event = threading.Event()
        # ``generate`` already derived the key of the identical request: hash the
        # conversation once, not once per layer.
        key = self._orphan_key(msgs, kwargs) if orphan_key is _UNSET else orphan_key

        def start(real_cancel):
            return self._generate_stream_impl(
                messages=msgs, cancel_event=real_cancel, **kwargs
            )

        async with contextlib.aclosing(  # type: ignore[type-var]  # an async generator at runtime
            orphan_adoption.run(
                key,
                start,
                cancel_event,
                float(settings.get("YUNSHU_ORPHAN_GRACE_SECONDS")),
                int(settings.get("YUNSHU_ORPHAN_MAX")),
                # only read to log why a request did not match an orphan
                parts=(lambda: orphan_adoption.key_parts(self.model_name, msgs, kwargs))
                if key
                else None,
            )
        ) as stream:
            async for out in stream:
                yield out

    async def _generate_stream_impl(
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

        from .structural_tag import constrains_initial_output

        tpl_extra = self._request_template_extra(kwargs)
        if cancel_event is None:
            cancel_event = threading.Event()

        # Extract images/audio once, reuse for both pipeline and generation.
        # Per-request identity-based temp tracking (see generate()): avoids the
        # positional-offset race that deleted concurrent/n>1 requests' files.
        _req_temp_files: list[str] = []
        _temp_token = _request_temp_files.set(_req_temp_files)
        try:
            image_paths = await self._extract_images(messages)
            audio_paths = await self._extract_audio(messages)
            video_frames = await self._extract_video_frames(messages)
            image_paths.extend(video_frames)

            enable_thinking = self._default_enable_thinking(
                enable_thinking,
                constrained=constrains_initial_output(kwargs.get("json_schema")),
            )
            self._track_pipeline(kwargs, messages, image_paths, audio_paths)
            self._check_request_supported(image_paths, audio_paths, kwargs)
            await self._settle_rehearsal_early(messages, enable_thinking, tpl_extra)
        except BaseException:
            # Nothing below owns these files yet: a bad request, a failed decode or a
            # cancel while extracting must not leave decoded media on disk.
            self._release_request_temp_files(_req_temp_files, _temp_token)
            raise

        import uuid

        req_id = f"vlm-{uuid.uuid4().hex[:8]}"

        queue: asyncio.Queue[RequestOutput | None] = asyncio.Queue(maxsize=256)
        # DFlash's copy-index setup can collect the Python heap after its first
        # target token is ready. Let the async consumer deliver that token first.
        first_delivery = (
            threading.Event()
            if getattr(self._batch_runner, "draft_kind", None) == "dflash"
            else None
        )

        # Capture event loop for thread-safe queue writes from executor thread.
        # asyncio.Queue.put_nowait() is NOT thread-safe — must schedule puts
        # via call_soon_threadsafe (same pattern as batched_engine.py).
        _loop_for_queue = asyncio.get_running_loop()

        _safe_queue = _ThreadSafeQueue(queue, _loop_for_queue)

        try:
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
        except BaseException:
            self._release_request_temp_files(_req_temp_files, _temp_token)
            raise

        runner_params["first_delivery"] = first_delivery
        from .request_tracker import current_request_info

        _latency_info = current_request_info.get()

        def _prepare():
            _marks = getattr(_latency_info, "latency_marks", {})
            _marks["template_start"] = time.perf_counter()
            ids, pkw, salt = self._runner_input(
                messages, image_paths, audio_paths, enable_thinking, tpl_extra
            )
            self._settle_rehearsal(ids)
            _marks["template_end"] = time.perf_counter()
            return _RunnerCall(
                self._stream_vlm_runner_text,
                ids,
                req_id,
                _safe_queue,
                prompt_kwargs=pkw,
                apc_semantic_hash=salt,
                prompt_cache_plan=self._resolve_prompt_cache_plan(
                    kwargs.get("prompt_cache_plan"),
                    messages,
                    ids,
                    pkw,
                    enable_thinking,
                    tpl_extra,
                    kwargs.get("prompt_cache_options")
                    or {"ttl": kwargs.get("prompt_cache_retention")},
                ),
                **runner_params,
            )

        with self._active_count_lock:
            self._active_count += 1
        loop = asyncio.get_running_loop()

        async def _stream_job():
            try:
                # Templating + media encoding on the MLX thread; tokens are then
                # consumed off it (the runner's driver needs that thread).
                from .grammar_compile import prepare_constraint

                call, _ = await asyncio.gather(
                    loop.run_in_executor(self._executor, _prepare),
                    prepare_constraint(kwargs.get("json_schema")),
                )
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
        _reply_parts: list[tuple[str, str]] = []
        _finish_seen: str | None = None

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
                    _finish_seen = output.finish_reason
                if output.new_text:
                    _reply_parts.append((output.new_text, output.current_state))
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
                # The caller has resumed after handling this chunk. This also
                # releases promptly when a parser needs another token before
                # it can produce a visible fragment.
                if first_delivery is not None and output.new_text:
                    first_delivery.set()
        finally:
            if first_delivery is not None:
                first_delivery.set()
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
            _safe_queue.close()
            # Drain remaining queue items to unblock the executor thread
            # so it can observe the cancellation and exit promptly.
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            if _ended_normally and not cancel_event.is_set():
                self._maybe_rehearse(
                    messages,
                    next_turn.reply_text(_reply_parts),
                    finish_reason=_finish_seen,
                    completion_tokens=_completion_tokens_count,
                    prompt_tokens=_prompt_tokens_count,
                    enable_thinking=enable_thinking,
                    tpl_extra=tpl_extra,
                    has_media=bool(image_paths or audio_paths),
                    kwargs=kwargs,
                )
            # This async generator is driven step-by-step from a separate task by
            # the SSE keepalive wrapper, so this finally can run in a different
            # contextvars.Context than the one .set() ran in — reset() would then
            # raise "Token was created in a different Context". File cleanup below
            # uses the closure list, not the ContextVar, so a failed reset is safe.
            self._release_request_temp_files(_req_temp_files, _temp_token)

    def _release_request_temp_files(self, req_files: list[str], token) -> None:
        """Drop one request's context var and delete exactly the files it registered."""
        with contextlib.suppress(ValueError):
            _request_temp_files.reset(token)
        _mine = list(req_files)
        if _mine:
            _mine_set = set(_mine)
            with self._temp_files_lock:
                if self._temp_files is not None:
                    self._temp_files[:] = [
                        f for f in self._temp_files if f not in _mine_set
                    ]
        self._cleanup_temp_files(_mine)

    # -- next-turn rehearsal (YUNSHU_NEXT_TURN_PREFILL) ---------------------------------

    def _cancel_rehearsal(self) -> None:
        """Stop an unfinished rehearsal at its next prefill atom (no prompt to compare)."""
        ev = getattr(self, "_rehearsal_cancel", None)
        if ev is not None:
            ev.set()

    def _peek_prompt_ids(
        self, messages: list[dict], enable_thinking: bool | None, tpl_extra: dict | None
    ) -> list[int]:
        """Token ids of a request's prompt without touching MLX (any thread)."""
        text = self._format_prompt(
            messages, enable_thinking=enable_thinking, template_extra=tpl_extra
        )
        bos = getattr(self._tokenizer, "bos_token", None)
        add_special = not (isinstance(bos, str) and bos and text.startswith(bos))
        return list(
            self._text_prompt_cache.prefix_tokens.encode(
                self._tokenizer, text, add_special_tokens=add_special
            )
        )

    async def _settle_rehearsal_early(
        self, messages: list[dict], enable_thinking: bool | None, tpl_extra: dict | None
    ) -> None:
        """A request arrived: adopt or cancel the running rehearsal BEFORE queueing behind it.

        The request's own tokenization runs on the single MLX executor, which is busy with the
        rehearsal's prefill; deciding there would only cancel after the atom. Decide here, from
        the event loop, with a token peek that needs no MLX."""
        state = getattr(self, "_rehearsal_state", None)
        if state is None:
            return
        has_media = any(
            isinstance(m.get("content"), list)
            and any(
                isinstance(p, dict) and p.get("type") not in (None, "text")
                for p in m["content"]
            )
            for m in messages
        )
        if has_media or not state.submitted:
            # media, or no work running yet (the start delay): nothing to adopt, no tokens needed
            state.cancel_event.set()
            return
        try:
            ids = await asyncio.get_running_loop().run_in_executor(
                None, self._peek_prompt_ids, messages, enable_thinking, tpl_extra
            )
        except Exception:
            logger.debug("rehearsal peek failed; cancelling", exc_info=True)
            state.cancel_event.set()
            return
        self._settle_rehearsal(ids)

    def _settle_rehearsal(self, ids: list[int]) -> None:
        """A real request's prompt is known: adopt the running rehearsal or cancel it."""
        state = getattr(self, "_rehearsal_state", None)
        if state is None:
            return
        covered = 0
        manager = getattr(self, "_apc_backend", None)
        probe = getattr(manager, "resident_prefix_len", None)
        if probe is not None:
            try:
                covered = int(probe(ids))
            except Exception:
                covered = 0
        verdict = state.arrive(ids, covered)
        logger.debug("next-turn rehearsal vs arriving request: %s", verdict)

    def _maybe_rehearse(
        self,
        messages: list[dict],
        reply: str,
        *,
        finish_reason: str | None,
        completion_tokens: int,
        enable_thinking: bool | None,
        tpl_extra: dict | None,
        has_media: bool,
        kwargs: dict,
        prompt_tokens: int = 0,
    ) -> None:
        mode = settings.get("YUNSHU_NEXT_TURN_PREFILL")
        if mode in (None, False, "", "off", "boundary"):
            return
        reason = next_turn.ineligible_reason(
            finish_reason=finish_reason,
            completion_tokens=completion_tokens,
            reply=reply,
            has_media=has_media,
            has_tools=bool(kwargs.get("_tool_spec") or kwargs.get("tools")),
            constrained=bool(kwargs.get("json_schema") or kwargs.get("grammar")),
        )
        if reason is not None:
            logger.debug("next-turn rehearsal ineligible: %s", reason)
            next_turn.record("ineligible")
            return
        from .kernels import preempt

        if not preempt.supported():
            # a diverging request would wait for the whole 2048-token atom
            logger.debug("next-turn rehearsal ineligible: prefill is not preemptible")
            next_turn.record("ineligible")
            return
        if self._active_count > 1 or self._batch_runner is None:
            logger.debug("next-turn rehearsal ineligible: engine busy")
            next_turn.record("ineligible")
            return  # other work is waiting: the GPU is not idle
        # Concurrent completions can both schedule a delayed rehearsal. Cancel the
        # previous owner before replacing its handle, or arriving requests can only
        # cancel the newest task and the orphan keeps prefilling beside decode.
        self._cancel_rehearsal()
        # created here, not in the task: a request arriving before the task first runs must
        # still be able to cancel it
        ev = threading.Event()
        ev.preemptible = True  # type: ignore[attr-defined]  # read by the runner (kernels/preempt)
        self._rehearsal_cancel = ev
        state = next_turn.RehearsalState(ev)
        self._rehearsal_state = state
        task = asyncio.ensure_future(
            self._rehearse(
                messages,
                reply,
                enable_thinking,
                tpl_extra,
                str(mode),
                state,
                prompt_tokens,
            )
        )
        self._rehearsal_task = task

    async def _rehearse(
        self,
        messages: list[dict],
        reply: str,
        enable_thinking: bool | None,
        tpl_extra: dict | None,
        mode: str = "stride",
        state: next_turn.RehearsalState | None = None,
        prompt_tokens: int = 0,
    ) -> None:
        loop = asyncio.get_running_loop()
        if state is None:
            state = next_turn.RehearsalState(threading.Event())
            self._rehearsal_cancel = state.cancel_event
            self._rehearsal_state = state
        ev = state.cancel_event
        rehearsal_messages = next_turn.with_reply(messages, reply)
        marker = self._chatml_head_marker()
        tokenizer = self._tokenizer

        def _prepare():
            full = self._tokenize_with_cache(
                rehearsal_messages,
                enable_thinking=enable_thinking,
                template_extra=tpl_extra,
            ).tolist()
            ids = next_turn.cut_before_user_turn(
                full,
                marker[0] if marker else None,
                lambda toks: tokenizer.decode(list(toks)),
            )
            if ids is None or ev.is_set():
                return None
            stride = int(getattr(self, "_apc_prefill_stride", 0) or 0)
            manager = self._apc_backend  # the YunshuAPCManager itself
            guard = int(getattr(manager, "exact_cache_guard_tokens", 1) or 1)
            if getattr(manager, "prefill_alignment", 0) == 64:
                target = next_turn.aligned_rehearsal_length(
                    len(ids), guard, prompt_tokens=prompt_tokens
                )
                if target is None:
                    return None
                ids = full[:target]
            elif mode == "exact":
                # end exactly at the next turn's header: the manager's turn boundary is
                # the checkpoint, the guard token is the header itself
                ids = full[: len(ids) + guard]
            else:
                target = next_turn.rehearsal_length(
                    len(ids), stride, guard, prompt_tokens=prompt_tokens
                )
                if target is None:
                    return None
                ids = ids[:target]
            params = self._runner_kwargs(
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                seed=None,
                stop=None,
                stop_token_ids=[],
                repetition_penalty=1.0,
                logprobs=False,
                top_logprobs=None,
                enable_thinking=enable_thinking,
                thinking_budget=None,
                cancel_event=ev,
                kwargs={},
            )
            state.set_ids(ids)
            if manager is not None:
                manager.keep_prior_for = tuple(map(int, ids))
            return _RunnerCall(
                self._generate_vlm_runner_text,
                ids,
                extras={},
                prompt_kwargs=None,
                apc_semantic_hash=None,
                prompt_cache_plan=self._resolve_prompt_cache_plan(
                    None,
                    rehearsal_messages[:-1],
                    ids,
                    None,
                    enable_thinking,
                    tpl_extra,
                    {"ttl": None},
                ),
                **params,
            )

        started = False
        call = None
        try:
            # Wait first, prepare after: preparing tokenizes the whole conversation on the MLX
            # executor, which a back-to-back request would queue behind (warm TTFT +10-17 ms).
            delay = float(settings.get("YUNSHU_NEXT_TURN_DELAY_S") or 0.0)
            if delay > 0:
                await asyncio.sleep(delay)
            if ev.is_set():
                next_turn.record("cancelled")
                return
            call = await loop.run_in_executor(self._executor, _prepare)
            if call is None:
                logger.debug("next-turn rehearsal skipped (no clean cut)")
                return
            if not state.submit():
                next_turn.record("cancelled")
                return
            next_turn.record("started")
            started = True
            t0 = time.perf_counter()
            extras: dict = call.keywords["extras"]
            outcome = await loop.run_in_executor(self._runner_consumers(), call)
            cached = int(outcome[5]) if outcome and len(outcome) > 5 else -1
            cancelled = ev.is_set()
            next_turn.record(
                "cancelled" if cancelled else "completed",
                0 if cancelled else int(extras.get("prompt_tokens") or 0),
            )
            logger.info(
                "next-turn rehearsal: prompt %s tokens, %s cached, done in %.2fs%s",
                extras.get("prompt_tokens"),
                cached,
                time.perf_counter() - t0,
                " (cancelled)" if cancelled else "",
            )
        except (Exception, asyncio.CancelledError):
            logger.debug("next-turn rehearsal ended", exc_info=True)
            if started and ev.is_set():
                next_turn.record("cancelled")
        finally:
            if getattr(self, "_rehearsal_cancel", None) is ev:
                self._rehearsal_cancel = None
            if getattr(self, "_rehearsal_state", None) is state:
                self._rehearsal_state = None
            # keep_prior_for stays set: the SSD write of the rehearsal's checkpoint completes in
            # the background after this task ends, and its supersede hook must still skip the
            # previous request (a tombstone there vetoes a retry's RAM hit). The next
            # rehearsal replaces it.

    def backend_capabilities(self, lm: Any = None) -> Any:
        """This backbone's cache layout via the shared ``model_backend`` layer
        (memoized); callers use it to describe cache layout."""
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
    _SPEC_MODEL_TYPES = ("qwen3_5", "qwen3_6", "qwen3_5_moe", "qwen4_exp")

    def _memory_total_bytes(self) -> int:
        from .apc_manager import total_memory_bytes
        from .memory_admission import effective_total_bytes

        return effective_total_bytes(
            total_memory_bytes(), settings.get("YUNSHU_MAX_MEMORY_GB")
        )

    def _memory_plan(self, weights: int, apc_gb: float, model_path):
        from .memory_admission import (
            GIB,
            default_reserve_bytes,
            kv_bytes_per_token,
            plan_load,
        )

        total = self._memory_total_bytes()
        reserve = settings.get("YUNSHU_MEMORY_RESERVE_GB")
        return plan_load(
            total,
            weights,
            apc_gb,
            kv_per_token=kv_bytes_per_token(getattr(self, "_config", {}) or {}),
            reserve_bytes=(
                int(float(reserve) * GIB)
                if reserve is not None
                else default_reserve_bytes(total)
            ),
            model_path=model_path,
        )

    def _build_memory_gate(self):
        if not settings.get("YUNSHU_MEMORY_ADMISSION"):
            return None
        from .memory_admission import build_gate

        gate = build_gate(
            self._config,
            total_bytes=self._memory_total_bytes(),
            ceiling_gb=settings.get("YUNSHU_MAX_MEMORY_GB"),
            reserve_gb=settings.get("YUNSHU_MEMORY_RESERVE_GB"),
        )
        if gate is not None:
            logger.info(
                "Memory admission: KV %.0f KiB/token, keeping %.1f GiB free",
                gate.kv_per_token / 1024,
                gate.reserve / (1 << 30),
            )
        return gate

    def _preflight_memory(self, model_path: Path) -> None:
        """Refuse a load that cannot fit before touching the weights: a model whose weights
        leave no room for the OS reserve and a minimum KV only ends in swap."""
        if not settings.get("YUNSHU_MEMORY_ADMISSION"):
            return
        from .memory_admission import weights_bytes_on_disk

        plan = self._memory_plan(weights_bytes_on_disk(model_path), 0.0, model_path)
        if plan.error:
            raise MemoryError(f"Cannot load {model_path.name}: {plan.error}")
        for note in plan.notes:
            logger.warning("Memory plan: %s", note)

    def _apc_memory_gb(self) -> float:
        """APC RAM budget in GiB: ``YUNSHU_VLM_APC_MEMORY_GB``, else sized from the
        machine's memory and the model's weights (0 disables the cache). Either way it
        shrinks to what the resident weights, the OS reserve and a minimum KV leave."""
        from .apc_manager import auto_memory_gb

        budget = settings.get("YUNSHU_VLM_APC_MEMORY_GB")
        weights = 0
        with contextlib.suppress(OSError):
            weights = sum(
                p.stat().st_size for p in Path(self._model_path).glob("*.safetensors")
            )
        if budget is not None:
            gb = float(budget)
        else:
            gb = auto_memory_gb(self._memory_total_bytes(), weights)
            logger.info("APC RAM budget: %.0f GiB (auto)", gb)
        if gb > 0 and settings.get("YUNSHU_MEMORY_ADMISSION"):
            import mlx.core as mx

            plan = self._memory_plan(
                int(mx.get_active_memory()) or weights, gb, self._model_path
            )
            for note in plan.notes:
                logger.warning("Memory plan: %s", note)
            gb = plan.apc_gb
        return gb

    def _chatml_head_marker(self) -> tuple[int, int] | None:
        """Token ids of ``<|im_start|>`` and ``user`` when the chat template is ChatML: the
        end of the system turn (system prompt + tools) is then a checkpoint position, so a
        new session with the same head reuses it."""
        tok = self._tokenizer
        template = getattr(tok, "chat_template", None) or ""
        if not (isinstance(template, str) and "<|im_start|>" in template):
            return None
        try:
            start = tok.convert_tokens_to_ids("<|im_start|>")
            user = tok.encode("user", add_special_tokens=False)
            if start is None or start == getattr(tok, "unk_token_id", None):
                return None
            return (int(start), int(user[0])) if len(user) == 1 else None
        except Exception:
            logger.debug("no ChatML head marker", exc_info=True)
            return None

    def _apc_disk_namespace(self) -> str:
        """Directory namespace of this checkpoint's states on disk.

        ``yunshu_kv.fingerprint``: the path, the weight files (names, sizes, mtimes), the
        config / tokenizer / chat-template / preprocessor contents and the cache layout
        (KV precision, mlx-vlm's adapter schema and exact-cache format). Re-quantizing,
        replacing or re-tokenizing a checkpoint in place, or a layout change, selects a
        fresh namespace and never reads back the old states.
        """
        from mlx_vlm.apc_adapters import ADAPTER_SCHEMA_VERSION

        from yunshu_kv.fingerprint import checkpoint_fingerprint

        return checkpoint_fingerprint(
            self._model_path,
            extra={
                "kv_precision": settings.get("YUNSHU_KV_PRECISION"),
                "apc_adapter_schema": ADAPTER_SCHEMA_VERSION,
                "apc_layout": "exact_cache_v1",
                **(
                    {"qwen4_norms": "direct-gamma-v1"}
                    if getattr(self, "_qwen4_direct_norms", False)
                    else {}
                ),
                "prefill_kernel": self._prefill_kernel_id(),
                "shared_prefix_invariant": bool(
                    getattr(self, "_prefix_invariant_dispatch", False)
                ),
            },
            digest_size=8,
        )

    def _prefill_kernel_id(self) -> str:
        """The arithmetic of prefill chunks (``lane_linear.prefill_kernel_id``) when the lane
        projections serve this model's prefill, else ``"stock"``."""
        from .kernels import lane_linear

        model = getattr(self, "_model", None)
        lm = getattr(model, "language_model", None)
        from .kernels import gdn_prefill

        matmul = "stock"
        single_row = ""
        if lm is not None:
            modules = [m for _, m in lm.named_modules()]
            lanes = any(isinstance(m, lane_linear.LaneLinear) for m in modules)
            invariant = lanes or any(
                getattr(m, "_yunshu_invariant", False) for m in modules
            )
            if lanes:
                matmul = lane_linear.prefill_kernel_id()
            # Old AR requests used stock attention for short prefill tails;
            # do not restore those checkpoints into the invariant lone-row path.
            if invariant:
                single_row = "+single-row-invariant-v1"
        stride = getattr(self, "_apc_prefill_stride", 0)
        grid = f"+apc-grid{stride}-v2" if stride else ""
        if gdn_prefill.split_enabled():
            from .kernels import attn_prefill

            grid = f"+split64-v1+{attn_prefill.arithmetic_id()}"
        from . import ane_ffn_runtime

        return f"{matmul}+{gdn_prefill.kernel_id()}{single_row}{grid}{ane_ffn_runtime.kernel_suffix()}"

    def _install_apc_identity(self, lm) -> None:
        """Make every APC key (RAM and SSD) depend on this checkpoint's identity, and
        have the SSD tier check what it reads back against the live model.

        mlx-vlm folds ``lm.apc_key_dependencies()`` into every request's extra hash, so a
        checkpoint revision (weights, config, tokenizer, layout) that differs from the one
        that wrote a state can never match it, even when the file sits in a shared
        directory. The validator rejects a state whose layer classes, offsets or
        geometry do not fit the model (a corrupt or foreign file).
        """
        identity = self._apc_disk_namespace()
        kernel = self._prefill_kernel_id()
        lm.apc_key_dependencies = lambda: (f"yunshu-checkpoint:{identity}:{kernel}",)
        disk = getattr(self._apc_backend, "disk", None)
        if disk is None or not hasattr(disk, "validator"):
            return
        from .apc_manager import check_loaded_cache

        try:
            template = lm.make_cache()
        except Exception:
            template = None
        args = getattr(lm, "args", None)
        kv_heads = getattr(args, "num_key_value_heads", None)
        # the config's explicit head_dim; the module property derives it from the hidden size
        # and heads, which is wrong when a family sets head_dim itself
        head_dim = getattr(args, "head_dim", None) or getattr(lm, "head_dim", None)
        if not isinstance(head_dim, int):
            head_dim = None

        disk.validator = lambda tokens, cache: check_loaded_cache(
            tokens, cache, template, kv_heads, head_dim
        )

    def _apc_disk_tier(self):
        """The APC SSD tier (on by default; ``YUNSHU_VLM_APC_DISK=0`` switches it off).

        Holds evicted prefix checkpoints (incl. hybrid recurrent state) so a
        long document or agent session revisited after RAM eviction (or a restart) is
        read back instead of re-prefilled. Namespaced by checkpoint; capped by
        ``YUNSHU_VLM_APC_DISK_GB``. Directory: ``YUNSHU_VLM_APC_DISK_DIR``, else
        ``~/.yunshu/cache/apc``.
        """
        from . import paths

        path = paths.apc_dir()
        if path is None:
            return None
        from yunshu_kv.disk_budget import resolve_cap_gb

        from .apc_manager import SpillDiskStore

        max_gb = resolve_cap_gb(settings.get("YUNSHU_VLM_APC_DISK_GB"), path)
        from .apc_storage import parse_tiers

        lower_specs = parse_tiers(settings.get("YUNSHU_VLM_APC_DISK_TIERS"))
        # with lower tiers the budget hands what it would delete to the background mover (it
        # deletes only when no lower tier can take the file, or the mover is far behind)
        hard_gb = max_gb
        try:
            if lower_specs:
                from .apc_storage import TieredDiskStore

                disk = TieredDiskStore(
                    path,
                    namespace=self._apc_disk_namespace(),
                    num_workers=1,
                    max_bytes=int(hard_gb * (1 << 30)) if hard_gb > 0 else None,
                )
                disk.soft_cap_bytes = int(max_gb * (1 << 30)) if max_gb > 0 else 0
            else:
                disk = SpillDiskStore(
                    path,
                    namespace=self._apc_disk_namespace(),
                    num_workers=1,
                    max_bytes=int(max_gb * (1 << 30)) if max_gb > 0 else None,
                )
        except Exception:
            logger.warning("APC disk tier unavailable at %s", path, exc_info=True)
            return None
        try:
            from yunshu_kv import disk_budget
            from yunshu_kv.fingerprint import checkpoint_fingerprint

            # one budget for the whole directory (every namespace / model together)
            budget = disk_budget.budget_for(
                path,
                cap_bytes=int(hard_gb * (1 << 30)) if hard_gb > 0 else 0,
                label="APC disk",
            )
            disk.attach_budget(budget)
            disk_budget.write_marker(
                disk.dir,
                str(self._model_path),
                checkpoint_fingerprint(self._model_path, digest_size=8),
            )
            budget.enforce(keep={disk.dir.name})
        except Exception:
            logger.warning("APC disk budget unavailable", exc_info=True)
        if lower_specs:
            self._attach_lower_tiers(disk, path, lower_specs)
        logger.info("APC disk tier at %s (cap %.0f GiB, root-wide)", disk.dir, max_gb)
        try:
            from . import dflash_apc_window

            # beside the APC root, not inside it: the root-wide budget owns that tree
            root = disk.dir.parent
            dflash_apc_window.set_disk(
                root.with_name(root.name + ".dflash-window") / disk.dir.name,
                getattr(disk, "budget", None),
            )
        except Exception:
            logger.warning("drafter window disk store unavailable", exc_info=True)
        return disk

    def _attach_lower_tiers(self, disk, primary_path, specs) -> None:
        """Profile the primary SSD and every lower tier, build the lower tiers (own budget and
        reserve per volume) and start the background mover."""
        from yunshu_kv import disk_budget
        from yunshu_kv.fingerprint import checkpoint_fingerprint

        from . import paths
        from .apc_storage import FileTier, ProfileStore, TierSpec, profile_for

        store = ProfileStore(paths.home() / "cache" / "apc-device-profiles.json")
        disk.profile = profile_for(TierSpec(primary_path), store)
        encoding = str(settings.get("YUNSHU_VLM_APC_DISK_ENCODING"))
        ns = disk.dir.name
        for spec in specs:
            prof = profile_for(spec, store)
            cap_gb = disk_budget.resolve_cap_gb(spec.cap_gib, spec.path)
            budget = None
            try:
                budget = disk_budget.budget_for(
                    spec.path,
                    cap_bytes=int(cap_gb * (1 << 30)) if cap_gb > 0 else 0,
                    label=f"APC disk tier {spec.path}",
                )
            except Exception:
                logger.warning(
                    "APC storage tier %s: no budget", spec.path, exc_info=True
                )
            try:
                tier = FileTier(
                    spec,
                    ns,
                    prof,
                    cap_bytes=int(cap_gb * (1 << 30)),
                    budget=budget,
                    encoding=encoding,
                )
            except Exception:
                logger.warning(
                    "APC storage tier %s unavailable", spec.path, exc_info=True
                )
                continue
            tier.reprofile = lambda spec=spec: profile_for(spec, store, force=True)
            with contextlib.suppress(Exception):
                disk_budget.write_marker(
                    tier.dir,
                    str(self._model_path),
                    checkpoint_fingerprint(self._model_path, digest_size=8),
                )
            disk.add_lower(tier)
            if prof is None:
                logger.warning(
                    "APC storage tier %s: not profiled, not used until it is", spec.path
                )
            else:
                logger.info(
                    "APC storage tier %s (%s): read %.0f MB/s, write %.0f MB/s, latency %.1f ms, cap %.0f GiB",
                    spec.path,
                    tier.name,
                    prof.read_bps / 1e6,
                    prof.write_bps / 1e6,
                    prof.latency_s * 1e3,
                    cap_gb,
                )
        disk.start_mover()

    def _build_batch_runner(self, model_path: str):
        """Build the runner that serves every request of this model.

        APC (prefix cache, ``YUNSHU_VLM_APC_MEMORY_GB``; 0 disables it) for every
        family whose cache has no sliding window. Speculative decoding
        (checkpoint MTP head, or ``YUNSHU_VLM_DRAFT``) and its verify kernels are
        Qwen3.5-family only (``_SPEC_MODEL_TYPES``); ``YUNSHU_MTP=0`` disables the
        MTP draft.
        """
        from .kernels import nax_prefill
        from .utils.hardware import is_paravirtual_metal
        from .vlm_batch_runner import VLMBatchRunner

        nax_prefill.disable()
        from .kernels import gdn_prefill

        gdn_prefill.disable()
        from .kernels import attn_prefill

        attn_prefill.disable()
        from .kernels import lane_linear as lane_sum_policy

        lane_sum_policy.configure_sum_policy(False)

        if is_paravirtual_metal():
            # air64's stock BF16 prefix arithmetic differs by prefill span.
            # Keep cold upstream forwards, without APC or custom verify/cache kernels.
            logger.warning(
                "Apple Paravirtual Metal: using stock VLM runner without APC or speculation"
            )
            self._prefix_invariant_dispatch = False
            self._apc_prefill_stride = 0
            runner = VLMBatchRunner(
                self._model, self._processor, executor=self._executor
            )
            runner.clear_on_idle = bool(getattr(self, "_mx_large_model", False))
            runner.stop_tokens = set(self._get_eos_ids())
            runner.inflight = lambda: self._active_count
            return runner

        spec_family = self._config.get("model_type") in self._SPEC_MODEL_TYPES
        lm = self._model.language_model
        qwen4 = self._config.get("model_type") == "qwen4_exp"
        drafter = None
        draft_kind = "mtp"
        from . import spec_select
        from .mlxvlm_mtp import is_mtp_capable, unindexed_mtp_warning

        warning = unindexed_mtp_warning(model_path)
        if warning:
            logger.warning(warning)
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
                # Quantized drafter (YUNSHU_DRAFT_BITS, default 8): drafts are
                # verified, so this only trades acceptance for drafter bytes per
                # cycle (27B server, 8-bit: faster than the shipped weights at
                # 1K-32K; 4-bit: +3-8% at 1K-8K, -7% on 32K code).
                from .dflash_tree import quantize_drafter

                draft_bits = int(settings.get("YUNSHU_DRAFT_BITS"))
                if draft_bits:
                    quantize_drafter(drafter, draft_bits)
                from .dflash_tree import share_masks, stack_projections

                share_masks()

                # K|V and gate|up as one matmul each (TensorFold lane_fuse): fewer dispatches.
                stack_projections(drafter)
                # every layer's context K|V from one (lane) matmul, as the ring reads them
                from .dflash_ring import install_context_projection

                install_context_projection(drafter)
                # stored drafter windows are this drafter's captured target layers
                from . import dflash_apc_window

                dflash_apc_window.set_tag(
                    tuple(
                        getattr(
                            getattr(drafter, "config", None), "target_layer_ids", ()
                        )
                        or ()
                    )
                )
                # Project only the context window the drafter attends to.
                from .dflash_context import install as install_dflash_context

                install_dflash_context(self._model.language_model)
        if drafter is None and choice.kind in ("mtp", "dflash"):
            # "dflash" here means the automatic drafter failed to load.
            from mlx_vlm.speculative.drafters import validate_drafter_compatibility

            from .mlxvlm_mtp import _load_drafter_in_memory

            if is_mtp_capable(model_path):
                if qwen4:
                    from .qwen4_mtp import load_native_head

                    drafter = load_native_head(
                        self._config,
                        getattr(self, "_qwen4_mtp_weights", None),
                        direct_norms=getattr(self, "_qwen4_direct_norms", False),
                    )
                    self._qwen4_mtp_weights = None
                else:
                    drafter = _load_drafter_in_memory(model_path)
                validate_drafter_compatibility(self._model, drafter, "mtp")
        block = settings.get("YUNSHU_MTP_BLOCK_SIZE")
        kernels = None
        from .kernels import sdpa_chunk

        # M1-M4 only; bounds the unfused head-dim-256 prefill score matrix
        sdpa_chunk.install()
        # Spec off on a spec family still runs the lane's plain-decode arithmetic
        # (invariant kernels), so spec on == spec off token-for-token.
        if qwen4:
            from .qwen4_mtp import configure_lane

            kernels, block = configure_lane(lm, drafter, block)
            self._qwen4_mtp_weights = None
        if not qwen4 and (drafter is not None or spec_family):
            from .kernels.omlx import apply as apply_verify_kernels

            kernels = apply_verify_kernels()
            # Lossless MTP decode: decode and verify share row-invariant kernels,
            # so speculative output == this engine's plain decode. On Qwen3.8-27B
            # (M5 Max) invariant + NAX-packed at block 6 decodes code/prose/json
            # at 88.6/59.9/67.3 tok/s vs 57-67/50-53/58-62 for row-exact kernels
            # at block 3 (the removed row-exact alternative) and
            # 83.8/59.9/66.7 for the non-exact fast verify, with parity on every
            # task (measured on Qwen3.8-27B, 2026-09-28).
            # DFlash verifies through the same target forward, so it gets the
            # same guarantee.
            from .kernels.batch_invariant import install as install_invariant
            from .kernels.batch_invariant import set_active

            # Target projections run TensorFold's row-invariant lane matmul
            # (M5-class tensor ops; sg8 elsewhere). 27B, in-process vs the
            # NAX packed kernel: MTP / DFlash2 decode +0..+17% at 1K-131K
            # with identical tokens, and 0.6 GiB less memory (no repacked copy).
            from .kernels.tensorfold.lane_qmm import ready

            if ready():
                from .kernels import lane_linear

                kernels["lane_linear"] = lane_linear.convert(
                    self._model.language_model
                )["converted"]
                if settings.get("YUNSHU_PREFILL_MATMUL") == "stock":
                    lane_linear.set_stock_rows(lane_linear.STOCK_MIN_ROWS)
                    text_cfg = self._config.get("text_config", self._config)
                    if (
                        kernels["lane_linear"]
                        and text_cfg.get("hidden_size") == 5120
                        and text_cfg.get("num_hidden_layers") == 64
                        and nax_prefill.enable()
                    ):
                        nax_prefill.warmup(
                            {
                                module.bits
                                for _, module in lm.named_modules()
                                if isinstance(module, lane_linear.LaneLinear)
                            }
                        )
                        kernels["nax_prefill"] = nax_prefill.arithmetic_id()
                kernels["prefill_matmul"] = lane_linear.prefill_kernel_id()
            if settings.get("YUNSHU_PREFILL_GDN") == "chunked":
                from .kernels import gdn_prefill

                gdn_prefill.install()
            if settings.get("YUNSHU_PREFILL_ATTENTION") == "tile":
                # Bit-identical to stock MLX attention, so no APC key change.
                kernels["attn_prefill"] = (
                    attn_prefill.arithmetic_id() if attn_prefill.enable() else None
                )
            kernels["invariant"] = install_invariant(
                self._model.language_model, model=self._model, packed=False
            )
            # Every prompt piece uses the same arithmetic, while decode and
            # speculative verify retain their own invariant kernels. Only the
            # jointly qualified 27B device/ABI/weight geometry can enter it.
            split_ready = bool(
                kernels.get("nax_prefill")
                and kernels.get("attn_prefill")
                and settings.get("YUNSHU_PREFILL_GDN") == "chunked"
                and settings.get("YUNSHU_KV_PRECISION") == "bf16"
                and all(
                    module.output_dims < lane_linear.NARROW
                    or module.output_dims >= 100_000
                    or nax_prefill.split_eligible(
                        64,
                        module.input_dims,
                        module.output_dims,
                        module.bits,
                        module.group_size,
                        module.tiled,
                    )
                    for _, module in lm.named_modules()
                    if isinstance(module, lane_linear.LaneLinear)
                )
            )
            gdn_prefill.configure_split(split_ready)
            if split_ready:
                gdn_prefill.install_split_forward(lm)
                kernels["split_prefill"] = "split64-v1"
                logger.info(
                    "Split-invariant prefill engaged: NAX/BQ32/GDN, alignment=64"
                )
            if settings.get("YUNSHU_ANE_FFN_SPLIT") == "splash" and split_ready:
                from . import ane_ffn_runtime

                runtime = ane_ffn_runtime.install(
                    lm,
                    Path(
                        settings.get("YUNSHU_ANE_FFN_CACHE")
                        or Path.home() / ".yunshu" / "ane_cache" / "ffn"
                    ),
                    str(model_path),
                    share=settings.get("YUNSHU_ANE_FFN_SHARE"),
                    minimum_rows=settings.get("YUNSHU_ANE_FFN_MIN_ROWS"),
                )
                if runtime and settings.get("YUNSHU_ANE_FFN_TRACE"):
                    runtime.trace = True
                    runtime.split.profile(True)
                kernels["ane_ffn_split"] = (
                    ane_ffn_runtime.kernel_suffix() if runtime else None
                )
            # The runner turns them on for spec and lone AR target steps.
            set_active(False)
            if draft_kind == "dflash" and drafter is not None:
                # Reduced candidate vocabulary for the DFlash lattice (draft_vocab.logits_subset).
                from .draft_vocab import DRAFT_VOCAB as DFLASH_KEEP
                from .draft_vocab import install as _install_dflash_vocab

                assert self._model is not None
                found = _install_dflash_vocab(
                    drafter, self._model.language_model, DFLASH_KEEP
                )
                kernels["draft_vocab"] = DFLASH_KEEP if found else None
            if draft_kind == "mtp" and drafter is not None:
                # The lane's own MTP rounds (one host read per cycle, the head
                # run over every verify row inside the verify's graph), fused
                # residual+norm layers and a reduced draft vocabulary. Same
                # tokens as upstream's loop; Qwen3.8-27B, single requests, HTTP
                # decode tok/s 1K/8K/32K/131K: code_python 45.8/68.6/53.4/55.9
                # -> 55.2/69.5/62.5/59.7, novel_en 48.5/42.2/43.3/32.7 ->
                # 55.3/49.9/44.5/35.0 (measured 2026-09-29).
                from . import mtp_lane
                from .draft_vocab import DRAFT_VOCAB
                from .draft_vocab import install as install_draft_vocab
                from .kernels import lane_layers

                kernels["mtp_lane"] = mtp_lane.install()
                kernels["copy_rows"] = mtp_lane.set_copy_rows(
                    settings.get("YUNSHU_SPEC_COPY_ROWS"),
                    mtp_lane.verify_max_rows(bool(kernels.get("lane_linear")), lm),
                )
                kernels["lane_layers"] = lane_layers.install()
                found = install_draft_vocab(
                    drafter, self._model.language_model, DRAFT_VOCAB
                )
                kernels["draft_vocab"] = DRAFT_VOCAB if found else None
            # Invariant kernels cost little per extra verify row and peak at 6
            # for MTP. A DFlash drafter proposes a whole block in one forward
            # and upstream adapts the depth to acceptance under this ceiling,
            # so the ceiling is the block it was trained on.
            if block is None and drafter is not None:
                if draft_kind == "dflash":
                    block = int(getattr(drafter.config, "block_size", 0)) or None
                else:
                    block = 6
        self._prefix_invariant_dispatch = bool(kernels and kernels.get("invariant"))
        # After the kernels: the identity names the prefill matmul they installed.
        from .vlm_batch_runner import PREFILL_STEP

        self._apc_prefill_stride = (
            PREFILL_STEP if spec_family and (kernels or {}).get("invariant") else 0
        )
        from .kernels import buffer_cache

        if spec_family and not qwen4:
            from .kernels import cache_restore, singleton_cache

            singleton_cache.install()
            cache_restore.install()
            from .kernels import preempt

            preempt.install()
        cache_gib = settings.get("YUNSHU_PREFILL_BUFFER_CACHE_GB")
        if cache_gib is None:
            import mlx.core as _mx

            from .apc_manager import total_memory_bytes

            ceiling = settings.get("YUNSHU_MAX_MEMORY_GB")
            cache_gib = buffer_cache.auto_limit_gib_for_weights(
                total_memory_bytes(),
                _mx.get_active_memory(),
                int(float(ceiling) * (1 << 30)) if ceiling else None,
            )
        buffer_cache.install(float(cache_gib))
        budget = self._apc_memory_gb()
        if self._apc_backend is None and budget > 0:
            # A rotating cache cannot be assembled from dense KV blocks, but
            # its native snapshot includes the window, absolute offset and
            # ring index. Upstream's grouped plan checks that every component
            # has a restore contract before admitting exact checkpoints.
            from mlx_vlm.apc import model_apc_plan, semantic_extra_hash

            from .apc_manager import YunshuAPCManager

            if model_apc_plan(lm).restorable and (
                self._config.get("model_type") == "gemma4"
                or not self.backend_capabilities(lm).cache.has_sliding_window
            ):
                warm_mode = str(settings.get("YUNSHU_VLM_APC_WARM"))
                share = float(settings.get("YUNSHU_VLM_APC_WARM_SHARE"))
                warm_gb = budget * share if warm_mode != "off" else 0.0
                boundary_tokens: list[int] = []
                assistant_header = None
                message_end = None
                if self._apc_prefill_stride and self._tokenizer is not None:
                    # Generation guards end at these tags. A later cold chat
                    # must finish the same prefix forward before assistant text.
                    for marker in ("<think>", "</think>"):
                        ids = self._tokenizer.encode(marker, add_special_tokens=False)
                        if len(ids) == 1:
                            boundary_tokens.append(int(ids[0]))
                    start_ids = self._tokenizer.encode(
                        "<|im_start|>", add_special_tokens=False
                    )
                    role_ids = self._tokenizer.encode(
                        "assistant", add_special_tokens=False
                    )
                    end_ids = self._tokenizer.encode(
                        "<|im_end|>", add_special_tokens=False
                    )
                    if len(start_ids) == len(role_ids) == len(end_ids) == 1:
                        assistant_header = (int(start_ids[0]), int(role_ids[0]))
                        message_end = int(end_ids[0])
                self._apc_backend = YunshuAPCManager(
                    num_blocks=512,
                    block_size=16,
                    disk=self._apc_disk_tier(),
                    overrides={"memory_max_gb": budget - warm_gb},
                    head_marker=self._chatml_head_marker(),
                    warm_mode=warm_mode,
                    warm_bytes=int(warm_gb * (1 << 30)),
                    prefill_stride=self._apc_prefill_stride,
                    prefill_alignment=64 if gdn_prefill.split_enabled() else 0,
                    prefill_boundary_tokens=tuple(boundary_tokens),
                    prefill_assistant_header=assistant_header,
                    prefill_message_end=message_end,
                    prefill_turn_boundary=(
                        settings.get("YUNSHU_NEXT_TURN_PREFILL")
                        in ("boundary", "exact")
                    ),
                    seam_checkpoint=settings.get("YUNSHU_APC_SEAM_CHECKPOINT"),
                )
                self._install_apc_identity(lm)
                self._apc_semantic_hash = semantic_extra_hash(
                    image_hash=0,
                    media={"audio": None, "video": None},
                    model=lm,
                    processor=self._processor,
                )
        if drafter is not None and draft_kind == "dflash":
            # Keep adaptive depth for the fallback. Qualified greedy requests
            # use the trained DFlash2 block and verified prompt-copy islands.
            from .spec_schedule import install_chain_budget

            install_chain_budget()
            if kernels is not None:
                from .dflash_copy import configure as configure_dflash_copy

                kernels["copy_rows"], kernels["dflash_copy"] = configure_dflash_copy(
                    lm,
                    settings.get("YUNSHU_SPEC_COPY_ROWS"),
                    invariant=bool(kernels.get("invariant")),
                    lane_projections=bool(kernels.get("lane_linear")),
                )
                if kernels["dflash_copy"] and kernels["copy_rows"] in (16, 32):
                    from . import dflash_fast

                    if dflash_fast.supported(lm, drafter):
                        lane_sum_policy.configure_sum_policy(True)
                        kernels["lane_sum_identity"] = "long-context"
                if (
                    settings.get("YUNSHU_SPEC_TREE") == "auto"
                    and kernels["dflash_copy"]
                ):
                    from . import dflash_fast

                    if dflash_fast.supported(lm, drafter):
                        dflash_fast.prepare(drafter)
                        kernels["dflash_fast"] = "short-context"
        if drafter is not None and not qwen4:
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
            prefix_invariant=self._prefix_invariant_dispatch,
        )
        runner.clear_on_idle = bool(getattr(self, "_mx_large_model", False))
        runner.stop_tokens = set(self._get_eos_ids())
        runner.inflight = lambda: self._active_count
        runner.memory_gate = self._build_memory_gate()
        # Per-row-length (ragged) KV + ragged decode attention for the shared
        # batch and the speculative lane: the layout for every model with
        # qwen3_5 attention (lossless; measured 2026-09-29).
        # KV precision is the user's memory/quality choice.
        precision = settings.get("YUNSHU_KV_PRECISION")
        from .kernels import ragged_kv

        if not qwen4 and ragged_kv.supports(self._model.language_model):
            ragged_kv.install()
            ragged_kv.enable(None)  # the runner sets the format while it steps
            runner.ragged_kv = precision
        elif precision != "bf16":
            logger.warning(
                "YUNSHU_KV_PRECISION=%s applies to Qwen3.5-family attention "
                "only; this model's KV stays bf16",
                precision,
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
            from .grammar_constraint import build_json_constraint

            if isinstance(json_schema, str) and json_schema == "json_object":
                return build_json_constraint(None, self._tokenizer)
            return build_json_constraint(json_schema, self._tokenizer)
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

    def _tool_call_marker_id(self) -> int | None:
        """Token id of ``<tool_call>`` when the tokenizer has it as one token, else None."""
        cached = self.__dict__.get("_tool_call_marker_cache", False)
        if cached is False:
            try:
                ids = self._tokenizer.encode("<tool_call>", add_special_tokens=False)
            except Exception:
                ids = []
            cached = int(ids[0]) if len(ids) == 1 else None
            self._tool_call_marker_cache = cached
        return None if cached is None else int(cached)

    def _runner_events(self, input_ids, **kw):
        """``_runner_events_impl`` plus the optional generated-vs-delivered capture
        (``YUNSHU_DEBUG_STREAM_CAPTURE``)."""
        tools = kw.pop("tool_recovery_tools", None)
        events = self._runner_events_impl(input_ids, tools_declared=bool(tools), **kw)
        if tools:
            from .tool_format import formats_for_tokenizer
            from .tool_thinking import recover_tool_events

            formats = formats_for_tokenizer(self._tokenizer)
            if any(fmt.name == "glm47" for fmt in formats):
                ids = self._tokenizer.encode(
                    "<|observation|>", add_special_tokens=False
                )
                if len(ids) == 1 and ids[0] not in (kw.get("stop_token_ids") or []):
                    events = recover_tool_events(
                        events, observation_id=ids[0], formats=formats, tools=tools
                    )
        path = settings.get_str("YUNSHU_DEBUG_STREAM_CAPTURE")
        if not path:
            yield from events
            return
        ids: list[int] = []
        pieces: list[str] = []
        finish = None
        prompt_list: list[int] = []
        round_trip = None
        with contextlib.suppress(Exception):
            from .roundtrip_capture import RoundTripTracker, flatten_ids

            prompt_list = flatten_ids(input_ids)
            tracker = getattr(self, "_roundtrip_tracker", None)
            if tracker is None:
                tracker = self._roundtrip_tracker = RoundTripTracker()
            round_trip = tracker.observe_prompt(prompt_list)
        try:
            for event in events:
                text, token, _state, finish_reason = event[:4]
                if token is not None:
                    ids.append(int(token))
                pieces.append(text)
                finish = finish_reason or finish
                yield event
        finally:
            with contextlib.suppress(Exception):
                row = {
                    "prompt_tokens": len(input_ids),
                    "token_ids": ids,
                    "pieces": pieces,
                    "text": "".join(pieces),
                    "finish": finish,
                    "round_trip": round_trip,
                }
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, "a") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                self._roundtrip_tracker.add(prompt_list, ids)

    def _runner_events_impl(
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
        prompt_cache_plan: dict | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        min_tokens: int = 0,
        ignore_eos: bool = False,
        suppress_tokens: list[int] | None = None,
        top_n_sigma: float = 0.0,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        tool_spec: dict | None = None,
        tools_declared: bool = False,
        first_delivery: threading.Event | None = None,
    ):
        """Yield ``(text, token_id, state, finish_reason, thinking_tokens, logprob)``.

        ``logprob`` is the token's {"token_id", "logprob", "top_logprobs"} entry
        when ``logprobs`` is requested, else ``None``.

        ``finish_reason`` is set only on the last event: "stop" (EOS or stop
        string), "length", "budget" (thinking budget reached) or "cancel".
        """
        from .structural_tag import constrains_initial_output
        from .text_utils import StopHoldbackBuffer
        from .vlm_batch_runner import (
            ConstraintProcessor,
            TokenMaskProcessor,
            build_penalty_processors,
        )

        processors = build_penalty_processors(
            repetition_penalty, frequency_penalty, presence_penalty, logit_bias
        )
        if not suppress_tokens:  # the request's own list wins over the model's default
            suppress_tokens = self._generation_suppress_ids() or None
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
        from .structured_report import backend_name, report

        stats.structured_output = report(
            "mlx-vlm",
            json_schema,
            backend_name(constraint),
            enforced=constraint is not None,
        )
        constraint_guide = None
        if constraint is not None:
            from .constrained_spec import ConstraintGuide

            assert self._tokenizer is not None
            constraint_guide = ConstraintGuide(
                ConstraintProcessor(constraint, self._tokenizer),
                int(
                    (self._config.get("text_config") or {}).get("vocab_size")
                    or self._config.get("vocab_size")
                    or len(getattr(self._tokenizer, "_tokenizer", self._tokenizer))
                ),
            )

        stop_ids = set() if ignore_eos else set(self._get_eos_ids())
        stop_ids.update(stop_token_ids or [])
        stop_strings = [s for s in (stop or []) if s]
        holdback = StopHoldbackBuffer(stop_strings)
        detok = self._tokenizer.detokenizer
        detok.reset()

        def drain_tail(reasoning: bool) -> tuple[str, bool]:
            """The text still held back at the end, and whether a stop string completed in
            it (only in the final detokenizer flush; stop strings never apply to reasoning)."""
            detok.finalize()
            tail = holdback.feed(detok.last_segment) + holdback.flush()
            hit = holdback.stop_hit and not reasoning
            if hit:
                stats.stop_string_hit = True
            return tail, hit

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
        tool_marker = (
            self._tool_call_marker_id() if (tool_spec or tools_declared) else None
        )
        guide = self._tool_guide(tool_spec, in_think)
        if constraint_guide is not None and constrains_initial_output(json_schema):
            # Generation starts inside an open <think>: the grammar binds after the close.
            from .reasoning_gate import ThinkGatedGuide, make_gate

            gate = make_gate(
                prompt_opens_thinking=in_think,
                close_id=think_end,
                eos_ids=self._get_eos_ids(),
                implicit_end_ids=[
                    t for t in (self._tool_call_marker_id(),) if t is not None
                ],
                report=stats.structured_output,
            )
            if gate is not None:
                constraint_guide = ThinkGatedGuide(constraint_guide, gate)
        if constraint_guide is not None:
            from .constrained_spec import CombinedGuide

            guide = (
                CombinedGuide(constraint_guide, guide) if guide else constraint_guide
            )
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
            prompt_cache_plan=prompt_cache_plan,
            cancel_event=cancel_event,
            stats=stats,
            logprobs=bool(logprobs),
            top_logprobs=int(top_logprobs or 0),
            thinking_budget=(
                thinking_budget
                if enable_thinking is not False
                and think_start is not None
                and not constrains_initial_output(json_schema)
                else None
            ),
            prompt_preopens_thinking=in_think,
            thinking_start_token=open_text,
            thinking_end_token=close_text,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            xtc_special_tokens=list(self._get_eos_ids()),
            guide=guide,
            first_delivery=first_delivery,
        ):
            count += 1
            stats.consumed = count
            lp = stats.last_logprob if logprobs else None
            if token in stop_ids:
                tail, _ = drain_tail(in_think)
                # EOS inside an open <think> ends the reasoning: its tail is reasoning
                yield (
                    tail,
                    token,
                    "reasoning" if in_think else "normal",
                    "stop",
                    thinking_tokens,
                    lp,
                )
                return
            if think_start is not None and token in (think_start, think_end):
                # The tags only switch state; they are never shown as text.
                in_think = token == think_start
                label_buf = "" if (channel and in_think) else None
                state = "reasoning" if in_think else "normal"
                if count >= max_tokens:
                    tail, hit = drain_tail(in_think)
                    yield (
                        tail,
                        token,
                        state,
                        "stop" if hit else "length",
                        thinking_tokens,
                        lp,
                    )
                    return
                continue
            if in_think and tool_marker is not None and token == tool_marker:
                # Qwen3.5-family models sometimes open a tool call inside the reasoning
                # block without closing it. The call marker ends reasoning implicitly
                # (vLLM Qwen3ReasoningParser, SGLang Qwen3 detector), else the call is
                # delivered as reasoning text and the client never sees it.
                in_think = False
            if in_think:
                thinking_tokens += 1
            state = "reasoning" if in_think else "normal"
            detok.add_token(token)
            segment = detok.last_segment
            if stop_strings and not in_think:
                text = holdback.feed(segment)
                if holdback.contains_stop():
                    stats.stop_string_hit = True
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
                tail, hit = drain_tail(in_think)
                yield (
                    text + tail,
                    token,
                    state,
                    "stop" if hit else "length",
                    thinking_tokens,
                    lp,
                )
                return
            yield text, token, state, None, thinking_tokens, lp
        if stats.finish_reason == "cancel":
            yield "", None, "normal", "cancel", thinking_tokens, None
            return
        tail, hit = drain_tail(in_think)
        yield (
            tail,
            None,
            "reasoning" if in_think else "normal",
            "stop" if hit or count < max_tokens else "length",
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
            return self._non_empty_ids(cast(list[int], ids.tolist())), pkw, salt
        ids = self._tokenize_with_cache(
            messages, enable_thinking=enable_thinking, template_extra=template_extra
        )
        return self._non_empty_ids(cast(list[int], ids.tolist())), None, None

    @staticmethod
    def _non_empty_ids(ids: list[int]) -> list[int]:
        from .context_window import require_non_empty_prompt

        require_non_empty_prompt(
            ids
        )  # EmptyPromptError, the gateway's 400 on every path
        return ids

    def _resolve_prompt_cache_plan(
        self,
        plan,
        messages,
        ids,
        prompt_kwargs,
        enable_thinking,
        template_extra,
        cache_options=None,
    ):
        if plan is None:
            from .prompt_caching import openai_plan

            plan = openai_plan(messages, cache_options)
        if plan is None:
            return None
        if not (plan.get("markers") or plan.get("tools")):
            if (cache_options or {}).get("mode") == "explicit":
                plan["resolved"] = True
                return plan
            return None
        from .prompt_caching import (
            expanded_boundaries,
            rendered_boundaries,
            trim_safe_shadow,
        )

        if prompt_kwargs is None:
            prompt = self._format_prompt(messages, enable_thinking, template_extra)
            shadow = self._format_prompt(
                trim_safe_shadow(plan["messages"], plan["markers"]),
                enable_thinking,
                template_extra,
            )
            rendered_ids = ids
        else:
            # The same pure template builder used for the prepared media request;
            # no second image/audio encoder invocation.
            count = sum(
                p.get("type") == "image_url"
                for m in messages
                for p in (
                    m.get("content") if isinstance(m.get("content"), list) else []
                )
                if isinstance(p, dict)
            )
            prompt = self._apply_vlm_template_with_cache(
                messages,
                enable_thinking=enable_thinking,
                max_images=count or None,
                template_extra=template_extra,
            )
            shadow = self._apply_vlm_template_with_cache(
                trim_safe_shadow(plan["messages"], plan["markers"]),
                enable_thinking=enable_thinking,
                max_images=count or None,
                template_extra=template_extra,
            )
            rendered_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        try:
            points = rendered_boundaries(
                prompt,
                shadow,
                plan["markers"],
                self._tokenizer,
                rendered_ids,
                tools=(template_extra or {}).get("tools"),
                tool_controls=plan["tools"],
                marker_ends=plan.get("marker_ends"),
                selection=plan,
            )
        except ValueError as exc:
            # cache_control / prompt_cache_breakpoint is a hint (Anthropic never fails a request over it): a chat
            # template that drops or repeats a marker (Qwen removes the reasoning of earlier turns, so a marker on a
            # thinking block vanishes) means no explicit checkpoint, not a failed turn.
            logger.warning(
                "explicit cache breakpoints ignored, serving without them: %s", exc
            )
            return None
        minimum = 0
        if prompt_kwargs is not None:
            from .prompt_caching import image_token_wrappers
            from .vlm_batch_runner import media_token_ids

            try:
                media = media_token_ids(self._model)
                wrappers = image_token_wrappers(getattr(self, "_processor", None))
                lookup = expanded_boundaries(
                    rendered_ids,
                    ids,
                    [(n, 0) for n in points["lookup_points"]],
                    media,
                    wrappers,
                )
                points = {
                    k: expanded_boundaries(rendered_ids, ids, value, media, wrappers)
                    for k, value in points.items()
                    if k != "lookup_points"
                }
                points["lookup_points"] = [n for n, _ in lookup]
                from mlx_vlm.apc import media_safe_prefix_min

                minimum = media_safe_prefix_min(ids, media)
                # Earlier system/text endpoints may precede a later image.
                # The upstream restore contract requires a text-only suffix;
                # keep safe endpoints instead of failing an otherwise valid turn.
                for key in ("points", "writes"):
                    points[key] = [
                        (n, ttl) for n, ttl in points[key] if minimum <= n < len(ids)
                    ]
                points["lookup_points"] = [
                    n for n in points["lookup_points"] if minimum <= n < len(ids)
                ]
            except ValueError as exc:
                logger.warning("explicit media cache breakpoints ignored: %s", exc)
                return None
        manager = getattr(self, "_apc_backend", None)
        alignment = getattr(manager, "prefill_alignment", 0)
        if prompt_kwargs is None and plan.get("protocol") != "openai":
            # Anthropic breakpoints sit on content blocks, so the last one can end well before the prompt
            # (the first turn's task text follows Claude Code's last marker). The prompt's own tail is what
            # the next turn of an append-only conversation shares, so it is always a checkpoint too.
            guard = int(getattr(manager, "exact_cache_guard_tokens", 1) or 1)
            tail = len(ids) - guard
            plan["tail"] = tail
            if tail > minimum and tail not in {n for n, _ in points["writes"]}:
                points["points"] = sorted({*points["points"], (tail, 300)})
                points["writes"] = sorted({*points["writes"], (tail, 300)})
                points["lookup_points"] = sorted({*points["lookup_points"], tail})
        if alignment:
            from .prompt_caching import align_prefill_plan

            points = align_prefill_plan(
                points,
                alignment,
                maximum=len(ids) - manager.exact_cache_guard_tokens,
                minimum=minimum,
            )
        plan.update(points)
        if plan.get("tail"):
            plan["tail"] = (
                plan["tail"] // alignment * alignment if alignment else plan["tail"]
            )
        plan["resolved"] = True
        logger.info("Prompt cache rendered token boundaries: %s", plan["points"])
        return plan

    def _runner_kwargs(self, **params) -> dict:
        kwargs = params.pop("kwargs")
        js = kwargs.get("json_schema")
        if js is None:
            js = kwargs.get("grammar")
        tb = params.pop("thinking_budget", None)
        if tb is None and kwargs.get("reasoning_effort") is not None:
            from .output_limits import effort_budget

            tb = effort_budget(kwargs["reasoning_effort"])
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
            tool_spec=kwargs.get("_tool_spec"),
            tool_recovery_tools=kwargs.get("_tool_recovery_tools"),
        )

    def validate_forced_tools(self, tools, choice, parallel=True):
        from .tool_call_grammar import validate_forced_tools

        validate_forced_tools(self, tools, choice, parallel)

    def _tool_guide(self, spec: dict | None, thinking_open: bool):
        """A ``ToolCallGuide`` for this request's tools (structural-tag constrained
        decoding), or None: no tools, ``YUNSHU_TOOL_GRAMMAR`` off, or a model / tool
        set the grammar cannot cover (the reply is then decoded unconstrained)."""
        if not spec:
            return None
        from . import tool_call_grammar as tcg

        cache = self.__dict__.setdefault("_tool_grammars", {})
        key = tcg.grammar_key(spec["tools"], spec["tool_choice"], spec["parallel"])
        if key not in cache:
            cfg = self._config or {}
            vocab = (cfg.get("text_config") or {}).get("vocab_size") or cfg.get(
                "vocab_size"
            )
            tok = self._tokenizer
            vocab = int(vocab or len(getattr(tok, "_tokenizer", tok)))
            if len(cache) >= 8:
                cache.pop(next(iter(cache)))
            cache[key] = tcg.compile_tool_grammar(
                spec["tools"],
                tok,
                vocab,
                tool_choice=spec["tool_choice"],
                parallel=spec["parallel"],
            )
        grammar = tcg.require_tool_grammar(cache[key], spec["tool_choice"])
        return grammar.guide(thinking_open=thinking_open) if grammar else None

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
            extras["stopped_by_stop_sequence"] = stats.stop_string_hit
            if params.get("logprobs"):
                extras["logprobs"] = lps
        if finish == "cancel":
            raise asyncio.CancelledError()
        return (
            "".join(parts),
            thinking,
            _completion_count(stats),
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
        first_sent = False
        try:
            for text, token, state, finish, thinking, lp in self._runner_events(
                input_ids, stats=stats, **params
            ):
                reason = {"budget": "stop"}.get(finish, finish)
                if not (text or reason or lp is not None):
                    continue
                # the first token's own event can carry no text (a think tag, a partial
                # character) and the runner can already be past token 1: flag the first
                # event that has a token, once
                first = token is not None and not first_sent
                first_sent = first_sent or token is not None
                ok = deliver(
                    RequestOutput(
                        request_id=req_id,
                        new_text=text,
                        new_token_ids=[token] if token is not None else [],
                        finish_reason=reason,
                        finished=reason is not None,
                        completion_tokens=_completion_count(stats),
                        prompt_tokens=prompt_tokens,
                        cached_tokens=stats.cached_tokens,
                        current_state=state,
                        reasoning_tokens=thinking,
                        ttft_ms=round(stats.first_token_s * 1000, 1) if first else 0.0,
                        logprobs=[lp] if lp is not None else None,
                        stopped_by_stop_sequence=bool(
                            reason == "stop" and stats.stop_string_hit
                        ),
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
                    error=(
                        MEMORY_LIMIT_MARK + str(exc)
                        if isinstance(exc, MemoryAdmissionError)
                        else str(exc)
                    ),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=_completion_count(stats),
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
                        if part.get("_yunshu_cache_marker"):
                            parts.append(
                                {"type": "text", "text": part["_yunshu_cache_marker"]}
                            )
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
            if isinstance(func, dict) and (
                func.get("arguments") is None or isinstance(func.get("arguments"), str)
            ):
                try:
                    parsed = _json.loads((func.get("arguments") or "").strip() or "{}")
                except Exception:
                    parsed = {"value": func["arguments"]}
                tc = dict(tc)
                tc["function"] = dict(func)
                tc["function"]["arguments"] = (
                    parsed
                    if isinstance(parsed, dict)
                    else {}
                    if parsed is None
                    else {"value": parsed}
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
        choice = kwargs.pop("tool_choice", None)
        parallel = kwargs.pop("parallel_tool_calls", True)
        if tools:
            extra["tools"] = tools
            kwargs["_tool_recovery_tools"] = tools
            from .structural_tag import is_lazy_constraint
            from .tool_call_grammar import is_forced, normalize_tool_choice

            lazy = is_lazy_constraint(kwargs.get("json_schema"))
            if lazy and is_forced(choice):
                raise ValueError(
                    "structural_tag cannot be combined with forced tool_choice"
                )
            if is_forced(choice) or (
                settings.get_bool("YUNSHU_TOOL_GRAMMAR") and not lazy
            ):
                choice = normalize_tool_choice(choice)
                if choice != "none":
                    kwargs["_tool_spec"] = {
                        "tools": tools,
                        "tool_choice": choice,
                        "parallel": parallel is not False,
                    }
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
        from .output_limits import THINKING_OFF_EFFORTS

        if str(effort).strip().lower() in THINKING_OFF_EFFORTS:
            kwargs.pop("reasoning_effort", None)  # thinking off, not a template rung
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

    def _prepare_template_messages(self, messages: list[dict]) -> list[dict]:
        """Role remap, family adapter and special-token defusing: every text render of a
        conversation (generation, ``/tokenize``, ``/apply-template``) starts here."""
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

            if self._config.get("model_type") != "deepseek_v4":
                messages = adapt_messages(messages, self.model_name)
        except Exception:
            logger.debug("VLM message adapter failed", exc_info=True)
        messages = neutralize_special_text(messages, getattr(self, "_tokenizer", None))
        return messages

    def _template_messages_clean(self, messages: list[dict]) -> list[dict]:
        """The ``{role, content, tool fields}`` list handed to ``apply_chat_template``."""
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
        return clean

    def _template_messages(self, messages: list[dict]) -> list[dict]:
        """Exactly the messages generation passes to ``apply_chat_template``."""
        return self._template_messages_clean(self._prepare_template_messages(messages))

    def _format_prompt(
        self,
        messages: list[dict],
        enable_thinking: bool | None = None,
        template_extra: dict | None = None,
    ) -> str:
        messages = self._prepare_template_messages(messages)

        if self._tokenizer is not None and hasattr(
            self._tokenizer, "apply_chat_template"
        ):
            try:
                clean = self._template_messages_clean(messages)
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
        # Exact reuse of the tokens of an earlier prompt's prefix ending at a
        # non-normalized <|im_end|> (multi-turn follow-ups); full encode otherwise.
        token_ids = self._text_prompt_cache.prefix_tokens.encode(
            self._tokenizer, prompt_text, add_special_tokens=_add_special
        )

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

        # The text path runs the family adapter (Qwen's template rejects a system message
        # after the first: coding agents send per-turn notes mid-conversation); the vision
        # path used to skip it, so any agent turn carrying an image failed in the template.
        try:
            from yunshu_engine.message_adapter import adapt_messages

            messages = adapt_messages(messages, self.model_name)
        except Exception:
            logger.debug("VLM message adapter failed", exc_info=True)
        messages = neutralize_special_text(messages, getattr(self, "_tokenizer", None))
        vlm_messages = self._build_vlm_messages(messages, max_images=max_images)
        tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            tpl_kwargs["enable_thinking"] = enable_thinking
        if num_audios > 0:
            tpl_kwargs["num_audios"] = num_audios
        tpl_kwargs.update(extra)

        try:
            try:
                template_text = self._processor.apply_chat_template(
                    vlm_messages,
                    **tpl_kwargs,
                )
            except TypeError as e:
                # Gemma4Processor counts the audio parts and passes num_audios itself: a second
                # num_audios is a TypeError (every audio request was a 500). Retry without ours.
                if "num_audios" not in str(e) or "num_audios" not in tpl_kwargs:
                    raise
                tpl_kwargs.pop("num_audios")
                processor: Any = self._processor
                template_text = processor.apply_chat_template(
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
            # Chat/Responses normally flatten text-only blocks with newlines.
            # Cache markers retain the parts; preserve that exact input text.
            separator = (
                "\n"
                if any(
                    isinstance(p, dict) and p.get("prompt_cache_breakpoint")
                    for p in content
                )
                else " "
            )
            return separator.join(texts)
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
        tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 — file must outlive function; cleanup via _temp_files registry
            suffix=f".{ext}", dir=private_media_dir(), delete=False
        )
        self._register_temp_file(tmp.name)  # a cancelled await still cleans up

        def _sync_decode_write() -> None:
            # MB-size base64 decode + write blocks event loop;
            # move to executor.
            try:
                tmp.write(base64.b64decode(data))
                tmp.close()
            except Exception:
                tmp.close()
                with contextlib.suppress(OSError):
                    os.unlink(tmp.name)
                raise

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_decode_write)
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
        tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 — file must outlive function; cleanup via _temp_files registry
            suffix=f".{ext}", dir=private_media_dir(), delete=False
        )
        self._register_temp_file(tmp.name)  # a cancelled await still cleans up

        def _sync_decode_write() -> None:
            # offload sync I/O off the event loop.
            try:
                tmp.write(base64.b64decode(data))
                tmp.close()
            except Exception:
                tmp.close()
                with contextlib.suppress(OSError):
                    os.unlink(tmp.name)
                raise

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_decode_write)
        return tmp.name

    async def _download_video(self, url: str) -> str:
        """Bounded, DNS-pinned HTTP video fetch; redirects pass the same SSRF guard."""
        await _resolve_media_target(url)
        from urllib.parse import urlsplit

        ext = Path(urlsplit(url).path).suffix.lower()
        if ext not in (".mp4", ".webm", ".mov", ".mkv", ".avi"):
            ext = ".mp4"
        tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115
            suffix=ext, dir=private_media_dir(), delete=False
        )
        tmp.close()
        self._register_temp_file(tmp.name)
        try:
            await netguard.download_to_file(
                url,
                tmp.name,
                max_bytes=settings.get("YUNSHU_VLM_MAX_VIDEO_BYTES"),
                timeout=60,
                allow_private=False,
                verify=True,
                headers={"User-Agent": "Yunshu/1.0"},
            )
        except Exception as exc:
            raise ValueError(f"Cannot download video: {exc}") from exc
        return tmp.name

    async def _download_image(self, url: str) -> str:
        """Download an image from HTTP/HTTPS URL to a temp file."""

        # SSRF validation up front (the download re-resolves, checks and pins every hop)
        await _resolve_media_target(url)

        ext = url.rsplit(".", 1)[-1].lower() if "." in url.split("?")[0] else "png"
        ext = ext if ext in ("png", "jpg", "jpeg", "webp", "gif") else "png"

        tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 — file must outlive function; cleanup via _temp_files registry
            suffix=f".{ext}", dir=private_media_dir(), delete=False
        )
        tmp.close()  # Close FD immediately — urlretrieve writes by path, not handle
        # Register temp file eagerly so cleanup happens even if download fails
        self._register_temp_file(tmp.name)

        # httpx through netguard.download_to_file: DNS resolved once per hop, checked and
        # pinned (Host/SNI keep the name), redirects re-checked, one total deadline, and a byte
        # cap enforced up front (Content-Length) and while streaming. "size limit" in the message
        # lets the outer handler skip the insecure-SSL retry (no re-download).
        async def _download(verify: bool) -> None:
            await netguard.download_to_file(
                url,
                tmp.name,
                max_bytes=settings.get("YUNSHU_VLM_MAX_IMAGE_BYTES"),
                timeout=30,
                allow_private=False,
                verify=verify,
                headers={"User-Agent": "Yunshu/1.0"},
            )

        try:
            await _download(True)
        except Exception as e:
            # a size-limit violation is final — don't fall through to the insecure-
            # SSL retry (which would re-download the oversized body). Map straight to a clean
            # error (→ 400/413 at the gateway).
            if "size limit" in str(e):
                logger.warning(f"Rejected oversized remote image from {url}: {e}")
                raise ValueError(str(e)) from e
            if "SSRF blocked" in str(e):
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
                await _download(False)
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
                            path = await self._download_video(url)
                            video_paths.append(path)
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

        tmpdir = tempfile.mkdtemp(prefix="yunshu_video_", dir=private_media_dir())
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
        why = ""
        try:
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd, capture_output=True, timeout=60, check=True
                ),
            )
        except FileNotFoundError:
            why = "ffmpeg is not installed"
        except subprocess.TimeoutExpired:
            why = "ffmpeg timed out"
        except subprocess.CalledProcessError as e:
            why = f"ffmpeg failed: {e.stderr.decode()[:200] if e.stderr else 'unknown'}"
        except Exception as e:
            why = f"ffmpeg failed: {e}"

        def _frames() -> list[str]:
            return sorted(
                os.path.join(tmpdir, f)
                for f in os.listdir(tmpdir)
                if f.startswith("frame_") and f.endswith(".jpg")
            )

        frames = _frames()
        if not frames:
            # OpenCV is a locked dependency: decode without ffmpeg (neither dev machine has it)
            logger.warning("video frames: %s; trying OpenCV", why or "no frames")
            await loop.run_in_executor(
                None, self._opencv_frames, video_path, tmpdir, fps, max_frames
            )
            frames = _frames()
        if not frames:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)
            # fail loud like every media input: a video the model never saw must not be answered
            raise ValueError(
                f"could not read any frames from the video ({why or 'empty output'}; "
                "install ffmpeg or check the file)"
            )
        # tmpdir was already eagerly registered above; don't append it a
        # second time (was a double entry in _temp_files).
        return frames

    @staticmethod
    def _opencv_frames(
        video_path: str, tmpdir: str, fps: float, max_frames: int
    ) -> None:
        """Write up to `max_frames` JPEG frames, about `fps` per second, with OpenCV."""
        try:
            import cv2
        except ImportError:
            return
        cap = cv2.VideoCapture(video_path)
        try:
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            step = max(1, round(src_fps / fps)) if src_fps > 0 and fps > 0 else 1
            i = written = 0
            while written < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % step == 0:
                    written += 1
                    cv2.imwrite(os.path.join(tmpdir, f"frame_{written:04d}.jpg"), frame)
                i += 1
        finally:
            cap.release()

    async def _save_base64_file(self, data: str, ext: str = "mp4") -> str:
        """Save base64-encoded data to a temp file."""
        tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 — file must outlive function; cleanup via _temp_files registry
            suffix=f".{ext}", dir=private_media_dir(), delete=False
        )
        self._register_temp_file(tmp.name)  # a cancelled await still cleans up

        def _sync_decode_write() -> None:
            # offload sync decode+write off the event loop.
            try:
                tmp.write(base64.b64decode(data))
                tmp.close()
            except Exception:
                tmp.close()
                with contextlib.suppress(OSError):
                    os.unlink(tmp.name)
                raise

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_decode_write)
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
            return cast(list[int], cached)
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

    def _generation_suppress_ids(self) -> list[int]:
        """``suppress_tokens`` from the checkpoint's generation_config.json (token ids the
        model's author bans from the output, e.g. audio/image placeholder tokens): the default
        when a request does not set its own."""
        cached = getattr(self, "_suppress_ids_cache", None)
        if cached is not None:
            return cast(list[int], cached)
        # Shared with the text path: a hub id resolves to its cached snapshot directory.
        from .generation_suppress import default_suppress_ids

        ids: list[int] = []
        with contextlib.suppress(Exception):
            ids = default_suppress_ids(
                getattr(self, "_model_path", None), getattr(self, "_tokenizer", None)
            )
        self._suppress_ids_cache = ids
        return self._suppress_ids_cache

    # ── Stats ──

    def apc_snapshot(self) -> dict | None:
        """Prefix-cache (APC) occupancy and counters; None when the cache is off."""
        apc = self._apc_backend
        if apc is None:
            return None
        try:
            return apc.snapshot() if hasattr(apc, "snapshot") else apc.stats_snapshot()
        except Exception:
            logger.debug("APC snapshot unavailable", exc_info=True)
            return None

    def apc_overview(self, max_entries: int = 200) -> dict | None:
        """Tiers, lookup counters and capped entry metadata (cache browser); None if APC is off."""
        apc = self._apc_backend
        fn = getattr(apc, "cache_overview", None)
        return fn(max_entries) if callable(fn) else None

    def apc_clear(self, tier: str) -> dict | None:
        """Drop one APC tier (``ram`` / ``warm`` / ``ssd``); None when the cache is off."""
        apc = self._apc_backend
        fn = getattr(apc, "clear_tier", None)
        return fn(tier) if callable(fn) else None

    def memory_admission_snapshot(self) -> dict | None:
        """Memory-admission counters and live headroom (None when the gate is off)."""
        runner = self._batch_runner
        gate = getattr(runner, "memory_gate", None)
        if runner is None or gate is None:
            return None
        snap: dict = gate.snapshot()
        snap["held_now"] = runner.memory_held()
        return snap

    def busy_snapshot(self) -> dict | None:
        """GPU-busy accounting of the batch runner (None before a model is loaded)."""
        runner = self._batch_runner
        return runner.busy_snapshot() if runner is not None else None

    def get_kv_cache_stats(self) -> dict:
        """Same shape as the text engine's, so /debug/kv-cache and /metrics cover both."""
        snap = self.apc_snapshot()
        if snap is None:
            return {"prefix_cache": {"enabled": False}}
        return {
            "apc": snap,
            "prefix_cache": {
                "enabled": True,
                "entries": snap.get("entries", 0),
                "hits": snap.get("lookups_hit", 0),
                "misses": snap.get("lookups_miss", 0),
            },
        }

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
        runner = self._batch_runner
        counts = getattr(runner, "phase_counts", None)
        if callable(counts):
            try:
                c = counts()
                # /debug/requests and the console's debug view read these two names.
                stats["active"] = c["prefill"] + c["decode"]
                stats["waiting"] = c["queued"]
                stats["phases"] = c
            except Exception:
                logger.debug("phase counts unavailable", exc_info=True)
        snap = self.apc_snapshot()
        if snap is not None:
            stats["apc"] = snap
        try:
            stats["pipeline"] = self._pipeline.get_stats()
        except Exception:
            logger.debug("operation failed", exc_info=True)
        return stats
