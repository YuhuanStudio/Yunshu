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
import contextlib
import contextvars
import gc
import hashlib
import importlib
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


def _build_noncached_sampler(
    temperature: float, top_p: float, top_k: int, min_p: float, seed: int | None
):
    """Build a numpy-backed sampler that bypasses mlx-lm's @mx.compile cache.

    The mlx-lm `categorical_sampling` is decorated with
    `@mx.compile(inputs=mx.random.state, outputs=mx.random.state)`. The
    compile cache traps the first call's PRNG state, so subsequent calls
    produce identical token streams even after `mx.random.seed()` between
    requests. This is the ⚠️ A "VLM determinism" bug — calling
    the same temp=2.0 prompt 3 times returns byte-identical output.

    For temperature==0 (greedy) we keep mlx-lm's compiled argmax path
    (deterministic anyway, faster). For temperature>0 we sample via
    `numpy.random.Generator` so each request has independent randomness.

    Caller responsibilities:
    - `seed=None` → request gets a time-based fresh seed
    - `seed=<int>` → identical seed produces identical output across calls
    """
    if temperature is None or temperature < 1e-6:
        from mlx_lm.sample_utils import make_sampler

        return make_sampler(
            temp=0.0, top_p=top_p, top_k=top_k if top_k > 0 else 0, min_p=min_p
        )

    import time as _t

    import numpy as _np

    base = (
        int(seed) & ((1 << 63) - 1)
        if seed is not None
        else _t.time_ns() & ((1 << 63) - 1)
    )
    rng = _np.random.default_rng(base)
    _t_ = float(temperature)
    _tp = float(top_p)
    _tk = int(top_k) if top_k and top_k > 0 else 0
    _mp = float(min_p) if min_p else 0.0

    def _sampler(logits):
        arr = _np.asarray(logits.astype(mx.float32))
        flat = arr.reshape(-1, arr.shape[-1])
        out = _np.empty(flat.shape[0], dtype=_np.int64)
        for i in range(flat.shape[0]):
            l = flat[i].astype(_np.float64) / _t_
            l = l - _np.max(l)
            p = _np.exp(l)
            p = p / p.sum()
            if _tk and _tk < len(p):
                idx = _np.argpartition(p, -_tk)[-_tk:]
                m = _np.zeros_like(p)
                m[idx] = 1.0
                p = p * m
                p = p / p.sum()
            if 0 < _tp < 1:
                order = _np.argsort(-p)
                cum = _np.cumsum(p[order])
                # Include the threshold-crossing token (matches mlx-lm apply_top_p);
                # `order[cum <= _tp]` dropped it → nucleus too narrow. See batched_engine.
                k = int(_np.searchsorted(cum, _tp, side="left")) + 1
                keep = order[: max(1, min(k, len(order)))]
                m = _np.zeros_like(p)
                m[keep] = 1.0
                p = p * m
                p = p / p.sum()
            if _mp > 0:
                pmax = p.max()
                m = (p >= _mp * pmax).astype(_np.float64)
                p = p * m
                p = p / p.sum()
            out[i] = int(rng.choice(len(p), p=p))
        return mx.array(out.reshape(arr.shape[:-1]).astype(_np.int64))

    return _sampler


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
    import os as _os
    from pathlib import Path as _Path

    if _os.environ.get("YUNSHU_ALLOW_LOCAL_FILES", "").lower() in ("true", "1", "yes"):
        return path

    media_dir = _os.environ.get("YUNSHU_MEDIA_DIR") or _os.path.join(
        _os.environ.get("TMPDIR", "/tmp"), "yunshu_media"
    )
    media_root = _Path(media_dir).resolve()
    resolved = _Path(path).resolve()
    try:
        resolved.relative_to(media_root)
    except ValueError as e:
        raise ValueError(
            f"Local file access blocked: {path} is not under YUNSHU_MEDIA_DIR={media_root}. "
            f"Set YUNSHU_ALLOW_LOCAL_FILES=1 to bypass (NOT recommended in multi-tenant)."
        ) from e
    return str(resolved)


def _get_model_classes_with_vlm_fallback(config: dict):
    """Resolve model classes.

    This is invoked by `mlx_lm.utils.load_model` (which passes a flat-dict
    config). mlx_lm.models.* `ModelArgs` is a dataclass that accepts a flat
    dict; mlx_vlm.models.* `ModelConfig` typically expects nested sub-configs
    (e.g., TextConfig) and chokes on flat dicts with
    `AttributeError: 'dict' object has no attribute 'model_type'`.

    Therefore: try mlx_lm first (it handles flat dicts natively). Only fall
    back to mlx_vlm when mlx_lm has no module for this model_type. The
    vision-encoder path is handled separately by `_load_vision_model`, which
    invokes mlx_vlm.utils.load directly with the proper nested config.
    """
    from mlx_lm.utils import MODEL_REMAPPING

    model_type = config.get("model_type", "")
    remapped = MODEL_REMAPPING.get(model_type, model_type)

    # Try mlx-lm first — it handles flat dict configs via dataclass ModelArgs
    try:
        arch = importlib.import_module(f"mlx_lm.models.{remapped}")
        return arch.Model, arch.ModelArgs
    except ImportError:
        pass

    # Fall back to mlx-vlm (covers VLM-only model types)
    try:
        arch = importlib.import_module(f"mlx_vlm.models.{remapped}")
        return arch.Model, arch.ModelConfig
    except ImportError:
        pass

    raise ValueError(f"Model type {model_type} not supported by mlx-lm or mlx-vlm.")


def _convert_nested_config(model_args_class, config_dict):
    """Convert nested dicts in config to proper config objects for mlx-vlm models.

    BaseModelConfig.from_dict() doesn't recurse into nested dicts, so
    vision_config/text_config stay as raw dicts. This function manually
    converts them to the correct dataclass types.
    """
    import dataclasses

    if not dataclasses.is_dataclass(model_args_class):
        return model_args_class.from_dict(config_dict)

    fields = dataclasses.fields(model_args_class)
    kwargs = {}
    for f in fields:
        if f.name not in config_dict:
            continue
        val = config_dict[f.name]
        if isinstance(val, dict) and dataclasses.is_dataclass(f.type):
            try:
                kwargs[f.name] = f.type(**val)
            except TypeError:
                kwargs[f.name] = val
        elif isinstance(f.type, str):
            kwargs[f.name] = val
        else:
            kwargs[f.name] = val

    try:
        return model_args_class(**kwargs)
    except TypeError:
        return model_args_class.from_dict(config_dict)


def _is_mlx_vlm_model(model) -> bool:
    """Check if a model was loaded from mlx-vlm's model classes (vs mlx-lm)."""
    return type(model).__module__.startswith("mlx_vlm.models.")


def _wrap_mlx_vlm_for_mlx_lm(model):
    """Adapt an mlx_vlm model so mlx_lm's generate_step can drive it text-only.

    Two upstream mlx_vlm quirks break the mlx_lm fast path:

    1. The top-level Model.__call__ returns a `LanguageModelOutput` dataclass
       (or sometimes another wrapper) instead of raw `mx.array` logits.
       mlx_lm.generate_step slices the return with ``[:, -1, :]`` and crashes
       on a dataclass.

    2. Some mlx_vlm thinkers (e.g. qwen3_omni_moe.thinker.Thinker.__call__)
       still try to unpack ``self.get_input_embeddings(...)`` into a 3-tuple,
       but the helper now returns an ``InputEmbeddingsFeatures`` dataclass.
       This raises ``InputEmbeddingsFeatures cannot be unpacked``.

    We fix (1) by replacing ``__call__`` with a thin shim that pulls ``.logits``
    out of any non-array return.  We fix (2) by replacing the offending
    ``get_input_embeddings`` with a wrapper that returns ``(inputs_embeds,
    visual_pos_masks, deepstack_visual_embeds)`` — the tuple shape its caller
    expects.  Both patches are idempotent and limited to the wrapped instance.
    """
    import mlx.core as mx

    # (2) Patch get_input_embeddings on the inner thinker if present.
    thinker = getattr(model, "thinker", None)
    if thinker is not None and hasattr(thinker, "get_input_embeddings"):
        original = thinker.get_input_embeddings

        def _tuple_input_embeddings(*args, **kwargs):
            out = original(*args, **kwargs)
            # Already a tuple/list — pass through.
            if isinstance(out, tuple):
                return out
            # Dataclass with the expected fields — shape into a 3-tuple.
            inputs_embeds = getattr(out, "inputs_embeds", None)
            if inputs_embeds is None:
                return out
            return (
                inputs_embeds,
                getattr(out, "visual_pos_masks", None),
                getattr(out, "deepstack_visual_embeds", None),
            )

        try:
            thinker.get_input_embeddings = _tuple_input_embeddings
        except Exception:
            # nn.Module may forbid attribute assignment; bind via __dict__.
            object.__setattr__(thinker, "get_input_embeddings", _tuple_input_embeddings)

    # (1) Wrap __call__ to coerce the return to raw logits.  Python looks up
    # ``__call__`` on the *type*, not the instance, so per-instance attribute
    # assignment is ignored by ``model(...)``.  Build a one-off subclass of the
    # model's type that overrides __call__ and rebind the instance to it.
    original_cls = type(model)
    if not getattr(original_cls, "_yunshu_mlx_lm_adapted", False):

        def _logits_only_call(self, *args, **kwargs):
            out = original_cls.__call__(self, *args, **kwargs)
            if isinstance(out, mx.array):
                return out
            logits = getattr(out, "logits", None)
            if logits is not None:
                return logits
            if isinstance(out, (tuple, list)) and out and isinstance(out[0], mx.array):
                return out[0]
            raise RuntimeError(
                f"mlx_vlm fallback for {original_cls.__module__}.{original_cls.__name__}: "
                f"model __call__ returned {type(out).__name__} with no .logits; "
                "cannot adapt to mlx_lm generate_step.  "
                "This model is not supported via the text-only mlx_lm fallback path."
            )

        adapted_cls = type(
            f"{original_cls.__name__}_YunshuMlxLmAdapter",
            (original_cls,),
            {
                "__call__": _logits_only_call,
                "_yunshu_mlx_lm_adapted": True,
            },
        )
        # In-place rebind: preserves all module weights and submodule registry.
        model.__class__ = adapted_cls
    return model


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


class _CachingVisionTower:
    """Cross-request vision-feature cache by wrapping a VLM's vision
    tower.

    The VLM vision-feature cache used to be wired as a `vision_cache` kwarg to
    mlx_vlm.generate/stream_generate — but the installed mlx_vlm has NO such
    parameter, so the adapter's get/put were NEVER called and the vision tower
    (ViT) re-encoded the image on EVERY request, even across a multi-turn
    conversation about the same image. This wrapper actually implements the
    cache: it intercepts the tower call, keys on the pixel-values hash (+ any
    grid/extra args), and returns the previously-computed image features on a
    repeat — skipping the expensive ViT forward. Bounded LRU to cap memory.

    It transparently proxies attribute access to the wrapped tower (some models
    read e.g. `vision_tower.patch_embed.proj.weight.dtype`), so it is a drop-in
    replacement for the module on its parent.
    """

    def __init__(self, tower, max_entries: int = 4):
        object.__setattr__(self, "_tower", tower)
        object.__setattr__(self, "_lru", __import__("collections").OrderedDict())
        object.__setattr__(self, "_max", max(1, int(max_entries)))
        object.__setattr__(self, "_hits", 0)
        object.__setattr__(self, "_misses", 0)

    @staticmethod
    def _key(pixel_values, args) -> str:
        import hashlib

        import numpy as _np

        h = hashlib.blake2b(digest_size=16)

        def _feed(x) -> None:
            # CORRECTNESS FIX: hash the REAL pixel bytes. The old code did
            # np.asarray(x, dtype=float32) then, on ANY failure (e.g. an mlx.array
            # whose dtype/format that call rejects), fell back to repr(x) — and a
            # large array's repr is TRUNCATED ("array([[[0.1, ...]]] )"), so two
            # DIFFERENT images produced the SAME key → the cache returned the wrong
            # image's features (read QUASAR as BANANA). Force a full materialization
            # via np.array (mlx arrays implement __array__); include shape+dtype; and
            # NEVER fall back to a colliding repr — use an id()-salted miss instead.
            try:
                arr = _np.array(x, copy=True)
                h.update(str(arr.shape).encode())
                h.update(str(arr.dtype).encode())
                h.update(arr.tobytes())
                return
            except Exception:
                pass
            try:
                import mlx.core as _mx

                if isinstance(x, _mx.array):
                    h.update(str(x.shape).encode())
                    h.update(_np.array(_mx.stop_gradient(x), copy=True).tobytes())
                    return
            except Exception:
                pass
            # Last resort: a UNIQUE (non-colliding) token so distinct objects never
            # share a key. This degrades to "always miss" (safe) rather than the
            # silent wrong-image collision the truncated repr caused.
            h.update(f"{type(x).__name__}:{getattr(x, 'shape', None)}:{id(x)}".encode())

        _feed(pixel_values)
        for a in args:
            _feed(a)
        return h.hexdigest()

    def __call__(self, pixel_values, *args, **kwargs):
        lru = object.__getattribute__(self, "_lru")
        key = self._key(pixel_values, args)
        if key in lru:
            lru.move_to_end(key)
            object.__setattr__(
                self, "_hits", object.__getattribute__(self, "_hits") + 1
            )
            return lru[key]
        out = object.__getattribute__(self, "_tower")(pixel_values, *args, **kwargs)
        try:
            import mlx.core as _mx

            _mx.eval(out)  # materialize before caching so the hit path is free
        except Exception:
            pass
        lru[key] = out
        object.__setattr__(
            self, "_misses", object.__getattribute__(self, "_misses") + 1
        )
        while len(lru) > object.__getattribute__(self, "_max"):
            lru.popitem(last=False)
        return out

    def cache_stats(self) -> dict:
        return {
            "vision_tower_cache_hits": object.__getattribute__(self, "_hits"),
            "vision_tower_cache_misses": object.__getattribute__(self, "_misses"),
            "vision_tower_cache_entries": len(object.__getattribute__(self, "_lru")),
        }

    def __getattr__(self, name):
        # Proxy everything else (parameters, sub-modules, dtype reads) to the tower.
        return getattr(object.__getattribute__(self, "_tower"), name)


def _wrap_vision_towers(model, max_entries: int = 4) -> list:
    """Find a VLM model's vision tower(s) and replace them with caching wrappers.

    Walks the model + its common container children (thinker/model/language_model)
    for attributes named vision_tower/vision_model/visual/image_encoder and swaps
    in a _CachingVisionTower. Returns the list of installed wrappers (for stats).
    Idempotent — skips already-wrapped towers.
    """
    _ATTRS = ("vision_tower", "vision_model", "visual", "image_encoder")
    wrappers = []
    seen = set()
    # Candidate parents: the model itself + one level of common containers.
    parents = [model]
    for cname in ("thinker", "model", "language_model", "vlm", "multi_modal"):
        child = getattr(model, cname, None)
        if child is not None:
            parents.append(child)
    for parent in parents:
        if id(parent) in seen:
            continue
        seen.add(id(parent))
        for attr in _ATTRS:
            tower = getattr(parent, attr, None)
            if tower is None or isinstance(tower, _CachingVisionTower):
                continue
            if not callable(tower):
                continue
            try:
                wrapper = _CachingVisionTower(tower, max_entries=max_entries)
                setattr(parent, attr, wrapper)
                wrappers.append(wrapper)
            except Exception:
                logger.debug(
                    "vision tower wrap failed for %s.%s",
                    type(parent).__name__,
                    attr,
                    exc_info=True,
                )
    return wrappers


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
                # Multi-image: hash actual file contents, not paths
                parts = []
                for p in image:
                    if os.path.exists(p):
                        with open(p, "rb") as f:
                            parts.append(f.read())
                    else:
                        parts.append(p.encode())
                img_hash = compute_image_hash(b"".join(parts))
            else:
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
                parts = []
                for p in image:
                    if os.path.exists(p):
                        with open(p, "rb") as f:
                            parts.append(f.read())
                    else:
                        parts.append(p.encode())
                img_hash = compute_image_hash(b"".join(parts))
            else:
                if os.path.exists(image):
                    with open(image, "rb") as f:
                        img_hash = compute_image_hash(f.read())
                else:
                    img_hash = compute_image_hash(image.encode())
            self._cache.put(img_hash, self._model_name, features)
        except Exception:
            logger.debug("vision cache adapter put failed", exc_info=True)


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
        self._active_count_lock = threading.Lock()
        self._num_requests_processed = 0
        self._total_reasoning_tokens = 0
        self._start_time = 0.0
        self._has_vision = False
        self._is_vlm = False
        self._temp_files: list[str] | None = None
        self._temp_files_lock = threading.Lock()

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
                "YUNSHU_VISION_CACHE_DIR",
                "~/.cache/yunshu/vision",
            )
            self._vision_cache = VisionFeatureCache(cache_dir=cache_dir)
            logger.info("Vision feature cache enabled (dir=%s)", cache_dir)

        # Adapter that wraps VisionFeatureCache for mlx_vlm's interface.
        # Created lazily after model load when model_name is known.
        self._vlm_vision_cache_adapter = None
        # installed _CachingVisionTower wrappers (cross-request vision
        # feature cache). Populated at load(); empty for text-only models.
        self._vision_tower_wrappers: list = []

        # Encoder cache — caches encoder hidden states (vision/audio/text encoder)
        # keyed by request_id. When the same image/audio is seen again in the
        # batch path, reuse cached encoder output instead of re-encoding.
        # Complements VisionFeatureCache (which caches at the mlx_vlm layer).
        from .encoder_cache import EncoderCacheManager

        self._encoder_cache = EncoderCacheManager(
            max_entries=int(os.environ.get("YUNSHU_ENCODER_CACHE_MAX", "64")),
            ttl_seconds=float(os.environ.get("YUNSHU_ENCODER_CACHE_TTL", "300")),
        )

        # Text prompt tokenization cache — caches _format_prompt() output and
        # tokenizer.encode() results keyed by message content hash.  Avoids
        # re-tokenizing identical prompts across requests.  Also caches
        # _processor.apply_chat_template() output for the VLM vision path.
        self._text_prompt_cache = _VLMTextPromptCache(
            max_entries=int(os.environ.get("YUNSHU_VLM_TEXT_CACHE_MAX", "256")),
        )

        # Per-image KV prefix cache state — maps image_hash to PromptCacheState.
        # When the same image appears with different text contexts, the KV cache
        # from the previous conversation is reused for the common image prefix.
        # Thread-safe: accessed from MLX executor thread and async stop() path.
        self._kv_prefix_states: dict[str, Any] = {}
        self._kv_prefix_lock = threading.Lock()
        self._kv_prefix_max_entries = 32

        # 4-tier KV prefix cache (HOT full-precision / WARM 4-bit-in-RAM
        # / SSD int8-on-disk) for the VLM *text* path. Previously VLM models had
        # NO cross-request KV prefix reuse on the text path — only the per-image
        # KV-state reuse above and the token-id/template _text_prompt_cache.
        #
        # DEFAULT ON (YUNSHU_VLM_KV_PREFIX=0 to disable). Verified BYTE-LOSSLESS on
        # mRoPE full-attention VLMs (GLM-OCR; the Qwen-VL family uses the same
        # mechanism): the text path supplies explicit sequential position_ids so a
        # reused prefix's suffix resumes at `matched`. Backbones that
        # CAN'T reuse losslessly are auto-bypassed by _text_prefix_reuse_safe
        # (shared model_backend layer): sliding-window (gemma RotatingKVCache) and
        # hybrid recurrent (Qwen3.5 ArraysCache). Stores the snapshot at the prompt
        # boundary (before decode pollutes it). See docs/VLM_TEXT_KV_PREFIX.md.
        self._text_kv_prefix_enabled = os.environ.get(
            "YUNSHU_VLM_KV_PREFIX", "1"
        ).strip() in ("1", "true", "yes")
        self._text_kv_prefix_cache = None
        if self._text_kv_prefix_enabled:
            from .kv_prefix_cache import KVPrefixCache

            _pc_max = int(os.environ.get("YUNSHU_PREFIX_MAX_ENTRIES", "128"))
            _pc_hot = int(os.environ.get("YUNSHU_PREFIX_HOT_LIMIT", "32"))
            self._text_kv_prefix_cache = KVPrefixCache(
                max_entries=_pc_max, hot_limit=_pc_hot, min_prefix_length=32
            )
            if os.environ.get("YUNSHU_SSD_CACHE", "").strip() in ("1", "true", "yes"):
                try:
                    ssd_dir = os.environ.get(
                        "YUNSHU_SSD_CACHE_DIR", "~/.cache/yunshu/kv-ssd-vlm"
                    )
                    _ssd_gb = int(
                        float(os.environ.get("YUNSHU_SSD_CACHE_MAX_GB", "10"))
                    )
                    self._text_kv_prefix_cache.enable_ssd_cache(
                        cache_dir=ssd_dir,
                        max_size_bytes=_ssd_gb * 1024**3,
                        model_name=os.path.basename(model_path.rstrip("/")),
                    )
                except Exception:
                    logger.debug("VLM text KV SSD tier init failed", exc_info=True)
        # Memoized backbone serving capabilities (see model_backend.py); decides
        # whether cross-request KV prefix reuse is lossless for this model.
        self._backend_caps: Any = None
        # Memoized empirical reuse-losslessness probe verdict (None = not run).
        self._reuse_probe_ok: bool | None = None
        # HYBRID VLM backbones (Qwen3.5/3.6-VL: KVCache +
        # GatedDeltaNet ArraysCache) reuse text prefixes via boundary snapshots
        # (no_trim) — the recurrent state can't be sliced, but a trim=0 snapshot
        # at a block boundary IS losslessly resumable (verified). Mirrors the LLM
        # fast path's YUNSHU_HYBRID_PREFIX. Opt-out via YUNSHU_VLM_HYBRID_PREFIX=0.
        self._text_hybrid_prefix_enabled = os.environ.get(
            "YUNSHU_VLM_HYBRID_PREFIX", "1"
        ).strip() in ("1", "true", "yes")
        self._text_hybrid_block = int(
            os.environ.get("YUNSHU_VLM_HYBRID_PREFIX_BLOCK", "128")
        )
        self._hybrid_reuse_probe_ok: bool | None = None

        # VLM cache stats (vision feature cache + KV prefix reuse)
        self._vlm_vision_hits = 0
        self._vlm_vision_misses = 0
        self._vlm_kv_prefix_hits = 0
        self._vlm_kv_prefix_misses = 0

        # (dead-code removal): the VisionEncoderFactory was constructed
        # here but NEVER used — real vision encoding goes through
        # mlx_vlm.stream_generate/generate. The dead instantiation AND the whole
        # vision_encoding.py module (790 lines, no other importer) were removed.

        # SpecPrefill for VLM text portion (opt-in via YUNSHU_VLM_SPEC_PREFILL)
        self._spec_prefill_enabled = False
        if os.environ.get("YUNSHU_VLM_SPEC_PREFILL", "").strip() in (
            "1",
            "true",
            "yes",
        ):
            self._spec_prefill_enabled = True
            logger.info("VLM SpecPrefill enabled")

        from .mlx_executor import get_mlx_executor

        self._executor = get_mlx_executor()

        # MultimodalPipelineCoordinator — unified 7-stage pipeline
        from .staged_pipeline import (
            MultimodalPipelineCoordinator,
        )

        self._pipeline = MultimodalPipelineCoordinator()
        self._register_pipeline_processors()

        # Async concurrent VLM engine (opt-in via YUNSHU_VLM_ASYNC=1)
        self._async_core = None

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
        """Load model and tokenizer with mlx-lm/mlx-vlm fallback.

        For mlx-vlm models, MUST run on the MLX executor thread so weights
        and compute share the same GPU stream.
        """
        from mlx_lm.utils import load_config, load_tokenizer

        model_path = Path(self._model_path)

        # Download if HF repo ID (not local path)
        if not model_path.exists():
            from mlx_lm.utils import _download

            model_path = Path(_download(self._model_path))

        self._config = load_config(model_path)

        # Check vision support
        thinker_cfg = self._config.get("thinker_config", {})
        self._has_vision = bool(
            self._config.get("vision_config") or thinker_cfg.get("vision_config")
        )

        # For vision models, use mlx_vlm's loader (handles nested config properly)
        if self._has_vision:
            try:
                with _quantize_shape_safety_patch():
                    self._model = self._load_vision_model(model_path)
                self._tokenizer = load_tokenizer(model_path)
                from .text_utils import cache_tokenizer_vocab

                cache_tokenizer_vocab(
                    self._tokenizer
                )  # avoid ~98ms/req get_vocab rebuild
                self._is_vlm = _is_mlx_vlm_model(self._model)
                logger.info(f"Loaded vision model via mlx_vlm: _is_vlm={self._is_vlm}")
                self._finish_vlm_load(model_path)
                return
            except Exception as e:
                err_str = str(e)
                # Quantization-shape failures (e.g. Qwen3-Omni 4-bit, where a
                # weight has last-dim 4304 which is not divisible by group 64)
                # indicate the published quant artefact is incompatible with
                # mlx_vlm's quantize layout. The mlx_lm text-only fallback
                # cannot rescue this — the same weights will misbehave during
                # forward and (on macOS) tend to crash the entire process with
                # no Python traceback. Surface a precise error instead.
                if "needs to be divisible by the quantization group size" in err_str:
                    raise RuntimeError(
                        f"VLM/Omni model {model_path.name} cannot be loaded: "
                        f"mlx_vlm quantization rejected weight shape "
                        f"({err_str.split('shape')[-1].strip().rstrip(').')} - "
                        f"last dim not divisible by group size 64). "
                        "The mlx_lm text-only fallback is unsafe for this "
                        "model because forward-pass quantize math will hit "
                        "the same shape constraint and crash the engine. "
                        "Use a bf16 build of this model, or a 4-bit quant "
                        "produced with group_size matching the weight shapes."
                    ) from e
                logger.warning(
                    f"mlx_vlm load failed ({e}), falling back to mlx_lm (text-only)"
                )
                # mlx_vlm failed → no processor will be set up. Mark as
                # text-only so image/audio requests fail loudly rather than
                # crashing later in apply_chat_template (processor=None).
                self._has_vision = False

        # Load model with mlx-lm fallback (text-only or failed mlx-vlm)
        from mlx_lm.utils import load_model

        model, config = load_model(
            model_path,
            get_model_classes=_get_model_classes_with_vlm_fallback,
        )
        self._tokenizer = load_tokenizer(model_path)
        from .text_utils import cache_tokenizer_vocab

        cache_tokenizer_vocab(self._tokenizer)  # avoid ~98ms/req get_vocab rebuild

        # Adapter: when an mlx_vlm model class was loaded via mlx_lm's loader,
        # mlx_vlm's __call__ returns a LanguageModelOutput dataclass (not raw
        # mx.array logits) and may also try to unpack an InputEmbeddingsFeatures
        # dataclass into a 3-tuple (broken in upstream mlx_vlm). Both crash
        # mlx_lm's generate_step. Wrap the model to (1) coerce the return value
        # to raw logits, and (2) monkey-patch get_input_embeddings on the
        # thinker (if present) so the tuple-unpack on the model's own __call__
        # does not raise. Falls through cleanly for mlx_lm-native models.
        if _is_mlx_vlm_model(model):
            model = _wrap_mlx_vlm_for_mlx_lm(model)

        self._model = model
        # Even though the model class may originate from mlx_vlm.models.*,
        # without mlx_vlm's processor the VLM code paths cannot work.
        # Treat as non-VLM regardless of model-class module origin.
        self._is_vlm = False
        self._finish_vlm_load(model_path)

    def _load_vision_model(self, model_path):
        """Load a vision model using mlx_vlm with proper nested config handling."""
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

        # Load processor for VLM vision input
        if self._has_vision and self._is_vlm:
            try:
                from pathlib import Path

                from mlx_vlm.utils import load_processor

                self._processor = load_processor(Path(model_path))
            except Exception as e:
                logger.warning(f"Could not load VLM processor: {e}")

        # Detect mRoPE support
        from .mrope import BatchRopeDeltaManager, detect_mrope

        self._mrope_info = detect_mrope(self._config)
        if self._mrope_info.enabled:
            self._rope_delta_manager = BatchRopeDeltaManager()
            logger.info(
                f"mRoPE detected: sections={self._mrope_info.sections}, "
                f"source={self._mrope_info.source_key}"
            )

        # Perf/stability: large VLMs (e.g. 30B-MoE) GPU-hang under
        # sustained load because — unlike BatchedEngine — the VLM paths never
        # released the MLX buffer pool between requests, so it grew until OOM/hang.
        # Flag large models so generation clears the cache + raises the wired
        # limit (prevents weight swap), mirroring BatchedEngine's _wired_limit_ctx.
        # Use the on-disk weight size (robust — an MoE's .parameters() can
        # undercount lazily-structured experts; recommended_max varies). >10GB →
        # large enough that the buffer pool must be released per request.
        self._mx_large_model = False
        try:
            import glob as _glob

            _bytes = 0
            for _f in _glob.glob(os.path.join(self._model_path, "*.safetensors")):
                with contextlib.suppress(OSError):
                    _bytes += os.path.getsize(_f)
            _thresh = int(os.environ.get("YUNSHU_VLM_LARGE_MODEL_GB", "10")) * 1024**3
            self._mx_large_model = _bytes > _thresh
            logger.info(
                "VLM model ~%.1fGB on disk, large=%s (per-request mem hygiene %s)",
                _bytes / 1e9,
                self._mx_large_model,
                "ON" if self._mx_large_model else "off",
            )
        except Exception:
            logger.debug("VLM model-size probe failed", exc_info=True)

        # shrink the text KV-prefix cache for large models. Each entry
        # holds the full prompt KV (~0.1GB/1.2k-tok on a 30B), so the default
        # 128-entry/32-hot cache could add ~6GB on top of a ~22GB-resident model
        # and push a 36GB Mac toward OOM/GPU-hang. Cap it tighter for big models.
        if self._mx_large_model and self._text_kv_prefix_cache is not None:
            try:
                self._text_kv_prefix_cache._max_entries = int(
                    os.environ.get("YUNSHU_VLM_LARGE_PREFIX_MAX", "24")
                )
                self._text_kv_prefix_cache._hot_limit = int(
                    os.environ.get("YUNSHU_VLM_LARGE_PREFIX_HOT", "8")
                )
                logger.info(
                    "VLM large-model KV prefix cache capped: hot=%d max=%d",
                    self._text_kv_prefix_cache._hot_limit,
                    self._text_kv_prefix_cache._max_entries,
                )
            except Exception:
                logger.debug("VLM large-model cache cap failed", exc_info=True)

        logger.info(
            f"VLM engine loaded: {self._model_path} "
            f"(vision={self._has_vision}, vlm_model={self._is_vlm}, "
            f"mrope={self._mrope_info.enabled})"
        )

        # run the KV-reuse losslessness probe NOW (load runs on the
        # MLX executor thread, so the probe's forwards land on the right GPU
        # stream). Memoizes self._reuse_probe_ok so _text_prefix_reuse_safe is a
        # pure memoized read on the hot path. Only probe reuse-capable caches.
        if self._is_vlm and self._text_kv_prefix_cache is not None:
            try:
                lm = getattr(self._model, "language_model", None)
                if lm is not None:
                    caps = self.backend_capabilities(lm)
                    if caps.supports_kv_prefix_reuse:
                        self._probe_text_reuse_lossless(lm)
                    elif (
                        caps.cache.is_hybrid
                        and not caps.cache.has_sliding_window
                        and self._text_hybrid_prefix_enabled
                    ):
                        # hybrid backbones reuse via no_trim
                        # boundary snapshots — probe that path instead.
                        self._probe_hybrid_reuse(lm)
            except Exception:
                logger.debug("VLM reuse probe at load failed", exc_info=True)

        # Create vision cache adapter now that model_name is known
        if self._vision_cache is not None:
            self._vlm_vision_cache_adapter = _MlxVlmVisionCacheAdapter(
                self._vision_cache,
                self.model_name,
            )

        # optional cross-request vision-feature cache (wraps the vision
        # tower to skip re-encoding a repeated image).
        # DEFAULT OFF. It proved to be a redundant LANDMINE — (1) its key
        # collided for mlx-array pixel_values (returned the WRONG image; fixed), and
        # (2) wrapping the Qwen3-Omni-30B vision tower HANGS its vision forward
        # (Metal GPU Hang) while the model runs fine without it. And it's redundant:
        # cross-request image reuse already works ~6x via the KV-prefix path
        # (verified — the tower cache was shadowed, hits=0). So it stays OPT-IN
        # (YUNSHU_VLM_VISION_CACHE=1) for experimentation only; production VLM image
        # reuse is the KV path.
        self._vision_tower_wrappers = []
        if (
            self._has_vision
            and self._model is not None
            and os.environ.get("YUNSHU_VLM_VISION_CACHE", "0")
            not in ("0", "false", "no")
        ):
            try:
                _max = int(os.environ.get("YUNSHU_VLM_VISION_CACHE_ENTRIES", "4"))
                self._vision_tower_wrappers = _wrap_vision_towers(
                    self._model, max_entries=_max
                )
                if self._vision_tower_wrappers:
                    logger.info(
                        "VLM vision-feature cache installed on %d tower(s)",
                        len(self._vision_tower_wrappers),
                    )
                else:
                    logger.debug(
                        "VLM vision-feature cache: no vision tower found to wrap"
                    )
            except Exception:
                logger.debug("vision tower cache install failed", exc_info=True)

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
        self._processor = None
        self._running = False

        # Clear per-model caches — stale entries from the old model would waste
        # memory and could return incorrect features if model_name happened to
        # collide.  The VisionFeatureCache itself is kept alive (its background
        # writer thread is daemon and shared), but in-memory entries are evicted.
        if self._vision_cache is not None:
            try:
                lock = getattr(self._vision_cache, "_memory_lock", None)
                cache = getattr(self._vision_cache, "_memory_cache", None)
                if lock is not None and cache is not None:
                    with lock:
                        cache.clear()
            except Exception:
                logger.debug("vision cache cleanup during stop failed", exc_info=True)
        self._vlm_vision_cache_adapter = None
        with self._kv_prefix_lock:
            self._kv_prefix_states.clear()
        self._encoder_cache.clear()
        self._text_prompt_cache.clear()
        if self._text_kv_prefix_cache is not None:
            try:
                self._text_kv_prefix_cache.close()
                self._text_kv_prefix_cache.clear()
            except Exception:
                logger.debug("VLM text KV prefix cache cleanup failed", exc_info=True)
        self._backend_caps = None
        self._reuse_probe_ok = None
        self._hybrid_reuse_probe_ok = None

        # Reset stats counters
        self._vlm_vision_hits = 0
        self._vlm_vision_misses = 0
        self._vlm_kv_prefix_hits = 0
        self._vlm_kv_prefix_misses = 0

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

        t0 = time.monotonic()

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
            _enable_thinking = enable_thinking
            # Gemma-4 default: its chat template enables thinking by default but
            # emits inline `thought` tokens that don't auto-stop, producing
            # output like "4thought\nThinking Process: …" for simple prompts.
            # Default to False for gemma-4 unless caller passed something explicit.
            if (
                _enable_thinking is None
                and isinstance(self.model_name, str)
                and "gemma-4" in self.model_name.lower()
            ):
                _enable_thinking = False
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
            stop_token_ids = kwargs.get("stop_token_ids") or []
            thinking_budget = kwargs.get("thinking_budget")
            reasoning_effort = kwargs.get("reasoning_effort")
            xtc_probability = kwargs.get("xtc_probability", 0.0)
            xtc_threshold = kwargs.get("xtc_threshold", 0.0)

            # Resolve reasoning_effort → thinking_budget
            if thinking_budget is None and reasoning_effort is not None:
                thinking_budget = {"low": 2048, "medium": 8192, "high": 32768}.get(
                    reasoning_effort, 8192
                )

            # logprobs is not supported by VLM engine (mlx_vlm.generate() and
            # model.language_model don't expose per-token logprobs).
            if logprobs or top_logprobs:
                logger.warning(
                    "VLMEngine does not support logprobs/top_logprobs — "
                    "parameter ignored. Use BatchedEngine for logprobs support."
                )

            def _generate_sync():
                # NOTE: don't call mx.random.seed(seed) here — the actual
                # sampling happens inside `with mx.stream(generation_stream)`
                # in _generate_vlm_text / _generate_vlm_vision, which uses
                # a separate stream PRNG. Setting seed outside that scope
                # has no effect on stream-scoped categorical sampling.
                # See _generate_vlm_text and _generate_vlm_vision below
                # where mx.random.seed is called INSIDE the stream context.

                if (image_paths and self._has_vision and self._is_vlm) or (
                    audio_paths and self._is_vlm
                ):
                    return self._generate_vlm_vision(
                        messages,
                        image_paths,
                        max_tokens,
                        temperature,
                        top_p,
                        top_k,
                        stop,
                        audio_paths=audio_paths,
                        enable_thinking=_enable_thinking,
                        thinking_budget=thinking_budget,
                        seed=seed,
                    )

                if image_paths or audio_paths:
                    raise RuntimeError(
                        f"Multimodal input provided ({len(image_paths or [])} image(s), "
                        f"{len(audio_paths or [])} audio) but model {self.model_name!r} loaded "
                        f"as text-only (has_vision={self._has_vision}, is_vlm={self._is_vlm}). "
                        f"This usually means mlx_vlm.load failed — check load-time logs. "
                        f"Use a different VLM model or fix the load error."
                    )

                input_ids = self._tokenize_with_cache(
                    messages, enable_thinking=_enable_thinking
                )

                if self._is_vlm:
                    freq_p = kwargs.get("frequency_penalty", 0.0)
                    pres_p = kwargs.get("presence_penalty", 0.0)
                    lb = kwargs.get("logit_bias")
                    js = kwargs.get("json_schema")
                    # Fallback: if grammar was passed directly (not via _parse_response_format),
                    # convert it to json_schema for the text generator.
                    if js is None:
                        js = kwargs.get("grammar")
                    return self._generate_vlm_text(
                        input_ids,
                        max_tokens,
                        temperature,
                        top_p,
                        top_k,
                        min_p,
                        stop,
                        stop_token_ids=stop_token_ids,
                        repetition_penalty=repetition_penalty,
                        frequency_penalty=freq_p,
                        presence_penalty=pres_p,
                        logit_bias=lb,
                        json_schema=js,
                        enable_thinking=_enable_thinking,
                        xtc_probability=xtc_probability,
                        xtc_threshold=xtc_threshold,
                        thinking_budget=thinking_budget,
                        cancel_event=kwargs.get("cancel_event"),
                        seed=seed,
                    )

                from mlx_lm.generate import generate_step, generation_stream

                # Determine seed for THIS request. Each request must get a
                # fresh PRNG state — calling mx.random.seed() once before
                # the loop only seeds the first call's compiled graph; the
                # second call reuses the same cached PRNG state because
                # categorical_sampling is @mx.compile-wrapped with
                # inputs=mx.random.state. Per-sample re-seed via a sampler
                # wrapper (mirrors the BatchedEngine fix).
                if seed is not None:
                    _base_seed = int(seed) & ((1 << 63) - 1)
                else:
                    import time as _t

                    _base_seed = _t.time_ns() & ((1 << 63) - 1)
                with mx.stream(generation_stream):
                    mx.random.seed(_base_seed)
                logger.debug(f"VLM fallback path base_seed={_base_seed}")

                sampler = _build_noncached_sampler(
                    temperature, top_p, top_k, min_p, seed
                )
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
                # Thinking token detection: only use token ID matching when
                # "<think"/"</think" encode to a SINGLE token.  Multi-token
                # encodings mean `encode(...)[-1]` picks a random last token,
                # causing false positives (any token sharing that ID triggers
                # a state transition).  Fall back to text-based suffix matching
                # for multi-token vocabularies.
                _think_single_token = False
                try:
                    _ts_ids = self._tokenizer.encode("<think")
                    _te_ids = self._tokenizer.encode("</think")
                    if len(_ts_ids) == 1 and len(_te_ids) == 1:
                        think_start_id = _ts_ids[0]
                        think_end_id = _te_ids[0]
                        _think_single_token = True
                    else:
                        think_start_id = think_end_id = None
                except Exception:
                    logger.debug("operation failed", exc_info=True)
                    think_start_id = think_end_id = None
                    _think_single_token = False

                _stop_hit = False
                _budget_hit = False
                _accumulated_text = ""  # for text-based thinking detection
                for token_id, _ in generate_step(
                    input_ids,
                    self._model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ):
                    tokens.append(token_id)
                    # Track thinking segment boundaries
                    if _think_single_token and think_start_id is not None:
                        if not _in_thinking and token_id == think_start_id:
                            _in_thinking = True
                        elif _in_thinking:
                            if token_id == think_end_id:
                                _in_thinking = False
                            else:
                                _thinking_tokens += 1
                    elif not _think_single_token:
                        # Text-based thinking detection for multi-token encodings
                        _tok_text = self._tokenizer.decode([token_id])
                        _accumulated_text += _tok_text
                        if not _in_thinking and _accumulated_text.endswith("<think"):
                            _in_thinking = True
                        elif _in_thinking:
                            if _accumulated_text.endswith("</think"):
                                _in_thinking = False
                            else:
                                _thinking_tokens += 1
                    if token_id in stop_ids:
                        _stop_hit = True
                        break
                    # Thinking budget enforcement — cap thinking tokens, not total tokens
                    if (
                        thinking_budget is not None
                        and _in_thinking
                        and _thinking_tokens >= thinking_budget
                    ):
                        # Append closing tag to keep output well-formed
                        if _think_single_token and think_end_id is not None:
                            tokens.append(think_end_id)
                        _budget_hit = True
                        break

                # Return (text, thinking, tokens, stop_hit, budget_hit, cached_tokens).
                # total_token_count includes the stop token if present.
                _decoded = self._tokenizer.decode(tokens, skip_special_tokens=True)
                return (
                    _decoded,
                    _thinking_tokens,
                    len(tokens),
                    _stop_hit,
                    _budget_hit,
                    0,
                )  # non-VLM text path: no text-prefix reuse

            loop = asyncio.get_running_loop()
            try:
                _timeout_seconds = kwargs.get("timeout_seconds") or 120.0
                (
                    result,
                    reasoning_tokens,
                    completion_token_count,
                    stop_hit,
                    budget_hit,
                    cached_token_count,
                ) = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, _generate_sync),
                    timeout=_timeout_seconds,
                )
            except TimeoutError:
                self._num_requests_processed += 1
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

            time.monotonic() - t0
            self._num_requests_processed += 1
            self._total_reasoning_tokens += reasoning_tokens

            # Use VLM template for vision path (includes image placeholders),
            # plain _format_prompt for text-only path
            if (image_paths and self._has_vision and self._is_vlm) or (
                audio_paths and self._is_vlm
            ):
                # pass the SAME max_images/num_audios as the generation call
                # (1979/2732) so prompt_tokens reflects the template the model actually
                # saw — otherwise SINGLE_IMAGE_ONLY_MODELS (and video) count a template
                # with a different placeholder count than was generated.
                _vlm_prompt = self._apply_vlm_template_with_cache(
                    messages,
                    enable_thinking=_enable_thinking,
                    num_audios=len(audio_paths) if audio_paths else 0,
                    max_images=len(image_paths) if image_paths else None,
                )
                prompt_tokens = self._count_text_tokens(_vlm_prompt)
                # Add per-image token estimate so prompt_tokens reflects the
                # actual model input size (text + image embeddings).
                if image_paths:
                    prompt_tokens += self._estimate_image_tokens() * len(image_paths)
            else:
                prompt_text = self._format_prompt(messages)
                prompt_tokens = self._count_text_tokens(prompt_text)
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
            }
        finally:
            # release the MLX buffer pool after each request on large
            # models so it doesn't grow across requests → OOM/GPU-hang under
            # sustained load (the Qwen3-Omni-30B hang). On the executor thread.
            if getattr(self, "_mx_large_model", False):
                try:
                    import mlx.core as _mxc

                    _loop = asyncio.get_running_loop()
                    await _loop.run_in_executor(
                        self._executor, lambda: (_mxc.synchronize(), _mxc.clear_cache())
                    )
                except Exception:
                    logger.debug("post-gen mx.clear_cache failed", exc_info=True)
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

        # logprobs is not supported by VLM engine
        if logprobs or top_logprobs:
            logger.warning(
                "VLMEngine does not support logprobs/top_logprobs — "
                "parameter ignored. Use BatchedEngine for logprobs support."
            )

        # Extract images/audio once, reuse for both pipeline and generation.
        # Per-request identity-based temp tracking (see generate()): avoids the
        # positional-offset race that deleted concurrent/n>1 requests' files.
        _req_temp_files: list[str] = []
        _temp_token = _request_temp_files.set(_req_temp_files)
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

        if (image_paths or audio_paths) and not (has_images or has_audio):
            raise RuntimeError(
                f"Multimodal input provided ({len(image_paths or [])} image(s), "
                f"{len(audio_paths or [])} audio) but model {self.model_name!r} loaded "
                f"as text-only (has_vision={self._has_vision}, is_vlm={self._is_vlm}). "
                f"This usually means mlx_vlm.load failed — check load-time logs."
            )

        # Capture event loop for thread-safe queue writes from executor thread.
        # asyncio.Queue.put_nowait() is NOT thread-safe — must schedule puts
        # via call_soon_threadsafe (same pattern as batched_engine.py).
        _loop_for_queue = asyncio.get_running_loop()

        # Eagerly resolve detokenizer availability so the error handler can
        # safely reference it even if the try-block fails before the point
        # where it was previously assigned inside _stream_sync.
        _has_detokenizer = (
            hasattr(self._tokenizer, "detokenizer") if self._tokenizer else False
        )

        # Thread-safe cancel check: wraps asyncio.Event so the executor
        # thread can read it without asyncio-specific thread-safety issues.
        # Reading ._value is a simple bool attribute read, GIL-protected.
        def _is_cancelled():
            if cancel_event is None:
                return False
            if isinstance(cancel_event, asyncio.Event):
                return cancel_event._value
            return cancel_event.is_set()

        # Thread-safe queue put for executor thread — asyncio.Queue is NOT
        # safe to call from non-event-loop threads.  Wrap it so all
        # put_nowait calls from the executor are routed through
        # call_soon_threadsafe (same pattern as batched_engine.py).

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

        _safe_queue = _ThreadSafeQueue(queue, _loop_for_queue)

        _stream_ttft_t0 = [time.perf_counter()]
        _stream_ttft_recorded = [False]
        _stream_ttft_val = [0.0]

        def _stream_sync():
            nonlocal _has_detokenizer
            # Initialize eagerly so the error handler can reference it
            # even if the exception fires before the point where it was
            # previously assigned inside the try block.
            has_detokenizer = _has_detokenizer
            detokenizer = (
                None  # Initialize before try so error handler can safely check
            )
            try:
                # NOTE: don't seed here — generation_stream has separate PRNG.
                # Seed is forwarded to _stream_vlm_text / _stream_vlm_vision
                # which set it inside their `with mx.stream()` block.

                if has_images or has_audio:
                    # Resolve reasoning_effort -> thinking_budget for VLM vision streaming
                    _tb = kwargs.get("thinking_budget")
                    if _tb is None:
                        _re = kwargs.get("reasoning_effort")
                        if _re is not None:
                            _tb = {"low": 2048, "medium": 8192, "high": 32768}.get(
                                _re, 8192
                            )
                    self._stream_vlm_vision(
                        messages,
                        image_paths,
                        max_tokens,
                        temperature,
                        top_p,
                        req_id,
                        _safe_queue,
                        top_k,
                        min_p,
                        stop,
                        audio_paths=audio_paths,
                        enable_thinking=enable_thinking,
                        cancel_event=cancel_event,
                        xtc_probability=xtc_probability,
                        xtc_threshold=xtc_threshold,
                        thinking_budget=_tb,
                        _ttft_t0=_stream_ttft_t0,
                        _ttft_recorded=_stream_ttft_recorded,
                        _ttft_val=_stream_ttft_val,
                        seed=seed,
                    )
                    return

                input_ids = self._tokenize_with_cache(
                    messages, enable_thinking=enable_thinking
                )

                if self._is_vlm:
                    freq_p = kwargs.get("frequency_penalty", 0.0)
                    pres_p = kwargs.get("presence_penalty", 0.0)
                    lb = kwargs.get("logit_bias")
                    js = kwargs.get("json_schema")
                    # Fallback: if grammar was passed directly (not via _parse_response_format),
                    # convert it to json_schema for the text generator.
                    if js is None:
                        js = kwargs.get("grammar")
                    _tb = kwargs.get("thinking_budget")
                    # Resolve reasoning_effort → thinking_budget
                    if _tb is None:
                        _re = kwargs.get("reasoning_effort")
                        if _re is not None:
                            _tb = {"low": 2048, "medium": 8192, "high": 32768}.get(
                                _re, 8192
                            )
                    self._stream_vlm_text(
                        input_ids,
                        max_tokens,
                        temperature,
                        top_p,
                        req_id,
                        _safe_queue,
                        top_k,
                        min_p,
                        stop,
                        repetition_penalty,
                        freq_p,
                        pres_p,
                        lb,
                        json_schema=js,
                        enable_thinking=enable_thinking,
                        cancel_event=cancel_event,
                        stop_token_ids=stop_token_ids,
                        xtc_probability=xtc_probability,
                        xtc_threshold=xtc_threshold,
                        thinking_budget=_tb,
                        _ttft_t0=_stream_ttft_t0,
                        _ttft_recorded=_stream_ttft_recorded,
                        _ttft_val=_stream_ttft_val,
                        seed=seed,
                    )
                    return

                from mlx_lm.generate import generate_step

                sampler = _build_noncached_sampler(
                    temperature, top_p, top_k, min_p, seed
                )
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
                if has_detokenizer:
                    detokenizer = self._tokenizer.detokenizer
                    detokenizer.reset()

                # Thinking state tracking for streaming fast path
                _in_thinking = False
                _thinking_tokens = 0
                thinking_budget = kwargs.get("thinking_budget")
                # Resolve reasoning_effort → thinking_budget
                if thinking_budget is None:
                    reasoning_effort = kwargs.get("reasoning_effort")
                    if reasoning_effort is not None:
                        thinking_budget = {
                            "low": 2048,
                            "medium": 8192,
                            "high": 32768,
                        }.get(reasoning_effort, 8192)
                # Thinking token detection: only use token ID matching when
                # "<think"/"</think" encode to a SINGLE token.  Multi-token
                # encodings mean `encode(...)[-1]` picks a random last token,
                # causing false positives.  Fall back to text-based detection.
                _think_single_token = False
                try:
                    _ts_ids = self._tokenizer.encode("<think")
                    _te_ids = self._tokenizer.encode("</think")
                    if len(_ts_ids) == 1 and len(_te_ids) == 1:
                        think_start_id = _ts_ids[0]
                        think_end_id = _te_ids[0]
                        _think_single_token = True
                    else:
                        think_start_id = think_end_id = None
                except Exception:
                    logger.debug("thinking token encode failed", exc_info=True)
                    think_start_id = think_end_id = None
                    _think_single_token = False

                accumulated = ""
                # Multi-token stop hold-back: withhold any streamed text that
                # could be the start of a stop string so its prefix never leaks
                # before the match completes. Previously this path emitted each
                # token immediately and only trimmed the stop on the token that
                # COMPLETED it — so the first token(s) of a multi-token stop
                # (e.g. "\n\n" as two "\n" tokens) leaked into the stream
                # (same class as the chat fast-path fix in).
                from .text_utils import StopHoldbackBuffer

                _hb = StopHoldbackBuffer([s for s in (stop or []) if s])
                _thinking_text = ""  # for text-based thinking detection
                token_count = 0
                _num_prompt_tokens = len(input_ids)
                _cur_state = "normal"  # Initialize before loop; referenced after loop if 0 iterations
                for token_id, _ in generate_step(
                    input_ids,
                    self._model,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ):
                    if _is_cancelled():
                        # Flush remaining detokenizer bytes + any text held by the
                        # stop buffer (no stop fired, so it's genuine output that
                        # was already pulled out of the detokenizer).
                        if has_detokenizer:
                            try:
                                detokenizer.finalize()
                                remaining = (
                                    _hb.feed(detokenizer.last_segment) + _hb.flush()
                                )
                                if remaining:
                                    _cancel_state = (
                                        "reasoning" if _in_thinking else "normal"
                                    )
                                    _safe_queue.put_nowait(
                                        RequestOutput(
                                            request_id=req_id,
                                            new_text=remaining,
                                            finish_reason=None,
                                            finished=False,
                                            current_state=_cancel_state,
                                        )
                                    )
                            except Exception:
                                logger.debug(
                                    "detokenizer finalize in cancel handler failed",
                                    exc_info=True,
                                )
                        _safe_queue.put_nowait(
                            RequestOutput(
                                request_id=req_id,
                                new_text="",
                                finish_reason="cancel",
                                finished=True,
                                completion_tokens=token_count,
                                prompt_tokens=_num_prompt_tokens,
                                reasoning_tokens=_thinking_tokens,
                            )
                        )
                        return
                    token_count += 1
                    # Record TTFT on first token
                    if not _stream_ttft_recorded[0]:
                        _stream_ttft_recorded[0] = True
                        _stream_ttft_val[0] = time.perf_counter() - _stream_ttft_t0[0]
                    is_eos = token_id in stop_ids

                    # Track thinking segment boundaries
                    if _think_single_token and think_start_id is not None:
                        if not _in_thinking and token_id == think_start_id:
                            _in_thinking = True
                        elif _in_thinking:
                            if token_id == think_end_id:
                                _in_thinking = False
                            else:
                                _thinking_tokens += 1
                    elif not _think_single_token:
                        # Text-based thinking detection for multi-token encodings
                        _tok_text = self._tokenizer.decode([token_id])
                        _thinking_text += _tok_text
                        if not _in_thinking and _thinking_text.endswith("<think"):
                            _in_thinking = True
                        elif _in_thinking:
                            if _thinking_text.endswith("</think"):
                                _in_thinking = False
                            else:
                                _thinking_tokens += 1

                    # Thinking budget enforcement
                    if (
                        thinking_budget is not None
                        and _in_thinking
                        and _thinking_tokens >= thinking_budget
                        and (think_end_id is not None or not _think_single_token)
                    ):
                        # Budget exceeded — stop generation (flush detok + buffered tail)
                        if has_detokenizer:
                            detokenizer.finalize()
                            remaining = _hb.feed(detokenizer.last_segment) + _hb.flush()
                            if remaining:
                                _safe_queue.put_nowait(
                                    RequestOutput(
                                        request_id=req_id,
                                        new_text=remaining,
                                        finish_reason=None,
                                        finished=False,
                                        current_state="reasoning"
                                        if _in_thinking
                                        else "normal",
                                    )
                                )
                        _safe_queue.put_nowait(
                            RequestOutput(
                                request_id=req_id,
                                new_text="",
                                finish_reason="stop",
                                finished=True,
                                completion_tokens=token_count,
                                prompt_tokens=_num_prompt_tokens,
                                current_state="reasoning" if _in_thinking else "normal",
                                reasoning_tokens=_thinking_tokens,
                            )
                        )
                        return

                    if not is_eos:
                        if has_detokenizer:
                            detokenizer.add_token(token_id)
                            token_text = detokenizer.last_segment
                        else:
                            token_text = self._tokenizer.decode(
                                [token_id], skip_special_tokens=True
                            )
                    else:
                        token_text = ""

                    accumulated += token_text

                    # Check multi-token stop suffixes (detection on full text)
                    finish_reason = None
                    _suffix_hit = False
                    if is_eos:
                        finish_reason = "stop"
                        token_text = ""  # Don't emit EOS token text
                    elif stop:
                        for s in stop:
                            if accumulated.endswith(s):
                                accumulated = accumulated[: -len(s)]
                                finish_reason = "stop"
                                _suffix_hit = True
                                break

                    _cur_state = "reasoning" if _in_thinking else "normal"

                    # Route text through the hold-back buffer so a multi-token
                    # stop never leaks its prefix. On a string-stop the completing
                    # token is fed in then take_stopped() drops the matched stop
                    # (and any held prefix that belonged to it); on EOS the held
                    # text is genuine output, so flush it.
                    if _suffix_hit:
                        # emit feed()'s pre-stop return too (else content fused with
                        # the stop token is silently lost). See _stream_vlm_text.
                        emit_text = _hb.feed(token_text) + _hb.take_stopped()
                    elif is_eos:
                        emit_text = _hb.flush()
                    else:
                        emit_text = _hb.feed(token_text)

                    if emit_text or finish_reason:
                        output = RequestOutput(
                            request_id=req_id,
                            new_text=emit_text,
                            new_token_ids=[token_id],
                            finish_reason=finish_reason,
                            finished=finish_reason is not None,
                            completion_tokens=token_count,
                            prompt_tokens=_num_prompt_tokens,
                            current_state=_cur_state,
                            reasoning_tokens=_thinking_tokens,
                            ttft_ms=round(_stream_ttft_val[0] * 1000, 1)
                            if _stream_ttft_val[0] > 0
                            else 0.0,
                        )
                        _safe_queue.put_nowait(output)

                    if finish_reason:
                        # Flush remaining detok bytes. On a string-stop the match
                        # was already dropped, so skip (avoids re-leaking the stop
                        # that finalize() may surface); on EOS emit the genuine tail.
                        if has_detokenizer and not _suffix_hit:
                            detokenizer.finalize()
                            remaining = detokenizer.last_segment
                            tail = (
                                (_hb.feed(remaining) + _hb.flush()) if remaining else ""
                            )
                            if tail:
                                _safe_queue.put_nowait(
                                    RequestOutput(
                                        request_id=req_id,
                                        new_text=tail,
                                        finish_reason=None,
                                        finished=False,
                                        current_state=_cur_state,
                                    )
                                )
                        return

                # Max tokens reached — finalize detokenizer and flush any text
                # still held back by the stop buffer (no stop fired, so it's all
                # genuine output).
                if has_detokenizer:
                    detokenizer.finalize()
                    remaining = detokenizer.last_segment
                    remaining = _hb.feed(remaining) + _hb.flush()
                    if remaining:
                        _safe_queue.put_nowait(
                            RequestOutput(
                                request_id=req_id,
                                new_text=remaining,
                                finish_reason=None,
                                finished=False,
                                current_state=_cur_state,
                            )
                        )
                else:
                    # No detokenizer: still flush any buffered tail.
                    _flush_tail = _hb.flush()
                    if _flush_tail:
                        _safe_queue.put_nowait(
                            RequestOutput(
                                request_id=req_id,
                                new_text=_flush_tail,
                                finish_reason=None,
                                finished=False,
                                current_state=_cur_state,
                            )
                        )
                _final_state = "reasoning" if _in_thinking else "normal"
                output = RequestOutput(
                    request_id=req_id,
                    new_text="",
                    finish_reason="length",
                    finished=True,
                    completion_tokens=token_count,
                    prompt_tokens=_num_prompt_tokens,
                    current_state=_final_state,
                    reasoning_tokens=_thinking_tokens,
                    ttft_ms=round(_stream_ttft_val[0] * 1000, 1)
                    if _stream_ttft_val[0] > 0
                    else 0.0,
                )
                _safe_queue.put_nowait(output)

            except Exception as e:
                logger.error(f"VLM stream error: {e}", exc_info=True)
                # Flush remaining detokenizer bytes on error
                if detokenizer is not None:
                    try:
                        detokenizer.finalize()
                        remaining = detokenizer.last_segment
                        if remaining:
                            _safe_queue.put_nowait(
                                RequestOutput(
                                    request_id=req_id,
                                    new_text=remaining,
                                    finish_reason=None,
                                    finished=False,
                                )
                            )
                    except Exception:
                        logger.debug(
                            "detokenizer finalize in error handler failed",
                            exc_info=True,
                        )
                # Emit error output so the consumer can distinguish error from normal end
                _safe_queue.put_nowait(
                    RequestOutput(
                        request_id=req_id,
                        new_text="",
                        finish_reason="error",
                        finished=True,
                        error=str(e),
                    )
                )
            finally:
                with contextlib.suppress(Exception):
                    _safe_queue.put_nowait(None)

        with self._active_count_lock:
            self._active_count += 1
        loop = asyncio.get_running_loop()

        stream_task = loop.run_in_executor(self._executor, _stream_sync)

        # the gateway passes timeout_seconds=req.timeout (chat.py), but this
        # read the wrong key 'timeout' → a user-set per-request timeout was SILENTLY
        # ignored and the inactivity timeout was permanently hardcoded to 300s. The
        # non-streaming twin (generate, ~line 1578) correctly reads 'timeout_seconds'.
        _timeout_seconds = kwargs.get("timeout_seconds") or kwargs.get("timeout") or 300
        _prompt_tokens_count = 0
        _completion_tokens_count = 0
        _model_id = self.model_name

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
                    # signal the GPU loop to STOP. stream_task.cancel() in the
                    # finally can't interrupt the executor thread running _stream_sync;
                    # the VLM decode loops (_stream_vlm_text/_stream_vlm_vision) only stop
                    # when they observe cancel_event. Without this the executor kept
                    # decoding to max_tokens, pinning the single VLM executor (                    # class — the BatchedEngine text path sets _timeout_cancel here).
                    try:
                        if cancel_event is not None:
                            cancel_event.set()
                    except Exception:
                        logger.debug(
                            "VLM stream timeout: cancel_event.set() failed",
                            exc_info=True,
                        )
                    yield RequestOutput(
                        request_id=req_id,
                        new_text="",
                        finish_reason="error",
                        finished=True,
                        error=f"Streaming timeout: no token for {_timeout_seconds}s",
                    )
                    break
                if output is None:
                    break
                if output.prompt_tokens > 0:
                    _prompt_tokens_count = output.prompt_tokens
                if output.completion_tokens > 0:
                    _completion_tokens_count = output.completion_tokens
                # Record TTFT in Prometheus on first token
                if output.ttft_ms > 0 and _stream_ttft_recorded[0]:
                    try:
                        from yunshu_gateway.middleware.prometheus_exporter import (
                            get_prometheus_metrics,
                        )

                        pm = get_prometheus_metrics()
                        pm.observe_histogram(
                            "ttft_seconds",
                            _stream_ttft_val[0],
                            labels={"model_id": self.model_name},
                        )
                    except Exception:
                        pass
                    _stream_ttft_recorded[0] = False
                yield output
        finally:
            # bound the MLX buffer pool on large models (see generate()).
            if getattr(self, "_mx_large_model", False):
                try:
                    import mlx.core as _mxc

                    _loop = asyncio.get_running_loop()
                    await _loop.run_in_executor(
                        self._executor, lambda: (_mxc.synchronize(), _mxc.clear_cache())
                    )
                except Exception:
                    logger.debug("post-stream mx.clear_cache failed", exc_info=True)
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
            with self._active_count_lock:
                # floor at 0 to match the non-streaming path (line ~1454).
                # Without it an unbalanced decrement underflows _active_count negative,
                # making is_busy() wrongly report idle (could unload the model mid-work).
                self._active_count = max(0, self._active_count - 1)
            if not stream_task.done():
                stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await stream_task
            # Drain remaining queue items to unblock the executor thread
            # so it can observe the cancellation and exit promptly.
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
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
        thinking_budget: int | None = None,
        seed: int | None = None,
    ) -> str:
        """Vision + text generation using mlx_vlm.generate().

        Passes vision_cache and prompt_cache_state to mlx_vlm so that:
        1. Image features are cached and reused when the same image appears again
        2. KV cache is reused for common prefix across conversations with same image
        """
        from mlx_vlm.generate import generate as vlm_generate

        # Apply chat template with caching to skip re-processing for repeated messages
        num_audios = len(audio_paths) if audio_paths else 0
        prompt = self._apply_vlm_template_with_cache(
            messages,
            enable_thinking=enable_thinking,
            num_audios=num_audios,
            max_images=len(image_paths) if image_paths else None,
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

        # Encoder cache key for post-generation storage. We don't look up
        # here because the actual encoder output reuse happens through the
        # vision_cache adapter (passed via gen_kwargs["vision_cache"]).
        # The encoder cache stores the raw encoder_outputs from vlm_generate's
        # result object when available.
        _encoder_cache_key = None
        if image_hash is not None:
            _encoder_cache_key = f"vlm-{image_hash}"

        # Seed the stream PRNG inside the generation_stream context. mlx_vlm's
        # vlm_generate uses generation_stream internally, so seeding here
        # (default stream) is insufficient; we set it here as best-effort.
        # See _generate_vlm_text for the proper stream-scoped seed pattern.
        if seed is not None:
            from mlx_lm.generate import generation_stream

            with mx.stream(generation_stream):
                mx.random.seed(int(seed) & ((1 << 63) - 1))

        gen_kwargs: dict = {
            "max_tokens": max_tokens,
            # mlx_vlm's generate/generate_step take `temperature`, not `temp`;
            # `temp` falls through **kwargs unread → greedy regardless of the
            # requested temperature.
            "temperature": temperature,
            "verbose": False,
        }
        if image_paths:
            gen_kwargs["image"] = (
                image_paths if len(image_paths) > 1 else image_paths[0]
            )
        if audio_paths:
            gen_kwargs["audio"] = self._audio_arg(audio_paths)

        # NOTE : the installed mlx_vlm has NO `vision_cache` parameter,
        # so this kwarg is IGNORED — it never drove any caching. The real
        # cross-request vision-feature cache is now the _CachingVisionTower
        # wrapper installed on the model's vision tower(s) at load. Kept here
        # (harmless) only for forward-compat if a future mlx_vlm adds the hook;
        # the VisionFeatureCache SSD tier remains available behind the adapter.
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
            new_hits = max(
                0, vc_stats_after.get("hits", 0) - vc_stats_before.get("hits", 0)
            )
            if new_hits > 0:
                self._vlm_vision_hits += new_hits
                logger.debug("VLM vision feature cache hit: reused encoded image")
            # count misses PER IMAGE (was +1 for any multi-image miss, which
            # skewed the hit rate vs the per-image hit count).
            _misses = max(0, (len(image_paths) if image_paths else 0) - new_hits)
            if _misses > 0:
                self._vlm_vision_misses += _misses

        # Store encoder output in encoder cache for future reuse.
        # Only store actual encoder_outputs from the result object.
        # Do NOT store placeholder markers (True) — the vision_cache adapter
        # already handles image feature caching, and storing fake markers
        # wastes memory and creates misleading cache hit stats.
        if _encoder_cache_key is not None:
            encoder_output = getattr(result, "encoder_outputs", None)
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

        # Extract text from result object and trim at stop sequences
        _result_text = result.text if hasattr(result, "text") else str(result)
        _stop_hit = False
        if stop:
            for s in stop:
                idx = _result_text.find(s)
                if idx >= 0:
                    _result_text = _result_text[:idx]
                    _stop_hit = True
                    break

        # Enforce thinking budget if provided
        # thinking_budget is a TOKEN count, not a character count.
        # Since vlm_generate doesn't expose raw tokens, tokenize the thinking
        # content to get an accurate token count for budget comparison.
        _budget_hit = False
        _thinking_tokens = 0
        # count reasoning tokens whenever the output HAS a think
        # block — not only when a thinking_budget was passed. The old code nested the
        # count inside `if thinking_budget is not None`, so a non-streaming vision
        # response with <think>…</think> but no explicit budget returned
        # reasoning_tokens=0 (the text path counts unconditionally; this was an
        # asymmetric undercount). _find_think_tag avoids <thinking>/<think_more> matches.
        if self._tokenizer is not None:
            think_start = self._find_think_tag(_result_text, "<think")
            if think_start >= 0:
                think_end = self._find_think_tag(
                    _result_text, "</think", search_start=think_start + 1
                )
                # if generation hit max_tokens while STILL inside the
                # thinking block (no closing tag), think_end < 0 and the whole
                # output is reasoning — count from the open tag to the end, else
                # _thinking_tokens stayed 0 (asymmetric undercount vs the streaming
                # and text paths, which count incrementally).
                _think_seg = (
                    _result_text[think_start:think_end]
                    if think_end >= 0
                    else _result_text[think_start:]
                )
                if think_end >= 0 or _think_seg:
                    think_content = _think_seg
                    try:
                        _thinking_tokens = len(self._tokenizer.encode(think_content))
                    except Exception:
                        # Fallback: rough character-to-token ratio (~4 chars/token)
                        _thinking_tokens = len(think_content) // 4
                    if (
                        think_end >= 0
                        and thinking_budget is not None
                        and _thinking_tokens > thinking_budget
                    ):
                        # Budget exceeded — truncate thinking content.
                        # Estimate character cutoff from token budget using
                        # the actual ratio observed in this thinking segment.
                        if _thinking_tokens > 0:
                            chars_per_token = len(think_content) / _thinking_tokens
                            max_chars = int(thinking_budget * chars_per_token)
                        else:
                            max_chars = thinking_budget * 4
                        _result_text = (
                            _result_text[:think_start]
                            + _result_text[think_start : think_start + max_chars]
                            + _result_text[think_end:]
                        )
                        _budget_hit = True

        # Capture mRoPE deltas after vision prefill
        if self._mrope_info and self._mrope_info.enabled:
            from .mrope import capture_rope_deltas

            delta = capture_rope_deltas(self._model)
            if delta is not None:
                logger.debug(f"mRoPE delta captured: {delta:.4f}")

        # Return 6-tuple: (text, thinking, tokens, stop_hit, budget_hit, cached_tokens)
        _est_tokens = getattr(result, "generation_tokens", 0) or 0
        if _stop_hit and _result_text and self._tokenizer:
            _corrected = len(self._tokenizer.encode(_result_text))
            if isinstance(_est_tokens, int) and _corrected < _est_tokens:
                _est_tokens = _corrected
        elif _est_tokens == 0 and _result_text and self._tokenizer:
            _est_tokens = len(self._tokenizer.encode(_result_text))
        return (
            _result_text,
            _thinking_tokens,
            _est_tokens,
            _stop_hit,
            _budget_hit,
            0,
        )  # vision: no text-prefix reuse

    # ── VLM text generation (for mlx-vlm models) ──

    def backend_capabilities(self, lm: Any = None) -> Any:
        """Derive this VLM backbone's serving capabilities via the shared,
        backbone-agnostic `model_backend` layer (Part 2/3). Memoized.

        Used to decide whether cross-request KV prefix reuse is lossless — the
        same classification logic BatchedEngine can adopt — instead of
        per-engine hard-coding. See docs/VLM_TEXT_KV_PREFIX.md."""
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
        # mRoPE detection: config (detect_mrope) is unreliable — Qwen3-Omni
        # nests rope_scaling under thinker_config.text_config, which detect_mrope
        # misses. Broaden with model introspection: a get_rope_index / _rope_deltas
        # method means position lives outside the cache (mRoPE-style), so the
        # caller must supply explicit positions for reuse.
        is_mrope = bool(
            (
                self._mrope_info is not None
                and getattr(self._mrope_info, "enabled", False)
            )
            or hasattr(lm, "get_rope_index")
            or hasattr(lm, "_rope_deltas")
        )
        caps = derive_capabilities(BackendKind.VLM, layers, is_mrope=is_mrope)
        self._backend_caps = caps
        if not caps.supports_kv_prefix_reuse:
            logger.info(
                "VLM text KV prefix cache: %s bypassing prefix reuse — %s (layers=%s)",
                self.model_name,
                caps.bypass_reason(),
                ",".join(sorted(set(caps.cache.layer_types))) or "?",
            )
        return caps

    def _text_prefix_reuse_safe(self, lm: Any) -> bool:
        """Whether cross-request text-prefix KV reuse is LOSSLESS for this model.

        Two gates: (1) the shared capability layer (cache must be resumable —
        bypasses sliding-window / hybrid), and (2) an EMPIRICAL load-time probe
        that runs the actual reuse path vs a full prefill and requires identical
        logits. The probe is the robust guard: it empirically confirms the
        model-native rope-state priming (_prime_mrope_reuse_state) reproduces a
        full prefill for THIS backbone, regardless of mRoPE variant. Runs ONCE at
        load (executor thread); this method is a pure memoized read — no GPU work
        — safe to call from any thread."""
        if self._text_kv_prefix_cache is None:
            return False
        if not self.backend_capabilities(lm).supports_kv_prefix_reuse:
            return False
        # Probe ran at load on the executor. None = not probed → bypass.
        return bool(self._reuse_probe_ok)

    def _probe_text_reuse_lossless(self, lm: Any) -> bool:
        """One-shot empirical check (memoized): does the reuse path produce the
        SAME GREEDY OUTPUT as a full prefill on this model? Compares argmax TOKEN
        SEQUENCES over several decode steps — the actual definition of lossless
        for greedy serving — rather than a single-prefill logit delta (too strict
        for MoE routing noise that never flips the greedy token). Runs on the
        executor thread. Conservative: any error → not safe (bypass)."""
        if self._reuse_probe_ok is not None:
            return self._reuse_probe_ok
        ok = False
        try:
            import mlx.core as mx
            from mlx_lm.generate import generation_stream
            from mlx_vlm.models.cache import make_prompt_cache

            txt = "The quick brown fox jumps over the lazy dog. " * 48
            ids = mx.array(self._tokenizer.encode(txt))
            if int(ids.shape[0]) < 64:
                ids = mx.concatenate([ids, ids])
            n = int(ids.shape[0])
            half = n // 2
            STEPS = 12
            needs_pos = bool(self.backend_capabilities(lm).requires_explicit_positions)

            def _greedy(cache, first_ids):
                cur = mx.argmax(
                    lm(first_ids[None], cache=cache).logits[:, -1, :], axis=-1
                )
                out = [int(cur.item())]
                for _ in range(STEPS - 1):
                    cur = mx.argmax(
                        lm(cur[None], cache=cache).logits[:, -1, :], axis=-1
                    )
                    out.append(int(cur.item()))
                return out

            with mx.stream(generation_stream):
                # Reference: full prefill + greedy.
                ref = _greedy(make_prompt_cache(lm), ids)
                # Reuse: prefill prefix, snapshot at the boundary, prime native
                # rope state, prefill the suffix on the snapshot + greedy.
                cW = make_prompt_cache(lm)
                lm(ids[:half][None], cache=cW)
                snap = self._text_kv_prefix_cache._snapshot_cache(cW, trim=0)
                if needs_pos:
                    self._prime_mrope_reuse_state(lm)
                got = _greedy(snap, ids[half:])
                ok = bool(ref == got)
            logger.info(
                "VLM text KV reuse probe for %s: %s (greedy %d-step match=%s, needs_pos=%s)",
                self.model_name,
                "LOSSLESS" if ok else "NOT lossless — bypassing",
                STEPS,
                ok,
                needs_pos,
            )
        except Exception:
            logger.warning(
                "VLM text KV reuse probe failed — bypassing reuse", exc_info=True
            )
            ok = False
        self._reuse_probe_ok = ok
        return ok

    def _text_reuse_needs_positions(self, lm: Any = None) -> bool:
        """True when this backbone needs its native rope state primed for lossless
        text-prefix reuse (mRoPE models — position lives outside the cache)."""
        return bool(self.backend_capabilities(lm).requires_explicit_positions)

    @staticmethod
    def _prime_mrope_reuse_state(lm: Any) -> None:
        """Prime an mRoPE language model's NATIVE rope state for prefix reuse.

        mRoPE models (GLM-OCR, Qwen-VL, Qwen3-Omni) compute the suffix's positions
        from ``cache_offset + self._rope_deltas`` — BUT only when ``_rope_deltas``
        is not None; after clear_rope_state() it's None, so they instead recompute
        from 0 (wrong context). Setting it to the TEXT-ONLY delta (0) makes the
        model's own logic produce cache_offset-based positions for the suffix and
        every decode step. This is model-native, so it works for BOTH simple
        (GLM-OCR) and interleaved (Qwen3-Omni) mRoPE — supplying our own
        position_ids only worked for the simple variant. No-op on a cold full
        prefill (offset 0 → the model's get_rope_index overwrites this). The
        empirical probe validates it per model; unsupported layouts just bypass."""
        try:
            lm._rope_deltas = mx.zeros((1, 1), dtype=mx.float32)
        except Exception:
            logger.debug("prime mRoPE reuse state failed", exc_info=True)

    def _text_hybrid_reuse_safe(self, lm: Any) -> bool:
        """Whether a HYBRID VLM backbone (Qwen3.5/3.6-VL: KVCache
        + non-sliceable GatedDeltaNet ArraysCache) can reuse text prefixes via
        boundary snapshots (no_trim mode). The recurrent state can't be trimmed,
        but a trim=0 snapshot at a block boundary IS losslessly resumable —
        verified by `_probe_hybrid_reuse`. Distinct from the standard (trimmable)
        reuse gate; only fires when the standard gate has declined for hybrid."""
        if self._text_kv_prefix_cache is None or not self._text_hybrid_prefix_enabled:
            return False
        caps = self.backend_capabilities(lm)
        if not caps.cache.is_hybrid or caps.cache.has_sliding_window:
            return False  # only pure hybrid (not sliding-window hybrid)
        return bool(self._hybrid_reuse_probe_ok)

    def _probe_hybrid_reuse(self, lm: Any) -> bool:
        """One-shot greedy probe (memoized): does no_trim boundary-snapshot reuse
        reproduce a full prefill for this hybrid model? Run at load on executor."""
        if self._hybrid_reuse_probe_ok is not None:
            return self._hybrid_reuse_probe_ok
        ok = False
        try:
            import mlx.core as mx
            from mlx_lm.generate import generation_stream
            from mlx_vlm.models.cache import make_prompt_cache

            txt = "The quick brown fox jumps over the lazy dog. " * 60
            ids = mx.array(self._tokenizer.encode(txt))
            n = int(ids.shape[0])
            blk = self._text_hybrid_block
            half = max(blk, (n // 2 // blk) * blk)
            if half >= n:
                half = blk
            STEPS = 12
            needs_pos = bool(self.backend_capabilities(lm).requires_explicit_positions)

            def _greedy(cache, first_ids):
                cur = mx.argmax(
                    lm(first_ids[None], cache=cache).logits[:, -1, :], axis=-1
                )
                out = [int(cur.item())]
                for _ in range(STEPS - 1):
                    cur = mx.argmax(
                        lm(cur[None], cache=cache).logits[:, -1, :], axis=-1
                    )
                    out.append(int(cur.item()))
                return out

            with mx.stream(generation_stream):
                ref = _greedy(make_prompt_cache(lm), ids)
                cW = make_prompt_cache(lm)
                lm(ids[:half][None], cache=cW)  # prefill to the boundary
                snap = self._text_kv_prefix_cache._snapshot_cache(cW, trim=0)
                if needs_pos:
                    self._prime_mrope_reuse_state(lm)
                got = _greedy(snap, ids[half:])  # resume from boundary
                ok = bool(ref == got)
            logger.info(
                "VLM HYBRID reuse probe for %s: %s (boundary=%d, greedy %d-step match=%s)",
                self.model_name,
                "LOSSLESS" if ok else "NOT lossless — bypassing",
                half,
                STEPS,
                ok,
            )
        except Exception:
            logger.warning("VLM hybrid reuse probe failed — bypassing", exc_info=True)
            ok = False
        self._hybrid_reuse_probe_ok = ok
        return ok

    def _capture_vlm_hybrid_prefix(self, lm, full_ids, cache, pc, start_offset, block):
        """Port of BatchedEngine._capture_hybrid_prefix for the VLM text path:
        advance `cache` through full_ids[start_offset:-1] in block chunks, storing
        a trim=0 boundary snapshot into `pc` at each boundary (so a later request
        sharing that N-token prefix reuses the EXACT recurrent state, no slicing).
        Leaves the final token for the caller to prefill. Returns tokens resident."""
        n = int(full_ids.shape[0])
        if n <= 1:
            return start_offset
        p = int(start_offset)
        end = n - 1
        while p < end:
            chunk = full_ids[p : min(p + block, end)]
            cn = int(chunk.shape[0])
            if cn == 0:
                break
            lm(chunk[None], cache=cache)
            p += cn
            if p % block == 0 and p >= pc._min_prefix:
                try:
                    pc.add(full_ids[:p], cache)
                except Exception:
                    logger.debug(
                        "VLM hybrid boundary snapshot add failed", exc_info=True
                    )
        return p

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
        thinking_budget: int | None = None,
        cancel_event: Any = None,
        seed: int | None = None,
    ) -> str:
        """Text generation for VLM models using model.language_model."""
        from mlx_lm.generate import generation_stream
        from mlx_vlm.models.cache import make_prompt_cache

        # Clear mRoPE state to prevent contamination from prior VLM request
        if self._mrope_info and self._mrope_info.enabled:
            from .mrope import clear_rope_state

            clear_rope_state(self._model)

        lm = self._model.language_model
        cache = make_prompt_cache(lm)
        sampler = _build_noncached_sampler(temperature, top_p, top_k, min_p, seed)
        eos_ids = self._get_eos_ids()

        # opt-in 4-tier KV prefix cache for the VLM TEXT path (text-only
        # by construction). Bypassed for backbones that can't reuse losslessly.
        _text_pc = (
            self._text_kv_prefix_cache if self._text_prefix_reuse_safe(lm) else None
        )
        # HYBRID backbones (Qwen3.5/3.6-VL) reuse via no_trim
        # boundary snapshots instead of bypassing. Same cache object, no_trim mode.
        _hybrid_mode = False
        if _text_pc is None and self._text_hybrid_reuse_safe(lm):
            _text_pc = self._text_kv_prefix_cache
            _text_pc._no_trim_mode = True
            _hybrid_mode = True

        # JSON schema / grammar constraint
        json_constraint = None
        if json_schema is not None:
            try:
                if isinstance(json_schema, dict) and json_schema.get("type") in (
                    "regex",
                    "choice",
                    "cfg",
                ):
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
                        json_constraint = ConstraintFactory.create(
                            gtype, grammar, self._tokenizer
                        )
                elif isinstance(json_schema, str) and json_schema == "json_object":
                    from .json_schema import JsonSchemaConstraint

                    json_constraint = JsonSchemaConstraint(None)
                else:
                    from .json_schema import JsonSchemaConstraint

                    json_constraint = JsonSchemaConstraint(json_schema)
            except Exception:
                logger.warning("Grammar constraint init failed", exc_info=True)

        has_penalty = (
            repetition_penalty != 1.0
            or frequency_penalty != 0.0
            or presence_penalty != 0.0
            or logit_bias
        )

        # Build stop IDs from string sequences + explicit stop_token_ids
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

        def _is_cancelled() -> bool:
            if cancel_event is None:
                return False
            if isinstance(cancel_event, asyncio.Event):
                return cancel_event._value
            return cancel_event.is_set()

        with mx.stream(generation_stream):
            # IMPORTANT: seed must be set INSIDE the stream context — the
            # generation_stream has a separate PRNG state from the default
            # stream. Setting mx.random.seed() outside this `with` block has
            # no effect on stream-scoped categorical sampling, which produces
            # identical output across requests / seeds (⚠️ A).
            if seed is not None:
                _eff_seed = int(seed) & ((1 << 63) - 1)
                mx.random.seed(_eff_seed)
                logger.debug(f"VLM _generate_vlm_text: seed={_eff_seed}")
            else:
                # No explicit seed — derive a per-call seed from current time
                # so successive requests don't repeat the same trajectory.
                _eff_seed = time.time_ns() & ((1 << 63) - 1)
                mx.random.seed(_eff_seed)
                logger.debug(f"VLM _generate_vlm_text: time-derived seed={_eff_seed}")

            # SpecPrefill: for long text prompts, use attention-based sparse
            # scoring to prioritize which tokens get prefill attention.
            # NOTE: We do NOT truncate input_ids — that would permanently
            # lose context. SpecPrefill is a no-op placeholder until proper
            # attention-weighted prefill is implemented.
            if self._spec_prefill_enabled and input_ids.shape[0] > 8192:
                logger.info(
                    f"SpecPrefill: input has {input_ids.shape[0]} tokens (>8192 threshold). "
                    "Full prefill will be used — sparse attention prefill not yet implemented."
                )

            # KV prefix cache lookup — reuse a cached prefix's KV and
            # prefill only the suffix. Lossless: the cached layers are exact
            # KV for the matched tokens (hybrid models reuse only on a trim=0
            # boundary). Falls back to full prefill on any miss/error.
            _prefill_ids = input_ids
            _pc_matched = 0
            if _text_pc is not None and len(input_ids) >= 32:
                try:
                    _cached_kv, _, _pc_matched = _text_pc.get(input_ids)
                    if _cached_kv is not None and _pc_matched > 0:
                        cache = _cached_kv
                        _prefill_ids = input_ids[_pc_matched:]
                        # On a FULL exact match the cache holds every prompt
                        # token, leaving nothing to prefill. A single-token
                        # re-feed is NOT numerically equal to the full prefill
                        # (a 1-token attention computes in a different context
                        # and flips the greedy argmax), so re-prefill the last
                        # BLOCK as a multi-token call — matching the LLM fast
                        # path's block-aligned reuse (verified bit-identical).
                        # Full exact match (empty suffix) — must still feed ≥1 token
                        # to get the first decode logits, else lm([]) → logits[:,-1]
                        # squeezes a 0-length axis and crashes.
                        if len(_prefill_ids) == 0:
                            if _hybrid_mode:
                                # ArraysCache can't be trimmed back a token; discard the
                                # full-match snapshot and cold-prefill (the hybrid capture
                                # below then re-stores boundaries + leaves the last token).
                                cache = make_prompt_cache(lm)
                                _prefill_ids = input_ids
                                _pc_matched = 0
                            else:
                                _refeed = min(len(input_ids) - 1, 128)
                                cache = _text_pc._snapshot_cache(cache, trim=_refeed)
                                _prefill_ids = input_ids[-_refeed:]
                                _pc_matched = len(input_ids) - _refeed
                        logger.debug(
                            "VLM text KV prefix hit: matched=%d/%d hybrid=%s",
                            _pc_matched,
                            len(input_ids),
                            _hybrid_mode,
                        )
                except Exception:
                    logger.warning(
                        "VLM text KV prefix get failed — full prefill", exc_info=True
                    )
                    cache = make_prompt_cache(lm)
                    _prefill_ids = input_ids
                    _pc_matched = 0

            # mRoPE backbones track token position OUTSIDE the KV
            # cache, and clear_rope_state() reset it to None — so on a reused
            # prefix the model recomputes the suffix's positions from 0 (wrong
            # context). FIX: prime the model's native rope state to the text-only
            # delta (0) so its OWN position logic computes cache_offset-based
            # positions for the suffix and every decode step. This is model-
            # native (works for BOTH simple GLM-OCR mRoPE and interleaved
            # Qwen3-Omni mRoPE, unlike supplying our own position_ids) and is a
            # no-op on a cold full prefill (offset 0 → the model's get_rope_index
            # overwrites it). Non-mRoPE backbones derive from cache.offset anyway.
            if _text_pc is not None and self._text_reuse_needs_positions(lm):
                self._prime_mrope_reuse_state(lm)

            # Prefill. HYBRID: chunk-prefill storing trim=0 boundary snapshots so
            # future shared-prefix requests reuse the exact recurrent state, then
            # prefill the final token for the first logits.
            if _hybrid_mode and _text_pc is not None and int(_prefill_ids.shape[0]) > 1:
                try:
                    _resident = self._capture_vlm_hybrid_prefix(
                        lm,
                        input_ids,
                        cache,
                        _text_pc,
                        _pc_matched,
                        self._text_hybrid_block,
                    )
                    _prefill_ids = input_ids[_resident:]
                except Exception:
                    logger.warning(
                        "VLM hybrid prefix capture failed — full prefill", exc_info=True
                    )
            _pf_t0 = time.perf_counter()
            output = lm(_prefill_ids[None], cache=cache)
            logits = output.logits[:, -1, :]
            current = sampler(logits)
            mx.eval(current)
            # Feed cold-prefill throughput to the text KV cache so its SSD tier can
            # auto-gate fast-prefill VLMs : GLM-OCR prefills ~6300 t/s,
            # for which restoring a prefix from disk is SLOWER than re-prefilling
            # (F-SSD 0.93×). Only on a cold prefill (matched==0) of a real prompt.
            if _text_pc is not None and _pc_matched == 0:
                try:
                    _n_pf = int(_prefill_ids.shape[0])
                    _pf_dt = time.perf_counter() - _pf_t0
                    if _n_pf >= 256 and _pf_dt > 0:
                        _text_pc.note_prefill_tps(_n_pf / _pf_dt)
                except Exception:
                    pass

            # store the prompt-boundary KV snapshot NOW, before the
            # decode loop pollutes the cache. CRITICAL for sliding-window models
            # (gemma RotatingKVCache): once generation rotates the window, the
            # prompt's tail KV is evicted and trimming back to prompt length
            # cannot restore it — a post-generation add() would store a corrupt
            # entry (verified: lossless for short answers, wrong for long ones).
            # add() makes a detached copy (trim=0 here since offset==prompt_len),
            # so subsequent decode on the live `cache` doesn't touch the entry.
            if _text_pc is not None:
                try:
                    _text_pc.add(input_ids, cache)
                except Exception:
                    logger.debug("VLM text KV prefix add failed", exc_info=True)

            tokens = [current.item()]
            if current.item() in stop_ids:
                return (
                    self._tokenizer.decode(tokens, skip_special_tokens=True),
                    0,
                    len(tokens),
                    True,
                    False,
                    int(_pc_matched),
                )

            _in_thinking = False
            _thinking_tokens = 0
            _stop_hit = False
            _budget_hit = False
            _think_single_token = False
            try:
                _ts_ids = self._tokenizer.encode("<think")
                _te_ids = self._tokenizer.encode("</think")
                if len(_ts_ids) == 1 and len(_te_ids) == 1:
                    think_start_id = _ts_ids[0]
                    think_end_id = _te_ids[0]
                    _think_single_token = True
                else:
                    think_start_id = think_end_id = None
            except Exception:
                logger.debug("operation failed", exc_info=True)
                think_start_id = think_end_id = None
                _think_single_token = False

            _thinking_text = ""  # for text-based thinking detection
            for _step_idx in range(max_tokens - 1):
                # Check cancellation every 16 steps to reduce overhead
                if _step_idx % 16 == 0 and _is_cancelled():
                    break

                output = lm(current[None], cache=cache)
                logits = output.logits[:, -1, :]

                if has_penalty:
                    if repetition_penalty != 1.0:
                        ctx = list(set(tokens))
                        sel = logits[..., ctx]
                        sel = mx.where(
                            sel < 0, sel * repetition_penalty, sel / repetition_penalty
                        )
                        logits[..., mx.array(ctx)] = sel
                    if frequency_penalty != 0.0:
                        for tid in set(tokens):
                            logits = logits.at[..., tid].add(
                                -frequency_penalty * tokens.count(tid)
                            )
                    if presence_penalty != 0.0:
                        for tid in set(tokens):
                            logits = logits.at[..., tid].add(-presence_penalty)
                    if logit_bias:
                        for tid, bias in logit_bias.items():
                            logits = logits.at[..., tid].add(bias)

                # JSON schema constraint masking
                if json_constraint is not None and tokens:
                    try:
                        allowed = json_constraint.get_allowed_tokens(
                            self._tokenizer, tokens
                        )
                        if allowed:
                            from .json_schema import apply_json_constraint

                            logits = apply_json_constraint(logits, allowed)
                    except Exception:
                        logger.debug("failed", exc_info=True)

                current = sampler(logits)
                mx.eval(current)
                tok_id = current.item()
                tokens.append(tok_id)

                # Advance JSON constraint state with the new token text
                if json_constraint is not None:
                    try:
                        token_text = self._tokenizer.decode([tok_id])
                        json_constraint.advance(token_text)
                    except Exception:
                        logger.debug("json constraint advance failed", exc_info=True)
                # Track thinking segment boundaries
                if _think_single_token and think_start_id is not None:
                    if not _in_thinking and tok_id == think_start_id:
                        _in_thinking = True
                    elif _in_thinking:
                        if tok_id == think_end_id:
                            _in_thinking = False
                        else:
                            _thinking_tokens += 1
                elif not _think_single_token:
                    # Text-based thinking detection for multi-token encodings
                    _tok_text = self._tokenizer.decode([tok_id])
                    _thinking_text += _tok_text
                    if not _in_thinking and _thinking_text.endswith("<think"):
                        _in_thinking = True
                    elif _in_thinking:
                        if _thinking_text.endswith("</think"):
                            _in_thinking = False
                        else:
                            _thinking_tokens += 1
                # Thinking budget enforcement — cap thinking tokens
                if (
                    thinking_budget is not None
                    and _in_thinking
                    and _thinking_tokens >= thinking_budget
                ):
                    # Append closing tag to keep output well-formed
                    if _think_single_token and think_end_id is not None:
                        tokens.append(think_end_id)
                    _budget_hit = True
                    break
                if tok_id in stop_ids:
                    _stop_hit = True
                    break

        text = self._tokenizer.decode(tokens, skip_special_tokens=True)
        # Handle multi-token stop sequences: check if decoded text ends with any suffix
        if stop_suffixes and not _stop_hit:
            for s in stop_suffixes:
                if text.endswith(s):
                    text = text[: -len(s)]
                    _stop_hit = True
                    break
        return (
            text,
            _thinking_tokens,
            len(tokens),
            _stop_hit,
            _budget_hit,
            int(_pc_matched),
        )

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
        thinking_budget: int | None = None,
        _ttft_t0: list | None = None,
        _ttft_recorded: list | None = None,
        _ttft_val: list | None = None,
        seed: int | None = None,
    ) -> None:
        """Streaming vision + text generation using mlx_vlm.stream_generate().

        Passes vision_cache and prompt_cache_state to mlx_vlm so that:
        1. Image features are cached and reused when the same image appears again
        2. KV cache is reused for common prefix across conversations with same image
        """
        from mlx_vlm.generate import stream_generate as vlm_stream_generate

        # Apply chat template with caching to skip re-processing for repeated messages
        num_audios = len(audio_paths) if audio_paths else 0
        prompt = self._apply_vlm_template_with_cache(
            messages,
            enable_thinking=enable_thinking,
            num_audios=num_audios,
            max_images=len(image_paths) if image_paths else None,
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

        sampler = _build_noncached_sampler(temperature, top_p, top_k, min_p, seed)
        stop_suffixes = stop or []
        token_count = 0
        accumulated = ""  # Accumulate text for multi-token stop suffix matching
        _emitted_pos = 0  # Track how much of accumulated has been emitted
        _in_thinking = False  # Track thinking state for current_state routing
        _thinking_token_count = 0  # Count tokens generated while in thinking state
        _think_scan_pos = 0  # Cursor for scanning thinking tags (avoids re-scanning already-seen text)
        _num_prompt_tokens = 0
        # Estimate prompt tokens for output metadata.
        # VLM models receive both text tokens and image placeholder tokens,
        # so we add a per-image estimate from the vision config (if available)
        # to avoid severely undercounting prompt_tokens when images are present.
        if self._tokenizer is not None:
            try:
                _num_prompt_tokens = len(self._tokenizer.encode(prompt))
            except Exception:
                logger.debug(
                    "prompt token estimation failed in stream_vlm_vision", exc_info=True
                )
        if image_paths:
            _tokens_per_image = self._estimate_image_tokens()
            _num_prompt_tokens += _tokens_per_image * len(image_paths)
        _cached_tokens = 0
        if (
            kv_prefix_state is not None
            and kv_prefix_state.token_ids is not None
            and self._tokenizer is not None
        ):
            try:
                _cached_tokens = kv_prefix_state.find_prefix_length(
                    self._tokenizer.encode(prompt)
                )
            except Exception:
                logger.debug("KV prefix length computation failed", exc_info=True)
        try:
            stream_kwargs: dict = {
                "max_tokens": max_tokens,
                "sampler": sampler,
            }
            if image_paths:
                stream_kwargs["image"] = (
                    image_paths if len(image_paths) > 1 else image_paths[0]
                )
            if audio_paths:
                stream_kwargs["audio"] = self._audio_arg(audio_paths)

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
                if cancel_event is not None and (
                    cancel_event._value
                    if isinstance(cancel_event, asyncio.Event)
                    else cancel_event.is_set()
                ):
                    queue.put_nowait(
                        RequestOutput(
                            request_id=req_id,
                            new_text="",
                            finish_reason="cancel",
                            finished=True,
                            completion_tokens=token_count,
                            prompt_tokens=_num_prompt_tokens,
                            current_state="reasoning" if _in_thinking else "normal",
                            reasoning_tokens=_thinking_token_count,
                        )
                    )
                    return
                token_count += 1
                # Record TTFT on first token
                if (
                    _ttft_t0 is not None
                    and _ttft_recorded is not None
                    and not _ttft_recorded[0]
                ):
                    _ttft_recorded[0] = True
                    _ttft_val[0] = time.perf_counter() - _ttft_t0[0]
                text = result.text if hasattr(result, "text") else ""
                accumulated += text
                # Track thinking state from text markers — scan only the
                # newly-appended portion to avoid permanent matches on tags
                # that appeared earlier in the accumulated text.
                _prev_in_thinking = _in_thinking
                _scan = accumulated[_think_scan_pos:]
                while _scan:
                    if _in_thinking:
                        # use _find_think_tag (guards the following
                        # char) so the streaming scanner doesn't false-match <thinking>/
                        # <think_more> like the raw .find did — matching the non-streaming
                        # vision path and avoiding spurious reasoning state.
                        idx = self._find_think_tag(_scan, "</think")
                        if idx >= 0:
                            _in_thinking = False
                            _scan = _scan[idx + len("</think") :]
                            _think_scan_pos = len(accumulated) - len(_scan)
                        else:
                            break
                    else:
                        idx = self._find_think_tag(_scan, "<think")
                        if idx >= 0:
                            _in_thinking = True
                            _scan = _scan[idx + len("<think") :]
                            _think_scan_pos = len(accumulated) - len(_scan)
                        else:
                            break
                _think_scan_pos = len(accumulated) - len(_scan)
                _cur_state = "reasoning" if _in_thinking else "normal"
                # Count tokens generated while in thinking state.
                # Skip counting on the token that triggered a state transition
                # (<think/</think tags themselves are not reasoning content).
                if _in_thinking and _prev_in_thinking:
                    _thinking_token_count += 1
                # Thinking budget enforcement — stop generation when the budget
                # is exceeded while in a thinking segment.  Flush held-back
                # text before emitting the terminal output.
                if (
                    thinking_budget is not None
                    and _in_thinking
                    and _thinking_token_count >= thinking_budget
                ):
                    _budget_state = "reasoning" if _in_thinking else "normal"
                    # Flush any held-back text before the terminal output
                    _held_budget_text = accumulated[_emitted_pos:]
                    # force-close the <think> tag so the truncated reasoning
                    # terminates cleanly (the text path injects </think> on budget exhaustion;
                    # the VLM path returned mid-<think> with no answer and no closing marker).
                    _budget_close = "</think>" if _budget_state == "reasoning" else ""
                    if _held_budget_text or _budget_close:
                        queue.put_nowait(
                            RequestOutput(
                                request_id=req_id,
                                new_text=_held_budget_text + _budget_close,
                                finish_reason=None,
                                finished=False,
                                prompt_tokens=_num_prompt_tokens,
                                current_state=_budget_state,
                            )
                        )
                    queue.put_nowait(
                        RequestOutput(
                            request_id=req_id,
                            new_text="",
                            # budget exhaustion is a LENGTH limit, not a natural stop —
                            # match the text path (was "stop", inconsistent across modalities).
                            finish_reason="length",
                            finished=True,
                            completion_tokens=token_count,
                            prompt_tokens=_num_prompt_tokens,
                            current_state=_budget_state,
                            reasoning_tokens=_thinking_token_count,
                        )
                    )
                    return
                finish_reason = None
                if hasattr(result, "finish_reason") and result.finish_reason:
                    finish_reason = result.finish_reason
                elif token_count >= max_tokens:
                    finish_reason = "length"
                # Check multi-token stop suffixes against accumulated text.
                # Search from a position that accounts for the longest suffix
                # length — a stop suffix may straddle the emit boundary (part
                # in already-emitted text, part in held-back text), so we must
                # look back far enough to catch it.
                if not finish_reason and stop_suffixes:
                    _max_suffix_len = max(len(s) for s in stop_suffixes)
                    _search_start = max(0, _emitted_pos - _max_suffix_len + 1)
                    _search_region = accumulated[_search_start:]
                    for s in stop_suffixes:
                        if s in _search_region:
                            # Full stop sequence found — trim it and everything after
                            idx = accumulated.find(s, _search_start)
                            # Don't trim into already-emitted text; only trim if
                            # the suffix starts at or after the emit boundary.
                            if idx < _emitted_pos:
                                # Suffix straddles emit boundary — clamp to avoid
                                # losing already-emitted text. The suffix itself is
                                # still detected and generation stops.
                                idx = _emitted_pos
                            accumulated = accumulated[:idx]
                            finish_reason = "stop"
                            # Reset thinking scan cursor since accumulated text
                            # was trimmed — old cursor may point into deleted
                            # portion where a partial <think or </think fragment
                            # was being tracked. Force a full rescan from the
                            # new end of the string.
                            _think_scan_pos = len(accumulated)
                            break

                # Compute the safe-to-emit text: everything up to _emitted_pos
                if finish_reason == "stop":
                    # Emit all held-back text up to the stop position.
                    # accumulated has already been trimmed (stop suffix removed),
                    # so everything from _emitted_pos to end is safe to emit.
                    _emit_text = accumulated[_emitted_pos:]
                    _emitted_pos = len(accumulated)
                else:
                    # Compute safe emit boundary — don't emit text that could be
                    # a partial prefix of a stop sequence.
                    _safe_end = len(accumulated)
                    _pending = accumulated[_emitted_pos:]
                    for s2 in stop_suffixes:
                        _max_hold = min(len(s2) - 1, len(_pending))
                        for _hold_len in range(1, _max_hold + 1):
                            if s2.startswith(_pending[-_hold_len:]):
                                _safe_end = min(_safe_end, len(accumulated) - _hold_len)
                                break
                    _emit_text = (
                        accumulated[_emitted_pos:_safe_end] if accumulated else ""
                    )
                    _emitted_pos = _safe_end

                _reasoning_tok = _thinking_token_count
                queue.put_nowait(
                    RequestOutput(
                        request_id=req_id,
                        new_text=_emit_text,
                        finish_reason=finish_reason,
                        finished=finish_reason is not None,
                        completion_tokens=token_count,
                        prompt_tokens=_num_prompt_tokens,
                        current_state=_cur_state,
                        reasoning_tokens=_reasoning_tok,
                        cached_tokens=_cached_tokens if token_count == 1 else 0,
                        ttft_ms=round(_ttft_val[0] * 1000, 1)
                        if _ttft_val is not None and _ttft_val[0] > 0
                        else 0.0,
                    )
                )
                if finish_reason:
                    return

            # Generator exhausted without a finish_reason — emit finished output.
            # This handles the case where vlm_stream_generate stops yielding
            # without setting result.finish_reason and token_count < max_tokens.
            # Always emit finished=True so the consumer never hangs waiting for
            # a final output, even when zero tokens were generated.
            #
            # Flush any held-back text (from stop suffix prefix detection)
            # before emitting the terminal output so text isn't lost.
            _exhausted_state = "reasoning" if _in_thinking else "normal"
            _held_text = accumulated[_emitted_pos:]
            if _held_text:
                queue.put_nowait(
                    RequestOutput(
                        request_id=req_id,
                        new_text=_held_text,
                        finish_reason=None,
                        finished=False,
                        prompt_tokens=_num_prompt_tokens,
                        current_state=_exhausted_state,
                    )
                )
            queue.put_nowait(
                RequestOutput(
                    request_id=req_id,
                    new_text="",
                    finish_reason="length" if token_count > 0 else "stop",
                    finished=True,
                    completion_tokens=token_count,
                    prompt_tokens=_num_prompt_tokens,
                    current_state=_exhausted_state,
                    reasoning_tokens=_thinking_token_count,
                )
            )
            # Post-streaming bookkeeping (cache stats, encoder cache, KV prefix).
            # Note: this only runs on the generator-exhausted path.  The
            # finally block below handles bookkeeping for ALL exit paths
            # (cancel, stop, budget, error).
        except Exception as e:
            _error_state = "reasoning" if _in_thinking else "normal"
            # Flush any held-back accumulated text before emitting the error
            # output. Without this, text that was buffered for stop suffix
            # prefix detection is silently lost on error.
            _held_error_text = accumulated[_emitted_pos:] if accumulated else ""
            if _held_error_text:
                # queue full or closed — best effort flush
                with contextlib.suppress(Exception):
                    queue.put_nowait(
                        RequestOutput(
                            request_id=req_id,
                            new_text=_held_error_text,
                            finish_reason=None,
                            finished=False,
                            prompt_tokens=_num_prompt_tokens,
                            current_state=_error_state,
                        )
                    )
            queue.put_nowait(
                RequestOutput(
                    request_id=req_id,
                    new_text="",
                    finish_reason="error",
                    finished=True,
                    completion_tokens=token_count,
                    prompt_tokens=_num_prompt_tokens,
                    error=str(e),
                    current_state=_error_state,
                    reasoning_tokens=_thinking_token_count,
                )
            )
        finally:
            # Always run bookkeeping regardless of how the stream ended:
            # cancel, stop suffix, thinking budget, error, or exhaustion.
            # This ensures vision cache stats, encoder cache entries, and KV
            # prefix states are updated even when the loop exits early.
            try:
                if self._vision_cache is not None:
                    vc_stats_after = self._vision_cache.stats
                    new_hits = max(
                        0,
                        vc_stats_after.get("hits", 0) - vc_stats_before.get("hits", 0),
                    )
                    if new_hits > 0:
                        self._vlm_vision_hits += new_hits
                        logger.debug(
                            "VLM stream vision feature cache hit: reused encoded image"
                        )
                    # per-image miss count (was +1 for any multi-image miss).
                    _misses = max(
                        0, (len(image_paths) if image_paths else 0) - new_hits
                    )
                    if _misses > 0:
                        self._vlm_vision_misses += _misses

                # Encoder cache: only store real encoder outputs, not markers.
                # The vision_cache adapter already handles image feature caching.
                # Do NOT store True markers — they waste memory and mislead stats.
                # (Actual encoder_outputs would be captured from stream results
                # if vlm_stream_generate exposed them.)

                # After streaming, save KV prefix state for this image
                if image_hash is not None:
                    if kv_prefix_state is not None:
                        # State was already updated by stream_generate
                        pass
                    else:
                        # First time seeing this image — create a state entry
                        self._ensure_kv_prefix_state(image_hash)
            except Exception:
                logger.debug("post-stream bookkeeping failed", exc_info=True)

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
        thinking_budget: int | None = None,
        _ttft_t0: list | None = None,
        _ttft_recorded: list | None = None,
        _ttft_val: list | None = None,
        seed: int | None = None,
    ) -> None:
        """Streaming text generation for VLM models."""
        from mlx_lm.generate import generation_stream
        from mlx_vlm.models.cache import make_prompt_cache

        # Clear mRoPE state to prevent contamination from prior VLM request
        if self._mrope_info and self._mrope_info.enabled:
            from .mrope import clear_rope_state

            clear_rope_state(self._model)

        lm = self._model.language_model
        cache = make_prompt_cache(lm)
        sampler = _build_noncached_sampler(temperature, top_p, top_k, min_p, seed)
        eos_ids = self._get_eos_ids()
        has_penalty = (
            repetition_penalty != 1.0
            or frequency_penalty != 0.0
            or presence_penalty != 0.0
            or logit_bias
        )

        # streaming text KV prefix reuse — same gate as the
        # non-streaming path (lossless-only bypass via _text_prefix_reuse_safe,
        # explicit mRoPE positions below). Both paths share YUNSHU_VLM_KV_PREFIX.
        _text_pc = (
            self._text_kv_prefix_cache if self._text_prefix_reuse_safe(lm) else None
        )

        # JSON schema / grammar constraint (streaming path)
        json_constraint = None
        if json_schema is not None:
            try:
                if isinstance(json_schema, dict) and json_schema.get("type") in (
                    "regex",
                    "choice",
                    "cfg",
                ):
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
                        json_constraint = ConstraintFactory.create(
                            gtype, grammar, self._tokenizer
                        )
                elif isinstance(json_schema, str) and json_schema == "json_object":
                    from .json_schema import JsonSchemaConstraint

                    json_constraint = JsonSchemaConstraint(None)
                else:
                    from .json_schema import JsonSchemaConstraint

                    json_constraint = JsonSchemaConstraint(json_schema)
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

        # Multi-token stop hold-back: withhold any streamed text that could be
        # the start of a (multi-token) stop string so its prefix never leaks
        # before the match completes. This is the real VLM text path; without
        # the buffer it emitted each token immediately and only trimmed the stop
        # on the COMPLETING token, leaking the earlier tokens' prefix
        # (same class as the chat fast-path fix). single-token stops
        # are handled via stop_ids (caught before emit), so the buffer only
        # needs the multi-token stop_suffixes; empty → pure passthrough.
        from .text_utils import StopHoldbackBuffer

        _hb = StopHoldbackBuffer(stop_suffixes)

        has_detokenizer = hasattr(self._tokenizer, "detokenizer")
        detokenizer = None
        if has_detokenizer:
            detokenizer = self._tokenizer.detokenizer
            detokenizer.reset()

        _num_prompt_tokens = len(input_ids)

        with mx.stream(generation_stream):
            # Seed stream PRNG inside the stream context — see _generate_vlm_text
            if seed is not None:
                mx.random.seed(int(seed) & ((1 << 63) - 1))

            # KV prefix lookup (gated). Reuse a cached prefix's KV.
            _prefill_ids = input_ids
            _pc_matched = 0
            if _text_pc is not None and len(input_ids) >= 32:
                try:
                    _cached_kv, _, _pc_matched = _text_pc.get(input_ids)
                    if _cached_kv is not None and _pc_matched > 0:
                        cache = _cached_kv
                        _prefill_ids = input_ids[_pc_matched:]
                        if len(_prefill_ids) == 0:
                            _refeed = min(len(input_ids) - 1, 128)
                            cache = _text_pc._snapshot_cache(cache, trim=_refeed)
                            _prefill_ids = input_ids[-_refeed:]
                            _pc_matched = len(input_ids) - _refeed
                        logger.debug(
                            "VLM stream text KV prefix hit: matched=%d/%d",
                            _pc_matched,
                            len(input_ids),
                        )
                    else:
                        _pc_matched = 0
                except Exception:
                    logger.warning(
                        "VLM stream text KV prefix get failed — full prefill",
                        exc_info=True,
                    )
                    cache = make_prompt_cache(lm)
                    _prefill_ids = input_ids
                    _pc_matched = 0

            # prime mRoPE native rope state for reuse (see _generate_vlm_text).
            if _text_pc is not None and self._text_reuse_needs_positions(lm):
                self._prime_mrope_reuse_state(lm)

            # Prefill
            _pf_t0 = time.perf_counter()
            output = lm(_prefill_ids[None], cache=cache)
            logits = output.logits[:, -1, :]
            current = sampler(logits)
            mx.eval(current)
            # Cold-prefill throughput → SSD auto-gate (see _generate_vlm_text)
            if _text_pc is not None and _pc_matched == 0:
                try:
                    _n_pf = int(_prefill_ids.shape[0])
                    _pf_dt = time.perf_counter() - _pf_t0
                    if _n_pf >= 256 and _pf_dt > 0:
                        _text_pc.note_prefill_tps(_n_pf / _pf_dt)
                except Exception:
                    pass
            token_count = 1

            # store the prompt-boundary snapshot NOW (before decode
            # pollutes the rotating window) — see _generate_vlm_text. This is
            # what makes streaming reuse lossless for long generations too.
            if _text_pc is not None:
                try:
                    _text_pc.add(input_ids, cache)
                except Exception:
                    logger.debug("VLM stream text KV prefix add failed", exc_info=True)

            # Record TTFT on first token (prefill complete)
            if (
                _ttft_t0 is not None
                and _ttft_recorded is not None
                and not _ttft_recorded[0]
            ):
                _ttft_recorded[0] = True
                _ttft_val[0] = time.perf_counter() - _ttft_t0[0]

            _in_thinking = False
            _thinking_tokens = 0
            _think_single_token = False
            try:
                _ts_ids = self._tokenizer.encode("<think")
                _te_ids = self._tokenizer.encode("</think")
                if len(_ts_ids) == 1 and len(_te_ids) == 1:
                    think_start_id = _ts_ids[0]
                    think_end_id = _te_ids[0]
                    _think_single_token = True
                else:
                    think_start_id = think_end_id = None
            except Exception:
                logger.debug("operation failed", exc_info=True)
                think_start_id = think_end_id = None
                _think_single_token = False

            _thinking_text = ""  # for text-based thinking detection
            token_id = current.item()
            is_eos = token_id in stop_ids
            finish_reason = "stop" if is_eos else None

            _seg = ""
            _suffix_hit = False
            if not is_eos:
                if has_detokenizer:
                    detokenizer.add_token(token_id)
                    _seg = detokenizer.last_segment
                    if stop_suffixes and any(
                        detokenizer.text.endswith(s) for s in stop_suffixes
                    ):
                        _suffix_hit = True
                else:
                    _seg = self._tokenizer.decode([token_id], skip_special_tokens=True)

            # Track thinking state for gateway routing
            # If the first token is the think-start token, mark _in_thinking so the
            # loop's thinking budget and state tracking work correctly.
            if (
                _think_single_token
                and think_start_id is not None
                and token_id == think_start_id
            ):
                _in_thinking = True
            elif not _think_single_token:
                # Text-based thinking detection for multi-token encodings
                _thinking_text += self._tokenizer.decode([token_id])
                if _thinking_text.endswith("<think"):
                    _in_thinking = True
            _state = "reasoning" if _in_thinking else "normal"

            finish_reason = "stop" if (is_eos or _suffix_hit) else None
            # Route text through the hold-back buffer (multi-token stop prefixes
            # are withheld; on a string-stop the matched stop is dropped).
            if _suffix_hit:
                # feed() RELEASES the text before the stop and removes it from the
                # buffer; take_stopped() returns only what remains up to the stop. When the
                # stop completes INSIDE a content-bearing token (e.g. "goodbyeEND" with stop
                # "END"), feed() returns "goodbye" — discarding it permanently lost it from
                # the append-only SSE stream. Emit BOTH (the text engine does this correctly).
                token_text = _hb.feed(_seg) + _hb.take_stopped()
            elif is_eos:
                token_text = _hb.flush()
            else:
                token_text = _hb.feed(_seg)

            if token_text or finish_reason:
                queue.put_nowait(
                    RequestOutput(
                        request_id=req_id,
                        new_text=token_text,
                        new_token_ids=[token_id],
                        finish_reason=finish_reason,
                        finished=finish_reason is not None,
                        completion_tokens=token_count,
                        prompt_tokens=_num_prompt_tokens,
                        current_state=_state,
                        reasoning_tokens=_thinking_tokens,
                        ttft_ms=round(_ttft_val[0] * 1000, 1)
                        if _ttft_val is not None and _ttft_val[0] > 0
                        else 0.0,
                    )
                )
            if finish_reason:
                return

            tokens_list = [
                token_id
            ]  # Include the first prefill token for JSON constraint
            try:
                for _ in range(max_tokens - 1):
                    if cancel_event is not None and (
                        cancel_event._value
                        if isinstance(cancel_event, asyncio.Event)
                        else cancel_event.is_set()
                    ):
                        # Flush remaining detokenizer bytes + buffered tail before cancelling
                        if has_detokenizer:
                            detokenizer.finalize()
                            remaining = _hb.feed(detokenizer.last_segment) + _hb.flush()
                            if remaining:
                                queue.put_nowait(
                                    RequestOutput(
                                        request_id=req_id,
                                        new_text=remaining,
                                        finish_reason=None,
                                        finished=False,
                                        prompt_tokens=_num_prompt_tokens,
                                    )
                                )
                        queue.put_nowait(
                            RequestOutput(
                                request_id=req_id,
                                new_text="",
                                finish_reason="cancel",
                                finished=True,
                                completion_tokens=token_count,
                                prompt_tokens=_num_prompt_tokens,
                                reasoning_tokens=_thinking_tokens,
                            )
                        )
                        return
                    output = lm(current[None], cache=cache)
                    logits = output.logits[:, -1, :]

                    if has_penalty:
                        if repetition_penalty != 1.0:
                            ctx = list(set(tokens_list))
                            sel = logits[..., ctx]
                            sel = mx.where(
                                sel < 0,
                                sel * repetition_penalty,
                                sel / repetition_penalty,
                            )
                            logits[..., mx.array(ctx)] = sel
                        if frequency_penalty != 0.0:
                            for tid in set(tokens_list):
                                logits = logits.at[..., tid].add(
                                    -frequency_penalty * tokens_list.count(tid)
                                )
                        if presence_penalty != 0.0:
                            for tid in set(tokens_list):
                                logits = logits.at[..., tid].add(-presence_penalty)
                        if logit_bias:
                            for tid, bias in logit_bias.items():
                                logits = logits.at[..., tid].add(bias)

                    # JSON schema constraint masking
                    if json_constraint is not None:
                        try:
                            allowed = json_constraint.get_allowed_tokens(
                                self._tokenizer, tokens_list
                            )
                            if allowed:
                                from .json_schema import apply_json_constraint

                                logits = apply_json_constraint(logits, allowed)
                        except Exception:
                            logger.debug("failed", exc_info=True)

                    current = sampler(logits)
                    mx.eval(current)
                    token_count += 1

                    token_id = current.item()
                    tokens_list.append(token_id)

                    # Advance JSON constraint state with the new token text
                    if json_constraint is not None:
                        try:
                            _tok_text = self._tokenizer.decode([token_id])
                            json_constraint.advance(_tok_text)
                        except Exception:
                            logger.debug(
                                "json constraint advance failed (stream)", exc_info=True
                            )
                    # Track thinking segment boundaries in VLM streaming
                    if _think_single_token and think_start_id is not None:
                        if not _in_thinking and token_id == think_start_id:
                            _in_thinking = True
                        elif _in_thinking:
                            if token_id == think_end_id:
                                _in_thinking = False
                            else:
                                _thinking_tokens += 1
                    elif not _think_single_token:
                        # Text-based thinking detection for multi-token encodings
                        _tok_text_vlm = self._tokenizer.decode([token_id])
                        _thinking_text += _tok_text_vlm
                        if not _in_thinking and _thinking_text.endswith("<think"):
                            _in_thinking = True
                        elif _in_thinking:
                            if _thinking_text.endswith("</think"):
                                _in_thinking = False
                            else:
                                _thinking_tokens += 1
                    # Thinking budget enforcement in VLM streaming
                    if (
                        thinking_budget is not None
                        and _in_thinking
                        and _thinking_tokens >= thinking_budget
                    ):
                        _state = "reasoning" if _in_thinking else "normal"
                        if has_detokenizer:
                            detokenizer.finalize()
                            remaining = _hb.feed(detokenizer.last_segment) + _hb.flush()
                            if remaining:
                                queue.put_nowait(
                                    RequestOutput(
                                        request_id=req_id,
                                        new_text=remaining,
                                        finish_reason=None,
                                        finished=False,
                                        current_state=_state,
                                        prompt_tokens=_num_prompt_tokens,
                                    )
                                )
                        queue.put_nowait(
                            RequestOutput(
                                request_id=req_id,
                                new_text="",
                                finish_reason="stop",
                                finished=True,
                                completion_tokens=token_count,
                                current_state=_state,
                                prompt_tokens=_num_prompt_tokens,
                                reasoning_tokens=_thinking_tokens,
                            )
                        )
                        return
                    is_eos = token_id in stop_ids
                    _seg = ""
                    suffix_hit = False
                    if not is_eos:
                        if has_detokenizer:
                            detokenizer.add_token(token_id)
                            _seg = detokenizer.last_segment
                            # Detect stop on the full decoded text (a multi-token
                            # stop only completes once all its tokens have arrived).
                            if stop_suffixes and any(
                                detokenizer.text.endswith(s) for s in stop_suffixes
                            ):
                                suffix_hit = True
                        else:
                            _seg = self._tokenizer.decode(
                                [token_id], skip_special_tokens=True
                            )

                    finish_reason = "stop" if (is_eos or suffix_hit) else None
                    _state = "reasoning" if _in_thinking else "normal"

                    # Route this token's text through the hold-back buffer so a
                    # multi-token stop never leaks its prefix. On a string-stop
                    # the completing token is fed then take_stopped() drops the
                    # matched stop (and any held prefix that belonged to it); on
                    # an EOS token the held text is genuine output → flush it.
                    if suffix_hit:
                        # emit feed()'s pre-stop return too (else content fused with
                        # the stop token, e.g. "goodbyeEND", is silently lost). See first site.
                        token_text = _hb.feed(_seg) + _hb.take_stopped()
                    elif is_eos:
                        token_text = _hb.flush()
                    else:
                        token_text = _hb.feed(_seg)

                    if token_text or finish_reason:
                        queue.put_nowait(
                            RequestOutput(
                                request_id=req_id,
                                new_text=token_text,
                                new_token_ids=[token_id],
                                finish_reason=finish_reason,
                                finished=finish_reason is not None,
                                completion_tokens=token_count,
                                prompt_tokens=_num_prompt_tokens,
                                current_state=_state,
                                reasoning_tokens=_thinking_tokens,
                            )
                        )

                    if finish_reason:
                        # On a string-stop the match was already dropped; only an
                        # EOS-terminated stream may have genuine trailing bytes.
                        if has_detokenizer and not suffix_hit:
                            detokenizer.finalize()
                            remaining = detokenizer.last_segment
                            tail = (
                                (_hb.feed(remaining) + _hb.flush()) if remaining else ""
                            )
                            if tail:
                                queue.put_nowait(
                                    RequestOutput(
                                        request_id=req_id,
                                        new_text=tail,
                                        finish_reason=None,
                                        finished=False,
                                        prompt_tokens=_num_prompt_tokens,
                                    )
                                )
                        return

                # Max tokens reached — finalize detokenizer + flush buffered tail
                _final_state = "reasoning" if _in_thinking else "normal"
                if has_detokenizer:
                    detokenizer.finalize()
                    remaining = _hb.feed(detokenizer.last_segment) + _hb.flush()
                    if remaining:
                        queue.put_nowait(
                            RequestOutput(
                                request_id=req_id,
                                new_text=remaining,
                                finish_reason=None,
                                finished=False,
                                prompt_tokens=_num_prompt_tokens,
                                current_state=_final_state,
                            )
                        )
                queue.put_nowait(
                    RequestOutput(
                        request_id=req_id,
                        new_text="",
                        finish_reason="length",
                        finished=True,
                        completion_tokens=token_count,
                        prompt_tokens=_num_prompt_tokens,
                        current_state=_final_state,
                        reasoning_tokens=_thinking_tokens,
                    )
                )
            except Exception as e:
                logger.error(f"VLM text streaming error: {e}", exc_info=True)
                _error_state = "reasoning" if _in_thinking else "normal"
                # Flush remaining detokenizer bytes before reporting error
                if has_detokenizer and detokenizer is not None:
                    try:
                        detokenizer.finalize()
                        remaining = detokenizer.last_segment
                        if remaining:
                            queue.put_nowait(
                                RequestOutput(
                                    request_id=req_id,
                                    new_text=remaining,
                                    finish_reason=None,
                                    finished=False,
                                    prompt_tokens=_num_prompt_tokens,
                                    current_state=_error_state,
                                )
                            )
                    except Exception:
                        logger.debug(
                            "detokenizer finalize in error handler failed",
                            exc_info=True,
                        )
                queue.put_nowait(
                    RequestOutput(
                        request_id=req_id,
                        new_text="",
                        finish_reason="error",
                        finished=True,
                        error=str(e),
                        prompt_tokens=_num_prompt_tokens,
                        current_state=_error_state,
                        reasoning_tokens=_thinking_tokens,
                    )
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

    def _format_prompt(
        self, messages: list[dict], enable_thinking: bool | None = None
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
            except Exception:
                logger.debug("chat template failed, using fallback", exc_info=True)

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
    ) -> mx.array:
        """Format prompt and tokenize, using the text prompt cache to skip work.

        Caches the tokenizer.encode() result keyed by a stable hash of the
        message content. On cache hit, skips _format_prompt() + encode().
        """
        cache_key = _VLMTextPromptCache._compute_messages_hash(
            messages,
            enable_thinking,
        )

        cached_ids = self._text_prompt_cache.get_token_ids(cache_key)
        if cached_ids is not None:
            logger.debug("VLM text prompt cache hit: %d tokens", len(cached_ids))
            return mx.array(cached_ids)

        prompt_text = self._format_prompt(messages, enable_thinking=enable_thinking)
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

        template_text = self._processor.apply_chat_template(
            vlm_messages,
            **tpl_kwargs,
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
        Thread-safe: acquires _kv_prefix_lock.
        """
        if image_hash is None:
            return None

        with self._kv_prefix_lock:
            state = self._kv_prefix_states.get(image_hash)
        if state is not None:
            self._vlm_kv_prefix_hits += 1
            return state

        self._vlm_kv_prefix_misses += 1
        return None

    def _ensure_kv_prefix_state(self, image_hash: str) -> Any:
        """Create a new PromptCacheState entry for this image hash.

        Thread-safe: acquires _kv_prefix_lock to protect concurrent
        reads/writes/evictions from the MLX executor thread.
        """
        from mlx_vlm.generate import PromptCacheState

        state = PromptCacheState()
        with self._kv_prefix_lock:
            # Evict old entries if over limit
            if len(self._kv_prefix_states) >= self._kv_prefix_max_entries:
                keys = list(self._kv_prefix_states.keys())
                for k in keys[:8]:
                    del self._kv_prefix_states[k]
            self._kv_prefix_states[image_hash] = state
        return state

    # ── Audio Extraction ──

    def _audio_arg(self, audio_paths: list[str]):
        """Convert audio file PATHS into loaded float32 sample arrays.

        mlx_vlm's `process_inputs` passes the `audio` argument straight to the
        model processor without loading it, so handing it a path string makes
        Qwen3-Omni (and other audio VLMs) try to treat the path as samples →
        "could not convert string to float: '/…/tmp.wav'". We must pre-load the
        path into an ndarray via mlx_vlm.load_audio at the model's sample rate.
        (2nd-pass live fix.) Returns a single array when there is
        one audio, else a list.
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
        arrays = [load_audio(p, sr) for p in audio_paths]
        return arrays if len(arrays) > 1 else arrays[0]

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
            _cap = int(
                os.environ.get("YUNSHU_VLM_MAX_IMAGE_BYTES", str(25 * 1024 * 1024))
            )
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
            _insecure_ok = os.environ.get("YUNSHU_VLM_INSECURE_SSL", "").lower() in (
                "1",
                "true",
                "yes",
            )
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

    @staticmethod
    def _find_think_tag(text: str, tag: str, search_start: int = 0) -> int:
        """Find a thinking tag (<think or </think) avoiding false positives.

        Matches the tag only when followed by a non-alphanumeric character
        (>, whitespace, newline, or end-of-string).  This prevents false
        matches on <thinking>, <think_more>, etc.
        """
        idx = search_start
        while True:
            pos = text.find(tag, idx)
            if pos < 0:
                return -1
            end = pos + len(tag)
            if end >= len(text) or not text[end].isalnum():
                return pos
            # False positive (e.g. <thinking>) — keep searching
            idx = end

    def _count_text_tokens(self, text: str) -> int:
        """Count tokens in a chat-template-rendered prompt WITHOUT double-counting
        BOS. The text came from apply_chat_template(tokenize=False),
        which already injected the literal bos_token for BOS-prepending models; a
        plain encode() defaults to add_special_tokens=True and prepends BOS again,
        inflating the billed prompt_tokens by 1. Mirrors BatchedEngine._encode_prompt."""
        if not self._tokenizer:
            return 0
        bos = getattr(self._tokenizer, "bos_token", None)
        add_special = not (isinstance(bos, str) and bos and text.startswith(bos))
        try:
            return len(self._tokenizer.encode(text, add_special_tokens=add_special))
        except TypeError:
            return len(self._tokenizer.encode(text))

    def _estimate_image_tokens(self) -> int:
        """Estimate the number of vision tokens per image for this model.

        Reads the vision config to get an accurate estimate when available,
        otherwise uses a conservative default.  This is used to adjust
        prompt_tokens in usage stats so they reflect the actual model input
        (text tokens + image placeholder tokens).
        """
        if not self._config:
            return 576  # conservative default (24x24 grid)
        vision_cfg = self._config.get("vision_config", {})
        if not vision_cfg:
            thinker_cfg = self._config.get("thinker_config", {})
            vision_cfg = thinker_cfg.get("vision_config", {})
        # spatial_merge_size (Qwen-VL etc.) merges merge×merge patches into ONE vision
        # token, so the patch-grid count must be divided by merge². : this was
        # ignored for the image_size paths (the spatial_merge branch below was dead
        # because they returned first), overcounting Qwen3-VL by ~4× (729 vs ~182).
        spatial_merge = vision_cfg.get("spatial_merge_size", 1)
        _merge_div = max(1, spatial_merge**2)
        # Common config keys for image token count
        for key in ("image_size", "image_resolution"):
            size = vision_cfg.get(key)
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                # Approximate: (H/patch_size) * (W/patch_size) / merge²
                patch_size = vision_cfg.get("patch_size", 14)
                return ((size[0] // patch_size) * (size[1] // patch_size)) // _merge_div
        # If image_size is a single int
        size = vision_cfg.get("image_size")
        if isinstance(size, int):
            patch_size = vision_cfg.get("patch_size", 14)
            return ((size // patch_size) ** 2) // _merge_div
        # Qwen-style models that only declare spatial_merge_size (no image_size)
        if spatial_merge > 1:
            return 256 // (spatial_merge**2)
        return 576  # default: 24x24 patch grid

    def _get_eos_ids(self) -> list[int]:
        from .text_utils import get_eos_token_ids

        return get_eos_token_ids(self._tokenizer)

    # ── Stats ──

    def get_stats(self) -> dict:
        uptime = time.monotonic() - self._start_time if self._start_time else 0.0
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
            # Vision feature cache (image encoder output reuse)
            "vision_cache_enabled": self._vision_cache is not None,
            "vlm_vision_feature_hits": self._vlm_vision_hits,
            "vlm_vision_feature_misses": self._vlm_vision_misses,
            "vlm_vision_feature_hit_rate": self._vlm_vision_hits / total_vision
            if total_vision > 0
            else 0.0,
            # KV prefix cache (per-image KV state reuse)
            "vlm_kv_prefix_entries": len(self._kv_prefix_states),
            "vlm_kv_prefix_hits": self._vlm_kv_prefix_hits,
            "vlm_kv_prefix_misses": self._vlm_kv_prefix_misses,
            "vlm_kv_prefix_hit_rate": self._vlm_kv_prefix_hits / total_kv
            if total_kv > 0
            else 0.0,
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
        # cross-request vision-feature cache (the one that actually runs).
        if self._vision_tower_wrappers:
            _vh = sum(
                w.cache_stats()["vision_tower_cache_hits"]
                for w in self._vision_tower_wrappers
            )
            _vm = sum(
                w.cache_stats()["vision_tower_cache_misses"]
                for w in self._vision_tower_wrappers
            )
            stats["vision_tower_cache"] = {
                "towers_wrapped": len(self._vision_tower_wrappers),
                "hits": _vh,
                "misses": _vm,
                "hit_rate": (_vh / (_vh + _vm)) if (_vh + _vm) else 0.0,
            }

        # Merge text prompt tokenization cache stats
        stats["text_prompt_cache"] = self._text_prompt_cache.stats

        # Merge MultimodalPipelineCoordinator stats
        try:
            stats["pipeline"] = self._pipeline.get_stats()
        except Exception:
            logger.debug("operation failed", exc_info=True)

        return stats
