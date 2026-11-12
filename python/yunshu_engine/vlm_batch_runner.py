"""VLM text runner: admission, delivery and stats around the round driver.

Every request of a loaded VLM target is one row of the round driver (``round_driver/``), which
combines, per request:

- APC prefix reuse (exact + recurrent checkpoints for hybrid models),
- a speculative draft on the Qwen3.5 family and Flash-Next (the checkpoint's native MTP head or an
  external DFlash2 drafter; token-identical to plain decode),
- mlx-native sampling (temperature / top-p / top-k / min-p / seed),
- logits processors: repetition / presence / frequency penalties, logit bias and
  Yunshu's grammar / JSON-schema constraints.

The runner only yields token ids. Detokenizing, stop strings, thinking state and
queue delivery stay in ``VLMEngine`` so every text path reports the same way.
GPU work runs on the serialized MLX executor thread; consumers read their
tokens from a queue on any other thread.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import mlx.core as mx

from . import hit_stages, settings
from .constraint_eos import GrammarDeadEndError
from .idle_memory import IdleMemory
from .sampling_defaults import effective_temperature
from .serving.busy_time import BusyMeter
from .serving.work_scheduler import (
    AGING_S,
)
from .text_logit_processors import GeneratedTokenPenalty

logger = logging.getLogger(__name__)


# Upstream prefill default; APC checkpoints land on these chunk boundaries and
# cancellation is honoured between chunks (~2 s at 2K tokens on a 27B model).
PREFILL_STEP = 2048


def media_token_ids(model) -> set[int]:
    """Media IDs can live on the VLM wrapper, not its text-only backbone."""
    from mlx_vlm.apc import multimodal_token_ids_from_config

    configs = [
        getattr(model, "config", None),
        getattr(getattr(model, "language_model", None), "config", None),
    ]
    ids: set[int] = set()
    for config in list(configs):
        configs.extend(
            getattr(config, key, None) for key in ("thinker_config", "text_config")
        )
    for config in configs:
        if config is None:
            continue
        ids.update(multimodal_token_ids_from_config(config))
        for key in ("audio_token_id", "audio_token_index"):
            token = getattr(config, key, None)
            if token is not None:
                ids.add(int(token))
    return ids


@dataclass
class RunStats:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    # Where the cached prefix came from ("ram", "ssd" or "none") and how long the lookup
    # (including an SSD reload) took; None until the request's cache lookup ran.
    cache_tier: str | None = None
    cache_reload_ms: float | None = None
    cache_device: str | None = (
        None  # the storage tier (volume) an SSD-tier hit came from
    )
    first_token_s: float = 0.0
    generated: int = 0
    # Tokens the engine's consumer has taken so far: what usage reports (``generated`` is the
    # runner's progress and can be ahead of it: a stop string ends the reply mid-round, the
    # consumer lags). 0 = not tracked.
    consumed: int = 0
    finish_reason: str | None = None
    stop_string_hit: bool = False  # a user stop string (not EOS) ended the request
    used_apc: bool = False
    used_draft: bool = False
    # Per-token {"token_id", "logprob", "top_logprobs": [...]} when requested.
    prompt_logprobs: list | None = None
    last_logprob: dict | None = None
    extra: dict = field(default_factory=dict)
    # Timing / progress (time.perf_counter() values; 0.0 = not reached yet). The
    # gateway reads them live for prefill-progress events and the per-response
    # ``x_yunshu`` stats.
    energy: dict | None = None  # populated off the MLX thread by gateway telemetry
    latency_marks: dict[str, float] = field(default_factory=dict)
    t_prefill_end: float = 0.0
    t_submit: float = 0.0  # handed to the runner
    t_admit: float = 0.0  # left the queue, handed to the generator
    # The first prefill atom of this request started running. Admission comes earlier: with several
    # requests the scheduler runs their atoms one at a time, so an admitted request waits for its turn.
    # ``tracks_prefill_start`` is set by the runner; paths that do not track it (the text fast path)
    # count admission as the start.
    t_prefill_start: float = 0.0
    tracks_prefill_start: bool = False
    t_first: float = 0.0  # first generated token
    t_last: float = 0.0  # latest generated token
    prefill_done: int = 0  # prompt tokens computed so far (cache hits excluded)
    prefill_total: int = 0  # prompt tokens to compute (cache hits excluded)
    # False from admission until the cache hit is known: before that ``prefill_total`` is the whole
    # prompt, so a percentage computed from it would start wrong and then jump when the hit arrives.
    prefill_known: bool = True
    spec_mode: str | None = None  # "mtp" / "dflash" while a drafter is in use
    spec_drafted: int = 0
    spec_accepted: int = 0
    spec_rounds: int = 0  # verify rounds this request took part in
    spec_copy_rounds: int = 0  # of those, rounds that verified a copied run
    spec_copy_tokens: int = 0  # tokens those rounds committed
    request_id: str | None = None
    cache_reason: str | None = None
    spec_reason: str | None = None
    spec_depth_drafted: list[int] = field(default_factory=list)
    spec_depth_accepted: list[int] = field(default_factory=list)
    structured_output: dict | None = None
    # Why the request still waits in the queue ("memory": the host has no headroom for it) and
    # the human-readable numbers; None once it is admitted or when it never waited.
    queue_reason: str | None = None
    queue_detail: str | None = None

    @property
    def phase(self) -> str:
        if self.finish_reason is not None:
            return "done"
        if not self.t_admit:
            return "queued"
        if not self.t_first:
            if self.t_prefill_start or not self.tracks_prefill_start:
                return "prefill"
            return "queued"
        return "decode"


class ConstraintProcessor:
    """Adapt a Yunshu grammar/JSON constraint to the runner's logits-processor protocol.

    The runner calls ``__call__`` for the first token (prompt stage) and
    ``process_last_token`` for every later step with the token it just sampled,
    so the constraint advances before masking the next distribution.
    """

    def __init__(self, constraint: Any, tokenizer: Any):
        from .json_schema import apply_json_constraint

        self._constraint = constraint
        self._tokenizer = tokenizer
        self._apply = apply_json_constraint
        self._generated: list[int] = []
        # Shared batch rows set ``isolate``: a dead end is recorded in ``fault`` (the runner
        # fails this one request) instead of raising out of the whole batch step.
        self.isolate = False
        self.fault: BaseException | None = None

    def _mask(self, logits: mx.array) -> mx.array:
        try:
            return self._mask_row(logits)
        except GrammarDeadEndError as exc:
            if not self.isolate:
                raise
            self.fault = exc
            return logits

    def _mask_row(self, logits: mx.array) -> mx.array:
        fast = getattr(self._constraint, "allowed_mask", None)
        if fast is not None:
            mask = fast(self._tokenizer, logits.shape[-1])
            if mask is not None:
                return mx.where(mask, logits, mx.array(float("-inf"), logits.dtype))
        allowed = self._constraint.get_allowed_tokens(self._tokenizer, self._generated)
        if not allowed:
            raise GrammarDeadEndError("Grammar constraint has no valid next token")
        return self._apply(logits, allowed)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        return self._mask(logits)

    def process_last_token(self, token: int, logits: mx.array) -> mx.array:
        token = int(token)
        self._generated.append(token)
        if hasattr(self._constraint, "advance_token"):
            self._constraint.advance_token(token)
        else:
            self._constraint.advance(self._tokenizer.decode([token]))
        return self._mask(logits)


def _row_fault(job: Any) -> BaseException | None:
    """The grammar dead end an isolated processor recorded for this row, if any."""
    for proc in (
        *(getattr(job, "processors", None) or ()),
        getattr(job, "guide", None),
    ):
        fault = getattr(proc, "fault", None)
        if isinstance(fault, BaseException):
            return fault
    return None


class Fp32LogitsProcessor:
    """Upcast logits to float32 so reported logprobs are exact.

    ``logprobs = logits - logsumexp(logits)`` is computed in the model's dtype. In bf16
    the logsumexp is rounded to about 0.1, so a near-certain token reads exactly 0.0
    while its alternatives read -4.25 in the same row. Appended last, only for requests
    that ask for logprobs (those already leave the fused greedy step).
    """

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        return logits.astype(mx.float32)

    def process_last_token(self, token: int, logits: mx.array) -> mx.array:
        return logits.astype(mx.float32)


def build_sampler(temperature: float, top_p: float, top_k: int, min_p: float):
    from mlx_lm.sample_utils import make_sampler

    if temperature is None or temperature < 1e-6:
        return None  # greedy argmax (fused when possible)
    return make_sampler(
        temp=float(temperature),
        top_p=float(top_p) if top_p and top_p < 1.0 else 0.0,
        min_p=float(min_p or 0.0),
        top_k=int(top_k or 0),
    )


def spec_safe_processors(processors) -> bool:
    """Every processor can be applied per row of a speculative window: mlx-lm's stateless
    repetition-penalty / logit-bias closures, the pure token mask and the generated-token
    penalty (``window_row``). Anything else keeps the request off the speculative lane."""
    for p in processors:
        if isinstance(
            p, (TokenMaskProcessor, LogitBiasProcessor, GeneratedTokenPenalty)
        ):
            continue
        if hasattr(p, "process_last_token"):
            return False
        if not getattr(p, "__module__", "").startswith("mlx_lm.sample_utils"):
            return False
    return True


def driver_allow_draft(
    allow_draft: bool,
    processors,
    logprobs: bool,
    guide,
    *,
    xtc: bool = False,
    sampled: bool = False,
) -> bool:
    """Whether a request's row may draft on the round driver (the driver's own ``add`` checks
    the rest: a sampled row needs the keyed sampler, ``xtc`` has none).

    Drafts: processors the driver applies one window row at a time (``driver.windowable``: token
    masks incl. min_tokens / top-n-sigma, logit bias, repetition penalty, one frequency /
    presence penalty), logprobs with or without them, and any invariant guide (tool-call or
    response-format, alone or combined) with either, on DFlash and on MTP head rows alike.
    Plain rows: XTC sampling (no keyed draw), a non-invariant guide, any other processor. A
    sampled row with logprobs drafts on MTP head rows too (``driver.row_logprobs`` reduces every
    window row on its own, so the report equals a plain step's). ``allow_draft`` False
    (the caller) always wins."""
    from .constrained_spec import driver_guide
    from .round_driver.driver import windowable

    # reporting adds an Fp32 processor; the driver does the fp32 readout itself
    rest = [p for p in (processors or []) if not isinstance(p, Fp32LogitsProcessor)]
    if not allow_draft or xtc:
        return False
    if guide is not None:
        return driver_guide(guide, logprobs, rest)
    return windowable(rest)


def static_masks_only(processors) -> bool:
    """True when every processor is a static ban (a checkpoint ``suppress_tokens`` list or
    ``ignore_eos``) -- vacuously true for none. Such a mask never needs the token history, so
    a row that carries only it stays eligible for drafting and for lane promotion; the lane
    applies the same ban to its verify rows (``ProcessorWindow``) and to its copy drafts."""
    return all(
        isinstance(p, TokenMaskProcessor) and p.is_static for p in (processors or [])
    )


class LogitBiasProcessor:
    """``logits[:, t] += bias[t]`` for the in-range ids only.

    mlx-lm's own bias closure scatters every key it is given, so a user id of ``-1`` wrapped
    to the LAST vocabulary entry (biasing a token nobody asked about) and one past the head
    was left to the scatter's undefined behaviour; OpenAI ignores ids outside the
    vocabulary and the text path does too (4634f98). In-range ids run the identical
    ``at[...].add`` scatter, so their logits are unchanged bit for bit. Stateless and
    context-free: safe for speculative verify rows and shared-batch rows alike.
    """

    context_free = True
    # the bias is a constant per id: a speculative window applies it to every row
    # (``pure_at``), as a plain row does, so a row that carries it still drafts
    is_static = True

    def pure_at(self, generated: int, logits: mx.array) -> mx.array:
        return self(None, logits)

    def __init__(self, biases: dict[int, float]):
        self._biases = {int(k): float(v) for k, v in biases.items()}
        self._vocab = -1
        self._scatter: tuple[mx.array, mx.array] | None = None

    def __call__(self, tokens, logits: mx.array) -> mx.array:
        vocab = logits.shape[-1]
        if vocab != self._vocab:
            self._vocab = vocab
            ids = [t for t in self._biases if 0 <= t < vocab]
            self._scatter = (
                (mx.array(ids), mx.array([self._biases[t] for t in ids]))
                if ids
                else None
            )
        if self._scatter is None:
            return logits
        return logits.at[..., self._scatter[0]].add(self._scatter[1])


def build_penalty_processors(
    repetition_penalty: float,
    frequency_penalty: float,
    presence_penalty: float,
    logit_bias: dict[int, float] | None,
) -> list:
    """Logit bias, repetition penalty, then frequency / presence penalty.

    frequency / presence follow OpenAI (and the text path): every generated token counts
    (``GeneratedTokenPenalty``). ``repetition_penalty`` is not an OpenAI field; it stays
    mlx-lm's, over the last ``PENALTY_CONTEXT`` (20) prompt + generated tokens.
    """
    if (
        repetition_penalty in (None, 1.0)
        and not frequency_penalty
        and not presence_penalty
        and not logit_bias
    ):
        return []
    from mlx_lm.sample_utils import make_logits_processors

    from .constrained_spec import PENALTY_CONTEXT

    out: list = [LogitBiasProcessor(logit_bias)] if logit_bias else []
    if repetition_penalty not in (None, 1.0):
        out += make_logits_processors(
            repetition_context_size=PENALTY_CONTEXT,
            repetition_penalty=repetition_penalty,
        )
    if frequency_penalty or presence_penalty:
        out.append(GeneratedTokenPenalty(frequency_penalty, presence_penalty))
    return out


class TokenMaskProcessor:
    """suppress_tokens / min_tokens / ignore_eos / top-nσ as one per-row
    logits processor (first token via ``__call__``, later tokens via
    ``process_last_token``, counting only generated tokens)."""

    def __init__(
        self,
        *,
        suppress: list[int] | None = None,
        eos_ids: list[int] | None = None,
        min_tokens: int = 0,
        ignore_eos: bool = False,
        top_n_sigma: float = 0.0,
    ):
        self._suppress = mx.array(sorted(set(suppress or [])), dtype=mx.int32)
        self._eos = mx.array(sorted(set(eos_ids or [])), dtype=mx.int32)
        self._min_tokens = int(min_tokens or 0)
        self._ignore_eos = bool(ignore_eos)
        self._n_sigma = float(top_n_sigma or 0.0)
        self._generated = 0
        self._block_cache: dict[tuple[int, int], mx.array] = {}

    @property
    def is_static(self) -> bool:
        """The mask does not depend on how many tokens were generated or on the logits
        (``suppress_tokens`` and ``ignore_eos`` only): a pure ban of a fixed id set."""
        return not self._min_tokens and not self._n_sigma

    def always_banned(self) -> frozenset[int]:
        """Ids this processor bans on every step (the fixed ban of a static mask)."""
        ids = set(cast(list[int], self._suppress.tolist()))
        if self._ignore_eos:
            ids.update(cast(list[int], self._eos.tolist()))
        return frozenset(ids)

    @staticmethod
    def active(**kw) -> bool:
        return bool(
            kw.get("suppress")
            or kw.get("min_tokens")
            or kw.get("ignore_eos")
            or kw.get("top_n_sigma")
        )

    def _block(self, ids: mx.array, vocab: int) -> mx.array:
        # The id set is fixed for the request, so the (vocab,) bool mask is built once
        # per vocab width instead of zeros + scatter on every decoded token.
        key = (id(ids), vocab)
        cached = self._block_cache.get(key)
        if cached is None:
            # ids outside [0, vocab) are ignored (a negative one would wrap onto the tail)
            keep = [t for t in cast(list[int], ids.tolist()) if 0 <= t < vocab]
            cached = mx.zeros((vocab,), dtype=mx.bool_)
            if keep:
                cached = cached.at[mx.array(keep, dtype=mx.int32)].add(True)
            self._block_cache[key] = cached
        return cached

    def _mask(self, logits: mx.array, generated: int | None = None) -> mx.array:
        # Functional (never writes into the caller's logits).
        generated = self._generated if generated is None else generated
        neg = mx.array(float("-inf"), logits.dtype)
        vocab = logits.shape[-1]
        if self._suppress.size:
            logits = mx.where(self._block(self._suppress, vocab), neg, logits)
        if self._eos.size and (self._ignore_eos or generated < self._min_tokens):
            logits = mx.where(self._block(self._eos, vocab), neg, logits)
        if self._n_sigma > 0:
            top = mx.max(logits, axis=-1, keepdims=True)
            finite = mx.where(mx.isinf(logits), top, logits)
            sigma = mx.sqrt(mx.var(finite, axis=-1, keepdims=True))
            logits = mx.where(logits < top - self._n_sigma * sigma, neg, logits)
        return logits

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        return self._mask(logits)

    def process_last_token(self, token: int, logits: mx.array) -> mx.array:
        self._generated += 1
        return self._mask(logits)

    def pure_at(self, generated: int, logits: mx.array) -> mx.array:
        """The mask after ``generated`` ``process_last_token`` calls, without counting (a
        speculative window applies it to every row, then ``advance`` by what committed)."""
        return self._mask(logits, generated)

    def advance(self, n: int) -> None:
        self._generated += int(n)


def _source_image_digest(paths, fingerprint: str, image_token_index) -> int | None:
    """Exact key: same bytes + same preprocessing config -> same key; any change differs."""
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    h.update(b"yunshu-source-image-v1")
    for part in (fingerprint, repr(image_token_index), str(len(paths))):
        data = part.encode()
        h.update(len(data).to_bytes(8, "little") + data)
    try:
        for path in paths:
            with open(path, "rb") as f:
                body = f.read()
            h.update(len(body).to_bytes(8, "little"))
            h.update(body)
    except OSError:
        return None
    return int.from_bytes(h.digest(), "little", signed=True)


class VLMBatchRunner:
    """Owns the APC manager, the drafter and the shared batch scheduler."""

    singleflight = True

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        apc_manager: Any = None,
        apc_semantic_hash: int | None = None,
        drafter: Any = None,
        draft_block_size: int | None = None,
        apc_admit: Any = None,
        draft_kind: str = "mtp",
        executor: Any = None,
        prefix_invariant: bool = False,
    ):
        self.model = model
        self._media_token_ids = frozenset(media_token_ids(model))
        self.prefix_invariant = prefix_invariant
        logger.info(
            "APC admission configuration: qualified=%s enabled=%s source=%s",
            prefix_invariant,
            self.singleflight,
            __file__,
        )
        self.processor = processor
        # MemoryGate, set by the engine (None = no memory admission)
        self.memory_gate: Any = None
        self._last_shed = 0.0
        self.apc_manager = apc_manager
        self.apc_semantic_hash = apc_semantic_hash
        self.drafter = drafter
        self.draft_kind = draft_kind
        self.finish_cleanup_failures: dict[
            str, int
        ] = {}  # cleanup step -> raises (isolated)
        from . import penalty_context

        penalty_context.install()  # penalty context = the full prompt on an APC hit too
        self.draft_block_size = draft_block_size
        # Callable(input_ids) -> bool: skip APC when its checkpoints cannot fit,
        # so a cold request does not pay APC bookkeeping for nothing.
        self._apc_admit = apc_admit
        # The serialized MLX executor the driver runs on (None: drive inline).
        self._executor = executor
        self._lock = threading.Lock()
        self._pending: list[_Job] = []
        self._prefix_producers: list[tuple[_Job, list[int]]] = []
        self._vocab_owner: Any = None
        self._driving = False
        self.clear_on_idle = False
        self._idle_memory: IdleMemory | None = None
        self.gc_guard: Any = None  # gc_policy.TtftGcGuard when YUNSHU_GC_POLICY=defer
        # Requests the engine has accepted, including ones still being
        # prepared (templating, image encoding) — the runner alone cannot see
        # those, and a request is only "alone" if the engine has no others.
        self.inflight = lambda: 0
        # All ids that end a turn (tokenizer + generation_config eos).
        self.stop_tokens: set[int] | None = None
        # KV precision of the Qwen3.5 layer's rows (YUNSHU_KV_PRECISION), None for other models.
        self.ragged_kv: str | None = None
        # Yunshu's round driver (round_driver/): every request of every model is a row in it
        # (packed Qwen3.5 / Flash-Next layer or the generic layer). Built by the engine.
        self.driver: Any = None
        self._aux_driver: Any = None
        self._driver_jobs: dict[int, _Job] = {}
        # GPU-busy seconds: every executor slice, and the round driver's steps inside them
        # (``/v1/yunshu/status`` and ``/metrics`` report the cumulative counters).
        self.busy_meter = BusyMeter()
        self.driver_busy_meter = BusyMeter()

    def _processor_fingerprint(self) -> str | None:
        """Everything that turns image bytes into pixel values, or None if unknown."""
        cached: str | None | int = getattr(self, "_proc_fp", 0)
        if cached != 0:
            return cached  # type: ignore[return-value]
        fp = None
        try:
            import json

            proc = self.processor
            parts = []
            for obj in (getattr(proc, "image_processor", None) or proc,):
                if obj is not None and hasattr(obj, "to_dict"):
                    parts.append(obj.to_dict())
            if parts:
                fp = type(proc).__name__ + json.dumps(
                    parts, sort_keys=True, default=str
                )
        except Exception:
            fp = None
        self._proc_fp = fp
        return fp

    def _source_image_key_async(self, image_paths, image_token_index):
        """Future of an exact digest of (processor config, image file bytes) or None.

        The processor fingerprint is also built on the worker thread, so a first
        request never serializes the processor config on the MLX thread.
        """
        pool = getattr(self, "_key_pool", None)
        if pool is None:
            from concurrent.futures import ThreadPoolExecutor

            pool = self._key_pool = ThreadPoolExecutor(
                1, thread_name_prefix="media-key"
            )

        def job():
            fp = self._processor_fingerprint()
            if fp is None:
                return None
            return _source_image_digest(tuple(image_paths), fp, image_token_index)

        return pool.submit(job)

    @staticmethod
    def _image_hash(_apc, pixel_values, source_key) -> int:
        if pixel_values is None:
            return 0
        if source_key is not None:
            try:
                digest = source_key.result()
            except Exception:
                digest = None
            if digest is not None:
                return int(digest)
        return int(_apc.hash_image_payload(pixel_values=pixel_values))

    def prepare_media(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        audio: list | None = None,
    ):
        """Preprocess + encode images / audio for one prompt on the MLX thread.

        ``audio`` is the preloaded waveform list from ``VLMEngine._audio_arg``;
        the processor turns it into ``input_features`` (and masks), which ride
        in the prompt kwargs to the model's ``get_input_embeddings``. Returns
        ``(input_ids, prompt_kwargs, apc_semantic_hash)``. The APC salt hashes
        the processed pixel values and audio features, so a prefix is only
        reused for the same media content (never across media or with
        text-only prompts).
        """
        from mlx_vlm import apc as _apc
        from mlx_vlm.utils import prepare_inputs

        from .mrope import clear_rope_state

        # The image key hashes the source file bytes + the processor config on a
        # CPU thread while the main thread preprocesses; hashing the processed
        # pixel tensor costs a device sync and a ~2.6 ms host copy per request.
        image_token_index = getattr(self.model.config, "image_token_index", None)
        source_key = (
            self._source_image_key_async(image_paths, image_token_index)
            if image_paths
            else None
        )
        raw = prepare_inputs(
            self.processor,
            images=image_paths or None,
            audio=audio or None,
            prompts=prompt,
            image_token_index=image_token_index,
            add_special_tokens=True,
        )
        input_ids = raw["input_ids"]
        pixel_values = raw.get("pixel_values")
        data = {
            k: v
            for k, v in raw.items()
            if k not in ("input_ids", "pixel_values", "attention_mask")
        }
        if not self.busy():
            # Stale mRoPE state from an earlier request; while a batch runs,
            # its rows carry their own positions / deltas.
            clear_rope_state(self.model)
        embed = self.model.get_input_embeddings(
            input_ids, pixel_values, mask=raw.get("attention_mask"), **data
        )
        kwargs = {**data, **{k: v for k, v in embed.to_dict().items() if v is not None}}
        salt = None
        if self.apc_manager is not None:
            salt = _apc.semantic_extra_hash(
                image_hash=self._image_hash(_apc, pixel_values, source_key),
                media={
                    "yunshu_media_boundary_schema": "wrapper-media-v1",
                    "audio": raw.get("input_features"),
                    "video": raw.get("pixel_values_videos"),
                    # Equal patch pixels can have different spatial grids;
                    # audio masks/lengths also affect what the encoder sees.
                    # Prompt-level masks/positions are excluded so a text-only
                    # continuation keeps the same media namespace.
                    **{
                        k: raw.get(k)
                        for k in (
                            "image_grid_thw",
                            "video_grid_thw",
                            "image_sizes",
                            "image_position_ids",
                            "video_position_ids",
                            "pixel_attention_mask",
                            "image_attention_mask",
                            "feature_attention_mask",
                            "input_features_mask",
                            "audio_features",
                            "audio_mask",
                            "audio_feature_lengths",
                            "audio_lengths",
                            "second_per_grid_ts",
                        )
                    },
                },
                model=self.model.language_model,
                processor=self.processor,
            )
        return input_ids[0], kwargs, salt

    # ── Scheduling ──────────────────────────────────────────────────────
    #
    # Requests submit a job and read tokens from their own queue on any thread.
    # One driver owns the GPU: it runs on the serialized MLX executor in short
    # slices (admit, one driver step, dispatch) and resubmits itself while work
    # remains, so prep for new requests (templating, image encoding) interleaves.
    #
    # - Every request is a row of the one round driver (round_driver/): the packed Qwen3.5 /
    #   Flash-Next layer with speculation, or the generic layer on the model's own forward. The row
    #   carries its sampling params and seed, its logits processors and guide; prefill is one
    #   request at a time.
    # - The runner owns admission (memory gate, prefix single-flight, background rows waiting for
    #   interactive ones), the stats and the delivery queues; the driver owns the GPU work.

    def iter_tokens(
        self,
        input_ids,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        seed: int | None = None,
        logits_processors: list | None = None,
        allow_draft: bool = True,
        ngram: bool = False,
        prompt_kwargs: dict | None = None,
        apc_semantic_hash: int | None = None,
        cancel_event: Any = None,
        stats: RunStats | None = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        prompt_logprobs: int | None = None,
        thinking_budget: int | None = None,
        prompt_preopens_thinking: bool = False,
        thinking_start_token: str = "<think>",
        thinking_end_token: str = "</think>",
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        xtc_special_tokens: list | None = None,
        guide: Any = None,
        prompt_cache_plan: dict | None = None,
        first_delivery: threading.Event | None = None,
    ) -> Iterator[int]:
        """Yield generated token ids; ``stats`` is filled in as generation runs.

        Must not be called on the MLX executor thread when the runner has an
        executor: the driver needs that thread.
        """
        stats = stats if stats is not None else RunStats()
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        if not ids:
            # a zero-token prompt cannot be prefilled: refuse before anything is queued.
            from .context_window import require_non_empty_prompt

            require_non_empty_prompt(ids)
        from .request_tracker import current_request_id

        stats.request_id = (
            getattr(cancel_event, "client_request_id", None) or current_request_id.get()
        )
        stats.prompt_tokens = len(ids)
        temperature = effective_temperature(temperature, top_k, min_p, xtc_probability)
        greedy = temperature is None or temperature < 1e-6
        processors = list(logits_processors or [])
        # A row may draft when the engine built a drafter; the driver decides per row (``driver_allow_draft``
        # below, then ``RoundDriver.add``) which request kinds draft: sampled rows draw from a position-keyed
        # sampler, so accepting a draft equals what serial sampling would have produced (keyed_sampling).
        use_draft = bool(allow_draft and self.drafter is not None)
        if logprobs:
            processors.append(Fp32LogitsProcessor())
        budget = None
        if thinking_budget is not None:
            from .thinking_budget import make_forced_close_criteria

            _tok = getattr(self.processor, "tokenizer", self.processor)
            budget = make_forced_close_criteria(
                _tok,
                self.processor,
                int(thinking_budget),
                thinking_end_token=thinking_end_token,
                thinking_start_token=thinking_start_token,
                enable_thinking=True,
                prompt_preopens_thinking=prompt_preopens_thinking,
            )
        job = _Job(
            ids=ids,
            max_tokens=int(max_tokens),
            greedy=greedy,
            sampling=(
                None
                if greedy
                else RowParams(
                    float(temperature),
                    float(top_p),
                    int(top_k),
                    float(min_p),
                    seed,
                    float(xtc_probability or 0.0),
                    float(xtc_threshold or 0.0),
                    xtc_special_tokens,
                )
            ),
            use_draft=use_draft,
            prompt_logprobs=prompt_logprobs,
            logprobs=bool(logprobs),
            top_logprobs=int(top_logprobs or 0) if logprobs else 0,
            processors=processors,
            guide=guide,
            ngram=bool(ngram),
            prompt_kwargs=prompt_kwargs,
            salt=apc_semantic_hash,
            seed=seed,
            cancel_event=cancel_event,
            stats=stats,
            priority=getattr(cancel_event, "scheduling_priority", 0),
            cache_plan=prompt_cache_plan,
            cache_store=bool(getattr(cancel_event, "cache_store", True)),
            budget=budget,
            first_delivery=first_delivery,
        )
        job.allow_draft = driver_allow_draft(
            allow_draft,
            processors,
            logprobs,
            guide,
            xtc=bool(xtc_probability),
            sampled=not greedy,
        )
        stats.used_draft = use_draft
        stats.t_submit = time.perf_counter()
        with contextlib.suppress(Exception):
            # The gateway reaches the live stats through the cancel event it owns.
            cancel_event.run_stats = stats
        self._submit(job)
        try:
            while True:
                if self._executor is None:
                    # No executor (tests / direct use): drive inline.
                    while job.out.empty():
                        self._drive_slice(resubmit=False)
                item = job.out.get()
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                token, lp = item
                if lp is not None:
                    stats.last_logprob = lp
                yield token
        finally:
            # A consumer that stops early (stop string, max length reached on
            # its side, disconnect) releases its row at the next slice.
            job.abandoned = True

    def _emit(self, job: _Job, item: Any) -> None:
        """Hand one item to a job's consumer; exactly one terminal per job.

        Data past OUT_LIMIT (a consumer that stopped reading) ends the job with
        an error instead of growing without bound; the row is dropped at the
        next slice. Only the first DFlash stream token can wait for a bounded
        consumer handoff; ordinary data never blocks the executor thread.
        """
        if job.terminal:
            return
        if item is _DONE or isinstance(item, BaseException):
            job.terminal = True
        elif job.out.qsize() >= OUT_LIMIT:
            job.overflowed = job.terminal = job.abandoned = True
            item = RuntimeError(
                f"output queue exceeded {OUT_LIMIT} undelivered tokens; "
                "request cancelled"
            )
        # The vocabulary context is only an identity guard. Once a request ends,
        # retaining its Job also retains grammar/media/cache-plan state at idle.
        if job.terminal and self._vocab_owner is job:
            self._vocab_owner = None
        if isinstance(item, tuple) and item and isinstance(item[0], int):
            job.generated.append(item[0])
        job.out.put(item)
        if self.gc_guard is not None:
            self.gc_guard.first_token(id(job))
        # The first target token is already complete. Starting copy lookup now
        # can run a full cyclic collection while the SSE consumer still needs
        # the GIL. Give that consumer one bounded opportunity to deliver first.
        # No tensor, sampler or proposal state changes, including on timeout.
        if (
            isinstance(item, tuple)
            and job.stats.generated == 1
            and job.first_delivery is not None
            and not job.first_delivery_waited
            and job.allow_draft
            and job.use_draft
            and self.draft_kind == "dflash"
            and self._executor is not None
            and job.max_tokens > 1
        ):
            job.first_delivery_waited = True
            completed = job.first_delivery.wait(0.004)
            if not getattr(self, "_first_delivery_logged", False):
                self._first_delivery_logged = True
                logger.info(
                    "First-content handoff engaged (DFlash, 4ms bound, completed=%s)",
                    completed,
                )

    def _fail_all(self, exc: BaseException) -> None:
        """Detach failed work before waking consumers; their terminal means drained."""
        with self._lock:
            pending, self._pending = self._pending, []
            drivers, self._driver_jobs = self._driver_jobs, {}
            self._driving = False
        for job in drivers.values():
            if self.driver is not None:
                with contextlib.suppress(Exception):
                    (job.driver or self.driver).remove(job)
        for job in pending:
            self._emit(job, exc)
        for job in drivers.values():
            self._emit(job, exc)

    def _schedule(self) -> None:
        try:
            self._executor.submit(self._drive_slice)
        except Exception as exc:
            logger.exception("VLM runner could not schedule a slice")
            self._fail_all(exc)

    def busy(self) -> bool:
        with self._lock:
            return bool(self._pending or self._driving or self._driver_jobs)

    def _submit(self, job: _Job) -> None:
        if self._idle_memory is not None:
            self._idle_memory.activity()
        if self.gc_guard is not None:
            self.gc_guard.begin(id(job))
        with self._lock:
            job.queued = job.last_service = time.perf_counter()
            self._pending.append(job)
            if self._executor is None:
                return
            if self._driving:
                return
            self._driving = True
        self._schedule()

    def _active_jobs(self) -> int:
        return len(self._driver_jobs)

    def _driver_guide(self, job: _Job) -> bool:
        """The DFlash driver advances an invariant tool-call guide itself (free text drafts,
        the call's rows are masked); any other guide runs as a ToolCallProcessor row."""
        if self.driver is None:
            return False
        # Forced windows need a live guide even without a model drafter (generic
        # families and spec-off packed rows). Preserve the old option-off route.
        if not self.driver.can_draft:
            if not settings.get("YUNSHU_JUMP_FORWARD"):
                return False
            # top-n-sigma reads the unmasked distribution on the serial route;
            # moving its guide before that processor would change the candidate set.
            if any(
                isinstance(p, TokenMaskProcessor) and p._n_sigma for p in job.processors
            ):
                return False
        from .constrained_spec import driver_guide

        rest = [p for p in job.processors if not isinstance(p, Fp32LogitsProcessor)]
        return job.guide is not None and driver_guide(job.guide, job.logprobs, rest)

    def _admit(self, job: _Job, alone: bool) -> None:
        """Hand ``job`` to the round driver: every request of every model is a row in it (the packed
        Qwen3.5 / Flash-Next layer or the generic layer), one row or many, speculation on or off, with
        or without explicit cache seams. Rows it cannot draft for (processors, constraints,
        non-invariant guides) decode one token a round through their own processors; media prompts ride
        as their prompt features."""
        del alone
        if self.driver is None:
            raise RuntimeError("VLM runner: no round driver was built for this model")
        hit_stages.begin()
        if self._decode_follows(job):
            # before this follow-up looks up its prefix: resume in place when the cache can
            # stand, else copy it into the APC as before
            with hit_stages.stage("standing"):
                if not self._stand_decode_cache():
                    self._flush_decode_cache()

        job.stats.t_admit = time.perf_counter()
        job.stats.tracks_prefill_start = True
        job.stats.prefill_total = len(job.ids)
        job.stats.prefill_known = (
            False  # the driver learns the cache hit when the row is added
        )

        if job.guide is not None:
            driver_guide = self._driver_guide(job)
            if not driver_guide:
                job.use_draft = job.allow_draft = False
            # Preserve serial processor order: penalties / suppression first,
            # grammar next, fp32 reporting last.
            from .tool_call_grammar import ToolCallProcessor

            pos = len(job.processors) - int(job.logprobs)
            if not driver_guide:
                job.processors.insert(pos, ToolCallProcessor(job.guide))

        use_apc = self.apc_manager is not None and job.priority >= 0
        if job.cache_plan is not None and job.cache_plan.get("writes") == []:
            use_apc = False
        if use_apc and self._apc_admit is not None:
            try:
                use_apc = bool(self._apc_admit(mx.array(job.ids)))
            except Exception:
                logger.debug("APC admission check failed; using APC", exc_info=True)
        self._admit_driver(job, use_apc)

    # ── round driver ────────────────────────────────────────────────────
    #: prompt-kwarg keys a driver row consumes; any other non-None feature the model's
    #: ``get_input_embeddings`` produced (deepstack features, cross attention, ...) is not
    #: carried by the driver and must not be dropped silently
    _MEDIA_KWARGS_USED = frozenset({"inputs_embeds", "position_ids", "rope_deltas"})
    _MEDIA_KWARGS_FEATURES = frozenset(
        {
            "attention_mask_4d",
            "visual_pos_masks",
            "deepstack_visual_embeds",
            "per_layer_inputs",
            "cross_attention_states",
            "cross_attention_mask",
            "full_text_row_masked_out_mask",
            "decoder_inputs_embeds",
            "attention_mask",
            "pos_hw",
        }
    )

    def _media_request(self, job: _Job) -> dict:
        """The driver ``Request`` fields of a media prompt: its embeddings (visual features
        already scattered in), mRoPE positions and decode offset. Text prompts: nothing."""
        pkw = job.prompt_kwargs
        if pkw is None:
            return {}
        unsupported = sorted(
            k for k in self._MEDIA_KWARGS_FEATURES if pkw.get(k) is not None
        )
        if unsupported:
            raise ValueError(
                f"round driver: prompt features {unsupported} are not carried (media model)"
            )
        embeds = pkw.get("inputs_embeds")
        if embeds is None:
            return {}
        pos = pkw.get("position_ids")
        rd = pkw.get("rope_deltas")
        return {
            "embeds": embeds[0],
            "position_ids": None if pos is None else pos[:, 0, :],
            "rope_delta": 0
            if rd is None
            else int(round(float(rd.reshape(-1)[0].item()))),
        }

    def _generic_request(self, job: _Job) -> dict:
        """The driver ``Request`` fields of the generic layer: the model's prompt inputs. A media prompt
        brings its own (``prepare_media``); a text prompt gets what the stock generator feeds it, the
        model's ``get_input_embeddings`` of the prompt ids (some families add inputs of their own:
        positions, per-layer inputs). The row owns its model state, so nothing is cleared here."""
        pkw = job.prompt_kwargs
        if pkw is None:
            pkw = self.model.get_input_embeddings(
                mx.array(job.ids)[None], None, mask=None
            ).to_dict()
        return {"prompt_kwargs": pkw, "media": job.prompt_kwargs is not None}

    def _driver_extra_hash(self, job: _Job) -> int:
        """The APC namespace of a driver row: the model / processor identity (a media prompt's own
        salt), and for explicit cache seams their own numerical span plan, as the stock path keeps it."""
        salt = (
            job.salt
            if job.prompt_kwargs is not None and job.salt is not None
            else self.apc_semantic_hash or 0
        )
        if job.cache_plan is not None:
            import hashlib

            salt = int.from_bytes(
                hashlib.blake2b(
                    f"{salt}:explicit-absolute-spans-v1".encode(), digest_size=8
                ).digest(),
                "little",
            )
        return int(salt)

    def _admit_driver(self, job: _Job, use_apc: bool) -> None:
        from .round_driver.driver import Request

        driver = self.driver
        if job.priority < 0:
            if self._aux_driver is None:
                from .round_driver.driver import RoundDriver

                # the same layer family as the main driver, bound to the same language model
                aux_layer = (
                    None
                    if self.driver is None
                    else type(self.driver.layer)(self.driver.lm)
                )
                self._aux_driver = RoundDriver(
                    self.model,
                    layer=aux_layer,
                    drafter=self.drafter if self.draft_kind == "mtp" else None,
                    dflash_drafter=self.drafter
                    if self.draft_kind == "dflash"
                    else None,
                    plain_verify=self.drafter is None,
                    stop_tokens=self.stop_tokens,
                    chunk=self.driver.chunk,
                    apc=None,
                    kv_format=self.driver.kv_format,
                )
            driver = self._aux_driver
        job.driver = driver
        driver.finish_hook = self._driver_store_decode
        # a grammar dead end fails this row, not the driver step (``_step_driver`` reads ``fault``)
        from .tool_call_grammar import ToolCallProcessor

        for proc in job.processors:
            if isinstance(proc, ConstraintProcessor | ToolCallProcessor):
                proc.isolate = True
        vocab = getattr(self.drafter, "_draft_vocab", None)
        if vocab is not None and driver.can_draft:
            vocab.skip_ids = self._media_token_ids
            # The reduced draft readout must cover this prompt's ids (the lane sets it at
            # every speculative start): a prompt's own tokens (code identifiers) are what its
            # continuation repeats. First row of an idle driver replaces the context; a row
            # joining running ones adds to it.
            with hit_stages.stage("vocab"):
                extra = (
                    vocab.add_context(job.ids)
                    if driver.rows
                    else vocab.set_context(job.ids)
                )
            logger.info("Round driver: draft vocabulary context +%s ids", extra)
        # the generic layer's rows always carry the model's prompt inputs (GenericLayer.admit requires them)
        from .round_driver.layer import GenericLayer

        generic_rows = isinstance(getattr(driver, "layer", None), GenericLayer)
        media = self._generic_request(job) if generic_rows else self._media_request(job)
        if media and not generic_rows:
            logger.info(
                "Round driver: media prompt %d tokens, rope_delta %d",
                len(job.ids),
                media["rope_delta"],
            )
        t_add = time.perf_counter()
        # The request id reaches the cache events only through the APC's id queue for these
        # ids (``checkpoint_lengths`` pops it while the row plans its restore). Only a row that
        # plans checkpoints queues one, and the queue entry is dropped again after ``add``.
        request_id = job.stats.request_id
        linked = use_apc and driver.apc is not None and self.apc_manager is not None
        if linked:
            self.apc_manager.register_console_request(job.ids, request_id)
        try:
            hit = driver.add(
                Request(
                    **media,
                    ids=job.ids,
                    max_tokens=job.max_tokens,
                    sampling=job.sampling,
                    # the driver reads logprobs out in fp32 itself (``Request.logprobs``)
                    processors=[
                        p
                        for p in (job.processors or [])
                        if not isinstance(p, Fp32LogitsProcessor)
                    ],
                    prompt_logprobs=job.prompt_logprobs,
                    logprobs=job.logprobs,
                    top_logprobs=job.top_logprobs,
                    draft=job.allow_draft,
                    ngram=job.ngram,
                    guide=job.guide if self._driver_guide(job) else None,
                    budget=job.budget,
                    handle=job,
                    extra_hash=self._driver_extra_hash(job),
                    use_apc=use_apc,
                    cache_store=job.cache_store,
                    cache_plan=job.cache_plan,
                )
            )
        finally:
            if linked:
                self.apc_manager.release_console_request(job.ids, request_id)
        hit_stages.add("add", time.perf_counter() - t_add)
        hit_stages.mark("add_end")
        self._vocab_owner = driver
        # ``t_admit`` stays what ``_admit`` set before the driver looked the prefix up (the cache
        # provenance record and the prefill-speed observation are measured from it)
        job.start = job.stats.t_prefill_start = time.perf_counter()
        job.stats.tracks_prefill_start = True
        hit = int(hit or 0)
        job.stats.cached_tokens = hit
        job.stats.prefill_total = len(job.ids) - hit
        job.stats.prefill_known = True
        job.stats.used_apc = use_apc
        # set by the driver's ``add``: n-gram / suffix, draft LM or Gemma-4 assistant row
        ngram_row = job.stats.spec_mode in ("ngram", "draft-lm", "gemma4-assistant")
        job.stats.used_draft = bool((job.allow_draft and driver.can_draft) or ngram_row)
        if job.stats.used_draft and not ngram_row:
            job.stats.spec_mode = (
                "dflash" if getattr(driver, "dflash", None) is not None else "mtp"
            )
        if not job.stats.used_draft:
            self._driver_plain_rows = getattr(self, "_driver_plain_rows", 0) + 1
            logger.info(
                "Round driver engaged without drafting (plain row #%d)",
                self._driver_plain_rows,
            )
        self._driver_jobs[id(job)] = job

    def _step_driver(self, primary: bool = False) -> None:
        for key, job in list(self._driver_jobs.items()):
            cancelled = job.cancel_event is not None and job.cancel_event.is_set()
            if job.abandoned or cancelled:
                (job.driver or self.driver).remove(job)
                del self._driver_jobs[key]
                job.stats.finish_reason = "cancel" if cancelled else None
                self._cleanup_step(
                    job, "observe_prefill", lambda job=job: self._observe_prefill(job)
                )
                self._emit(job, _DONE)
        if not self._driver_jobs:
            return
        events = []
        for driver in (self.driver, self._aux_driver):
            if driver is None or (primary and driver is self._aux_driver):
                continue
            vocab = getattr(self.drafter, "_draft_vocab", None)
            if (
                vocab is not None
                and driver.can_draft
                and self._vocab_owner is not driver
                and driver.rows
            ):
                vocab.set_context([t for row in driver.rows for t in row.req.ids])
                self._vocab_owner = driver
            with self.driver_busy_meter.span():
                events.extend(driver.step())
            self._note_driver_prefill(driver)
        faulted: set[int] = set()
        for event in events:
            job = event.handle
            fault = _row_fault(job)
            if fault is not None:
                # This row's grammar hit a dead end: its token is meaningless. Fail only
                # this request; the other rows keep decoding (as in the batch path).
                if id(job) not in faulted:
                    faulted.add(id(job))
                    logger.warning(
                        "VLM runner row failed (%s); other rows continue", fault
                    )
                    (job.driver or self.driver).remove(job)
                    self._driver_jobs.pop(id(job), None)
                    self._emit(job, fault)
                continue
            stats = job.stats
            if event.prompt_logprobs is not None:
                stats.prompt_logprobs = event.prompt_logprobs
            now = time.perf_counter()
            first = stats.generated == 0
            if first:
                stats.first_token_s = now - job.start
                stats.t_first = now
                stats.prefill_done = stats.prefill_total
                # per-request prefix-cache provenance (tier, lookup time) for ``x_yunshu.cache``
                self._cleanup_step(
                    job, "note_cache", lambda job=job: self._note_cache(job)
                )
            stats.t_last = now
            stats.generated += 1
            self._emit(job, (int(event.token), event.logprob))
            if first:
                hit_stages.finish(
                    len(job.ids),
                    int(stats.cached_tokens or 0),
                    time.perf_counter() - now,
                )
            if event.finish is not None:
                stats.finish_reason = event.finish
                self._driver_jobs.pop(id(job), None)
                # the storage tiers learn how fast prefill really is (a lone, long prefill counts)
                self._cleanup_step(
                    job, "observe_prefill", lambda job=job: self._observe_prefill(job)
                )
                self._emit(job, _DONE)
        if events:
            # the first tokens are out: evaluate the captured APC checkpoints now
            for driver in (self.driver, self._aux_driver):
                if driver is not None:
                    driver.flush_checkpoints()

    def _cleanup_step(self, job: _Job, step: str, fn) -> bool:
        """Run one finish cleanup step; a raise is logged and counted, never propagated."""
        try:
            fn()
        except Exception:
            self.finish_cleanup_failures[step] = (
                self.finish_cleanup_failures.get(step, 0) + 1
            )
            logger.exception(
                "request %s: finish step %s failed; still publishing its terminal",
                getattr(job.stats, "request_id", None),
                step,
            )
            return False
        return True

    @staticmethod
    def _store_decode_cache(
        group: Any,
        job: _Job,
        caches=None,
        window_tail=None,
        salt=None,
        stand=False,
        owned=False,
    ) -> bool:
        """Store the cache the reply was decoded into, keyed by the prompt plus the tokens it holds:
        a follow-up that repeats the conversation then prefills only its new message
        (YUNSHU_NEXT_TURN_PREFILL=decode, user 2026-11-02). Fewer stored tokens than the cache
        holds never happens; a cache that disagrees with the emitted tokens is not stored."""
        coordinator = getattr(group.gen, "apc", None)
        if coordinator is None or not job.generated or not caches:
            logger.info("decode cache not stored: no APC, no caches or no reply tokens")
            return False

        offsets = {n for n in (_held_rows(c) for c in caches) if n is not None}
        tokens = list(job.ids) + list(job.generated)
        if len(offsets) != 1:
            logger.info(
                "decode cache not stored: attention caches hold %s rows",
                sorted(offsets),
            )
            return False
        held = offsets.pop()
        if not len(job.ids) < held <= len(tokens):
            logger.info(
                "decode cache not stored: %d rows held for %d prompt + %d reply tokens",
                held,
                len(job.ids),
                len(job.generated),
            )
            return False
        salt = int(job.apc_salt or 0) if salt is None else int(salt)
        if stand:
            # the follow-up is admitted now: hand it the live cache (no copies)
            offer = getattr(coordinator, "offer_standing", None)
            if offer is None:
                return False
            from mlx_vlm.apc import snapshot_prompt_cache_row

            from . import dflash_apc_window

            rows = snapshot_prompt_cache_row(caches, 0, clone=False)
            if rows is None:
                return False
            offer(tokens[:held], rows, extra_hash=salt)
            window = dflash_apc_window.store_decode_window(
                tokens[:held], salt, tail=window_tail
            )
            logger.info(
                "decode cache standing for its follow-up: %d prompt + %d reply tokens "
                "(drafter window %s)",
                len(job.ids),
                held - len(job.ids),
                "stored" if window else "none",
            )
            return True
        store = getattr(coordinator, "store_decode_tail", None)
        if store is None:
            logger.info("decode cache not stored: the APC has no reply-cache store")
            return False
        t0 = time.perf_counter()
        # owned: the APC keeps ``caches`` as they are (a private cut-out copy), no second copy
        extra = {"owned": True} if owned else {}
        stored = store(tokens[:held], caches, extra_hash=salt, **extra)
        t1 = time.perf_counter()
        window = False
        if stored:
            from . import dflash_apc_window

            # the drafter's context window that ends at this cache (DFlash lane)
            window = dflash_apc_window.store_decode_window(
                tokens[:held], salt, tail=window_tail
            )
        t2 = time.perf_counter()
        logger.info(
            "decode cache %s for the next turn: %d prompt + %d reply tokens (drafter window "
            "%s; store %.1f ms, window %.1f ms)",
            "stored" if stored else "not stored",
            len(job.ids),
            held - len(job.ids),
            "stored" if window else "none",
            (t1 - t0) * 1000,
            (t2 - t1) * 1000,
        )
        return bool(stored)

    def _driver_store_decode(self, row: Any) -> bool:
        from .round_driver import flashnext

        """Round driver hook: the last row of a driver leaves finished, its K/V still in its own
        caches (the driver's rows never reach ``_finish``). The same deferred store as the
        speculative lane's. True: the store holds the buffers, the driver must not pool them (the
        flush offers them)."""
        job = row.req.handle
        if (
            settings.get("YUNSHU_NEXT_TURN_PREFILL") != "decode"
            or row.apc is None
            or not row.req.use_apc
        ):
            return False
        skip = [
            why
            for why, bad in (
                ("cache_store off", not row.req.cache_store),
                ("overflowed", bool(getattr(job, "overflowed", False))),
                ("MTP batch row", row.cache is None),
                (
                    # int8 reply caches are not handed to the follow-up (its turn-2 output was
                    # wrong); the prefix cache stays bf16 and the follow-up re-prefills the reply
                    "int8 KV",
                    getattr(getattr(job, "driver", None), "kv_format", "bf16")
                    == "int8",
                ),
                (
                    "MTP head row without its head cache",
                    bool(
                        row.drafting
                        and getattr(job.driver, "dflash", None) is None
                        and flashnext.decode_caches(job.driver, row) is None
                        and getattr(row, "head_cache", None) is None
                    ),
                ),
                ("no reply tokens", len(row.context) <= len(row.req.ids)),
            )
            if bad
        ]
        if skip:
            logger.info("decode cache not stored: %s", ", ".join(skip))
            return False
        group = SimpleNamespace(
            gen=SimpleNamespace(apc=row.apc), driver_salt=int(row.salt or 0)
        )
        # a Flash-Next head row stores its head's KV beside the target's (the drafting salt's layout)
        caches = flashnext.decode_caches(job.driver, row) if row.drafting else None
        if caches is None and getattr(row, "head_cache", None) is not None:
            caches = list(row.cache) + list(
                row.head_cache
            )  # the packed MTP batch's row
        self._defer_decode_cache(group, job, caches=caches or list(row.cache))
        return self._decode_pending_job() is job

    def _decode_pending_job(self):
        pending = self.__dict__.get("_decode_pending")
        return pending[1] if pending is not None else None

    def _defer_decode_cache(self, group: Any, job: _Job, caches=None) -> None:
        """Hold the finished reply's cache instead of copying it into the APC now: the copy
        (~45 ms at 8K) ran ahead of whatever request came next, so the same prompt sent again
        waited for it (8K warm TTFT 20 -> 58 ms). It is stored when the next request is this
        conversation's follow-up (before its lookup) or when the runner goes idle."""
        from . import dflash_apc_window

        if self.__dict__.get("_decode_pending") is not None:
            # A newer reply finished before the runner went idle: the older one is stored
            # now. Dropping it (as before 2026-11-09) made a follow-up restore its reply when
            # it ran alone but re-prefill it when another reply finished first, and decode
            # state and a re-prefill differ in the last bits: concurrent != solo output.
            logger.info("decode cache stored early: a newer reply finished first")
            self._flush_decode_cache()
        coordinator = getattr(group.gen, "apc", None)
        if coordinator is not None and hasattr(coordinator, "take_standing"):
            unclaimed = (
                coordinator.take_standing()
            )  # a standing cache no follow-up claimed
            if unclaimed is not None:
                from .kernels.buffer_cache import offer_restore_buffers

                with contextlib.suppress(Exception):
                    offer_restore_buffers(unclaimed[1])
        pending = (group, job, list(caches), dflash_apc_window.take_decode_tail())
        self._decode_pending = pending
        executor = getattr(self, "_executor", None)
        if executor is None:
            return
        import threading

        def _schedule() -> None:
            with contextlib.suppress(RuntimeError):  # the executor shut down meanwhile
                executor.submit(self._flush_decode_cache_if_idle, pending)

        timer = threading.Timer(DECODE_STORE_IDLE_S, _schedule)
        timer.daemon = True
        timer.start()

    def _flush_decode_cache(self) -> None:
        pending = self.__dict__.pop("_decode_pending", None)
        if pending is None:
            return
        group, job, caches, tail = pending
        staged = owned = False
        try:
            plan = _standing_plan(caches)
            owned = plan == "copy"
            stored = self._store_decode_cache(
                group,
                job,
                caches=caches,
                window_tail=tail,
                salt=getattr(group, "driver_salt", None),
                **({"owned": True} if owned else {}),
            )
            owned = owned and bool(stored)
            if stored and plan is not None:
                staged = _stage_standing(group, job, caches, copy=owned)
        except Exception:  # an optimisation only: the next turn re-prefills
            logger.warning("deferred decode-cache store failed", exc_info=True)
        finally:
            from .kernels.buffer_cache import offer_restore_buffers

            # buffers the APC took (owned) are its own: never pooled for overwriting
            if not staged and not owned:
                with contextlib.suppress(Exception):
                    offer_restore_buffers(caches)

    def _stand_decode_cache(self) -> bool:
        """Hand the held reply's cache to its follow-up's lookup as is (TensorFold's standing
        slot): no APC store copy, no restore copy. Its buffers then belong to the follow-up,
        so they are not offered back to the restore pool."""
        pending = self.__dict__.pop("_decode_pending", None)
        if pending is None:
            return False
        group, job, caches, tail = pending
        try:
            if self._store_decode_cache(
                group,
                job,
                caches=caches,
                window_tail=tail,
                salt=getattr(group, "driver_salt", None),
                stand=True,
            ):
                return True
        except Exception:  # fall back to the copy
            logger.warning("standing decode cache failed", exc_info=True)
        self._decode_pending = pending
        return False

    def _decode_follows(self, job: _Job) -> bool:
        """``job`` continues the held reply (its prompt holds the reply's first token)."""
        pending = self.__dict__.get("_decode_pending")
        if pending is None:
            return False
        prev = pending[1]
        head = list(prev.ids) + list(prev.generated[:1])
        return len(job.ids) > len(head) and list(job.ids[: len(head)]) == head

    @staticmethod
    def _stopped(job: _Job) -> bool:
        return job.abandoned or (
            job.cancel_event is not None and job.cancel_event.is_set()
        )

    def _observe_prefill(self, job: _Job) -> None:
        """Tell the storage tiers how fast prefill really is (their cost model compares a
        restore against it); only a request that prefilled alone and a lot counts."""
        if job.priority < 0:
            return  # auxiliary prompts must not train the agent APC cost model
        disk = getattr(self.apc_manager, "disk", None)
        observe = getattr(disk, "observe_prefill", None)
        st = job.stats
        cost = getattr(self.apc_manager, "media_restore_cost", None)
        if (
            cost is not None
            and getattr(job, "prompt_kwargs", None) is not None
            and st.t_first
            and st.t_admit
            and self._active_jobs() == 0
        ):
            cost.observe(len(job.ids), st.cached_tokens, st.t_first - st.t_admit)
        if (
            observe is None
            or not st.t_first
            or not st.t_admit
            or self._active_jobs() > 0
        ):
            return
        reload_s = (st.cache_reload_ms or 0.0) / 1000.0
        # tokens computed this request: the prompt minus what the cache served (prefill_total
        # can still count the cached ones while the prompt is being processed)
        fresh = len(job.ids) - st.cached_tokens
        observe(fresh, st.t_first - st.t_admit - reload_s)

    def _note_cache(self, job: _Job) -> None:
        """Record the tier and lookup time of ``job``'s prefix-cache hit (per-request
        provenance in ``x_yunshu.cache``)."""
        mgr = self.apc_manager
        take = getattr(mgr, "take_lookup", None)
        if take is None or job.stats.cache_tier is not None:
            return
        rec = take(len(job.ids), job.stats.cached_tokens, since=job.stats.t_admit)
        if rec is not None:
            job.stats.cache_reason = "hit" if rec.cached else "no_reusable_prefix"
            job.stats.cache_tier = rec.tier
            job.stats.cache_reload_ms = rec.ms
            job.stats.cache_device = rec.device
            job.stats.latency_marks["prefill_start"] = rec.t

    def _note_driver_prefill(self, driver=None) -> None:
        """Publish prefill progress of the round driver's rows."""
        try:
            for row in (driver or self.driver).rows:
                st = row.req.handle.stats
                if st.t_first == 0.0:
                    st.prefill_done = min(row.done - row.hit, st.prefill_total)
                    if st.prefill_done >= st.prefill_total and not st.t_prefill_end:
                        st.t_prefill_end = time.perf_counter()
        except Exception:
            logger.debug("driver prefill progress unavailable", exc_info=True)

    def memory_held(self) -> int:
        """Requests waiting in the queue for memory headroom right now."""
        with self._lock:
            return sum(1 for j in self._pending if j.stats.queue_reason == "memory")

    def phase_counts(self) -> dict[str, int]:
        """Requests by phase right now: waiting for admission, prefilling, decoding."""
        with self._lock:
            queued = len(self._pending)
        jobs = list(self._driver_jobs.values())
        prefill = sum(1 for j in jobs if not j.stats.t_first)
        return {"queued": queued, "prefill": prefill, "decode": len(jobs) - prefill}

    def busy_snapshot(self) -> dict:
        """Cumulative GPU-busy accounting: all slices and the round driver's share."""
        return {
            "slices": self.busy_meter.snapshot(),
            "round_driver": self.driver_busy_meter.snapshot(),
        }

    def _committed_bytes(self, running: list) -> int:
        """Memory the running requests will still grow into: their reservation minus what
        their KV already occupies (the live host reading counts the occupied part)."""
        gate = self.memory_gate
        total = 0
        for j in running:
            if j.mem_est:
                done = (
                    j.stats.cached_tokens + j.stats.prefill_done + j.stats.generated
                ) * gate.kv_per_token
                total += max(0, j.mem_est - done)
        return total

    def _memory_wait(self, job: _Job, active: list) -> bool:
        """Hold ``job`` in the queue (True) while the host has no headroom for it. The first
        request of an idle runner is always admitted; the reason is reported through
        ``stats.queue_reason`` (x_yunshu, the console queue, the metrics)."""
        gate = self.memory_gate
        if gate is None:
            return False
        try:
            running = [j for j in active if not j.terminal]
            if job.mem_est == 0:
                job.mem_est = gate.estimate(len(job.ids), job.max_tokens)
            decision = gate.decide(
                job.mem_est, self._committed_bytes(running), any_active=bool(running)
            )
        except Exception:
            logger.debug("memory gate failed; admitting", exc_info=True)
            return False
        if decision.refuse:
            from .memory_admission import MemoryAdmissionError, PromptTooLongError

            logger.warning("Memory admission: request refused (%s)", decision.detail)
            n = len(job.ids)
            if decision.limit_bytes is not None:
                fits = gate.max_prompt_tokens(n, job.max_tokens, decision.limit_bytes)
                if fits < n:
                    raise PromptTooLongError(n, fits, decision.detail)
            raise MemoryAdmissionError(decision.detail)
        if decision.admit:
            if job.mem_held_at:
                job.stats.extra["memory_wait_ms"] = round(
                    (time.perf_counter() - job.mem_held_at) * 1000, 1
                )
            job.stats.queue_reason = None
            job.stats.queue_detail = None
            return False
        if not job.mem_held_at:
            job.mem_held_at = time.perf_counter()
            gate.holds += 1
            logger.warning(
                "Memory admission: request held in queue (%s)", decision.detail
            )
        job.stats.queue_reason = decision.reason
        job.stats.queue_detail = decision.detail
        return True

    def _shed_under_pressure(self) -> None:
        """Host memory is short: give APC RAM entries back (to SSD where configured) before
        the kernel starts swapping."""
        gate = self.memory_gate
        shed = getattr(self.apc_manager, "shed_ram", None)
        now = time.monotonic()
        if gate is None or shed is None or now - self._last_shed < 1.0:
            return
        self._last_shed = now
        want = gate.pressure_shortfall()
        if want <= 0:
            return
        freed = int(shed(want))
        if freed:
            gate.sheds += 1
            gate.shed_bytes += freed
            logger.warning(
                "Memory pressure: shed %.2f GiB of APC RAM entries (short by %.2f GiB)",
                freed / (1 << 30),
                want / (1 << 30),
            )

    def _prefix_wait(self, job: _Job) -> bool:
        """Wait only for existing, exact hybrid checkpoint positions.

        The serialized executor owns this table. No shared mutable KV, sampling
        state or future survives a producer failure. Releasing always returns
        through the normal content/media/kernel-validated APC lookup.
        """
        if (
            not self.singleflight
            or not self.prefix_invariant
            or self.apc_manager is None
        ):
            return False
        for producer, points in self._prefix_producers:
            cancel = producer.cancel_event
            if (
                producer.terminal
                or producer.abandoned
                or producer.stats.t_first
                or (cancel is not None and cancel.is_set())
                or producer.salt != job.salt
                or (producer.cache_plan is None) != (job.cache_plan is None)
            ):
                continue
            eligible = [
                n
                for n in points
                if n < len(job.ids) and n > producer.stats.cached_tokens
            ]
            if job.cache_plan is not None:
                requested = set(
                    job.cache_plan.get(
                        "lookup_points", [n for n, _ in job.cache_plan["points"]]
                    )
                )
                eligible = [n for n in eligible if n in requested]
            for n in reversed(eligible):
                if job.ids[:n] != producer.ids[:n]:
                    continue
                # Demand uses an existing canonical position and matching
                # earlier split plan, never invents a recurrent checkpoint.
                if producer.cache_plan is not None and job.cache_plan is not None:

                    def prior(plan):
                        return tuple(p for p, _ in plan["points"] if p <= n)

                    if prior(producer.cache_plan) != prior(job.cache_plan):
                        continue
                ready = getattr(self.apc_manager, "checkpoint_ready", None)
                if ready is not None and ready(producer.ids[:n], producer.apc_salt):
                    return False
                # Prefill progress alone is insufficient when publication is
                # deferred. Completion/error/cancel above bounds this wait;
                # missing snapshots eventually fall back to a cold admission.
                if "prefix_wait_tokens" not in job.stats.extra:
                    job.stats.extra["prefix_wait_tokens"] = n
                    logger.info(
                        "APC single-flight wait: prefix=%d producer=%s waiter=%s",
                        n,
                        id(producer),
                        id(job),
                    )
                return True
        return False

    def _track_prefix_producer(self, job: _Job):
        if not job.stats.used_apc or not job.cache_store:
            return  # stores nothing: no producer to wait for
        if id(job) in self._driver_jobs:
            # The driver owns its APC: its planned checkpoints (stored by its prefill steps and
            # published after the first token) are what a same-prefix waiter waits for.
            planned = (job.driver or self.driver).planned_checkpoints(job)
            if planned is not None and planned[1]:
                job.apc_salt = planned[0]
                self._prefix_producers.append((job, planned[1]))

    def _drive_slice(self, resubmit: bool = True) -> None:
        """One scheduling slice on the MLX thread, timed into ``busy_meter``."""
        with self.busy_meter.span():
            self._drive_slice_body(resubmit)

    def _busy_for_decode_store(self) -> bool:
        with self._lock:
            return bool(self._pending) or bool(self.__dict__.get("_driver_jobs"))

    def _flush_decode_cache_if_idle(self, pending) -> None:
        """The idle timer's store, on the MLX thread: only if that reply is still held and
        nothing else is queued or running."""
        if self.__dict__.get("_decode_pending") is pending and not (
            self._busy_for_decode_store()
        ):
            self._flush_decode_cache()

    def _drive_slice_body(self, resubmit: bool) -> None:
        # Jobs already popped from ``_pending`` live only in these locals until they are
        # admitted or re-queued; the failure handler below must still reach them.
        pending: list[_Job] = []
        waiting: list[_Job] = []
        try:
            with self._lock:
                cancelled_pending = [j for j in self._pending if self._stopped(j)]
                self._pending = [j for j in self._pending if not self._stopped(j)]
                active = list(self._driver_jobs.values())
                # A call still preparing on the MLX executor has no row yet.
                # Conservatively treat it as primary until its metadata arrives.
                preparing = self.inflight() > len(active) + len(self._pending)
                primary = preparing or any(
                    j.priority >= 0 for j in [*active, *self._pending]
                )
                now = time.perf_counter()
                # background (priority < 0) requests wait while interactive work exists, until they age
                pending = [
                    j
                    for j in self._pending
                    if j.priority >= 0 or not primary or now - j.last_service >= AGING_S
                ]
                selected_ids = {id(j) for j in pending}
                self._pending = [j for j in self._pending if id(j) not in selected_ids]
                auxiliary = sum(
                    j.priority < 0 for j in [*active, *self._pending, *pending]
                )
            for job in cancelled_pending:
                job.stats.finish_reason = (
                    "cancel"
                    if job.cancel_event is not None and job.cancel_event.is_set()
                    else None
                )
                self._emit(job, _DONE)
            alone = (
                len(pending) == 1
                and not any(j.priority >= 0 for j in active)
                and (not active or pending[0].priority >= 0)
                and self.inflight() - auxiliary <= 1
            )
            self._prefix_producers = [
                (j, points)
                for j, points in self._prefix_producers
                if not j.terminal and not j.stats.t_first
            ]
            admitted: list[_Job] = []
            for job in list(pending):
                # A job leaves ``pending`` once it is admitted, parked in ``waiting`` or
                # terminal, so the failure handler below only reaches jobs not yet handled.
                cancelled = job.cancel_event is not None and job.cancel_event.is_set()
                if job.abandoned or cancelled:
                    # Never prefill a request nobody is waiting for.
                    job.stats.finish_reason = "cancel" if cancelled else None
                    self._emit(job, _DONE)
                    pending.remove(job)
                    continue
                try:
                    memory_hold = self._memory_wait(job, [*active, *admitted])
                except MemoryError as exc:  # refused: cannot fit even alone
                    self._emit(job, exc)
                    pending.remove(job)
                    continue
                if memory_hold:
                    waiting.append(job)
                    pending.remove(job)
                    continue
                try:
                    if self._prefix_wait(job):
                        waiting.append(job)
                        pending.remove(job)
                        continue
                    admitted.append(job)
                    self._admit(job, alone)
                    self._track_prefix_producer(job)
                except Exception as exc:
                    logger.exception("VLM runner admission failed")
                    self._emit(job, exc)
                pending.remove(job)
            if waiting:
                with self._lock:
                    self._pending = waiting + self._pending
                waiting = []
            pending = []
            try:
                self._shed_under_pressure()
            except Exception:
                logger.debug("pressure shedding failed", exc_info=True)
            if self._driver_jobs:
                self._step_driver(primary)
        except Exception as exc:
            logger.exception("VLM runner step failed; failing active requests")
            for job in [*pending, *waiting]:
                self._emit(job, exc)
            for job in self._driver_jobs.values():
                self._emit(job, exc)
                if self.driver is not None:
                    (job.driver or self.driver).remove(job)
            self._driver_jobs.clear()
        if not resubmit:
            return
        with self._lock:
            if self._pending or self._driver_jobs:
                again = True
            else:
                self._driving = False
                again = False
        if again:
            self._schedule()
        else:
            self._drained()

    def _drained(self) -> None:
        if self._executor is not None:
            if self._idle_memory is None:
                self._idle_memory = IdleMemory(self._executor.submit, self.busy)
            self._idle_memory.drained()
        if self.clear_on_idle:
            # Large models: release the buffer pool once everything drains
            # (clearing under active batches would only force reallocation). Up to
            # YUNSHU_PREFILL_BUFFER_CACHE_GB stays: the next request's cache restore
            # reuses it (a cleared pool costs ~45 ms per 32K-token cache copy).
            with contextlib.suppress(Exception):
                from .kernels import buffer_cache

                mx.synchronize()
                buffer_cache.clear_if_over()


# Uids of the rows the upstream batch is sampling right now (set by the
# wrappers below). Upstream calls the sampler for the whole batch and passes
# row_ids=[0]*n, so a per-row sampler has no other way to know which request
# each row is.
_STEP_UIDS: list | None = None


@dataclass
class RowParams:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    seed: int | None = None
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.0
    xtc_special_tokens: list | None = None


_DONE = object()
# Undelivered tokens a job may hold before it is cancelled (slow / stuck client).
OUT_LIMIT = 65536
# a finished reply's decode cache is copied into the APC once the runner has been idle this
# long (or at once when its follow-up arrives): a request sent right away does not wait on it
DECODE_STORE_IDLE_S = 0.25


def _held_rows(c) -> int | None:
    """Rows a reply's attention cache holds. KVCache: an int offset; the lane's single-row
    BatchKVCache: _idx past its left padding; None for a cache without keys."""
    if getattr(c, "keys", None) is None:
        return None
    idx = getattr(c, "_idx", None)
    pad = getattr(c, "left_padding", None)
    if idx is not None and pad is not None:
        return int(idx) - int(pad.reshape(-1)[0].item())
    off = getattr(c, "offset", None)
    if off is None:
        return None
    return int(off) if isinstance(off, int) else int(off.reshape(-1)[0].item())


# Spare keys a reply's standing copy keeps past its rows: its follow-up appends its new message in
# place when it fits (no restore copy; prefill attention reads the same bits at any KV capacity, probe
# 1112-233219-00-fu-kv-stride-probe). 128 MiB per 27B reply; a longer message pays the one copy its
# own prefill (> 2 s) dwarfs. Only the standing copy has it: the cut-out and the stored checkpoint stay
# exact (a cut-out with this room raised the 96K memory-suite peak by 7 GiB, fu-mem-attrib 2026-11-13).
STANDING_ROOM = 2048


def _standing_plan(caches: list) -> str | None:
    """How the idle store can leave a reply standing for its follow-up: "own" (the caches already
    have STANDING_ROOM spare keys, e.g. a DFlash row decoded in place: the APC copies them as before
    and they stand), "copy" (exact cut-out caches: the APC keeps them as they are, and the one copy
    the store made before becomes the standing copy, with room), None (another layout: as before)."""
    from mlx_vlm.models.cache import ArraysCache, KVCache

    spare: set = set()
    for c in caches:
        if type(c) is ArraysCache:
            continue
        if type(c) is not KVCache or c.keys is None or c.values is None:
            return None
        spare.add(int(c.keys.shape[-2]) - int(c.offset))
    if not spare:
        return None
    if min(spare) >= STANDING_ROOM:
        return "own"
    return "copy" if spare == {0} else None


def _standing_copy(caches: list, n: int) -> list | None:
    """New cache objects for a standing reply: K/V copied into buffers with STANDING_ROOM zero
    keys after the ``n`` rows (one layer at a time), recurrent state as separate handles (the
    follow-up's prefill replaces its cache's arrays: the stored checkpoint's must not change)."""
    import mlx.core as mx
    from mlx_vlm.models.cache import ArraysCache, KVCache

    from .apc_manager import _single_native_arrays_row

    out: list = []
    for c in caches:
        if type(c) is ArraysCache:
            row = _single_native_arrays_row(c)
            if row is None:
                return None
            out.append(row)
            continue
        kv = KVCache()
        parts = []
        for a in (c.keys, c.values):
            pad = mx.zeros((*a.shape[:2], STANDING_ROOM, a.shape[3]), dtype=a.dtype)
            parts.append(mx.concatenate([a[..., :n, :], pad], axis=2))
        mx.eval(*parts)
        kv.keys, kv.values = parts
        kv.offset = n
        out.append(kv)
    return out


def _stage_standing(group: Any, job: Any, caches: list, *, copy: bool = False) -> bool:
    """After the idle store, leave the reply standing for its follow-up, which then resumes it in
    place: no restore copy (40 ms of a 33K follow-up's 51 ms lookup, 2026-11-12), while the stored
    checkpoint still serves a retry or a branch. Memory as before: the reply's buffers held standing
    went to the restore pool; with ``copy`` the store kept them and this copy is the one the store
    used to make. False (the caller pools uncopied buffers) when the pool would not keep them."""
    coordinator = getattr(group.gen, "apc", None)
    offer = getattr(coordinator, "offer_standing", None)
    if offer is None:
        return False
    from .kernels.buffer_cache import pool_accepts

    held = {n for n in (_held_rows(c) for c in caches) if n is not None}
    if len(held) != 1:
        return False
    n = next(iter(held))
    size = sum(
        int(a.nbytes)
        for c in caches
        for a in (getattr(c, "keys", None), getattr(c, "values", None))
        if a is not None and not isinstance(a, tuple)
    )
    if copy:
        size = size * (n + STANDING_ROOM) // max(1, n)
    if not pool_accepts(size):
        return False
    # "own": plain single-row caches the APC copied: they stand as they are (a row snapshot would
    # view them cut to their rows and drop the room)
    rows = _standing_copy(caches, n) if copy else list(caches)
    if rows is None:
        return False
    salt = getattr(group, "driver_salt", None)
    salt = int(job.apc_salt or 0) if salt is None else int(salt)
    offer((list(job.ids) + list(job.generated))[:n], rows, extra_hash=salt)
    logger.info("decode cache standing after its store: %d tokens", n)
    return True


@dataclass
class _Job:
    ids: list[int]
    max_tokens: int
    greedy: bool
    sampling: RowParams | None
    use_draft: bool
    logprobs: bool
    top_logprobs: int
    processors: list
    prompt_kwargs: dict | None
    salt: int | None
    seed: int | None
    cancel_event: Any
    stats: RunStats
    prompt_logprobs: int | None = None
    priority: int = 0
    queued: float = 0.0
    last_service: float = 0.0
    driver: Any = None
    cache_plan: dict | None = None
    cache_store: bool = (
        True  # False: look the prefix up, store nothing (X-Yunshu-Cache-Store)
    )
    apc_salt: int | None = None
    out: queue.Queue = field(default_factory=queue.Queue)
    start: float = 0.0
    abandoned: bool = False
    budget: Any = None
    allow_draft: bool = False
    ngram: bool = (
        False  # n-gram / suffix speculation requested (generic layer, greedy rows)
    )
    guide: Any = None
    terminal: bool = False
    overflowed: bool = False
    first_delivery: threading.Event | None = None
    first_delivery_waited: bool = False
    mem_est: int = 0  # bytes the memory gate reserved for this request (KV + workspace)
    mem_held_at: float = 0.0
    generated: list = field(
        default_factory=list
    )  # emitted token ids (the decode cache's tail)
