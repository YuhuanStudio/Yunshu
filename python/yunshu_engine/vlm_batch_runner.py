# Patches upstream mlx-vlm symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""VLM text runner on upstream mlx-vlm ``BatchGenerator`` with shared batching.

One decode path for a loaded VLM target that combines, per request:

- APC prefix reuse (exact + recurrent checkpoints for hybrid models),
- a speculative draft (greedy, no logits processors) — the checkpoint's native
  MTP head or an external DFlash2 block-diffusion drafter; token-identical to AR,
- mlx-native sampling (temperature / top-p / top-k / min-p / seed),
- logits processors: repetition / presence / frequency penalties, logit bias and
  Yunshu's grammar / JSON-schema constraints.

Measured on Qwen3.8-27B (docs/research/runs/2026-09-28-matrix/): APC + MTP in one
BatchGenerator stays token-identical to AR while a repeated 5K prompt drops from
5.4 s to 0.11 s and long decode rises from 32 to ~57 tok/s.

The runner only yields token ids. Detokenizing, stop strings, thinking state and
queue delivery stay in ``VLMEngine`` so every text path reports the same way.
GPU work runs on the serialized MLX executor thread; consumers read their
tokens from a queue on any other thread.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np

from . import keyed_sampling, settings
from .keyed_sampling import top_k_filter, top_p_filter
from .serving.busy_time import BusyMeter
from .serving.work_scheduler import (
    AGING_S,
    DECODE_QUANTUM_S,
    PRIMARY_HANDOFF_S,
    Work,
)

logger = logging.getLogger(__name__)

# Upstream prefill default; APC checkpoints land on these chunk boundaries and
# cancellation is honoured between chunks (~2 s at 2K tokens on a 27B model).
PREFILL_STEP = 2048


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
    finish_reason: str | None = None
    used_apc: bool = False
    used_draft: bool = False
    # Per-token {"token_id", "logprob", "top_logprobs": [...]} when requested.
    last_logprob: dict | None = None
    extra: dict = field(default_factory=dict)
    # Timing / progress (time.perf_counter() values; 0.0 = not reached yet). The
    # gateway reads them live for prefill-progress events and the per-response
    # ``x_yunshu`` stats.
    t_submit: float = 0.0  # handed to the runner
    t_admit: float = 0.0  # left the queue, prefill started
    t_first: float = 0.0  # first generated token
    t_last: float = 0.0  # latest generated token
    prefill_done: int = 0  # prompt tokens computed so far (cache hits excluded)
    prefill_total: int = 0  # prompt tokens to compute (cache hits excluded)
    spec_mode: str | None = None  # "mtp" / "dflash" while a drafter is in use
    spec_drafted: int = 0
    spec_accepted: int = 0
    spec_rounds: int = 0  # verify rounds this request took part in
    spec_copy_rounds: int = 0  # of those, rounds that verified a copied run
    spec_copy_tokens: int = 0  # tokens those rounds committed

    @property
    def phase(self) -> str:
        if self.finish_reason is not None:
            return "done"
        if not self.t_admit:
            return "queued"
        if not self.t_first:
            return "prefill"
        return "decode"


class ConstraintProcessor:
    """Adapt a Yunshu grammar/JSON constraint to a BatchGenerator logits processor.

    BatchGenerator calls ``__call__`` for the first token (prompt stage) and
    ``process_last_token`` for every later step with the token it just sampled,
    so the constraint advances before masking the next distribution.
    """

    def __init__(self, constraint: Any, tokenizer: Any):
        from .json_schema import apply_json_constraint

        self._constraint = constraint
        self._tokenizer = tokenizer
        self._apply = apply_json_constraint
        self._generated: list[int] = []

    def _mask(self, logits: mx.array) -> mx.array:
        fast = getattr(self._constraint, "allowed_mask", None)
        if fast is not None:
            mask = fast(self._tokenizer, logits.shape[-1])
            if mask is not None:
                return mx.where(mask, logits, mx.array(float("-inf"), logits.dtype))
        allowed = self._constraint.get_allowed_tokens(self._tokenizer, self._generated)
        if not allowed:
            raise ValueError("Grammar constraint has no valid next token")
        return self._apply(logits, allowed)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        return self._mask(logits)

    def process_last_token(self, token: int, logits: mx.array) -> mx.array:
        token = int(token)
        self._generated.append(token)
        self._constraint.advance(self._tokenizer.decode([token]))
        return self._mask(logits)


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
        return None  # BatchGenerator's greedy argmax (fused when possible)
    return make_sampler(
        temp=float(temperature),
        top_p=float(top_p) if top_p and top_p < 1.0 else 0.0,
        min_p=float(min_p or 0.0),
        top_k=int(top_k or 0),
    )


def build_penalty_processors(
    repetition_penalty: float,
    frequency_penalty: float,
    presence_penalty: float,
    logit_bias: dict[int, float] | None,
) -> list:
    if (
        repetition_penalty in (None, 1.0)
        and not frequency_penalty
        and not presence_penalty
        and not logit_bias
    ):
        return []
    from mlx_lm.sample_utils import make_logits_processors

    return list(
        make_logits_processors(
            logit_bias=logit_bias or None,
            repetition_penalty=(
                repetition_penalty if repetition_penalty not in (None, 1.0) else None
            ),
            presence_penalty=presence_penalty or None,
            frequency_penalty=frequency_penalty or None,
        )
    )


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

    @staticmethod
    def active(**kw) -> bool:
        return bool(
            kw.get("suppress")
            or kw.get("min_tokens")
            or kw.get("ignore_eos")
            or kw.get("top_n_sigma")
        )

    def _block(self, ids: mx.array, vocab: int) -> mx.array:
        return mx.zeros((vocab,), dtype=mx.bool_).at[ids].add(True)

    def _mask(self, logits: mx.array) -> mx.array:
        # Functional (never writes into the caller's logits).
        neg = mx.array(float("-inf"), logits.dtype)
        vocab = logits.shape[-1]
        if self._suppress.size:
            logits = mx.where(self._block(self._suppress, vocab), neg, logits)
        if self._eos.size and (self._ignore_eos or self._generated < self._min_tokens):
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
        self.prefix_invariant = prefix_invariant
        logger.info(
            "APC admission configuration: qualified=%s enabled=%s source=%s",
            prefix_invariant,
            self.singleflight,
            __file__,
        )
        self.processor = processor
        self.apc_manager = apc_manager
        self.apc_semantic_hash = apc_semantic_hash
        self.drafter = drafter
        self.draft_kind = draft_kind
        self.draft_block_size = draft_block_size
        # Callable(input_ids) -> bool: skip APC when its checkpoints cannot fit,
        # so a cold request does not pay APC bookkeeping for nothing.
        self._apc_admit = apc_admit
        # The serialized MLX executor the driver runs on (None: drive inline).
        self._executor = executor
        self._lock = threading.Lock()
        self._pending: list[_Job] = []
        self._prefix_producers: list[tuple[_Job, list[int]]] = []
        # Shared batches keyed by (top_logprobs_k, use_apc, priority). Each
        # priority may retain a speculative lane; only the highest steps.
        self._batches: dict[tuple, _Group] = {}
        self._spec: _Group | None = None
        self._aux_spec: _Group | None = None
        self._vocab_owner: Any = None
        self._driving = False
        self._decode_debt = 0.0
        self._primary_handoff_at: float | None = None
        self.clear_on_idle = False
        # Requests the engine has accepted, including ones still being
        # prepared (templating, image encoding) — the runner alone cannot see
        # those, and a request is only "alone" if the engine has no others.
        self.inflight = lambda: 0
        # All ids that end a turn (tokenizer + generation_config eos).
        self.stop_tokens: set[int] | None = None
        # Per-row-length KV layout for the shared batch and the speculative
        # lane (models with qwen3_5 attention): None (stock caches) or its
        # precision, "bf16" / "int8" (YUNSHU_KV_PRECISION).
        self.ragged_kv: str | None = None
        self._ragged_logged = False
        # Yunshu's round driver (YUNSHU_ROUND_DRIVER, round_driver/): serves
        # text requests of dense Qwen3.5-family models; None = upstream only.
        self.driver: Any = None
        self._aux_driver: Any = None
        self._driver_jobs: dict[int, _Job] = {}
        # GPU-busy seconds: every executor slice, and the round driver's steps inside them
        # (``/v1/yunshu/status`` and ``/metrics`` report the cumulative counters).
        self.busy_meter = BusyMeter()
        self.driver_busy_meter = BusyMeter()

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

        raw = prepare_inputs(
            self.processor,
            images=image_paths or None,
            audio=audio or None,
            prompts=prompt,
            image_token_index=getattr(self.model.config, "image_token_index", None),
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
                image_hash=(
                    _apc.hash_image_payload(pixel_values=pixel_values)
                    if pixel_values is not None
                    else 0
                ),
                media={
                    "audio": raw.get("input_features"),
                    "video": raw.get("pixel_values_videos"),
                },
                model=self.model.language_model,
                processor=self.processor,
            )
        return input_ids[0], kwargs, salt

    # ── Scheduling ──────────────────────────────────────────────────────
    #
    # Requests submit a job and read tokens from their own queue on any thread.
    # One driver owns the GPU: it runs on the serialized MLX executor in short
    # slices (admit, one step per batch, dispatch) and resubmits itself while
    # work remains, so prep for new requests (templating, image encoding)
    # interleaves.
    #
    # - Every request joins one shared continuous batch (upstream
    #   BatchGenerator) with its own sampling params and seed (RowSampler),
    #   its own logits processors, and one-request-at-a-time prefill (as oMLX
    #   does; upstream's mixed warm/cold multi-row prefill mis-assigned rows).
    # - A request that is alone uses speculative decoding (MTP / DFlash) in an
    #   exclusive lane; spec on == spec off holds there (batch-invariant
    #   kernels). Requests arriving meanwhile go to the shared batch.
    #   Upstream keeps drafter/round state for exactly one speculative batch
    #   and cannot add rows to it, so multi-row speculation needs our own
    #   round driver (not done yet).

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
        prompt_kwargs: dict | None = None,
        apc_semantic_hash: int | None = None,
        cancel_event: Any = None,
        stats: RunStats | None = None,
        logprobs: bool = False,
        top_logprobs: int = 0,
        thinking_budget: int | None = None,
        prompt_preopens_thinking: bool = False,
        thinking_start_token: str = "<think>",
        thinking_end_token: str = "</think>",
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        xtc_special_tokens: list | None = None,
        guide: Any = None,
        prompt_cache_plan: dict | None = None,
    ) -> Iterator[int]:
        """Yield generated token ids; ``stats`` is filled in as generation runs.

        Must not be called on the MLX executor thread when the runner has an
        executor: the driver needs that thread.
        """
        stats = stats if stats is not None else RunStats()
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        stats.prompt_tokens = len(ids)
        greedy = temperature is None or temperature < 1e-6
        processors = list(logits_processors or [])
        # Upstream drops logprobs while drafting, so logprob requests decode AR.
        # A thinking budget forces "\n</think>" through upstream's
        # ThinkingBudgetCriteria, which only the non-speculative batch applies.
        # Sampled requests draft too: their tokens are drawn by a position-keyed sampler, so
        # accepting a draft equals what serial sampling would have produced (keyed_sampling).
        keyed_ok = greedy or keyed_sampling.supports(
            RowParams(
                float(temperature),
                float(top_p),
                int(top_k),
                float(min_p),
                seed,
                float(xtc_probability or 0.0),
            )
        )
        use_draft = bool(
            allow_draft
            and self.drafter is not None
            and keyed_ok
            and not processors
            and not logprobs
            and thinking_budget is None
            and (guide is None or self._lane_takes_guide(guide))
        )
        if logprobs:
            processors.append(Fp32LogitsProcessor())
        budget = None
        if thinking_budget is not None:
            from mlx_vlm.utils import ThinkingBudgetCriteria

            budget = ThinkingBudgetCriteria(
                getattr(self.processor, "tokenizer", self.processor),
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
            logprobs=bool(logprobs),
            top_logprobs=int(top_logprobs or 0) if logprobs else 0,
            processors=processors,
            guide=guide,
            prompt_kwargs=prompt_kwargs,
            salt=apc_semantic_hash,
            seed=seed,
            cancel_event=cancel_event,
            stats=stats,
            priority=getattr(cancel_event, "scheduling_priority", 0),
            cache_plan=prompt_cache_plan,
            budget=budget,
        )
        # The round driver drafts for any row without logits processors, logprobs
        # or a tool-call guide (a thinking budget is fine there); sampled rows draw
        # with the position-keyed sampler, so a draft is accepted exactly when serial
        # sampling would have produced it.
        job.allow_draft = bool(
            allow_draft and not processors and not logprobs and guide is None
        )
        stats.used_draft = use_draft
        stats.t_submit = time.perf_counter()
        with contextlib.suppress(Exception):
            # The gateway reaches the live stats through the cancel event it owns.
            cancel_event.run_stats = stats
        if not greedy and use_draft:
            job.keyed = keyed_sampling.KeyedSampler(
                job.sampling,
                seed if seed is not None else int.from_bytes(os.urandom(8), "little"),
            )
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

    def _lane_takes_guide(self, guide: Any) -> bool:
        """The speculative lane masks a tool-call guide's verify window itself when the
        row is greedy MTP and starts unconstrained (its first token is sampled
        before the lane runs)."""
        from . import mtp_lane

        return (
            self.draft_kind == "mtp"
            and mtp_lane.can_guide(self.drafter)
            and not guide.constrained
        )

    def _emit(self, job: _Job, item: Any) -> None:
        """Hand one item to a job's consumer; exactly one terminal per job.

        Data past OUT_LIMIT (a consumer that stopped reading) ends the job with
        an error instead of growing without bound; the row is dropped at the
        next slice. Never blocks the executor thread.
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
        job.out.put(item)

    def _fail_all(self, exc: BaseException) -> None:
        """Terminate every pending / active job with ``exc`` (executor gone)."""
        with self._lock:
            pending, self._pending = self._pending, []
            self._driving = False
        for job in pending:
            self._emit(job, exc)
        for group in self._groups():
            for job in group.jobs.values():
                self._emit(job, exc)
            with contextlib.suppress(Exception):
                group.gen.close()
        self._batches.clear()
        self._spec = None
        self._aux_spec = None
        for job in self._driver_jobs.values():
            self._emit(job, exc)
            if self.driver is not None:
                with contextlib.suppress(Exception):
                    (job.driver or self.driver).remove(job)
        self._driver_jobs.clear()

    def _schedule(self) -> None:
        try:
            self._executor.submit(self._drive_slice)
        except Exception as exc:
            logger.exception("VLM runner could not schedule a slice")
            self._fail_all(exc)

    def busy(self) -> bool:
        with self._lock:
            return bool(
                self._pending
                or self._batches
                or self._spec
                or self._aux_spec
                or self._driving
                or self._driver_jobs
            )

    def _submit(self, job: _Job) -> None:
        with self._lock:
            job.queued = job.last_service = time.perf_counter()
            self._pending.append(job)
            if self._executor is None or self._driving:
                return
            self._driving = True
        self._schedule()

    def _groups(self) -> list[_Group]:
        return [g for g in (self._spec, self._aux_spec) if g is not None] + list(
            self._batches.values()
        )

    def _active_jobs(self) -> int:
        return sum(len(g.jobs) for g in self._groups()) + len(self._driver_jobs)

    def _new_generator(
        self,
        *,
        spec: bool,
        use_apc: bool,
        top_logprobs: int,
        sampler,
        greedy=True,
        auxiliary=False,
    ):
        from mlx_vlm.generate.ar import BatchGenerator

        from .cache_prefill import install

        install()
        if self.prefix_invariant:
            from .cache_decode import install as install_decode

            install_decode()

        manager = self.apc_manager if use_apc else None
        if auxiliary and callable(getattr(self.apc_manager, "coordinator", None)):
            from .serving.auxiliary_prefill import AuxiliaryPrefillPolicy

            manager = AuxiliaryPrefillPolicy(self.apc_manager)
        gen = BatchGenerator(
            self.model.language_model,
            self.processor,
            stop_tokens=self.stop_tokens,
            sampler=sampler,
            apc_manager=manager,
            draft_model=self.drafter if spec else None,
            draft_kind=self.draft_kind if spec else None,
            draft_block_size=self.draft_block_size if spec else None,
            greedy_sampling=spec and greedy,
            compute_logprobs=not spec,
            top_logprobs_k=top_logprobs,
            prefill_step_size=PREFILL_STEP,
            prefill_batch_size=1,
        )
        # Upstream binds a stock APCCoordinator; ours places the extra checkpoints
        # (end of the system turn) and numbers requests for superseding.
        bind = getattr(manager, "coordinator", None)
        if (
            manager is not None
            and bind is not None
            and getattr(gen, "apc", None) is not None
        ):
            gen.apc = bind(gen.model)
        return gen

    def _admit(self, job: _Job, alone: bool) -> None:
        from .mrope import clear_rope_state

        job.stats.t_admit = time.perf_counter()
        job.stats.prefill_total = len(job.ids)

        if job.guide is not None:
            lane = (
                job.use_draft
                and alone
                and self._spec is None
                and not (self.driver is not None and job.prompt_kwargs is None)
            )
            if not lane:
                # Rows that decode one token at a time mask through a processor.
                from .tool_call_grammar import ToolCallProcessor

                job.use_draft = job.allow_draft = False
                job.processors = [*job.processors, ToolCallProcessor(job.guide)]

        use_apc = self.apc_manager is not None and job.priority >= 0
        if job.cache_plan is not None and job.cache_plan.get("writes") == []:
            use_apc = False
        if use_apc and self._apc_admit is not None:
            try:
                use_apc = bool(self._apc_admit(mx.array(job.ids)))
            except Exception:
                logger.debug("APC admission check failed; using APC", exc_info=True)
        if (
            self.driver is not None
            and job.prompt_kwargs is None
            and job.cache_plan is None
        ):
            self._admit_driver(job, use_apc)
            return
        job.stats.used_apc = use_apc
        spec_lane = self._aux_spec if job.priority < 0 else self._spec
        spec = job.use_draft and alone and spec_lane is None
        if not spec:
            job.use_draft = False
            job.stats.used_draft = False
        else:
            job.stats.spec_mode = _spec_mode(self.drafter)
            job.spec_base = _spec_counters(self.drafter)
            job.spec_last = job.spec_base
        pkw = job.prompt_kwargs
        if pkw is None:
            if alone:
                clear_rope_state(self.model)
            pkw = self.model.get_input_embeddings(
                mx.array(job.ids)[None], None, mask=None
            ).to_dict()
        salt = job.salt if job.salt is not None else self.apc_semantic_hash
        if job.cache_plan is not None:
            # Explicit seams have their own numerical span plan. Automatic
            # checkpoints keep their existing namespace and persistence.
            import hashlib

            salt = int.from_bytes(
                hashlib.blake2b(
                    f"{salt}:explicit-absolute-spans-v1".encode(), digest_size=8
                ).digest(),
                "little",
            )
        if salt is not None:
            pkw["_apc_semantic_hash"] = salt
        rd = pkw.get("rope_deltas")
        job.rope_delta = float(rd.reshape(-1)[0].item()) if rd is not None else 0.0
        if spec:
            vocab = getattr(self.drafter, "_draft_vocab", None)
            if vocab is not None:  # the reduced draft readout covers this prompt's ids
                vocab.set_context(job.ids)
                self._vocab_owner = job
            group = _Group(
                gen=self._new_generator(
                    spec=True,
                    use_apc=use_apc,
                    top_logprobs=0,
                    sampler=job.keyed,
                    greedy=job.keyed is None,
                    auxiliary=job.priority < 0,
                ),
                spec=True,
            )
            if job.priority < 0:
                self._aux_spec = group
            else:
                self._spec = group
        else:
            key = (job.top_logprobs, use_apc, job.priority, job.cache_plan is not None)
            group = self._batches.get(key)
            if group is None:
                sampler = RowSampler()
                group = self._batches[key] = _Group(
                    gen=self._new_generator(
                        spec=False,
                        use_apc=use_apc,
                        top_logprobs=job.top_logprobs,
                        sampler=sampler,
                        auxiliary=job.priority < 0,
                    ),
                    spec=False,
                    sampler=sampler,
                )
        if getattr(group.gen, "apc", None) is not None and hasattr(
            group.gen.apc, "set_request"
        ):
            group.gen.apc.set_request(job.ids, job.cache_plan)
        extra_hash = getattr(group.gen, "_apc_extra_hash", None)
        if extra_hash is not None:
            job.apc_salt = extra_hash(pkw)
        (uid,) = group.gen.insert(
            [job.ids],
            max_tokens=job.max_tokens,
            prompt_kwargs=[pkw],
            logits_processors=[job.processors or None],
            thinking_budget_criteria=[job.budget],
        )
        job.uid = uid
        job.start = time.perf_counter()
        group.jobs[uid] = job
        if group.sampler is not None and job.sampling is not None:
            group.sampler.add(uid, job.sampling)

    # ── round driver ────────────────────────────────────────────────────
    def _admit_driver(self, job: _Job, use_apc: bool) -> None:
        from .round_driver.driver import Request

        driver = self.driver
        if job.priority < 0:
            if self._aux_driver is None:
                from .round_driver.driver import RoundDriver

                self._aux_driver = RoundDriver(
                    self.model,
                    drafter=self.drafter if self.draft_kind == "mtp" else None,
                    stop_tokens=self.stop_tokens,
                    chunk=self.driver.chunk,
                    apc=None,
                )
            driver = self._aux_driver
        job.driver = driver
        hit = driver.add(
            Request(
                ids=job.ids,
                max_tokens=job.max_tokens,
                sampling=job.sampling,
                processors=list(job.processors or []),
                logprobs=job.logprobs,
                top_logprobs=job.top_logprobs,
                draft=job.allow_draft,
                budget=job.budget,
                handle=job,
                extra_hash=self.apc_semantic_hash or 0,
                use_apc=use_apc,
            )
        )
        self._vocab_owner = driver
        job.start = job.stats.t_admit = time.perf_counter()
        hit = int(hit or 0)
        job.stats.cached_tokens = hit
        job.stats.prefill_total = len(job.ids) - hit
        job.stats.used_apc = use_apc
        job.stats.used_draft = bool(job.allow_draft and driver.head is not None)
        if job.stats.used_draft:
            job.stats.spec_mode = "mtp"
        self._driver_jobs[id(job)] = job

    def _step_driver(self, primary: bool = False) -> None:
        for key, job in list(self._driver_jobs.items()):
            cancelled = job.cancel_event is not None and job.cancel_event.is_set()
            if job.abandoned or cancelled:
                (job.driver or self.driver).remove(job)
                del self._driver_jobs[key]
                job.stats.finish_reason = "cancel" if cancelled else None
                self._emit(job, _DONE)
        if not self._driver_jobs:
            return
        events = []
        for driver in (self.driver, self._aux_driver):
            if driver is None or (primary and driver is self._aux_driver):
                continue
            vocab = getattr(self.drafter, "_draft_vocab", None)
            if vocab is not None and self._vocab_owner is not driver and driver.rows:
                vocab.set_context([t for row in driver.rows for t in row.req.ids])
                self._vocab_owner = driver
            with self.driver_busy_meter.span():
                events.extend(driver.step())
            self._note_driver_prefill(driver)
        for event in events:
            job = event.handle
            stats = job.stats
            now = time.perf_counter()
            if stats.generated == 0:
                stats.first_token_s = now - job.start
                stats.t_first = now
                stats.prefill_done = stats.prefill_total
            stats.t_last = now
            stats.generated += 1
            self._emit(job, (int(event.token), event.logprob))
            if event.finish is not None:
                stats.finish_reason = event.finish
                self._driver_jobs.pop(id(job), None)
                self._emit(job, _DONE)

    def _finish(self, group: _Group, uid: int, reason: str | None) -> None:
        job = group.jobs.pop(uid, None)
        if group.sampler is not None:
            group.sampler.drop(uid)
        if job is None:
            return
        if reason is not None:
            job.stats.finish_reason = reason
        if (
            self.prefix_invariant
            and job.priority >= 0
            and reason != "cancel"
            and not job.abandoned
        ):
            now = time.perf_counter()
            auxiliary = [
                j for g in self._groups() for j in g.jobs.values() if j.priority < 0
            ]
            auxiliary += [j for j in self._pending if j.priority < 0]
            aged = [j for j in auxiliary if now - j.last_service >= AGING_S]
            oldest = min(aged, key=lambda j: j.queued, default=None)
            if oldest is None or not oldest.handoff_graced:
                self._primary_handoff_at = now
                if oldest is not None:
                    oldest.handoff_graced = True
        self._observe_prefill(job)
        coordinator = getattr(group.gen, "apc", None)
        if (
            job.cache_plan is not None
            and coordinator is not None
            and hasattr(coordinator, "release_request")
        ):
            coordinator.release_request(job.ids, job.cache_plan)
        self._emit(job, _DONE)

    @staticmethod
    def _stopped(job: _Job) -> bool:
        return job.abandoned or (
            job.cancel_event is not None and job.cancel_event.is_set()
        )

    def _prune_group(self, group: _Group) -> None:
        # Cleanup also runs for paused auxiliary rows.
        for uid, job in list(group.jobs.items()):
            cancelled = job.cancel_event is not None and job.cancel_event.is_set()
            if job.abandoned or cancelled:
                group.prefills.pop(uid, None)
                discard = getattr(
                    getattr(group.gen, "apc", None),
                    "discard_deferred_checkpoints",
                    None,
                )
                if discard is not None:
                    discard()
                with contextlib.suppress(Exception):
                    group.gen.remove(uid)
                self._finish(group, uid, "cancel" if cancelled else None)

    def _step_group(self, group: _Group, *, decode_only: bool = False) -> None:
        from .kernels import batch_invariant

        apc = getattr(group.gen, "apc", None)
        discard = getattr(apc, "discard_deferred_checkpoints", None)
        if not getattr(self, "_kernel_configuration_logged", False):
            self._kernel_configuration_logged = True
            logger.info(
                "APC runtime kernels: qualified=%s installed=%s module=%s",
                self.prefix_invariant,
                batch_invariant.is_installed(),
                batch_invariant.__file__,
            )
        self._prune_group(group)
        if not group.jobs:
            if discard is not None:
                discard()
            return
        # Prefix consumers and producers must use the same per-row arithmetic
        # regardless of batch arrival order. The same existing kernels already
        # make the speculative lane invariant.
        invariant = batch_invariant.is_installed()
        # With ragged KV on, the speculative lane's decode and verify
        # attention run the ragged kernel over its one-row cache, so both
        # share per-row arithmetic (and the shared batch's).
        dense_lane = bool(self.ragged_kv)
        from . import cache_decode

        cache_decode.set_active(invariant and self.prefix_invariant)
        if invariant:
            if self.prefix_invariant and not getattr(
                self, "_prefix_invariant_logged", False
            ):
                self._prefix_invariant_logged = True
                logger.info(
                    "APC prefix-invariant dispatch engaged: shared=%s dense_attention=%s",
                    not group.spec,
                    dense_lane,
                )
            batch_invariant.set_active(True)
        if self.ragged_kv:
            from .kernels import ragged_kv

            # Joins build ragged caches only while this runner's model steps.
            ragged_kv.set_format(self.ragged_kv)
            if dense_lane:
                ragged_kv.set_dense_lane(True)
        try:
            self._step_generator(group, decode_only=decode_only)
        finally:
            cache_decode.set_active(False)
            if invariant:
                batch_invariant.set_active(False)
            if self.ragged_kv:
                ragged_kv.set_format(None)
                if dense_lane:
                    ragged_kv.set_dense_lane(False)

    def _step_generator(self, group: _Group, *, decode_only: bool = False) -> None:
        if group.spec:
            # Upstream's speculative verify reads mRoPE deltas from model
            # state, which the shared batch's steps overwrite.
            (job,) = group.jobs.values()
            counter_now = _spec_counters(self.drafter)
            if (
                counter_now is not None
                and job.spec_base is not None
                and job.spec_last is not None
            ):
                # Exclude counters accumulated by another lane while this row paused.
                job.spec_base = tuple(
                    base + current - last
                    for base, current, last in zip(
                        job.spec_base, counter_now, job.spec_last, strict=True
                    )
                )
            # A paused auxiliary lane shares the drafter weights, but its own
            # generator retains every cache and sampler. Restore the prompt's
            # reduced readout context when switching lanes.
            vocab = getattr(self.drafter, "_draft_vocab", None)
            if vocab is not None and self._vocab_owner is not job:
                vocab.set_context(job.ids)
                self._vocab_owner = job
            lm = self.model.language_model
            if hasattr(lm, "_rope_deltas"):
                lm._rope_deltas = mx.array([[job.rope_delta]], dtype=mx.float32)
        if group.spec:
            from . import mtp_lane

            mtp_lane.set_guide(job.guide)
            mtp_lane.set_context(job.ids)
        apc = getattr(group.gen, "apc", None)
        flush = getattr(apc, "flush_deferred_checkpoints", None)
        if flush is not None and apc is not None:
            # One prefilling request owns these captures. A shared group can
            # decode one row while admitting another, whose first token has not
            # been delivered yet; that admission keeps synchronous stores.
            apc.defer_checkpoint_stores = len(group.jobs) == 1
        try:
            if decode_only:
                # Upstream returns after decode once capacity is reached. Zero
                # capacity yields without starting/resuming any prefill atom.
                capacity = group.gen.completion_batch_size
                group.gen.completion_batch_size = 0
                try:
                    prompt_progress, responses = group.gen.next()
                finally:
                    group.gen.completion_batch_size = capacity
            else:
                prompt_progress, responses = group.gen.next()
        except BaseException:
            discard = getattr(apc, "discard_deferred_checkpoints", None)
            if discard is not None:
                discard()
            raise

        finally:
            if flush is not None and apc is not None:
                apc.defer_checkpoint_stores = False
            if group.spec:
                mtp_lane.set_guide(None)
                mtp_lane.set_context(None)
        self._note_prefill(group)
        for job in group.jobs.values():
            self._note_cache(job)
        if group.spec:
            for job in group.jobs.values():
                _note_spec(self.drafter, job)
        if self.ragged_kv and not self._ragged_logged and not group.spec:
            # Engagement proof in the server log (a no-op path once cost a
            # full MMLU run to notice). Joins build the ragged caches
            # (``ragged_kv.enable``); a lone request keeps its stock cache.
            from .kernels.ragged_kv import RaggedKVCache

            batch = getattr(group.gen, "_generation_batch", None)
            caches = getattr(batch, "prompt_cache", None) or []
            n = sum(isinstance(c, RaggedKVCache) for c in caches)
            if n:
                self._ragged_logged = True
                logger.info(
                    "Ragged KV engaged (%s): %d attention caches", self.ragged_kv, n
                )
        for progress in prompt_progress or []:
            job = group.jobs.get(getattr(progress, "uid", None))
            if job is not None:
                job.stats.cached_tokens = int(getattr(progress, "cached_tokens", 0))
                self._note_cache(job)
        finished = []
        for response in responses:
            job = group.jobs.get(response.uid)
            if job is None:
                continue
            stats = job.stats
            now = time.perf_counter()
            stats.t_last = now
            if stats.generated == 0:
                stats.first_token_s = now - job.start
                stats.t_first = now
                stats.prefill_done = stats.prefill_total
            if response.token is None:
                finished.append((response.uid, response.finish_reason or "stop"))
                continue
            stats.generated += 1
            lp = None
            if job.logprobs:
                lp = {
                    "token_id": int(response.token),
                    "logprob": float(response.token_logprob),
                    "top_logprobs": [
                        {"token_id": int(t), "logprob": float(v)}
                        for t, v in (response.top_logprobs or [])
                    ],
                }
            self._emit(job, (int(response.token), lp))
            if response.finish_reason is not None:
                finished.append((response.uid, response.finish_reason))
        # The consumer can detokenize/send the first token while checkpoint
        # copies and admission run on this same serialized MLX thread.
        if responses and flush is not None:
            flush()
        # Completion/usage must observe publication even for a one-token reply.
        # The first token was already emitted, so this keeps stores off TTFT.
        for uid, reason in finished:
            self._finish(group, uid, reason)

    def _observe_prefill(self, job: _Job) -> None:
        """Tell the storage tiers how fast prefill really is (their cost model compares a
        restore against it); only a request that prefilled alone and a lot counts."""
        if job.priority < 0:
            return  # auxiliary prompts must not train the agent APC cost model
        disk = getattr(self.apc_manager, "disk", None)
        observe = getattr(disk, "observe_prefill", None)
        st = job.stats
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
            job.stats.cache_tier = rec.tier
            job.stats.cache_reload_ms = rec.ms
            job.stats.cache_device = rec.device

    def _note_driver_prefill(self, driver=None) -> None:
        """Publish prefill progress of the round driver's rows."""
        try:
            for row in (driver or self.driver).rows:
                st = row.req.handle.stats
                if st.t_first == 0.0:
                    st.prefill_done = min(row.done - row.hit, st.prefill_total)
        except Exception:
            logger.debug("driver prefill progress unavailable", exc_info=True)

    @staticmethod
    def _note_prefill(group: _Group) -> None:
        """Publish prefill progress (uncached tokens computed / to compute) for
        the rows of the batch that is prefilling right now."""
        gen = group.gen
        pb = getattr(gen, "_prompt_batch", None)
        try:
            if pb is not None:
                done = int(getattr(pb, "_processed_prompt_columns", 0))
                rest = int(pb._input_ids.shape[1])
                cached = list(getattr(pb, "_cached_tokens_per_row", []) or [])
                for i, uid in enumerate(getattr(pb, "_prompt_uids", [])):
                    job = group.jobs.get(uid)
                    if job is None:
                        continue
                    hit = int(cached[i]) if i < len(cached) else 0
                    if hit:
                        job.stats.cached_tokens = max(job.stats.cached_tokens, hit)
                    job.stats.prefill_total = max(len(job.ids) - hit, done + rest)
                    job.stats.prefill_done = min(done, job.stats.prefill_total)
                return
            waiting = {s[0] for s in getattr(gen, "_unprocessed_sequences", [])}
            # Suspended canonical atoms live in the runner, not the generator.
            waiting.update(getattr(group, "prefills", {}))
        except Exception:
            logger.debug("prefill progress unavailable", exc_info=True)
            return
        for uid, job in group.jobs.items():
            st = getattr(job, "stats", None)
            if st is not None and uid not in waiting and st.t_first == 0.0:
                st.prefill_done = st.prefill_total

    def busy_snapshot(self) -> dict:
        """Cumulative GPU-busy accounting: all slices and the round driver's share."""
        return {
            "slices": self.busy_meter.snapshot(),
            "round_driver": self.driver_busy_meter.snapshot(),
        }

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
        if not job.stats.used_apc or id(job) in self._driver_jobs:
            return
        for group in self._groups():
            if job.uid not in group.jobs or group.jobs[job.uid] is not job:
                continue
            coordinator = getattr(group.gen, "apc", None)
            if coordinator is not None and hasattr(coordinator, "set_request"):
                points = coordinator.checkpoint_lengths(
                    job.ids, group.gen._apc_media_token_ids(), begin=False
                )
                if job.cache_plan is not None:
                    points = [
                        n
                        for n, _ in job.cache_plan.get(
                            "writes", job.cache_plan["points"]
                        )
                    ]
                self._prefix_producers.append((job, points))
            break

    def _drive_slice(self, resubmit: bool = True) -> None:
        """One scheduling slice on the MLX thread, timed into ``busy_meter``."""
        with self.busy_meter.span():
            self._drive_slice_body(resubmit)

    def _work(self, job: _Job) -> Work:
        remaining = max(
            1, len(job.ids) - job.stats.cached_tokens - job.stats.prefill_done
        )
        if (
            not job.stats.prefill_done
            and not job.stats.cached_tokens
            and job.priority >= 0
        ):
            # Read metadata only: no clone, SSD read, LRU touch or cache pin.
            # A later ordinary lookup revalidates availability and identity.
            mgr = self.apc_manager
            lock = getattr(mgr, "lock", None)
            entries = getattr(mgr, "_exact_cache", None)
            salt = job.apc_salt
            if salt is None:
                salt = job.salt if job.salt is not None else self.apc_semantic_hash
            if lock is not None and isinstance(entries, dict):
                tokens = tuple(job.ids)
                with lock:
                    hit = max(
                        (
                            len(e.token_ids)
                            for e in entries.values()
                            if e.extra_hash == (salt or 0)
                            and 0 < len(e.token_ids) < len(tokens)
                            and tokens[: len(e.token_ids)] == e.token_ids
                        ),
                        default=0,
                    )
                remaining = max(1, len(tokens) - hit)
        return Work(
            job.queued,
            job.last_service,
            remaining,
            job.priority,
            skips=job.prefill_skips,
        )

    def _step_work_groups(self, primary: bool) -> None:
        now = time.perf_counter()
        groups = self._groups()

        def eligible(job: _Job) -> bool:
            # Re-evaluate after a primary completes earlier in this slice.
            held = self._primary_handoff_at is not None and (
                time.perf_counter() - self._primary_handoff_at < PRIMARY_HANDOFF_S
            )
            return job.priority >= 0 or (
                not held and (not primary or now - job.last_service >= AGING_S)
            )

        candidates = [
            (g, j)
            for g in groups
            for j in g.jobs.values()
            if not j.stats.t_first
            and j.uid
            not in getattr(getattr(g.gen, "_generation_batch", None), "uids", [])
            and eligible(j)
        ]
        chosen = min(
            candidates, key=lambda gj: self._work(gj[1]).key(now), default=None
        )
        aged_yield = None
        primary_choice = min(
            ((g, j) for g, j in candidates if j.priority >= 0),
            key=lambda gj: self._work(gj[1]).key(now),
            default=None,
        )
        if (
            chosen is not None
            and chosen[1].priority < 0
            and chosen[1].prefill_skips < 1
            and primary_choice is not None
            and self._work(primary_choice[1]).uncached_tokens <= PREFILL_STEP
        ):
            aged_yield = chosen[1]
            chosen = primary_choice
        # Paused auxiliary decode cannot repay debt. Counting it here makes
        # a primary prefill wait for each auxiliary aging interval (~20 s).
        decoding = any(
            len(getattr(g.gen, "_generation_batch", []))
            and any(eligible(j) for j in g.jobs.values())
            for g in groups
        )
        # Once the selected primary reaches its last canonical atom, carry it
        # through generate() and first-token delivery before repaying decode.
        # Upstream still owns every checkpoint cut and all sampling state.
        finishing = next(((g, j) for g, j in candidates if j.finishing_prefill), None)
        if finishing is not None:
            chosen = finishing
        delivering_first = any(
            j.finishing_prefill
            and not j.stats.t_first
            and j.uid in getattr(g.gen._generation_batch, "uids", [])
            for g in groups
            for j in g.jobs.values()
        )
        final_atom = (
            chosen is not None
            and chosen[1].priority >= 0
            and (
                chosen[1].finishing_prefill
                or self._work(chosen[1]).uncached_tokens <= PREFILL_STEP
            )
        )
        if delivering_first or (self._decode_debt > 0 and decoding and not final_atom):
            chosen = None
        for group in groups:
            self._prune_group(group)
            allowed = [j for j in group.jobs.values() if eligible(j)]
            if not allowed:
                continue
            # Generators without the upstream atom API (test fakes) retain
            # ordinary dispatch. The optional round driver has its own policy.
            if not hasattr(group.gen, "_unprocessed_sequences"):
                self._step_group(group)
                continue
            selected_job = (
                chosen[1] if chosen is not None and chosen[0] is group else None
            )
            selected = selected_job is not None
            has_decode = bool(len(group.gen._generation_batch))
            if not selected and not has_decode:
                continue
            if selected_job is not None:
                if aged_yield is not None:
                    aged_yield.prefill_skips += 1
                if final_atom:
                    selected_job.finishing_prefill = True
                uid = selected_job.uid
                group.gen._prompt_batch = group.prefills.pop(uid, None)
                group.gen._unprocessed_sequences.sort(key=lambda seq: seq[0] != uid)
            first_pending = any(
                j.finishing_prefill and not j.stats.t_first for j in allowed
            )
            before = time.perf_counter()
            self._step_group(group, decode_only=not selected)
            elapsed = time.perf_counter() - before
            if any(j.priority < 0 for j in allowed):
                self._primary_handoff_at = None
            for job in allowed:
                if job.stats.t_first or (selected and job is selected_job):
                    job.last_service = time.perf_counter()
                    if job.priority < 0:
                        job.handoff_graced = False
            if selected_job is not None:
                if selected_job.priority < 0:
                    selected_job.prefill_skips = 0
                # Revalidate after actual lookup/progress. A metadata-only warm
                # estimate can lose its checkpoint before this atom executes.
                selected_job.finishing_prefill = (
                    selected_job.priority >= 0
                    and not selected_job.stats.t_first
                    and self._work(selected_job).uncached_tokens <= PREFILL_STEP
                )
                # Bound overtaking in executed atoms, independent of arrival
                # rate and the latency of the competing short requests.
                for _, waiter in candidates:
                    if (
                        waiter is not selected_job
                        and waiter.priority == selected_job.priority
                    ):
                        waiter.prefill_skips += 1
                batch = group.gen._prompt_batch
                if batch is not None:
                    (uid,) = batch.uids  # prefill_batch_size is always one
                    group.prefills[uid] = batch
                    group.gen._prompt_batch = None
                self._decode_debt = DECODE_QUANTUM_S
            elif not first_pending:
                self._decode_debt = max(0.0, self._decode_debt - elapsed)

    def _handoff_delay(self) -> float:
        if (
            not settings.get_bool("YUNSHU_UNCACHED_SCHEDULING")
            or self.driver is not None
        ):
            return 0.0
        at = self._primary_handoff_at
        if at is None:
            return 0.0
        remaining = PRIMARY_HANDOFF_S - (time.perf_counter() - at)
        if remaining <= 0 or any(
            j.priority >= 0 for g in self._groups() for j in g.jobs.values()
        ):
            return 0.0
        with self._lock:
            if any(j.priority >= 0 for j in self._pending):
                return 0.0
        return min(0.002, remaining)

    def _drive_slice_body(self, resubmit: bool) -> None:
        _install_row_context()
        try:
            for group in self._groups():
                self._prune_group(group)
            with self._lock:
                cancelled_pending = [j for j in self._pending if self._stopped(j)]
                self._pending = [j for j in self._pending if not self._stopped(j)]
                active = [j for g in self._groups() for j in g.jobs.values()]
                active += list(self._driver_jobs.values())
                # A call still preparing on the MLX executor has no row yet.
                # Conservatively treat it as primary until its metadata arrives.
                preparing = self.inflight() > len(active) + len(self._pending)
                primary = preparing or any(
                    j.priority >= 0 for j in [*active, *self._pending]
                )
                now = time.perf_counter()
                pending = [
                    j
                    for j in self._pending
                    if not primary or j.priority >= 0 or now - j.last_service >= AGING_S
                ]
                selected_ids = {id(j) for j in pending}
                self._pending = [j for j in self._pending if id(j) not in selected_ids]
                work_order = (
                    settings.get_bool("YUNSHU_UNCACHED_SCHEDULING")
                    and self.driver is None
                )
                auxiliary = sum(
                    j.priority < 0 for j in [*active, *self._pending, *pending]
                )
                if not auxiliary:
                    self._primary_handoff_at = None
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
            if work_order:
                pending.sort(key=lambda j: self._work(j).key(time.perf_counter()))
            self._prefix_producers = [
                (j, points)
                for j, points in self._prefix_producers
                if not j.terminal and not j.stats.t_first
            ]
            waiting = []
            for job in pending:
                cancelled = job.cancel_event is not None and job.cancel_event.is_set()
                if job.abandoned or cancelled:
                    # Never prefill a request nobody is waiting for.
                    job.stats.finish_reason = "cancel" if cancelled else None
                    self._emit(job, _DONE)
                    continue
                if self._prefix_wait(job):
                    waiting.append(job)
                    continue
                try:
                    self._admit(job, alone)
                    self._track_prefix_producer(job)
                except Exception as exc:
                    logger.exception("VLM runner admission failed")
                    self._emit(job, exc)
            if waiting:
                with self._lock:
                    self._pending = waiting + self._pending
            if work_order:
                self._step_work_groups(primary)
            else:
                for group in self._groups():
                    self._prune_group(group)
                    if (
                        not primary
                        or not any(j.priority < 0 for j in group.jobs.values())
                        or any(
                            time.perf_counter() - j.last_service >= AGING_S
                            for j in group.jobs.values()
                        )
                    ):
                        self._step_group(group)
                        for j in group.jobs.values():
                            j.last_service = time.perf_counter()
            if self._driver_jobs:
                self._step_driver(primary)
            if self._spec is not None and not self._spec.jobs:
                if not self._spec.gen.has_work:
                    self._spec.gen.close()
                    self._spec = None
            if self._aux_spec is not None and not self._aux_spec.jobs:
                if not self._aux_spec.gen.has_work:
                    self._aux_spec.gen.close()
                    self._aux_spec = None
            for key, group in list(self._batches.items()):
                if not group.jobs and not group.gen.has_work:
                    group.gen.close()
                    del self._batches[key]
        except Exception as exc:
            logger.exception("VLM runner step failed; failing active requests")
            for group in self._groups():
                for job in group.jobs.values():
                    self._emit(job, exc)
                with contextlib.suppress(Exception):
                    group.gen.close()
            self._batches.clear()
            self._spec = None
            self._aux_spec = None
            for job in self._driver_jobs.values():
                self._emit(job, exc)
                if self.driver is not None:
                    (job.driver or self.driver).remove(job)
            self._driver_jobs.clear()
        if not resubmit:
            return
        with self._lock:
            if (
                self._pending
                or self._batches
                or self._spec is not None
                or self._aux_spec is not None
                or self._driver_jobs
            ):
                again = True
            else:
                self._driving = False
                again = False
        if again:
            delay = self._handoff_delay()
            if delay:
                # Free the Metal worker and GIL for preparing/submitting the
                # next request. The timer only submits; it never executes MLX.
                timer = threading.Timer(delay, self._schedule)
                timer.daemon = True
                timer.start()
            else:
                self._schedule()
        elif self.clear_on_idle:
            # Large models: release the buffer pool once everything drains
            # (clearing under active batches would only force reallocation). Up to
            # YUNSHU_PREFILL_BUFFER_CACHE_GB stays: the next request's cache restore
            # reuses it (a cleared pool costs ~45 ms per 32K-token cache copy).
            with contextlib.suppress(Exception):
                from .kernels import buffer_cache

                mx.synchronize()
                buffer_cache.clear_if_over()


def _spec_counters(drafter: Any) -> tuple | None:
    """(rounds, accepted, drafted) lifetime counters the round loops keep on the drafter."""
    if drafter is None:
        return None
    return (
        getattr(drafter, "speculative_total_rounds", 0),
        float(getattr(drafter, "speculative_total_accepted", 0.0)),
        getattr(drafter, "speculative_total_drafted", 0),
        getattr(drafter, "copy_total_rounds", 0),
        getattr(drafter, "copy_total_tokens", 0),
    )


def _note_spec(drafter: Any, job: _Job) -> None:
    """Per-request drafted / accepted draft tokens of the single-row speculative lane: the
    drafter's counters (bumped by mtp_lane, mtp_tree, dflash_tree and upstream's loops)
    minus their baseline, adjusted at slice entry to exclude other lanes' work."""
    now, base = _spec_counters(drafter), job.spec_base
    if now is None or base is None:
        return
    job.stats.spec_drafted = max(int(now[2] - base[2]), 0)
    job.stats.spec_rounds = max(int(now[0] - base[0]), 0)
    job.stats.spec_accepted = max(int(round(now[1] - base[1])), 0)
    job.stats.spec_copy_rounds = max(int(now[3] - base[3]), 0)
    job.stats.spec_copy_tokens = max(int(now[4] - base[4]), 0)
    job.spec_last = now


def _spec_mode(drafter: Any) -> str:
    name = type(drafter).__name__.lower() if drafter is not None else ""
    if "dflash" in name:
        return "dflash"
    if "mtp" in name:
        return "mtp"
    return name or "draft"


# Uids of the rows the upstream batch is sampling right now (set by the
# wrappers below). Upstream calls the sampler for the whole batch and passes
# row_ids=[0]*n, so a per-row sampler has no other way to know which request
# each row is.
_STEP_UIDS: list | None = None


def _install_row_context() -> None:
    import importlib

    ar = importlib.import_module("mlx_vlm.generate.ar")
    if getattr(ar, "_yunshu_row_context", False):
        return
    step = ar.GenerationBatch._step
    generate = ar.PromptProcessingBatch.generate

    def _step(self):
        global _STEP_UIDS
        _STEP_UIDS = list(self.uids)
        return step(self)

    def _generate(self, *args, **kwargs):
        global _STEP_UIDS
        _STEP_UIDS = list(self.uids)
        return generate(self, *args, **kwargs)

    ar.GenerationBatch._step = _step
    ar.PromptProcessingBatch.generate = _generate
    ar._yunshu_row_context = True


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


class RowSampler:
    """Per-row sampling for one shared BatchGenerator.

    Rows without params are greedy (argmax). Sampled rows apply their own
    top-p / min-p / top-k and temperature (same order as mlx-lm's
    make_sampler) and draw with their own PRNG key when seeded, so a seeded
    request is reproducible regardless of what else is in the batch.
    """

    def __init__(self):
        # uid -> (params, mx PRNG key for stateful XTC rows | None, keyed sampler | None,
        # generation index of the next draw)
        self._rows: dict[int, list] = {}

    def add(self, uid: int, params: RowParams) -> None:
        if keyed_sampling.supports(params):
            # Position-keyed draws: the token at generation index g is a function of
            # (logits, seed, g) only, so a seeded request gives the same stream here as on
            # the speculative lane, alone or in a mixed batch.
            seed = (
                params.seed
                if params.seed is not None
                else int.from_bytes(os.urandom(8), "little")
            )
            self._rows[uid] = [params, None, keyed_sampling.seed_base(seed), 0]
            return
        # XTC is stateful by design (its coin flips advance a PRNG): stays on the old path.
        key = mx.random.key(int(params.seed)) if params.seed is not None else None
        self._rows[uid] = [params, key, None, 0]

    def drop(self, uid: int) -> None:
        self._rows.pop(uid, None)

    def sample_target(self, logprobs, row_ids=None, positions=None):
        return self(logprobs)

    def __call__(self, logprobs: mx.array) -> mx.array:
        from mlx_lm.sample_utils import (
            apply_min_p,
            apply_xtc,
        )

        tokens = mx.argmax(logprobs, axis=-1)
        if not self._rows:
            return tokens
        uids = _STEP_UIDS
        if uids is None or len(uids) != logprobs.shape[0]:
            # Never fall back to greedy for sampled requests silently: this
            # means upstream changed how it calls the sampler.
            raise RuntimeError(
                f"RowSampler cannot map {logprobs.shape[0]} rows to requests "
                f"(step uids: {uids})"
            )
        keyed: dict[tuple, list[int]] = {}
        for i, uid in enumerate(uids):
            entry = self._rows.get(uid)
            if entry is None:
                continue
            p, key, base, pos = entry
            if base is not None:
                keyed.setdefault((p.temperature, p.top_p, p.top_k, p.min_p), []).append(
                    i
                )
                continue
            row = logprobs[i : i + 1]
            if p.top_p < 1.0:
                row = top_p_filter(row, max(p.top_p, 0.0))
            if p.min_p:
                row = apply_min_p(row, p.min_p)
            if p.xtc_probability > 0.0:
                row = apply_xtc(
                    row,
                    p.xtc_probability,
                    p.xtc_threshold,
                    list(p.xtc_special_tokens or []),
                )
            row = top_k_filter(row, p.top_k)
            row = row * (1 / p.temperature)
            if key is not None:
                key, sub = mx.random.split(key)
                entry[1] = key
                token = mx.random.categorical(row, key=sub)
            else:
                token = mx.random.categorical(row)
            tokens[i] = token[0]
        for idx in keyed.values():
            # one vectorized draw per distinct parameter set (usually one per step)
            ents = [self._rows[uids[i]] for i in idx]
            bases = mx.array(np.array([e[2] for e in ents], dtype=np.uint64))
            pos = mx.array([e[3] for e in ents])
            for e in ents:
                e[3] += 1
            whole = len(idx) == logprobs.shape[0]
            sub = logprobs if whole else logprobs[mx.array(idx)]
            drawn = keyed_sampling.sample_rows(sub, ents[0][0], bases, pos)
            if whole:
                tokens = drawn.astype(tokens.dtype)
            else:
                tokens[mx.array(idx)] = drawn.astype(tokens.dtype)
        return tokens


_DONE = object()
# Undelivered tokens a job may hold before it is cancelled (slow / stuck client).
OUT_LIMIT = 65536


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
    priority: int = 0
    queued: float = 0.0
    last_service: float = 0.0
    prefill_skips: int = 0
    finishing_prefill: bool = False
    handoff_graced: bool = False
    driver: Any = None
    cache_plan: dict | None = None
    apc_salt: int | None = None
    out: queue.Queue = field(default_factory=queue.Queue)
    uid: int | None = None
    start: float = 0.0
    abandoned: bool = False
    budget: Any = None
    rope_delta: float = 0.0
    allow_draft: bool = False
    guide: Any = None
    keyed: Any = (
        None  # KeyedSampler of a sampled request served by the speculative lane
    )
    # Lifetime counter baseline, adjusted to exclude other lanes while paused.
    spec_base: tuple | None = None
    spec_last: tuple | None = None
    terminal: bool = False
    overflowed: bool = False


@dataclass
class _Group:
    gen: Any
    spec: bool
    sampler: RowSampler | None = None
    jobs: dict = field(default_factory=dict)
    prefills: dict = field(default_factory=dict)
