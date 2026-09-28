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
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# Upstream prefill default; APC checkpoints land on these chunk boundaries and
# cancellation is honoured between chunks (~2 s at 2K tokens on a 27B model).
PREFILL_STEP = 2048


@dataclass
class RunStats:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    first_token_s: float = 0.0
    generated: int = 0
    finish_reason: str | None = None
    used_apc: bool = False
    used_draft: bool = False
    # Per-token {"token_id", "logprob", "top_logprobs": [...]} when requested.
    last_logprob: dict | None = None
    extra: dict = field(default_factory=dict)


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

    return make_logits_processors(
        logit_bias=logit_bias or None,
        repetition_penalty=(
            repetition_penalty if repetition_penalty not in (None, 1.0) else None
        ),
        presence_penalty=presence_penalty or None,
        frequency_penalty=frequency_penalty or None,
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
    ):
        self.model = model
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
        # Shared continuous batches keyed by (top_logprobs_k, use_apc) and the
        # exclusive speculative lane (one request, only while it is alone).
        self._batches: dict[tuple, _Group] = {}
        self._spec: _Group | None = None
        self._driving = False
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
        # Fused chunked prefill (YUNSHU_FUSED_PREFILL_TOKENS): prefill tokens
        # per step run inside the decode forward while rows decode; 0 = off.
        self.fused_prefill_tokens = 0

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
        use_draft = bool(
            allow_draft
            and self.drafter is not None
            and greedy
            and not processors
            and not logprobs
            and thinking_budget is None
        )
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
            prompt_kwargs=prompt_kwargs,
            salt=apc_semantic_hash,
            seed=seed,
            cancel_event=cancel_event,
            stats=stats,
            budget=budget,
        )
        stats.used_draft = use_draft
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

    def busy(self) -> bool:
        with self._lock:
            return bool(self._pending or self._batches or self._spec or self._driving)

    def _submit(self, job: _Job) -> None:
        with self._lock:
            self._pending.append(job)
            if self._executor is None or self._driving:
                return
            self._driving = True
        self._executor.submit(self._drive_slice)

    def _groups(self) -> list[_Group]:
        return ([self._spec] if self._spec is not None else []) + list(
            self._batches.values()
        )

    def _active_jobs(self) -> int:
        return sum(len(g.jobs) for g in self._groups())

    def _new_generator(self, *, spec: bool, use_apc: bool, top_logprobs: int, sampler):
        from mlx_vlm.generate.ar import BatchGenerator

        return BatchGenerator(
            self.model.language_model,
            self.processor,
            stop_tokens=self.stop_tokens,
            sampler=sampler,
            apc_manager=self.apc_manager if use_apc else None,
            draft_model=self.drafter if spec else None,
            draft_kind=self.draft_kind if spec else None,
            draft_block_size=self.draft_block_size if spec else None,
            greedy_sampling=spec,
            compute_logprobs=not spec,
            top_logprobs_k=top_logprobs,
            prefill_step_size=PREFILL_STEP,
            prefill_batch_size=1,
        )

    def _admit(self, job: _Job, alone: bool) -> None:
        from .mrope import clear_rope_state

        use_apc = self.apc_manager is not None
        if use_apc and self._apc_admit is not None:
            try:
                use_apc = bool(self._apc_admit(mx.array(job.ids)))
            except Exception:
                logger.debug("APC admission check failed; using APC", exc_info=True)
        job.stats.used_apc = use_apc
        spec = job.use_draft and alone and self._spec is None
        if not spec:
            job.use_draft = False
            job.stats.used_draft = False
        pkw = job.prompt_kwargs
        if pkw is None:
            if alone:
                clear_rope_state(self.model)
            pkw = self.model.get_input_embeddings(
                mx.array(job.ids)[None], None, mask=None
            ).to_dict()
        salt = job.salt if job.salt is not None else self.apc_semantic_hash
        if salt is not None:
            pkw["_apc_semantic_hash"] = salt
        rd = pkw.get("rope_deltas")
        job.rope_delta = float(rd.reshape(-1)[0].item()) if rd is not None else 0.0
        if spec:
            group = self._spec = _Group(
                gen=self._new_generator(
                    spec=True, use_apc=use_apc, top_logprobs=0, sampler=None
                ),
                spec=True,
            )
        else:
            key = (job.top_logprobs, use_apc)
            group = self._batches.get(key)
            if group is None:
                sampler = RowSampler()
                group = self._batches[key] = _Group(
                    gen=self._new_generator(
                        spec=False,
                        use_apc=use_apc,
                        top_logprobs=job.top_logprobs,
                        sampler=sampler,
                    ),
                    spec=False,
                    sampler=sampler,
                )
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

    def _finish(self, group: _Group, uid: int, reason: str | None) -> None:
        job = group.jobs.pop(uid, None)
        if group.sampler is not None:
            group.sampler.drop(uid)
        if job is None:
            return
        if reason is not None:
            job.stats.finish_reason = reason
        job.out.put(_DONE)

    def _step_group(self, group: _Group) -> None:
        from .kernels import batch_invariant

        # Drop rows whose consumer left or whose request was cancelled.
        for uid, job in list(group.jobs.items()):
            cancelled = job.cancel_event is not None and job.cancel_event.is_set()
            if job.abandoned or cancelled:
                with contextlib.suppress(Exception):
                    group.gen.remove(uid)
                self._finish(group, uid, "cancel" if cancelled else None)
        if not group.jobs:
            return
        # The invariant kernels are what make spec on == spec off; they are on
        # only while the single-row speculative lane steps (and off for every
        # other user of the model, including the shared batch).
        invariant = group.spec and batch_invariant.is_installed()
        # With ragged KV on, the speculative lane's decode and verify
        # attention run the ragged kernel over its one-row cache, so both
        # share per-row arithmetic (and the shared batch's).
        dense_lane = group.spec and bool(self.ragged_kv)
        if invariant:
            batch_invariant.set_active(True)
        if self.ragged_kv:
            from .kernels import ragged_kv

            # Joins build ragged caches only while this runner's model steps.
            ragged_kv.set_format(self.ragged_kv)
            if dense_lane:
                ragged_kv.set_dense_lane(True)
        try:
            self._step_generator(group)
        finally:
            if invariant:
                batch_invariant.set_active(False)
            if self.ragged_kv:
                ragged_kv.set_format(None)
                if dense_lane:
                    ragged_kv.set_dense_lane(False)

    def _step_generator(self, group: _Group) -> None:
        if group.spec:
            # Upstream's speculative verify reads mRoPE deltas from model
            # state, which the shared batch's steps overwrite.
            (job,) = group.jobs.values()
            lm = self.model.language_model
            if hasattr(lm, "_rope_deltas"):
                lm._rope_deltas = mx.array([[job.rope_delta]], dtype=mx.float32)
        fused = None
        if self.fused_prefill_tokens > 0 and not group.spec:
            from . import fused_prefill

            # The decode step and the waiting prompt's prefill chunk of this
            # next() run as one forward (see fused_prefill).
            fused = fused_prefill.plan(
                group.gen, self.fused_prefill_tokens, PREFILL_STEP
            )
        try:
            prompt_progress, responses = group.gen.next()
        finally:
            if self.fused_prefill_tokens > 0 and not group.spec:
                fused_prefill.finish(fused)
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
        for response in responses:
            job = group.jobs.get(response.uid)
            if job is None:
                continue
            stats = job.stats
            if stats.generated == 0:
                stats.first_token_s = time.perf_counter() - job.start
            if response.token is None:
                self._finish(group, response.uid, response.finish_reason or "stop")
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
            job.out.put((int(response.token), lp))
            if response.finish_reason is not None:
                self._finish(group, response.uid, response.finish_reason)

    def _drive_slice(self, resubmit: bool = True) -> None:
        """One scheduling slice on the MLX thread."""
        _install_row_context()
        try:
            with self._lock:
                pending, self._pending = self._pending, []
            alone = (
                len(pending) == 1 and self._active_jobs() == 0 and self.inflight() <= 1
            )
            for job in pending:
                try:
                    self._admit(job, alone)
                except Exception as exc:
                    logger.exception("VLM runner admission failed")
                    job.out.put(exc)
            for group in self._groups():
                self._step_group(group)
            if self._spec is not None and not self._spec.jobs:
                if not self._spec.gen.has_work:
                    self._spec.gen.close()
                    self._spec = None
            for key, group in list(self._batches.items()):
                if not group.jobs and not group.gen.has_work:
                    group.gen.close()
                    del self._batches[key]
        except Exception as exc:
            logger.exception("VLM runner step failed; failing active requests")
            for group in self._groups():
                for job in group.jobs.values():
                    job.out.put(exc)
                with contextlib.suppress(Exception):
                    group.gen.close()
            self._batches.clear()
            self._spec = None
        if not resubmit:
            return
        with self._lock:
            if self._pending or self._batches or self._spec is not None:
                again = True
            else:
                self._driving = False
                again = False
        if again:
            self._executor.submit(self._drive_slice)
        elif self.clear_on_idle:
            # Large models: release the buffer pool once everything drains
            # (clearing under active batches would only force reallocation).
            with contextlib.suppress(Exception):
                mx.synchronize()
                mx.clear_cache()


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
        self._rows: dict[int, tuple[RowParams, Any]] = {}

    def add(self, uid: int, params: RowParams) -> None:
        key = mx.random.key(int(params.seed)) if params.seed is not None else None
        self._rows[uid] = (params, key)

    def drop(self, uid: int) -> None:
        self._rows.pop(uid, None)

    def sample_target(self, logprobs, row_ids=None, positions=None):
        return self(logprobs)

    def __call__(self, logprobs: mx.array) -> mx.array:
        from mlx_lm.sample_utils import (
            apply_min_p,
            apply_top_k,
            apply_top_p,
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
        for i, uid in enumerate(uids):
            entry = self._rows.get(uid)
            if entry is None:
                continue
            p, key = entry
            row = logprobs[i : i + 1]
            if 0 < p.top_p < 1.0:
                row = apply_top_p(row, p.top_p)
            if p.min_p:
                row = apply_min_p(row, p.min_p)
            if p.xtc_probability > 0.0:
                row = apply_xtc(
                    row,
                    p.xtc_probability,
                    p.xtc_threshold,
                    list(p.xtc_special_tokens or []),
                )
            if p.top_k > 0:
                row = apply_top_k(row, p.top_k)
            row = row * (1 / p.temperature)
            if key is not None:
                key, sub = mx.random.split(key)
                self._rows[uid] = (p, key)
                token = mx.random.categorical(row, key=sub)
            else:
                token = mx.random.categorical(row)
            tokens[i] = token[0]
        return tokens


_DONE = object()


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
    out: queue.Queue = field(default_factory=queue.Queue)
    uid: int | None = None
    start: float = 0.0
    abandoned: bool = False
    budget: Any = None
    rope_delta: float = 0.0


@dataclass
class _Group:
    gen: Any
    spec: bool
    sampler: RowSampler | None = None
    jobs: dict = field(default_factory=dict)
