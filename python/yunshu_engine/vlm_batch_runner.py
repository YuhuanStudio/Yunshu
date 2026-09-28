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
GPU work runs on the serialized MLX executor thread; consumers read their\ntokens from a queue on any other thread.
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
# Speculative batches are cohorts: wait this long for simultaneous arrivals.
SPEC_COALESCE_S = 0.03
SPEC_MIN_COALESCE_S = 0.005
SPEC_MAX_ROWS = 8


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
        self._groups: dict[int, _Group] = {}
        self._driving = False
        self.clear_on_idle = False
        # Requests the engine has accepted (incl. ones still being prepared);
        # speculative cohorts only wait for arrivals when others are coming.
        self.inflight = lambda: 0

    def prepare_images(self, prompt: str, image_paths: list[str]):
        """Preprocess + encode images for one prompt on the MLX thread.

        Returns ``(input_ids, prompt_kwargs, apc_semantic_hash)``. The APC salt
        hashes the processed pixel values, so a prefix is only reused for the
        same image content (never across images or with text-only prompts).
        """
        from mlx_vlm import apc as _apc
        from mlx_vlm.utils import prepare_inputs

        from .mrope import clear_rope_state

        raw = prepare_inputs(
            self.processor,
            images=image_paths,
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
        clear_rope_state(self.model)
        embed = self.model.get_input_embeddings(
            input_ids, pixel_values, mask=raw.get("attention_mask"), **data
        )
        kwargs = {**data, **{k: v for k, v in embed.to_dict().items() if v is not None}}
        salt = None
        if self.apc_manager is not None:
            salt = _apc.semantic_extra_hash(
                image_hash=_apc.hash_image_payload(pixel_values=pixel_values),
                media={"audio": None, "video": raw.get("pixel_values_videos")},
                model=self.model.language_model,
                processor=self.processor,
            )
        return input_ids[0], kwargs, salt

    # ── Shared continuous batching ──────────────────────────────────────
    #
    # Requests submit a job and consume tokens from their own queue on any
    # thread. One driver owns the GPU: it runs on the serialized MLX executor
    # in short slices (admit new jobs, one generator step per group, dispatch),
    # resubmitting itself while work remains so other MLX jobs (tokenizing,
    # image encoding for the next request) interleave. Jobs that share
    # sampling settings share one upstream BatchGenerator, so concurrent
    # requests decode together instead of queueing behind each other.

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
        use_draft = bool(
            allow_draft
            and self.drafter is not None
            and greedy
            and not processors
            and not logprobs
        )
        job = _Job(
            ids=ids,
            max_tokens=int(max_tokens),
            greedy=greedy,
            sampling=(
                None
                if greedy
                else (float(temperature), float(top_p), int(top_k), float(min_p))
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
            return bool(self._pending or self._groups or self._driving)

    def _submit(self, job: _Job) -> None:
        with self._lock:
            self._pending.append(job)
            if self._executor is None or self._driving:
                return
            self._driving = True
        self._executor.submit(self._drive_slice)

    def _group_key(self, job: _Job, use_apc: bool) -> tuple:
        # Requests with logits processors (constraints, penalties) get their own
        # generator: in a shared batch a JSON constraint was applied to another
        # row after mixed warm/cold APC admission (a math answer "102" came out
        # as "1" + EOS). Plain requests batch together.
        solo = id(job) if job.processors else None
        return (
            job.use_draft,
            job.sampling,
            job.logprobs,
            job.top_logprobs,
            use_apc,
            solo,
        )

    def _new_group(self, job: _Job, use_apc: bool) -> _Group:
        from mlx_vlm.generate.ar import BatchGenerator

        sampler = build_sampler(*job.sampling) if job.sampling is not None else None
        gen = BatchGenerator(
            self.model.language_model,
            self.processor,
            max_tokens=job.max_tokens,
            sampler=sampler,
            apc_manager=self.apc_manager if use_apc else None,
            draft_model=self.drafter if job.use_draft else None,
            draft_kind=self.draft_kind if job.use_draft else None,
            draft_block_size=self.draft_block_size if job.use_draft else None,
            greedy_sampling=job.greedy,
            compute_logprobs=job.logprobs,
            top_logprobs_k=job.top_logprobs,
            prefill_step_size=PREFILL_STEP,
        )
        return _Group(gen=gen, use_draft=job.use_draft)

    def _admit(self, job: _Job) -> None:
        from .mrope import clear_rope_state

        use_apc = self.apc_manager is not None
        if use_apc and self._apc_admit is not None:
            try:
                use_apc = bool(self._apc_admit(mx.array(job.ids)))
            except Exception:
                logger.debug("APC admission check failed; using APC", exc_info=True)
        job.stats.used_apc = use_apc
        if job.use_draft and any(
            g.use_draft and g.sealed for g in self._groups.values()
        ):
            # The drafter holds state for one speculative batch, and a running
            # speculative batch cannot take new rows: decode this request in
            # the continuous (non-speculative) batch alongside it instead.
            job.use_draft = False
            job.stats.used_draft = False
        key = self._group_key(job, use_apc)
        group = next(
            (g for g in self._groups.values() if g.key == key and not g.sealed), None
        )
        if group is None:
            group = self._new_group(job, use_apc)
            group.key = key
            self._groups[id(group)] = group
        if job.seed is not None and not group.jobs:
            mx.random.seed(int(job.seed) & ((1 << 63) - 1))
        pkw = job.prompt_kwargs
        if pkw is None:
            if not self._active_jobs():
                clear_rope_state(self.model)
            pkw = self.model.get_input_embeddings(
                mx.array(job.ids)[None], None, mask=None
            ).to_dict()
        salt = job.salt if job.salt is not None else self.apc_semantic_hash
        if salt is not None:
            pkw["_apc_semantic_hash"] = salt
        rd = pkw.get("rope_deltas")
        job.rope_delta = float(rd.reshape(-1)[0].item()) if rd is not None else 0.0
        (uid,) = group.gen.insert(
            [job.ids],
            max_tokens=job.max_tokens,
            prompt_kwargs=[pkw],
            logits_processors=[job.processors or None],
        )
        job.uid = uid
        job.start = time.perf_counter()
        group.jobs[uid] = job

    def _active_jobs(self) -> int:
        return sum(len(g.jobs) for g in self._groups.values())

    def _finish(self, group: _Group, uid: int, reason: str | None) -> None:
        job = group.jobs.pop(uid, None)
        if job is None:
            return
        group.done[uid] = job
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
        if group.use_draft and not group.sealed:
            # A running speculative batch cannot take new rows (upstream), so
            # give near-simultaneous arrivals a moment to join this cohort;
            # later ones start their own group.
            age = time.perf_counter() - group.created
            if len(group.jobs) < SPEC_MAX_ROWS and (
                age < SPEC_MIN_COALESCE_S
                or (age < SPEC_COALESCE_S and self.inflight() > self._active_jobs())
            ):
                return
            group.sealed = True
        if batch_invariant.is_installed():
            # Spec on == spec off is only guaranteed for one drafting row
            # (the invariant kernels take <= 8 verify rows); everything else
            # takes the faster verify kernels.
            batch_invariant.set_active(group.use_draft and len(group.jobs) == 1)
        if group.use_draft:
            self._set_spec_rope_deltas(group)
        prompt_progress, responses = group.gen.next()
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

    def _set_spec_rope_deltas(self, group: _Group) -> None:
        """Speculative verify reads mRoPE deltas from model state, which any
        other prefill (another group, a mixed warm/cold APC batch) overwrites;
        set them for this batch's rows before every step."""
        batch = getattr(group.gen, "_generation_batch", None)
        uids = getattr(batch, "_all_uids", None)
        lm = self.model.language_model
        if not uids or not getattr(batch, "is_speculative", False):
            return
        if not hasattr(lm, "_rope_deltas"):
            return
        deltas = [
            getattr(group.jobs.get(uid) or group.done.get(uid), "rope_delta", 0.0)
            for uid in uids
        ]
        lm._rope_deltas = mx.array(deltas, dtype=mx.float32)[:, None]

    def _drive_slice(self, resubmit: bool = True) -> None:
        """One scheduling slice on the MLX thread."""
        try:
            with self._lock:
                pending, self._pending = self._pending, []
            for job in pending:
                try:
                    self._admit(job)
                except Exception as exc:
                    logger.exception("VLM runner admission failed")
                    job.out.put(exc)
            for key, group in list(self._groups.items()):
                self._step_group(group)
                if not group.jobs and not group.gen.has_work:
                    group.gen.close()
                    del self._groups[key]
        except Exception as exc:
            logger.exception("VLM runner step failed; failing active requests")
            for group in self._groups.values():
                for job in group.jobs.values():
                    job.out.put(exc)
                with contextlib.suppress(Exception):
                    group.gen.close()
            self._groups.clear()
        if not resubmit:
            return
        with self._lock:
            if self._pending or self._groups:
                again = True
            else:
                self._driving = False
                again = False
        if again:
            self._executor.submit(self._drive_slice)
        elif self.clear_on_idle:
            # Large models: release the buffer pool once the batch drains
            # (clearing under an active batch would only force reallocation).
            with contextlib.suppress(Exception):
                mx.synchronize()
                mx.clear_cache()


_DONE = object()


@dataclass
class _Job:
    ids: list[int]
    max_tokens: int
    greedy: bool
    sampling: tuple | None
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
    rope_delta: float = 0.0


@dataclass
class _Group:
    gen: Any
    use_draft: bool
    key: tuple = ()
    sealed: bool = False
    created: float = field(default_factory=time.perf_counter)
    jobs: dict = field(default_factory=dict)
    done: dict = field(default_factory=dict)
