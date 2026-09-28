"""Single-request VLM text runner on upstream mlx-vlm ``BatchGenerator``.

One decode path for a loaded VLM target that combines, per request:

- APC prefix reuse (exact + recurrent checkpoints for hybrid models),
- a native MTP draft head (greedy, no logits processors) — token-identical to AR,
- mlx-native sampling (temperature / top-p / top-k / min-p / seed),
- logits processors: repetition / presence / frequency penalties, logit bias and
  Yunshu's grammar / JSON-schema constraints.

Measured on Qwen3.8-27B (docs/research/runs/2026-09-28-matrix/): APC + MTP in one
BatchGenerator stays token-identical to AR while a repeated 5K prompt drops from
5.4 s to 0.11 s and long decode rises from 32 to ~57 tok/s.

The runner only yields token ids. Detokenizing, stop strings, thinking state and
queue delivery stay in ``VLMEngine`` so every text path reports the same way.
Everything here runs on the serialized MLX executor thread.
"""

from __future__ import annotations

import contextlib
import logging
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
    """Owns the per-model APC manager and MTP drafter; builds one generator per request."""

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
    ):
        self.model = model
        self.processor = processor
        self.apc_manager = apc_manager
        self.apc_semantic_hash = apc_semantic_hash
        self.drafter = drafter
        self.draft_block_size = draft_block_size
        # Callable(input_ids) -> bool: skip APC when its checkpoints cannot fit,
        # so a cold request does not pay APC bookkeeping for nothing.
        self._apc_admit = apc_admit

    def iter_tokens(
        self,
        input_ids: mx.array,
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
        cancel_event: Any = None,
        stats: RunStats | None = None,
    ) -> Iterator[int]:
        """Yield generated token ids; ``stats`` is filled in as generation runs."""
        from mlx_vlm.generate.ar import BatchGenerator

        from .mrope import clear_rope_state

        stats = stats if stats is not None else RunStats()
        stats.prompt_tokens = int(input_ids.shape[0])
        greedy = temperature is None or temperature < 1e-6
        processors = list(logits_processors or [])
        use_draft = bool(
            allow_draft and self.drafter is not None and greedy and not processors
        )
        apc = self.apc_manager
        if apc is not None and self._apc_admit is not None:
            try:
                if not self._apc_admit(input_ids):
                    apc = None
            except Exception:
                logger.debug("APC admission check failed; using APC", exc_info=True)
        stats.used_apc = apc is not None
        stats.used_draft = use_draft

        if seed is not None:
            mx.random.seed(int(seed) & ((1 << 63) - 1))
        clear_rope_state(self.model)
        if prompt_kwargs is None:
            prompt_kwargs = self.model.get_input_embeddings(
                input_ids[None], None, mask=None
            ).to_dict()
        if self.apc_semantic_hash is not None:
            prompt_kwargs["_apc_semantic_hash"] = self.apc_semantic_hash

        matched_before = apc.stats.matched_tokens if apc is not None else 0
        generator = BatchGenerator(
            self.model.language_model,
            self.processor,
            max_tokens=max_tokens,
            sampler=build_sampler(temperature, top_p, top_k, min_p),
            apc_manager=apc,
            draft_model=self.drafter if use_draft else None,
            draft_kind="mtp" if use_draft else None,
            draft_block_size=self.draft_block_size if use_draft else None,
            greedy_sampling=greedy,
            compute_logprobs=False,
            prefill_step_size=PREFILL_STEP,
        )
        uid = None
        start = time.perf_counter()
        try:
            uid = generator.insert(
                [input_ids.tolist()],
                max_tokens=max_tokens,
                prompt_kwargs=[prompt_kwargs],
                logits_processors=[processors or None],
            )[0]
            # Upper bound on generator steps: prefill chunks + one token per step.
            for _ in range(stats.prompt_tokens // 256 + max_tokens + 64):
                if cancel_event is not None and cancel_event.is_set():
                    stats.finish_reason = "cancel"
                    return
                _, responses = generator.next()
                for response in responses:
                    if response.uid != uid:
                        continue
                    if stats.generated == 0:
                        stats.first_token_s = time.perf_counter() - start
                        if apc is not None:
                            stats.cached_tokens = (
                                apc.stats.matched_tokens - matched_before
                            )
                    stats.generated += 1
                    if response.finish_reason is not None:
                        stats.finish_reason = response.finish_reason
                    yield int(response.token)
                    if stats.finish_reason is not None:
                        return
            raise RuntimeError("VLM batch runner exceeded its step bound")
        finally:
            if uid is not None and stats.finish_reason in (None, "cancel"):
                with contextlib.suppress(Exception):
                    generator.remove(uid)
            generator.close()
