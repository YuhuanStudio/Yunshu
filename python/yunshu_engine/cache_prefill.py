# Patches upstream mlx-vlm dispatch; kernels remain unchanged.
"""Absolute prefill spans for the upstream runner's exact checkpoints.

Span dispatch and native KV capacity reservation are changed. Kernels and
checkpoint publication retain their existing contracts. A suffix starts by finishing the same atom as a cold request.
"""

from __future__ import annotations

import contextlib
import importlib

from . import prompt_end_replay
from .kernels import gdn_prefill

TAIL_FAST_ROWS = 64  # a whole tail (1-64 rows after the last aligned boundary) on the lane projections; attention stays BQ32 above 16 rows


def reserve_prompt_kv(prompt_cache, end: int) -> None:
    """Reserve native KV backing once its first chunk establishes the head shape.

    The logical offset and canonical prefill boundaries stay unchanged. Native
    reservation preserves the existing prefix and APC snapshots keep their
    detached ownership contract. Attention reads the reserved backing's strides.
    """
    from mlx_vlm.models.cache import BatchKVCache, KVCache

    for cache in prompt_cache:
        if type(cache) is KVCache:
            cache.prefix_cache_reserve(end)
        elif (
            type(cache) is BatchKVCache
            and cache.keys is not None
            and cache.values is not None
            and cache.keys.shape[0] == 1
            and cache.left_padding.tolist() == [0]
            and cache._right_padding is None
        ):
            # The singleton forward extracts a native view, then merges it back
            # into this batch wrapper after every chunk. Reserve that wrapper's
            # backing so the next native view retains the capacity too.
            row = KVCache()
            row.keys, row.values = cache.keys, cache.values
            row.offset = cache._idx
            row.prefix_cache_reserve(end)
            cache.keys, cache.values = row.keys, row.values


def install():
    ar = importlib.import_module("mlx_vlm.generate.ar")
    base = ar.PromptProcessingBatch
    if getattr(base, "_yunshu_absolute_spans", False):
        return

    class AbsolutePromptBatch(base):
        _yunshu_absolute_spans = True

        def generate(self, *args, **kwargs):
            coordinator = getattr(self, "_apc_coordinator", None)
            meta = getattr(self, "_apc_meta", None) or []
            descriptors = (
                [
                    (m["full_input_ids"], coordinator.request(m["full_input_ids"]))
                    for m in meta
                    if m is not None
                ]
                if coordinator is not None and hasattr(coordinator, "release_request")
                else []
            )
            session = prompt_end_replay.open_session(self)
            try:
                with self._tail_scope():
                    gen_batch = super().generate(*args, **kwargs)
                if session is not None:
                    session.end()
                    try:
                        session.store(gen_batch)
                    except Exception:  # the stored block is an optimisation only
                        prompt_end_replay.clear()
                return gen_batch
            finally:
                if session is not None:
                    session.end()
                # A decoding row no longer needs its prefill descriptor. This
                # runs before the next identical queued prompt can look up APC.
                for ids, policy in descriptors:
                    coordinator.release_request(ids, policy)

        def _prefill_scope(self):
            embeds = getattr(self, "_inputs_embeds", None)
            if (
                gdn_prefill.split_enabled()
                and embeds is not None
                and embeds.shape[0] == 1
            ):
                return gdn_prefill.prefill_scope()
            return contextlib.nullcontext()

        def _tail_scope(self):
            # The final rows after the last 64-aligned boundary (1-64 tokens) always start at
            # the same absolute position for a given prompt, cold or restored, so they need
            # no long kernels to stay split-invariant. Every tail runs on the lane projections: a
            # 49-row tail took 70 ms longer on the long prefill scope than a 48-row one (32K).
            embeds = getattr(self, "_inputs_embeds", None)
            if (
                gdn_prefill.split_enabled()
                and embeds is not None
                and embeds.shape[0] == 1
                and embeds.shape[1] <= TAIL_FAST_ROWS
            ):
                return gdn_prefill.tail_scope()
            return self._prefill_scope()

        def _split_position(self):
            embeds = getattr(self, "_inputs_embeds", None)
            if not (
                gdn_prefill.split_enabled()
                and embeds is not None
                and embeds.shape[0] == 1
            ):
                return None
            meta = getattr(self, "_apc_meta", None)
            prefix = (
                int(meta[0].get("prefix_len", 0))
                if meta and len(meta) == 1 and meta[0]
                else int((getattr(self, "_cached_tokens_per_row", None) or [0])[0])
            )
            start = self._processed_prompt_columns + prefix
            if start % 64:
                raise ValueError(
                    "split-invariant prefill cannot resume an unaligned state"
                )
            return start, start + embeds.shape[1]

        def needs_processing(self):
            position = self._split_position()
            if position is None:
                return super().needs_processing()
            start, end = position
            # Leave the entire last recurrence block for generate(), including
            # its real prompt logits. The upstream last-one-row cut is unaligned.
            return (end - 1) // 64 * 64 > start

        def prompt_step(self):
            from . import ane_ffn_runtime

            with self._prefill_scope():
                return ane_ffn_runtime.guarded(
                    getattr(self, "prompt_cache", None), self._absolute_prompt_step
                )

        def _absolute_prompt_step(self):
            step = self.prefill_step_size
            position = self._split_position()
            if position is not None:
                start, end = position
                reserve_prompt_kv(getattr(self, "prompt_cache", None) or (), end)
                step = step or 2048
                boundary = min((start // step + 1) * step, (end - 1) // 64 * 64)
                if boundary <= start or boundary % 64:
                    raise ValueError(
                        "split-invariant prefill needs an aligned forward boundary"
                    )
                previous_step = self.prefill_step_size
                self.prefill_step_size = boundary - start
                try:
                    return super().prompt_step()
                finally:
                    self.prefill_step_size = previous_step
            meta = getattr(self, "_apc_meta", None)
            coordinator = getattr(self, "_apc_coordinator", None)
            policy = (
                coordinator.request(meta[0]["full_input_ids"])
                if coordinator is not None
                and hasattr(coordinator, "request")
                and meta
                and len(meta) == 1
                and meta[0]
                else None
            )
            if not policy or not step or not meta or len(meta) != 1 or not meta[0]:
                return super().prompt_step()
            start = self._processed_prompt_columns + int(meta[0].get("prefix_len", 0))
            self.prefill_step_size = step - start % step
            try:
                return super().prompt_step()
            finally:
                self.prefill_step_size = step

    ar.PromptProcessingBatch = AbsolutePromptBatch
    root = importlib.import_module("mlx_vlm.generate")
    root.PromptProcessingBatch = AbsolutePromptBatch
