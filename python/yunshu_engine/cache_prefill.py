# Patches upstream mlx-vlm dispatch; kernels remain unchanged.
"""Absolute prefill spans for the upstream runner's exact checkpoints.

Only span dispatch is changed. Kernels and checkpoint publication are owned by
other workers. A suffix starts by finishing the same atom as a cold request.
"""

from __future__ import annotations

import contextlib
import importlib

from .kernels import gdn_prefill

TAIL_FAST_ROWS = 48  # tail rows on the lane projections (LaneLinear row pieces; attention stays BQ32)


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
            try:
                with self._tail_scope():
                    return super().generate(*args, **kwargs)
            finally:
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
            # no long kernels to stay split-invariant. Up to TAIL_FAST_ROWS of them run on the
            # decode/verify kernels: a 16-row tail took 168 ms on the long prefill path
            # (host-bound: ~2000 small ops) against ~40 ms for a verify window.
            embeds = getattr(self, "_inputs_embeds", None)
            if (
                gdn_prefill.split_enabled()
                and embeds is not None
                and embeds.shape[0] == 1
                and embeds.shape[1] <= TAIL_FAST_ROWS
            ):
                # Upstream consumes only these final logits (including any
                # right padding), but Qwen3.5 ignores its logits_to_keep kwarg.
                # The lane head is row invariant, so omit the unused rows.
                keep = 1 + max(getattr(self, "_right_pad_per_row", None) or [0])
                return gdn_prefill.tail_scope(logits_to_keep=keep)
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
            with self._prefill_scope():
                return self._absolute_prompt_step()

        def _absolute_prompt_step(self):
            step = self.prefill_step_size
            position = self._split_position()
            if position is not None:
                start, end = position
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
