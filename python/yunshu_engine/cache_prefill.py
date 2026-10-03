# Patches upstream mlx-vlm dispatch; kernels remain unchanged.
"""Absolute prefill spans for the upstream runner's exact checkpoints.

Only span dispatch is changed. Kernels and checkpoint publication are owned by
other workers. A suffix starts by finishing the same atom as a cold request.
"""

from __future__ import annotations

import importlib


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
                return super().generate(*args, **kwargs)
            finally:
                # A decoding row no longer needs its prefill descriptor. This
                # runs before the next identical queued prompt can look up APC.
                for ids, policy in descriptors:
                    coordinator.release_request(ids, policy)

        def prompt_step(self):
            step = self.prefill_step_size
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
