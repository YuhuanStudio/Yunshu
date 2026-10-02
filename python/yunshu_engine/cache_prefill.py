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
