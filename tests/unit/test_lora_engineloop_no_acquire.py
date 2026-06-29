"""in the opt-in engine-loop (batched) mode, add_request acquired the per-request
LoRA adapter and then IMMEDIATELY released it (per-request LoRA is unsupported there — the
model is shared across the batch). acquire_adapter mutates the shared _base_model in-place
ON THE EVENT-LOOP THREAD, racing the executor's generate_step → a rejected LoRA request
could transiently corrupt a concurrent batched request's weights. Now the acquire is gated
on `not _running`, removing the off-thread mutation entirely; the request is still rejected.
"""

from __future__ import annotations

import inspect

from yunshu_engine import engine_core


def test_acquire_gated_on_not_running():
    src = inspect.getsource(engine_core.EngineCore.add_request)
    # the acquire is now guarded so it never runs in engine-loop mode
    assert "if lora_adapter and not getattr(self, '_running', False):" in src
    # the SPECIFIC racy "acquire anyway, then release when _running" block is gone
    # (the legitimate fast-path release-after-generation calls stay)
    assert "if loaded_lora and getattr(self, '_running', False):" not in src
    # the engine-loop rejection path is preserved (request still refused, just no
    # off-thread mutation first)
    assert "Per-request LoRA is not supported in batched (engine loop) mode" in src
