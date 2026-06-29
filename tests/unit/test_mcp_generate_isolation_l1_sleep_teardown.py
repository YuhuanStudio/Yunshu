"""+ 957: MCP generate model-isolation regardless of args shape + L1 sleep teardown.

the MCP tools/call `generate` model-isolation gate fired only when arguments was a
  dict (`and isinstance(arguments, dict)`). A tools/call for `generate` with arguments
  OMITTED (or a non-dict) skipped the check → _tool_generate fell through to the default
  engine, so a key scoped away from the default model could drive it by omitting arguments.
  Now gate REGARDLESS of args shape (resolve the default engine's model_name when omitted).
L1 sleep leaked the entire engine subsystem — it manually nulled
  engine._model/_loaded/_running but never called engine.stop(), leaving the engine_core
  loop+thread, KV/prefix caches, KV-transfer sockets, lora manager, spec decoders alive
  while wake built a BRAND-NEW engine (per-cycle thread/socket/buffer leak). Also the
  manager-unload loop was gated `if level >= 2`, so multi-model L1 (get_engine()==None) was
  a pure no-op that flipped _sleeping=True with every model still resident. Now L1 calls
  engine.stop() and frees manager entries.
"""

from __future__ import annotations

import inspect


def test_mcp_generate_gate_regardless_of_args_shape():
    from yunshu_gateway.routers import mcp

    src = inspect.getsource(mcp)
    # the OLD guard required arguments to be a dict before checking access — gone
    assert 'name") == "generate" and isinstance(_p.get("arguments"), dict)' not in src
    # the gate now fires on name=="generate" alone, then resolves the model
    assert 'if _p.get("name") == "generate":' in src
    # when model is omitted it falls back to the default engine's model_name and gates it
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    assert "_check_model_access(request, _gen_model)" in code
    assert 'getattr(_ge(), "model_name", None)' in code


def test_l1_sleep_stops_engine_and_frees_manager():
    from yunshu_gateway.routers import sleep

    src = inspect.getsource(sleep)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # contextlib is imported (the edit uses contextlib.suppress)
    assert "import contextlib" in code
    # L1 sleep tears down via engine.stop() (not just nulling _model)
    assert "await engine.stop()" in code
    # the manager-unload loop is reachable for L1 (was `if level >= 2:`)
    assert "if level >= 2:" not in code
    assert "if level >= 1:" in code
    # the leak point that was the bug — nulling without stop — no longer stands alone
    assert "engine._model = None" in code  # still nulls, but AFTER stop()


def test_engine_stop_precedes_null():
    """stop() must run BEFORE _model is nulled, else stop() can't reach the live model."""
    from yunshu_gateway.routers import sleep

    src = inspect.getsource(sleep)
    stop_idx = src.index("await engine.stop()")
    null_idx = src.index("engine._model = None")
    assert stop_idx < null_idx, "engine.stop() must precede nulling engine._model"


def test_repetition_penalty_zero_is_noop_not_nan():
    """repetition_penalty=0.0 (allowed by the gateway ge=0.0 schemas) hit the
    `logits / penalty` branch → ±inf → NaN through softmax. It must no-op instead."""
    import math

    import mlx.core as mx

    from yunshu_engine.batch_sampler import LogitsProcessorBatch, LogitsProcessorConfig

    fn = LogitsProcessorBatch._apply_repetition_penalty
    cfg0 = LogitsProcessorConfig(repetition_penalty=0.0, generated_tokens=[1, 2, 3])
    # penalty=0 must be treated as a no-op (returns None → logits untouched)
    assert fn(mx.array([[0.5, -0.5, 0.3, 0.1, -0.2]]), cfg0) is None
    # a valid penalty still applies and stays finite
    cfg = LogitsProcessorConfig(repetition_penalty=1.3, generated_tokens=[1, 2, 3])
    out = fn(mx.array([[0.5, -0.5, 0.3, 0.1, -0.2]]), cfg)
    mx.eval(out)
    assert all(math.isfinite(x) for x in out.tolist()[0])
