"""engine-level response cache (batched_engine) missed the W666 hardening.

The HTTP middleware cache was hardened (auth-namespaced + skips non-deterministic), but
the SEPARATE engine-level cache inside BatchedEngine.generate() — consulted on every middleware
miss — was not. A fresh hunt found 3 issues:
  HIGH — key omitted lora_adapter: a request for adapter Y collided with a cached adapter X
    response → WRONG fine-tuned weights served (LoRA is the product feature).
  HIGH — no determinism guard: a temperature>0/seed=None request cached its first random sample
    and replayed it for every identical request (frozen creative output).
  MEDIUM — key omitted stop_token_ids/min_tokens/ignore_eos/suppress_tokens.
Fix: add a temperature>0&&seed-None skip guard (mirroring the middleware) + add the missing
output-affecting params to the cache key.
"""
from __future__ import annotations

import inspect

from yunshu_engine.gateway_optimizer import ResponseCache


def test_lora_adapter_differentiates_key():
    base = ResponseCache.hash_request("m", "p", temperature=0.0)
    kx = ResponseCache.hash_request("m", "p", temperature=0.0, lora_adapter="X")
    ky = ResponseCache.hash_request("m", "p", temperature=0.0, lora_adapter="Y")
    assert kx != ky                 # different adapters → different cache entries
    assert kx != base and ky != base


def test_missing_sampling_params_differentiate_key():
    for k in ("stop_token_ids", "min_tokens", "ignore_eos", "suppress_tokens"):
        a = ResponseCache.hash_request("m", "p", **{k: "1"})
        b = ResponseCache.hash_request("m", "p", **{k: "2"})
        assert a != b, f"{k} does not differentiate the cache key"


def test_engine_cache_has_determinism_guard_and_lora_in_key():
    from yunshu_engine import batched_engine
    src = inspect.getsource(batched_engine.BatchedEngine.generate)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # determinism guard mirrors the middleware (skip temp>0 && seed is None)
    assert "_rc_deterministic" in code
    assert "temperature > 0 and seed is None" in code
    assert "if not spec_decode and _rc_deterministic:" in code
    # the missing params are now in the cache key
    assert "lora_adapter=str(lora_adapter)" in code
    assert "stop_token_ids=str(stop_token_ids)" in code
    assert "suppress_tokens=str(suppress_tokens)" in code
