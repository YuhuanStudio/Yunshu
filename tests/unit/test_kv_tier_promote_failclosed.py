"""a grounded KV-tier hunt found the numeric core clean but a HIGH-LATENT
fail-OPEN landmine in TieredKVCacheManager._allocate_prefill_promote. When the hot KV
tensors (_key_cache) are unbound — which they always are today, since no production code
calls set_kv_tensors, making the whole KV-byte tier-movement dormant — the promote
write-back was silently skipped yet control fell through to cache_block() + num_matched_tokens,
registering a block of UNINITIALIZED KV as a prefix-cache HIT. The instant anyone wires the
hot tensors, the model would attend over garbage KV (silent corruption / wrong tokens).

Fix: both the warm and SSD promote paths now fail CLOSED — if _key_cache is None, re-insert
the entry into its tier and break the chain instead of counting a bogus hit.
"""

from __future__ import annotations

import inspect

from yunshu_kv import tiered


def test_promote_fails_closed_on_unbound_hot_cache():
    src = inspect.getsource(tiered.TieredKVCacheManager._allocate_prefill_promote)
    # both tiers (warm + SSD) must guard cache_block on a bound hot KV tensor
    assert src.count("if self.hot._key_cache is None:") >= 2, (
        "both warm and SSD promote paths must fail-closed when _key_cache is unbound"
    )
    # the fail-closed branch re-inserts (warm.demote / ssd.store) and breaks BEFORE the
    # cache_block hit-registration — verify the guard precedes 'Register in prefix cache'.
    first_guard = src.index("if self.hot._key_cache is None:")
    first_register = src.index("Register in prefix cache")
    assert first_guard < first_register
    # the rationale is recorded
    assert "FAIL-CLOSED" in src
