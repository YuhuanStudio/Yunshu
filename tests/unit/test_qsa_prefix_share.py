"""Flash-Next QSA checkpoints: a shorter checkpoint's K/V and indexer rows ARE the longer one's rows."""

import pytest

mx = pytest.importorskip("mlx.core")

from tests.unit.test_round_driver_flashnext import make_tiny  # noqa: E402


def _snap(cache):
    from mlx_vlm.apc_adapters import clone_cache_entry

    out = []
    for c in cache:
        s = clone_cache_entry(c, min_capacity_tokens=None, eval_targets=[])
        out.append(s)
    return out


def test_prefix_rows_of_qsa_cache_equal_after_further_prefill():
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache

    lm, _ = make_tiny()
    ids = mx.array([[(7 * i + 3) % 50 for i in range(40)]], dtype=mx.int32)
    cache = lm.make_cache()
    lm(ids[:, :24], cache=cache, return_hidden=True, skip_logits=True)
    mx.eval([c.state for c in cache if hasattr(c, "state")][0:1])
    short = _snap(cache)
    mx.eval([a for c in short if isinstance(c, QSAKVCache) for a in c.state[:4]])
    lm(ids[:, 24:], cache=cache, return_hidden=True, skip_logits=True)
    long = _snap(cache)
    n = 0
    for a, b in zip(short, long):  # noqa: B905
        if not isinstance(a, QSAKVCache) or a.keys is None:
            continue
        n += 1
        assert mx.array_equal(a.keys, b.keys[..., :24, :])
        assert mx.array_equal(a.values, b.values[..., :24, :])
        assert mx.array_equal(a.index_keys, b.index_keys[:, :24])
        assert mx.array_equal(a.index_position_ids, b.index_position_ids[..., :24])
    assert n > 0, "the tiny model has no QSA cache"
