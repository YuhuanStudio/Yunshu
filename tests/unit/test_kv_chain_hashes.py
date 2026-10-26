"""Vectorised block-hash chains equal the per-token struct.pack chains they replaced."""

import numpy as np
import pytest

from yunshu_engine import kv_prefix_cache
from yunshu_kv import hash as kvhash
from yunshu_kv.hash import (
    compute_block_hash,
    compute_chain_hashes,
    compute_prompt_hashes,
)


def old_prompt_hashes(token_ids, block_size, extra_keys=()):
    hashes = []
    parent_hash = None
    for i in range(0, len(token_ids) - (len(token_ids) % block_size), block_size):
        h = compute_block_hash(parent_hash, token_ids[i : i + block_size], extra_keys)
        hashes.append(h)
        parent_hash = h
    return hashes


def old_block_hashes(tokens, block_size):
    hashes = []
    parent = None
    for i in range(0, len(tokens), block_size):
        parent = compute_block_hash(parent, tokens[i : i + block_size])
        hashes.append(parent)
    return hashes


LENGTHS = [0, 1, 63, 64, 65, 127, 128, 1000, 4096 + 17]


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32])
def test_prefix_cache_chain_matches_the_old_loop(n, dtype):
    rng = np.random.default_rng(n)
    toks = rng.integers(0, 248_320, size=n).astype(dtype)
    want = old_block_hashes(toks, kv_prefix_cache._BLOCK_SIZE)
    assert kv_prefix_cache._compute_block_hashes(toks) == want
    assert kv_prefix_cache._compute_block_hashes(toks.tolist()) == want


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("extra", [(), (123456789,), ("model-a",), (7, "x")])
@pytest.mark.parametrize("block_size", [16, 64])
def test_prompt_hashes_match_the_old_loop(n, extra, block_size):
    rng = np.random.default_rng(n + block_size)
    toks = rng.integers(0, 248_320, size=n).tolist()
    assert compute_prompt_hashes(toks, block_size, extra) == old_prompt_hashes(
        toks, block_size, extra
    )


def test_same_hashes_with_the_blake2b_fallback(monkeypatch):
    import hashlib

    def blake(data: bytes) -> int:
        return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")

    monkeypatch.setattr(kvhash, "_hash_bytes", blake)
    toks = list(range(1000, 1300))
    assert compute_chain_hashes(toks, 64) == old_prompt_hashes(toks, 64)


def test_a_hash_covers_everything_before_it():
    a = list(range(256))
    b = list(a)
    b[3] += 1
    ha, hb = compute_chain_hashes(a, 64), compute_chain_hashes(b, 64)
    assert ha[0] != hb[0] and all(x != y for x, y in zip(ha, hb, strict=True))
