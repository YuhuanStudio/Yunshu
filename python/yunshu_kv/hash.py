"""Block hashing utilities for KV prefix caching.

Uses xxhash for speed (vs oMLX's SHA256). Chain hashing: each block's hash
depends on the parent hash + token IDs, enabling O(1) prefix lookup.
"""

import struct

import numpy as np
import xxhash


def _hash_bytes(data: bytes) -> int:
    return int(xxhash.xxh64(data).intdigest())


def compute_block_hash(
    parent_hash: int | None,
    token_ids: list[int],
    extra_keys: tuple = (),
) -> int:
    """Compute a chain hash for a KV block.

    Args:
        parent_hash: Hash of the previous block (None for the first block).
        token_ids: Token IDs in this block.
        extra_keys: Additional keys to mix in (e.g., model name hash).

    Returns:
        A 64-bit integer hash.
    """
    # Pack into bytes: parent_hash (8 bytes) + token_ids (4 bytes each)
    parts = []
    if parent_hash is not None:
        parts.append(struct.pack("<Q", parent_hash))
    else:
        parts.append(struct.pack("<Q", 0))
    for tid in token_ids:
        parts.append(struct.pack("<I", tid))
    for key in extra_keys:
        if isinstance(key, int):
            parts.append(struct.pack("<Q", key))
        elif isinstance(key, str):
            parts.append(key.encode("utf-8"))
    return _hash_bytes(b"".join(parts))


def _extra_key_bytes(extra_keys: tuple) -> bytes:
    parts = []
    for key in extra_keys:
        if isinstance(key, int):
            parts.append(struct.pack("<Q", key))
        elif isinstance(key, str):
            parts.append(key.encode("utf-8"))
    return b"".join(parts)


def compute_chain_hashes(
    token_ids,
    block_size: int,
    extra_keys: tuple = (),
    *,
    partial_tail: bool = False,
) -> list[int]:
    """Chain hashes of consecutive ``block_size`` blocks, one per block.

    Same values as folding :func:`compute_block_hash` over the blocks (each hash covers
    the parent hash, the block's little-endian uint32 token ids and the extra keys), but
    the token ids are converted to bytes once instead of one ``struct.pack`` per token
    (3.5 ms -> 0.5 ms per 32k tokens). ``token_ids`` is a list or an integer ndarray.
    ``partial_tail`` also hashes a final incomplete block; by default only complete
    blocks are hashed.
    """
    raw = np.ascontiguousarray(token_ids, dtype="<u4").tobytes()
    n = len(raw) // 4
    end = n if partial_tail else n - n % block_size
    tail = _extra_key_bytes(extra_keys)
    hashes = []
    parent = 0
    for i in range(0, end, block_size):
        parent = _hash_bytes(
            struct.pack("<Q", parent) + raw[4 * i : 4 * (i + block_size)] + tail
        )
        hashes.append(parent)
    return hashes


def compute_prompt_hashes(
    token_ids: list[int],
    block_size: int,
    extra_keys: tuple = (),
) -> list[int]:
    """Compute chain hashes for all complete blocks in a prompt.

    Returns:
        List of block hashes, one per complete block.
    """
    return compute_chain_hashes(token_ids, block_size, extra_keys)
