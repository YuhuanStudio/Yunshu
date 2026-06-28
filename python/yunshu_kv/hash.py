"""Block hashing utilities for KV prefix caching.

Uses xxhash for speed (vs oMLX's SHA256). Chain hashing: each block's hash
depends on the parent hash + token IDs, enabling O(1) prefix lookup.
"""


import struct

try:
    import xxhash
    def _hash_bytes(data: bytes) -> int:
        return xxhash.xxh64(data).intdigest()
except ImportError:
    import hashlib
    def _hash_bytes(data: bytes) -> int:
        return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


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


def compute_prompt_hashes(
    token_ids: list[int],
    block_size: int,
    extra_keys: tuple = (),
) -> list[int]:
    """Compute chain hashes for all complete blocks in a prompt.

    Returns:
        List of block hashes, one per complete block.
    """
    hashes = []
    parent_hash: int | None = None

    for i in range(0, len(token_ids) - (len(token_ids) % block_size), block_size):
        block_tokens = token_ids[i : i + block_size]
        h = compute_block_hash(parent_hash, block_tokens, extra_keys)
        hashes.append(h)
        parent_hash = h

    return hashes
