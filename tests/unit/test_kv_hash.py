"""Unit tests for KV hash module."""
from yunshu_kv.hash import compute_block_hash, compute_prompt_hashes


class TestComputeBlockHash:
    def test_consistent_hash(self):
        tokens = [1, 2, 3, 4]
        h1 = compute_block_hash(None, tokens)
        h2 = compute_block_hash(None, tokens)
        assert h1 == h2

    def test_different_tokens_different_hash(self):
        h1 = compute_block_hash(None, [1, 2, 3])
        h2 = compute_block_hash(None, [4, 5, 6])
        assert h1 != h2

    def test_parent_hash_changes_result(self):
        tokens = [1, 2, 3]
        h_no_parent = compute_block_hash(None, tokens)
        h_with_parent = compute_block_hash(12345, tokens)
        assert h_no_parent != h_with_parent

    def test_different_parent_different_hash(self):
        tokens = [1, 2, 3]
        h1 = compute_block_hash(100, tokens)
        h2 = compute_block_hash(200, tokens)
        assert h1 != h2

    def test_empty_tokens(self):
        h = compute_block_hash(None, [])
        assert isinstance(h, int)

    def test_extra_keys_int(self):
        h1 = compute_block_hash(None, [1, 2], extra_keys=(42,))
        h2 = compute_block_hash(None, [1, 2], extra_keys=(99,))
        assert h1 != h2

    def test_extra_keys_string(self):
        h1 = compute_block_hash(None, [1, 2], extra_keys=("model-a",))
        h2 = compute_block_hash(None, [1, 2], extra_keys=("model-b",))
        assert h1 != h2

    def test_hash_is_64bit(self):
        h = compute_block_hash(None, [1, 2, 3])
        assert 0 <= h < 2**64


class TestComputePromptHashes:
    def test_exact_blocks(self):
        tokens = list(range(128))  # 2 blocks of size 64
        hashes = compute_prompt_hashes(tokens, block_size=64)
        assert len(hashes) == 2

    def test_leftover_tokens_excluded(self):
        tokens = list(range(100))  # 1 full block + 36 leftover
        hashes = compute_prompt_hashes(tokens, block_size=64)
        assert len(hashes) == 1

    def test_empty_input(self):
        hashes = compute_prompt_hashes([], block_size=64)
        assert hashes == []

    def test_single_full_block(self):
        tokens = list(range(64))
        hashes = compute_prompt_hashes(tokens, block_size=64)
        assert len(hashes) == 1

    def test_chain_property(self):
        """Changing one token changes all subsequent hashes."""
        tokens_a = list(range(128))
        tokens_b = list(range(128))
        tokens_b[0] = 999
        hashes_a = compute_prompt_hashes(tokens_a, 64)
        hashes_b = compute_prompt_hashes(tokens_b, 64)
        assert hashes_a[0] != hashes_b[0]
        assert hashes_a[1] != hashes_b[1]

    def test_block_size_variations(self):
        tokens = list(range(100))
        h32 = compute_prompt_hashes(tokens, block_size=32)
        h64 = compute_prompt_hashes(tokens, block_size=64)
        assert len(h32) == 3
        assert len(h64) == 1

    def test_extra_keys_propagated(self):
        h1 = compute_prompt_hashes([1] * 64, 64, extra_keys=("a",))
        h2 = compute_prompt_hashes([1] * 64, 64, extra_keys=("b",))
        assert h1[0] != h2[0]

    def test_hash_stability(self):
        tokens = list(range(256))
        for _ in range(5):
            hashes = compute_prompt_hashes(tokens, 64)
            assert len(hashes) == 4
