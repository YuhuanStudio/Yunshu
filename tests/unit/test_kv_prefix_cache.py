"""Tests for KV Prefix Cache integration.

Tests:
- compute_block_hash() chain hashing correctness
- compute_prompt_hashes() for various prompt lengths
- KVCacheManager.allocate_for_prefill() with cache hits and misses
- KVCacheManager.cache_completed_blocks() registration
- KVCacheManager.free_request() block freeing
- Prefix cache hit across two requests with same prefix
- Memory estimation via compute_num_blocks()
- Hit rate tracking in KVCacheManager
- EngineCore KV cache wiring
- BatchedEngine model architecture extraction
- Admin /kv-cache endpoint
"""
import pytest

from yunshu_kv.block import BlockPool, KVBlock
from yunshu_kv.block_table import BlockTable
from yunshu_kv.hash import compute_block_hash, compute_prompt_hashes
from yunshu_kv.manager import (
    KVCacheConfig,
    KVCacheManager,
    PrefixMatch,
    compute_num_blocks,
)


# ── Chain Hashing Correctness ──


class TestChainHashCorrectness:
    """Tests for compute_block_hash() chain hashing properties."""

    def test_deterministic_same_inputs(self):
        """Same parent + tokens + extra_keys produces same hash."""
        tokens = [10, 20, 30, 40]
        h1 = compute_block_hash(None, tokens)
        h2 = compute_block_hash(None, tokens)
        assert h1 == h2

    def test_different_parent_produces_different_hash(self):
        """Different parent hash changes the result (chain property)."""
        tokens = [1, 2, 3, 4]
        h_no_parent = compute_block_hash(None, tokens)
        h_with_parent = compute_block_hash(42, tokens)
        assert h_no_parent != h_with_parent

    def test_different_tokens_produces_different_hash(self):
        """Different token IDs produce different hashes."""
        h1 = compute_block_hash(None, [1, 2, 3, 4])
        h2 = compute_block_hash(None, [4, 3, 2, 1])
        assert h1 != h2

    def test_extra_keys_change_hash(self):
        """Extra keys (model hash) affect the hash."""
        tokens = [1, 2, 3, 4]
        h1 = compute_block_hash(None, tokens, extra_keys=())
        h2 = compute_block_hash(None, tokens, extra_keys=(99,))
        assert h1 != h2

    def test_string_extra_key(self):
        """String extra keys are mixed into the hash."""
        tokens = [1, 2, 3]
        h1 = compute_block_hash(None, tokens, extra_keys=("model_a",))
        h2 = compute_block_hash(None, tokens, extra_keys=("model_b",))
        assert h1 != h2

    def test_chain_order_matters(self):
        """Hash(A->B) != Hash(A) then separately Hash(B).

        Chain hashing: the parent hash is mixed in, so the order of
        blocks in the chain affects the hash.
        """
        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]

        h_a = compute_block_hash(None, tokens_a)
        h_b_from_a = compute_block_hash(h_a, tokens_b)

        h_b_standalone = compute_block_hash(None, tokens_b)

        assert h_b_from_a != h_b_standalone

    def test_parent_zero_equivalent_to_none(self):
        """Parent hash of 0 is the same as None (both pack as 0)."""
        tokens = [1, 2, 3]
        h_none = compute_block_hash(None, tokens)
        h_zero = compute_block_hash(0, tokens)
        # Both pack 0 for parent, so they should be equal
        assert h_none == h_zero


# ── compute_prompt_hashes ──


class TestComputePromptHashes:
    """Tests for compute_prompt_hashes() with various prompt lengths."""

    def test_exact_multiple_of_block_size(self):
        """256 tokens / 64 block_size = 4 hashes."""
        tokens = list(range(256))
        hashes = compute_prompt_hashes(tokens, block_size=64)
        assert len(hashes) == 4

    def test_remainder_tokens_not_hashed(self):
        """Tokens that don't fill a complete block are not hashed."""
        tokens = list(range(200))
        hashes = compute_prompt_hashes(tokens, block_size=64)
        assert len(hashes) == 3  # 192 / 64 = 3, remainder 8 not hashed

    def test_shorter_than_one_block(self):
        """Fewer tokens than block_size produces no hashes."""
        tokens = list(range(10))
        hashes = compute_prompt_hashes(tokens, block_size=64)
        assert len(hashes) == 0

    def test_empty_prompt(self):
        """Empty token list produces no hashes."""
        hashes = compute_prompt_hashes([], block_size=64)
        assert len(hashes) == 0

    def test_exactly_one_block(self):
        """Exactly block_size tokens produce exactly one hash."""
        tokens = list(range(64))
        hashes = compute_prompt_hashes(tokens, block_size=64)
        assert len(hashes) == 1

    def test_chain_property(self):
        """Hashes are chained: hash[i] depends on hash[i-1]."""
        tokens = list(range(128))
        hashes = compute_prompt_hashes(tokens, block_size=64)
        assert len(hashes) == 2

        # Verify manually
        h0 = compute_block_hash(None, tokens[0:64])
        h1 = compute_block_hash(h0, tokens[64:128])
        assert hashes[0] == h0
        assert hashes[1] == h1

    def test_different_prompts_different_hashes(self):
        """Different prompts produce different hash chains."""
        tokens1 = list(range(64))
        tokens2 = list(range(64, 128))
        h1 = compute_prompt_hashes(tokens1, block_size=64)
        h2 = compute_prompt_hashes(tokens2, block_size=64)
        assert h1 != h2

    def test_extra_keys_propagate(self):
        """Extra keys are mixed into each block hash."""
        tokens = list(range(64))
        h1 = compute_prompt_hashes(tokens, block_size=64, extra_keys=())
        h2 = compute_prompt_hashes(tokens, block_size=64, extra_keys=(42,))
        assert h1 != h2


# ── KVCacheManager.allocate_for_prefill ──


class TestAllocateForPrefill:
    """Tests for KVCacheManager.allocate_for_prefill() with hits and misses."""

    def _make_manager(self, num_blocks: int = 200, block_size: int = 16):
        config = KVCacheConfig(block_size=block_size, enable_caching=True)
        return KVCacheManager(config, num_blocks=num_blocks)

    def test_first_request_no_cache_hit(self):
        """First request always has zero matched tokens."""
        mgr = self._make_manager()
        tokens = list(range(64))
        table, match = mgr.allocate_for_prefill(tokens, model_hash=42)

        assert match.num_matched_tokens == 0
        assert len(match.unmatched_token_ids) == 64
        assert table.num_blocks == 4  # 64/16

    def test_allocate_for_short_prompt(self):
        """Prompt shorter than block_size still gets one block."""
        mgr = self._make_manager()
        tokens = [1, 2, 3]
        table, match = mgr.allocate_for_prefill(tokens, model_hash=42)

        assert match.num_matched_tokens == 0
        assert table.num_blocks == 1  # ceil(3/16) = 1

    def test_allocate_returns_prefix_match_dataclass(self):
        """allocate_for_prefill returns PrefixMatch with correct fields."""
        mgr = self._make_manager()
        tokens = list(range(32))
        table, match = mgr.allocate_for_prefill(tokens, model_hash=42)

        assert isinstance(match, PrefixMatch)
        assert hasattr(match, "matched_blocks")
        assert hasattr(match, "num_matched_tokens")
        assert hasattr(match, "unmatched_token_ids")

    def test_different_model_hash_no_hit(self):
        """Different model_hash prevents cache hits."""
        mgr = self._make_manager()
        tokens = list(range(64))
        table1, _ = mgr.allocate_for_prefill(tokens, model_hash=1)
        mgr.cache_completed_blocks(table1, tokens, model_hash=1)
        mgr.free_request(table1)

        # Same tokens but different model hash
        table2, match2 = mgr.allocate_for_prefill(tokens, model_hash=2)
        assert match2.num_matched_tokens == 0


# ── KVCacheManager.cache_completed_blocks ──


class TestCacheCompletedBlocks:
    """Tests for KVCacheManager.cache_completed_blocks() registration."""

    def _make_manager(self, num_blocks: int = 200, block_size: int = 16):
        config = KVCacheConfig(block_size=block_size, enable_caching=True)
        return KVCacheManager(config, num_blocks=num_blocks)

    def test_caches_full_blocks(self):
        """cache_completed_blocks registers full blocks in prefix cache."""
        mgr = self._make_manager()
        tokens = list(range(64))
        table, _ = mgr.allocate_for_prefill(tokens, model_hash=42)

        cached = mgr.cache_completed_blocks(table, tokens, model_hash=42)
        assert cached == 4  # 4 full blocks

        # Verify blocks are now in hash lookup
        blocks = table.get_blocks()
        for b in blocks:
            assert b.block_hash is not None

    def test_does_not_cache_partial_block(self):
        """Incomplete final block is not cached."""
        mgr = self._make_manager()
        tokens = list(range(50))  # 3 full blocks (48) + 2 remaining
        table, _ = mgr.allocate_for_prefill(tokens, model_hash=42)

        cached = mgr.cache_completed_blocks(table, tokens, model_hash=42)
        assert cached == 3

        blocks = table.get_blocks()
        assert blocks[0].block_hash is not None
        assert blocks[1].block_hash is not None
        assert blocks[2].block_hash is not None
        # Last block is partial, may or may not be cached depending on implementation

    def test_no_double_caching(self):
        """Already cached blocks are not re-cached."""
        mgr = self._make_manager()
        tokens = list(range(32))
        table, _ = mgr.allocate_for_prefill(tokens, model_hash=42)

        cached1 = mgr.cache_completed_blocks(table, tokens, model_hash=42)
        assert cached1 == 2

        cached2 = mgr.cache_completed_blocks(table, tokens, model_hash=42)
        assert cached2 == 0  # already cached

    def test_caching_disabled(self):
        """With enable_caching=False, no blocks are cached."""
        config = KVCacheConfig(block_size=16, enable_caching=False)
        mgr = KVCacheManager(config, num_blocks=200)
        tokens = list(range(32))
        table, _ = mgr.allocate_for_prefill(tokens, model_hash=42)

        cached = mgr.cache_completed_blocks(table, tokens, model_hash=42)
        assert cached == 0


# ── KVCacheManager.free_request ──


class TestFreeRequest:
    """Tests for KVCacheManager.free_request() block freeing."""

    def _make_manager(self, num_blocks: int = 200, block_size: int = 16):
        config = KVCacheConfig(block_size=block_size, enable_caching=True)
        return KVCacheManager(config, num_blocks=num_blocks)

    def test_frees_all_blocks(self):
        """free_request returns all blocks to the free pool."""
        mgr = self._make_manager()
        initial_free = mgr.num_free_blocks

        tokens = list(range(64))
        table, _ = mgr.allocate_for_prefill(tokens, model_hash=42)
        assert mgr.num_free_blocks == initial_free - 4

        mgr.free_request(table)
        assert mgr.num_free_blocks == initial_free

    def test_clears_block_table(self):
        """free_request empties the block table."""
        mgr = self._make_manager()
        tokens = list(range(32))
        table, _ = mgr.allocate_for_prefill(tokens, model_hash=42)

        assert table.num_blocks > 0
        mgr.free_request(table)
        assert table.num_blocks == 0

    def test_shared_blocks_ref_counted(self):
        """Shared blocks (prefix cache hit) are not freed until all refs drop."""
        mgr = self._make_manager()
        block_size = mgr.block_size

        # Request 1: allocate and cache
        tokens1 = list(range(64))
        table1, _ = mgr.allocate_for_prefill(tokens1, model_hash=42)
        mgr.cache_completed_blocks(table1, tokens1, model_hash=42)

        # Request 2: hits prefix cache (shares blocks with table1)
        tokens2 = list(range(64)) + list(range(100, 116))
        table2, match2 = mgr.allocate_for_prefill(tokens2, model_hash=42)
        assert match2.num_matched_tokens == 64

        # Free request 1 — shared blocks should still be alive
        mgr.free_request(table1)

        # Free request 2
        mgr.free_request(table2)

        # All blocks back to free pool
        assert mgr.num_free_blocks == 199  # 200 - 1 (null block)


# ── Prefix Cache Hit Across Requests ──


class TestPrefixCacheHitAcrossRequests:
    """Tests for prefix cache reuse across two requests with same prefix."""

    def _make_manager(self, num_blocks: int = 200, block_size: int = 16):
        config = KVCacheConfig(block_size=block_size, enable_caching=True)
        return KVCacheManager(config, num_blocks=num_blocks)

    def test_full_prefix_hit(self):
        """Second request fully reuses first request's prefix."""
        mgr = self._make_manager()

        tokens1 = list(range(64))
        table1, match1 = mgr.allocate_for_prefill(tokens1, model_hash=42)
        assert match1.num_matched_tokens == 0

        mgr.cache_completed_blocks(table1, tokens1, model_hash=42)
        mgr.free_request(table1)

        # Same tokens, should fully hit
        tokens2 = list(range(64)) + list(range(100, 116))
        table2, match2 = mgr.allocate_for_prefill(tokens2, model_hash=42)
        assert match2.num_matched_tokens == 64
        assert len(match2.unmatched_token_ids) == 16
        mgr.free_request(table2)

    def test_partial_prefix_hit(self):
        """Second request partially matches prefix."""
        mgr = self._make_manager()

        tokens1 = list(range(48))  # 3 full blocks
        table1, _ = mgr.allocate_for_prefill(tokens1, model_hash=42)
        mgr.cache_completed_blocks(table1, tokens1, model_hash=42)
        mgr.free_request(table1)

        # First 48 tokens match, rest don't
        tokens2 = list(range(48)) + list(range(100, 132))  # 80 tokens total
        table2, match2 = mgr.allocate_for_prefill(tokens2, model_hash=42)
        assert match2.num_matched_tokens == 48
        assert len(match2.unmatched_token_ids) == 32
        mgr.free_request(table2)

    def test_no_prefix_hit_different_content(self):
        """Different prompts produce no prefix hit."""
        mgr = self._make_manager()

        tokens1 = list(range(64))
        table1, _ = mgr.allocate_for_prefill(tokens1, model_hash=42)
        mgr.cache_completed_blocks(table1, tokens1, model_hash=42)
        mgr.free_request(table1)

        # Completely different tokens
        tokens2 = list(range(200, 264))
        table2, match2 = mgr.allocate_for_prefill(tokens2, model_hash=42)
        assert match2.num_matched_tokens == 0
        mgr.free_request(table2)

    def test_chain_hash_breaks_on_mismatch(self):
        """If block N doesn't match, block N+1 can't match (chain hash)."""
        mgr = self._make_manager()

        # First request: tokens 0-31 (2 blocks)
        tokens1 = list(range(32))
        table1, _ = mgr.allocate_for_prefill(tokens1, model_hash=42)
        mgr.cache_completed_blocks(table1, tokens1, model_hash=42)
        mgr.free_request(table1)

        # Second request: tokens 0-15 match block 0, but tokens 16-31 differ
        # Block 0 matches (tokens 0-15), block 1 doesn't (tokens 16-31 differ)
        # So only 1 block (16 tokens) matches
        tokens2 = list(range(16)) + list(range(200, 216)) + list(range(100, 116))
        table2, match2 = mgr.allocate_for_prefill(tokens2, model_hash=42)
        assert match2.num_matched_tokens == 16  # only block 0 matches
        mgr.free_request(table2)

    def test_three_requests_incremental_prefix(self):
        """Three requests with incrementally longer shared prefixes."""
        mgr = self._make_manager()

        # Request 1: 32 tokens (2 blocks)
        tokens1 = list(range(32))
        t1, _ = mgr.allocate_for_prefill(tokens1, model_hash=0)
        mgr.cache_completed_blocks(t1, tokens1, model_hash=0)
        mgr.free_request(t1)

        # Request 2: same 32 tokens + 16 more (3 blocks total)
        tokens2 = list(range(48))
        t2, m2 = mgr.allocate_for_prefill(tokens2, model_hash=0)
        assert m2.num_matched_tokens == 32
        mgr.cache_completed_blocks(t2, tokens2, model_hash=0)
        mgr.free_request(t2)

        # Request 3: same 48 tokens + 16 more
        tokens3 = list(range(64))
        t3, m3 = mgr.allocate_for_prefill(tokens3, model_hash=0)
        assert m3.num_matched_tokens == 48
        mgr.free_request(t3)


# ── Memory Estimation via compute_num_blocks ──


class TestComputeNumBlocks:
    """Tests for compute_num_blocks() memory estimation."""

    def test_typical_m3_ultra_config(self):
        """192GB UMA, 40GB weights, 32L/8KV/128d model."""
        config = KVCacheConfig(
            block_size=64,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_bytes=2,
        )
        # 192 GB UMA, 40 GB model weights
        n = compute_num_blocks(config, 192 * 2**30, 40 * 2**30)
        assert n > 0
        # bytes per block = 64 * 32 * 8 * 128 * 2 * 2 = 8,388,608 = 8MB
        # available = (192 - 40) * 0.85 = 129.2 GB
        # blocks = 129.2 GB / 8 MB ≈ 16,691
        assert 15000 < n < 18000

    def test_small_model_large_uma(self):
        """Small model on large UMA yields many blocks."""
        config = KVCacheConfig(
            block_size=64,
            num_layers=12,
            num_kv_heads=4,
            head_dim=64,
            dtype_bytes=2,
        )
        n = compute_num_blocks(config, 192 * 2**30, 2 * 2**30)
        assert n > 50000

    def test_zero_layers_yields_zero(self):
        """Zero layers means zero bytes per block."""
        config = KVCacheConfig(
            block_size=64,
            num_layers=0,
            num_kv_heads=8,
            head_dim=128,
            dtype_bytes=2,
        )
        n = compute_num_blocks(config, 192 * 2**30, 0)
        assert n == 0

    def test_activation_reserve_reduces_blocks(self):
        """Higher activation_reserve_ratio reduces available blocks."""
        config = KVCacheConfig(
            block_size=64,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_bytes=2,
        )
        n_low = compute_num_blocks(config, 192 * 2**30, 40 * 2**30, activation_reserve_ratio=0.1)
        n_high = compute_num_blocks(config, 192 * 2**30, 40 * 2**30, activation_reserve_ratio=0.5)
        assert n_low > n_high

    def test_large_weights_reduce_blocks(self):
        """Larger model weights leave less room for KV cache."""
        config = KVCacheConfig(
            block_size=64,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_bytes=2,
        )
        n_small_weights = compute_num_blocks(config, 192 * 2**30, 10 * 2**30)
        n_large_weights = compute_num_blocks(config, 192 * 2**30, 100 * 2**30)
        assert n_small_weights > n_large_weights


# ── Hit Rate Tracking ──


class TestHitRateTracking:
    """Tests for hit rate tracking in KVCacheManager."""

    def test_initial_hit_rate_zero(self):
        """Fresh manager has 0.0 hit rate."""
        config = KVCacheConfig(block_size=16, enable_caching=True)
        mgr = KVCacheManager(config, num_blocks=200)
        assert mgr.hit_rate == 0.0
        assert mgr._total_lookups == 0
        assert mgr._total_hits == 0

    def test_hit_rate_after_miss(self):
        """First request (all misses) has 0.0 hit rate.

        The lookup loop breaks at the first miss (chain hash), so
        a fresh cache with 4-block prompt does 1 lookup (miss on first block).
        """
        config = KVCacheConfig(block_size=16, enable_caching=True)
        mgr = KVCacheManager(config, num_blocks=200)

        tokens = list(range(64))
        table, _ = mgr.allocate_for_prefill(tokens, model_hash=42)
        # Loop breaks at first miss: 1 lookup, 0 hits
        assert mgr._total_lookups == 1
        assert mgr._total_hits == 0
        assert mgr.hit_rate == 0.0
        mgr.free_request(table)

    def test_hit_rate_after_hit(self):
        """Second request with prefix hit updates hit rate.

        Request 1: 64 tokens (4 blocks), first hash misses -> 1 lookup, 0 hits.
        Request 2: 80 tokens (5 block hashes checked), first 4 hit then break
        on 5th which doesn't exist -> 5 lookups, 4 hits.
        Total: 6 lookups, 4 hits = 66.7% hit rate.
        """
        config = KVCacheConfig(block_size=16, enable_caching=True)
        mgr = KVCacheManager(config, num_blocks=200)

        tokens1 = list(range(64))
        t1, _ = mgr.allocate_for_prefill(tokens1, model_hash=42)
        mgr.cache_completed_blocks(t1, tokens1, model_hash=42)
        mgr.free_request(t1)

        # After request 1: 1 lookup (first hash miss, break), 0 hits
        assert mgr._total_lookups == 1
        assert mgr._total_hits == 0

        tokens2 = list(range(64)) + list(range(100, 116))
        t2, m2 = mgr.allocate_for_prefill(tokens2, model_hash=42)
        assert m2.num_matched_tokens == 64

        # Request 2: checks 5 hashes (4 hit + 1 miss on 5th), then break
        assert mgr._total_lookups == 1 + 5
        assert mgr._total_hits == 4
        assert abs(mgr.hit_rate - 4 / 6) < 0.001
        mgr.free_request(t2)


# ── EngineCore KV Cache Wiring ──


class TestEngineCoreKVWiring:
    """Tests for EngineCore KV cache configuration and wiring."""

    def test_paged_kv_disabled_by_default(self):
        """EngineCore without enable_paged_kv uses base Scheduler."""
        from yunshu_engine.engine_core import EngineCore, EngineCoreConfig
        from yunshu_engine.scheduler import Scheduler

        config = EngineCoreConfig()
        core = EngineCore(None, None, config=config, executor=_fake_executor())

        assert isinstance(core.scheduler, Scheduler)
        assert core._kv_manager is None
        assert core.get_kv_cache_stats() == {"enabled": False}

    def test_paged_kv_enabled_creates_paged_scheduler(self):
        """EngineCore with enable_paged_kv creates PagedScheduler."""
        from yunshu_engine.engine_core import EngineCore, EngineCoreConfig
        from yunshu_engine.paged_scheduler import PagedScheduler

        config = EngineCoreConfig(
            enable_paged_kv=True,
            kv_block_size=16,
            kv_num_blocks=100,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
        )
        core = EngineCore(None, None, config=config, executor=_fake_executor())

        assert isinstance(core.scheduler, PagedScheduler)
        assert core._kv_manager is not None

    def test_kv_stats_with_paged_enabled(self):
        """get_kv_cache_stats returns meaningful stats when paged KV is on."""
        from yunshu_engine.engine_core import EngineCore, EngineCoreConfig

        config = EngineCoreConfig(
            enable_paged_kv=True,
            kv_block_size=16,
            kv_num_blocks=100,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
        )
        core = EngineCore(None, None, config=config, executor=_fake_executor())

        stats = core.get_kv_cache_stats()
        assert stats["enabled"] is True
        assert stats["block_size"] == 16
        assert stats["total_blocks"] == 100
        assert stats["free_blocks"] == 99  # -1 for null block
        assert "usage" in stats
        assert "hit_rate" in stats
        assert "active_block_tables" in stats

    def test_engine_core_config_arch_fields(self):
        """EngineCoreConfig has model architecture fields."""
        from yunshu_engine.engine_core import EngineCoreConfig

        config = EngineCoreConfig(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            kv_num_blocks=1024,
        )
        assert config.num_layers == 32
        assert config.num_kv_heads == 8
        assert config.head_dim == 128
        assert config.kv_num_blocks == 1024


# ── BatchedEngine Model Architecture Extraction ──


class TestBatchedEngineArchExtraction:
    """Tests for BatchedEngine._extract_model_arch()."""

    def test_extracts_from_standard_config(self):
        """Extracts num_layers, num_kv_heads, head_dim from HuggingFace config."""
        from yunshu_engine.batched_engine import BatchedEngine

        class FakeConfig:
            num_hidden_layers = 32
            num_key_value_heads = 8
            hidden_size = 4096
            num_attention_heads = 32

        class FakeModel:
            config = FakeConfig()

        result = BatchedEngine._extract_model_arch(FakeModel())
        assert result["num_layers"] == 32
        assert result["num_kv_heads"] == 8
        assert result["head_dim"] == 128  # 4096 / 32

    def test_returns_empty_for_none_model(self):
        """Returns empty dict for None model."""
        from yunshu_engine.batched_engine import BatchedEngine

        result = BatchedEngine._extract_model_arch(None)
        assert result == {}

    def test_returns_empty_for_model_without_config(self):
        """Returns empty dict when model has no config."""
        from yunshu_engine.batched_engine import BatchedEngine

        class BareModel:
            pass

        result = BatchedEngine._extract_model_arch(BareModel())
        assert result == {}

    def test_returns_empty_for_incomplete_config(self):
        """Returns empty dict when config is missing required fields."""
        from yunshu_engine.batched_engine import BatchedEngine

        class PartialConfig:
            num_hidden_layers = 32
            # missing num_key_value_heads

        class PartialModel:
            config = PartialConfig()

        result = BatchedEngine._extract_model_arch(PartialModel())
        assert result == {}

    def test_handles_args_attribute(self):
        """Also checks model.args as fallback (some MLX models use args)."""
        from yunshu_engine.batched_engine import BatchedEngine

        class FakeArgs:
            num_hidden_layers = 24
            num_key_value_heads = 4
            hidden_size = 2048
            num_attention_heads = 16

        class FakeModel:
            args = FakeArgs()

        result = BatchedEngine._extract_model_arch(FakeModel())
        assert result["num_layers"] == 24
        assert result["num_kv_heads"] == 4
        assert result["head_dim"] == 128

    def test_batched_engine_kv_stats_no_engine(self):
        """get_kv_cache_stats returns disabled when no engine is loaded."""
        from yunshu_engine.batched_engine import BatchedEngine

        engine = BatchedEngine()
        assert engine.get_kv_cache_stats() == {"enabled": False}


# ── Admin KV Cache Endpoint ──


class TestAdminKVCacheEndpoint:
    """Tests for /admin/kv-cache endpoint."""

    @pytest.mark.asyncio
    async def test_endpoint_returns_disabled_without_engine(self):
        """Endpoint returns {enabled: false} when no engine is set."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        from yunshu_api.routers.admin import router

        app = FastAPI()
        app.include_router(router)

        # Reset engine state
        import yunshu_gateway.engine as engine_mod
        old_engine = engine_mod._engine
        engine_mod._engine = None
        old_manager = engine_mod._model_manager
        engine_mod._model_manager = None

        try:
            client = TestClient(app)
            response = client.get("/admin/kv-cache")
            assert response.status_code == 200
            assert response.json() == {"enabled": False}
        finally:
            engine_mod._engine = old_engine
            engine_mod._model_manager = old_manager

    @pytest.mark.asyncio
    async def test_endpoint_returns_stats_with_engine(self):
        """Endpoint returns KV stats when engine has paged KV enabled."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from unittest.mock import MagicMock

        from yunshu_api.routers.admin import router

        app = FastAPI()
        app.include_router(router)

        # Mock engine with get_kv_cache_stats
        import yunshu_gateway.engine as engine_mod
        old_engine = engine_mod._engine

        mock_engine = MagicMock()
        mock_engine.get_kv_cache_stats.return_value = {
            "enabled": True,
            "block_size": 16,
            "total_blocks": 100,
            "free_blocks": 80,
            "usage": 0.2,
        }
        engine_mod._engine = mock_engine

        try:
            client = TestClient(app)
            response = client.get("/admin/kv-cache")
            assert response.status_code == 200
            data = response.json()
            assert data["enabled"] is True
            assert data["block_size"] == 16
            assert data["total_blocks"] == 100
        finally:
            engine_mod._engine = old_engine


# ── Helper ──


def _fake_executor():
    """Create a fake executor that runs sync functions immediately."""
    from concurrent.futures import ThreadPoolExecutor
    return ThreadPoolExecutor(max_workers=1)
