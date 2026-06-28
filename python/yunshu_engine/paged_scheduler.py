"""Yunshu PagedScheduler — Scheduler with KVCacheManager integration.

Extends the base Scheduler with:
- Paged KV cache via KVCacheManager (BlockPool + BlockTable)
- Prefix caching: reuse cached blocks across requests
- Memory-aware scheduling: reject requests that exceed KV budget
- Block-wise cache management integrated with BatchGenerator lifecycle
- Periodic KV block compaction (KVBlockCompactor from kv_optimizations)
- Prediction-based eviction (KVEvictionPredictor from kv_optimizations)
"""

import logging
import os
from typing import Any

from .kv_optimizations import KVBlockCompactor, KVEvictionPredictor
from .request import Request, RequestOutput, RequestStatus
from .scheduler import Scheduler, SchedulerConfig, SchedulerOutput

logger = logging.getLogger(__name__)


class PagedScheduler(Scheduler):
    """Scheduler with paged KV cache and prefix caching."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: SchedulerConfig | None = None,
        kv_cache_manager: Any = None,
    ) -> None:
        super().__init__(model, tokenizer, config)
        self._kv_manager = kv_cache_manager
        self._block_tables: dict[str, Any] = {}
        # Tracks request IDs that have already been finalized to prevent
        # double finalization (e.g., _manage_kv_cache finishes a request,
        # then _cleanup_finished calls _finalize_request_blocks again).
        self._finalized_requests: set[str] = set()
        # Boundary snapshot store for non-sliceable cache layers
        self._boundary_store = None
        ssd_dir = os.environ.get("YUNSHU_SSD_CACHE_DIR")
        if ssd_dir:
            try:
                from pathlib import Path

                from yunshu_kv.boundary_snapshot import BoundarySnapshotSSDStore
                self._boundary_store = BoundarySnapshotSSDStore(Path(ssd_dir))
                self._boundary_store.start()
                logger.info("BoundarySnapshotSSDStore started for non-sliceable layers")
            except Exception as e:
                logger.debug(f"Boundary snapshot store not available: {e}")
        # KV block compactor for periodic defragmentation (kv_optimizations)
        self._compactor: KVBlockCompactor | None = None
        # Eviction predictor for prediction-based block retention (kv_optimizations)
        self._eviction_predictor: KVEvictionPredictor | None = None

    def set_kv_cache_manager(self, manager: Any) -> None:
        self._kv_manager = manager

    def set_compactor(self, compactor: KVBlockCompactor) -> None:
        """Set the KV block compactor for periodic defragmentation."""
        self._compactor = compactor

    def set_eviction_predictor(self, predictor: KVEvictionPredictor) -> None:
        """Set the eviction predictor for prediction-based block retention."""
        self._eviction_predictor = predictor

    def prefill_progress(self, request_id: str) -> tuple[int, int]:
        """Return (matched_tokens, total_tokens) for a request's prefill."""
        req = self.requests.get(request_id)
        if req is None:
            return (0, 0)
        total = len(req.prompt_token_ids)
        matched = getattr(req, 'cached_tokens', 0)
        return (matched, total)

    def get_memory_status(self) -> dict:
        """Return KV cache memory status."""
        if self._kv_manager is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "total_blocks": self._kv_manager.block_pool.num_blocks,
            "free_blocks": self._kv_manager.num_free_blocks,
            "usage_pct": round(self._kv_manager.usage * 100, 1),
            "active_requests": len(self._block_tables),
        }

    def add_request(self, request: Request) -> None:
        if self._kv_manager is not None:
            self._add_with_paged_cache(request)
        else:
            super().add_request(request)

    def _add_with_paged_cache(self, request: Request) -> None:
        token_ids = request.prompt_token_ids

        needed_blocks = (len(token_ids) + self._kv_manager.block_size - 1) // self._kv_manager.block_size
        decode_reserve = min(
            (request.sampling_params.max_tokens + self._kv_manager.block_size - 1) // self._kv_manager.block_size,
            16,
        )

        # Estimate prefix cache savings: if the RadixTree has a matching
        # prefix, those blocks don't need to be allocated from the free pool.
        # Without this, the admission check rejects requests that would fit
        # due to prefix reuse (false rejection).
        cached_blocks_estimate = 0
        if hasattr(self._kv_manager, '_radix_tree') and self._kv_manager.config.enable_caching:
            try:
                bs = self._kv_manager.block_size
                if len(token_ids) >= bs:
                    matched_node, _ = self._kv_manager._radix_tree.match(token_ids)
                    cached_tokens = matched_node.total_tokens()
                    cached_blocks_estimate = cached_tokens // bs
            except Exception:
                pass  # Best-effort estimate
        effective_needed = max(0, needed_blocks - cached_blocks_estimate) + decode_reserve

        if self._kv_manager.num_free_blocks < effective_needed:
            if not self._kv_manager.evict_for_memory(effective_needed):
                logger.warning(
                    f"Rejecting request {request.request_id}: "
                    f"need {effective_needed} blocks (of which {cached_blocks_estimate} cached), "
                    f"only {self._kv_manager.num_free_blocks} free"
                )
                request.set_finished(RequestStatus.FINISHED_ERROR, reason="kv_cache_full")
                return

        try:
            table, prefix_match = self._kv_manager.allocate_for_prefill(token_ids)
        except (MemoryError, ValueError) as exc:
            # OOM recovery: allocate_for_prefill may have partially allocated
            # blocks before failing.  Free whatever was allocated and reject.
            logger.warning(
                f"allocate_for_prefill failed for {request.request_id}: {exc}"
            )
            # The table returned on exception may be partial; if it exists in
            # _block_tables from a prior attempt, clean it up.
            partial = self._block_tables.pop(request.request_id, None)
            if partial is not None:
                self._kv_manager.free_request(partial, request.request_id)
            request.set_finished(RequestStatus.FINISHED_ERROR, reason="kv_cache_oom")
            return
        self._block_tables[request.request_id] = table
        request.cached_tokens = getattr(prefix_match, 'num_matched_tokens', 0)

        # Check queue capacity BEFORE calling super(), which may reject.
        # If rejected, free the already-allocated blocks to avoid leaks.
        if len(self.waiting) >= self.config.max_waiting_requests:
            self._kv_manager.free_request(table, request.request_id)
            self._block_tables.pop(request.request_id, None)
            request.set_finished(RequestStatus.FINISHED_ERROR, reason="queue_full")
            return

        super().add_request(request)

    def step(self) -> SchedulerOutput:
        output = super().step()
        if self._kv_manager is not None and output.outputs:
            self._manage_kv_cache(output.outputs)
        # Periodic KV block compaction (every 50 steps)
        # Use the step counter already incremented by super().step()
        if self._compactor is not None:
            self._compactor.maybe_compact(self._step_counter)
        return output

    def _manage_kv_cache(self, outputs: list[RequestOutput]) -> None:
        error_outputs: list[RequestOutput] = []
        for req_output in list(outputs):
            req_id = req_output.request_id

            if req_output.finished:
                self._finalize_request_blocks(req_id)
            else:
                req = self.requests.get(req_id)
                if req is not None:
                    table = self._block_tables.get(req_id)
                    if table is not None and self._kv_manager is not None:
                        total_tokens = len(req.prompt_token_ids) + req.num_output_tokens
                        # Allocate a new block when the current blocks can no
                        # longer hold all tokens.  The old check
                        # (total_tokens % block_size == 0) missed the case
                        # where the prompt length was an exact multiple of
                        # block_size and the first decode token spilled into a
                        # new block that hadn't been allocated yet.
                        current_capacity = table.num_blocks * self._kv_manager.block_size
                        if total_tokens > current_capacity:
                            try:
                                self._kv_manager.allocate_block_for_decode(table)
                            except (ValueError, MemoryError, RuntimeError) as alloc_err:
                                # Block allocation failed — propagate error by
                                # marking request finished and scheduling cleanup.
                                # Catch all alloc errors (not just ValueError) to
                                # prevent silent failures that leak KV blocks.
                                logger.warning(
                                    "KV block allocation failed for request %s: %s",
                                    req_id, alloc_err,
                                )
                                if req.batch_uid is not None and req.batch_uid not in self._uids_to_remove:
                                    self._uids_to_remove.append(req.batch_uid)
                                req.set_finished(RequestStatus.FINISHED_ERROR, reason="kv_cache_oom")
                                self._uid_to_req.pop(getattr(req, 'batch_uid', None), None)
                                self._finalize_request_blocks(req_id)
                                # Generate an error output so the engine loop
                                # delivers a response to the client instead of
                                # silently dropping the request.
                                error_outputs.append(RequestOutput(
                                    request_id=req_id,
                                    output_text="",
                                    finished=True,
                                    finish_reason="error",
                                    error="KV block allocation failed",
                                ))
        outputs.extend(error_outputs)

    def trim_sliding_window_blocks(self, req_id: str, num_blocks: int) -> int:
        """Free physical KV blocks that slid outside the attention window.

        Called by EngineCore when SlidingWindowKVManager determines that some
        prefix blocks are no longer needed.  The blocks are removed from the
        request's BlockTable and returned to the BlockPool so they can be
        reused by other requests.

        This is the missing physical free that caused a memory leak: before
        this fix, the sliding-window manager only updated its own logical
        bookkeeping and trimmed the MLX KV arrays, but never released the
        physical KVBlock objects back to the pool.

        Args:
            req_id: Request ID whose blocks should be trimmed.
            num_blocks: Number of prefix blocks to evict.

        Returns:
            Number of blocks actually freed (may be less than num_blocks if
            the request has fewer blocks or is not tracked).
        """
        if num_blocks <= 0 or self._kv_manager is None:
            return 0

        table = self._block_tables.get(req_id)
        if table is None:
            return 0

        # Don't trim more blocks than the table holds
        actual = min(num_blocks, table.num_blocks)
        if actual <= 0:
            return 0

        try:
            trimmed = table.trim_prefix_blocks(actual)
        except (ValueError, IndexError):
            logger.warning(
                "trim_prefix_blocks failed for request %s (n=%d, table has %d blocks)",
                req_id, actual, table.num_blocks,
                exc_info=True,
            )
            return 0

        if trimmed:
            # Free the physical blocks back to the pool.  block_pool.free()
            # decrements ref_count and puts blocks with ref_count==0 into
            # the free list.
            self._kv_manager.block_pool.free(trimmed)
            logger.debug(
                "Sliding window: freed %d physical blocks for request %s "
                "(%d blocks remaining, %d free in pool)",
                len(trimmed), req_id, table.num_blocks,
                self._kv_manager.num_free_blocks,
            )

        return len(trimmed)

    def _finalize_request_blocks(self, req_id: str) -> None:
        """Cache completed blocks and free the block table for a finished request.

        Extracted from _manage_kv_cache and _cleanup_finished to avoid
        duplication.  Safe to call multiple times — the second call is a
        no-op because _block_tables.pop returns None and the
        _finalized_requests guard catches the race between
        _manage_kv_cache and _cleanup_finished.
        """
        if req_id in self._finalized_requests:
            return
        table = self._block_tables.pop(req_id, None)
        if table is None:
            return
        self._finalized_requests.add(req_id)
        try:
            req = self.requests.get(req_id)
            if req is not None:
                prompt_ids = req.prompt_token_ids or []
                all_tokens = prompt_ids
                cache_tokens = prompt_ids
                if cache_tokens:
                    self._kv_manager.cache_completed_blocks(table, cache_tokens)
                    blocks = table.get_blocks()
                    # Only include blocks within the prompt range — decode-phase
                    # blocks must NOT be stored in the radix tree because the
                    # token-to-block mapping would be wrong.
                    num_prompt_blocks = len(prompt_ids) // self._kv_manager.block_size
                    cached_blocks = []
                    cached_hashes = []
                    for i, b in enumerate(blocks):
                        if i >= num_prompt_blocks:
                            break
                        if b.block_hash is not None:
                            cached_blocks.append(b)
                            cached_hashes.append(b.block_hash)
                    if cached_hashes:
                        self._kv_manager.cache_to_radix_tree(
                            all_tokens, cached_blocks, cached_hashes,
                        )
        finally:
            self._kv_manager.free_request(table, request_id=req_id)

    def _preempt_request(self, request) -> None:
        """Preempt a running request, releasing decode KV blocks but preserving prefix.

        Unlike finalization (which frees all blocks), preemption keeps prefix
        blocks alive in the radix tree so the request can resume from its cached
        prefix when re-scheduled — avoiding a full re-prefill.

        Order:
        1. Cache prefix blocks to radix tree (increases ref_count)
        2. Free the full block table (decrements all ref_counts, but radix-held
           prefix blocks remain allocated because radix holds an extra ref)
        3. Delegate to base class for KV extraction and waiting queue push
        """
        req_id = request.request_id
        # Save prefix blocks to radix tree FIRST — this increments ref_counts
        # on cached blocks so they survive the subsequent free_request().
        try:
            table = self._block_tables.get(req_id)
            if table is not None and self._kv_manager is not None:
                req = self.requests.get(req_id)
                if req is not None:
                    prompt_ids = req.prompt_token_ids or []
                    if prompt_ids:
                        self._kv_manager.cache_completed_blocks(table, prompt_ids)
                        blocks = table.get_blocks()
                        num_prompt_blocks = len(prompt_ids) // self._kv_manager.block_size
                        cached_blocks = []
                        cached_hashes = []
                        for i, b in enumerate(blocks):
                            if i >= num_prompt_blocks:
                                break
                            if b.block_hash is not None:
                                cached_blocks.append(b)
                                cached_hashes.append(b.block_hash)
                        if cached_hashes:
                            self._kv_manager.cache_to_radix_tree(
                                prompt_ids, cached_blocks, cached_hashes,
                            )
        except Exception:
            logger.error(
                "Failed to cache prefix blocks during preemption of %s",
                req_id, exc_info=True,
            )

        # Free the block table — radix-held prefix blocks keep their extra ref
        # and remain allocated. Decode-only blocks are fully released.
        if req_id not in self._finalized_requests:
            table = self._block_tables.pop(req_id, None)
            if table is not None:
                self._finalized_requests.add(req_id)
                if self._kv_manager is not None:
                    self._kv_manager.free_request(table, request_id=req_id)
        # Remove from finalized set so re-scheduled request's blocks can
        # be cleaned up on its eventual second completion.
        self._finalized_requests.discard(req_id)
        # Delegate to base class for the actual preemption logic
        # (extract from BatchGenerator, save prefix cache, move to waiting queue)
        super()._preempt_request(request)

    def _cleanup_finished(self) -> None:
        if self._kv_manager is not None:
            for req_id in list(self.running.keys()):
                req = self.running[req_id]
                if RequestStatus.is_finished(req.status):
                    self._finalize_request_blocks(req_id)
        super()._cleanup_finished()
        # Clean up finalized tracking set entries that have been fully
        # removed by the base class _cleanup_finished (no longer in
        # self.running or self.waiting).
        self._finalized_requests -= {
            rid for rid in self._finalized_requests
            if rid not in self.running and rid not in self.waiting
        }

    def _process_aborts(self) -> None:
        """Override to also free block tables for aborted requests."""
        for req_id in list(self._pending_abort_ids):
            if req_id in self._block_tables:
                self._finalize_request_blocks(req_id)
                self._finalized_requests.discard(req_id)
        super()._process_aborts()

    def get_stats(self) -> dict:
        stats = super().get_stats()
        if self._kv_manager is not None:
            stats["kv_cache"] = {
                "usage": round(self._kv_manager.usage, 3),
                "free_blocks": self._kv_manager.num_free_blocks,
                "block_size": self._kv_manager.block_size,
                "active_block_tables": len(self._block_tables),
            }
        if self._compactor is not None:
            stats["kv_compactor"] = self._compactor.get_stats()
        if self._eviction_predictor is not None:
            stats["kv_eviction_predictor"] = self._eviction_predictor.get_stats()
        return stats
