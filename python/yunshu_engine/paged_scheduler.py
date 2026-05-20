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

from .scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from .request import Request, RequestOutput, RequestStatus
from .kv_optimizations import KVBlockCompactor, KVEvictionPredictor

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
        if self._compactor is not None:
            step_counter = getattr(self, '_step_counter', 0) + 1
            self._step_counter = step_counter
            self._compactor.maybe_compact(step_counter)
        return output

    def _manage_kv_cache(self, outputs: list[RequestOutput]) -> None:
        for req_output in outputs:
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
                            except ValueError:
                                logger.warning(f"KV cache exhausted for request {req_id}")
                                if req.batch_uid is not None and req.batch_uid not in self._uids_to_remove:
                                    self._uids_to_remove.append(req.batch_uid)
                                req.set_finished(RequestStatus.FINISHED_ERROR, reason="kv_cache_oom")
                                # Do NOT pop from self.running here — let _cleanup_finished
                                # produce the finished output and clean up naturally.
                                self._uid_to_req.pop(getattr(req, 'batch_uid', None), None)
                                self._finalize_request_blocks(req_id)

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
        req = self.requests.get(req_id)
        if req is not None:
            prompt_ids = req.prompt_token_ids or []
            output_ids = list(req.output_token_ids) if hasattr(req, 'output_token_ids') else []
            # Only cache prompt tokens to the radix tree — output tokens are
            # specific to a single generation run and should not be served as
            # prefix matches for future requests.
            all_tokens = prompt_ids
            cache_tokens = prompt_ids + output_ids
            if cache_tokens:
                self._kv_manager.cache_completed_blocks(table, cache_tokens)
                # Insert prompt-only prefix into RadixTree for O(k) matching.
                # Only pass blocks that have a hash -- uncached blocks would
                # misalign with the hashes list since they are filtered.
                blocks = table.get_blocks()
                cached_blocks = []
                cached_hashes = []
                for b in blocks:
                    if b.block_hash is not None:
                        cached_blocks.append(b)
                        cached_hashes.append(b.block_hash)
                if cached_hashes:
                    self._kv_manager.cache_to_radix_tree(
                        all_tokens, cached_blocks, cached_hashes,
                    )
        self._kv_manager.free_request(table, request_id=req_id)

    def _preempt_request(self, request) -> None:
        """Preempt a running request, releasing its KV blocks back to the pool.

        Without this override, the base Scheduler._preempt_request would move
        the request back to the waiting queue but leave its paged KV blocks
        allocated — a resource leak under memory pressure (which is the very
        reason preemption is triggered).
        """
        # Release KV blocks BEFORE base class removes the request from
        # the batch generator (which invalidates the cache reference).
        self._finalize_request_blocks(request.request_id)
        # Remove from finalized set so re-scheduled request's blocks can
        # be cleaned up on its eventual second completion.
        self._finalized_requests.discard(request.request_id)
        # Delegate to base class for the actual preemption logic
        # (save prefix cache, move to waiting queue, etc.)
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
            if rid not in self.requests
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
