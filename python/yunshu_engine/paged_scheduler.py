"""Yunshu PagedScheduler — Scheduler with KVCacheManager integration.

Extends the base Scheduler with:
- Paged KV cache via KVCacheManager (BlockPool + BlockTable)
- Prefix caching: reuse cached blocks across requests
- Memory-aware scheduling: reject requests that exceed KV budget
- Block-wise cache management integrated with BatchGenerator lifecycle
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from .request import Request, RequestOutput, RequestStatus

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

    def set_kv_cache_manager(self, manager: Any) -> None:
        self._kv_manager = manager

    def get_block_table(self, request_id: str) -> Any | None:
        return self._block_tables.get(request_id)

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
        total_needed = needed_blocks + decode_reserve

        if self._kv_manager.num_free_blocks < total_needed:
            if not self._kv_manager.evict_for_memory(total_needed):
                logger.warning(
                    f"Rejecting request {request.request_id}: "
                    f"need {total_needed} blocks, only {self._kv_manager.num_free_blocks} free"
                )
                request.status = RequestStatus.FINISHED_ERROR
                request.finish_reason = "kv_cache_full"
                return

        table, prefix_match = self._kv_manager.allocate_for_prefill(token_ids)
        self._block_tables[request.request_id] = table
        request.cached_tokens = getattr(prefix_match, 'num_matched_tokens', 0)

        super().add_request(request)

    def step(self) -> SchedulerOutput:
        output = super().step()
        if self._kv_manager is not None and output.outputs:
            self._manage_kv_cache(output.outputs)
        return output

    def _manage_kv_cache(self, outputs: list[RequestOutput]) -> None:
        for req_output in outputs:
            req_id = req_output.request_id

            if req_output.finished:
                table = self._block_tables.pop(req_id, None)
                if table is not None:
                    req = self.requests.get(req_id)
                    if req is not None:
                        prompt_ids = req.prompt_token_ids or []
                        output_ids = list(req.output_token_ids) if hasattr(req, 'output_token_ids') else []
                        all_tokens = prompt_ids + output_ids
                        if all_tokens:
                            self._kv_manager.cache_completed_blocks(table, all_tokens)
                            # Insert completed blocks into RadixTree for O(k) prefix matching
                            blocks = table.get_blocks()
                            hashes = [b.block_hash for b in blocks if b.block_hash is not None]
                            if hashes:
                                self._kv_manager.cache_to_radix_tree(all_tokens, blocks, hashes)
                    self._kv_manager.free_request(table)
            else:
                req = self.running.get(req_id)
                if req is not None:
                    table = self._block_tables.get(req_id)
                    if table is not None and self._kv_manager is not None:
                        total_tokens = len(req.prompt_token_ids) + req.num_output_tokens
                        if total_tokens % self._kv_manager.block_size == 0:
                            try:
                                self._kv_manager.allocate_block_for_decode(table)
                            except ValueError:
                                logger.warning(f"KV cache exhausted for request {req_id}")
                                self.abort_request(req_id)

    def _cleanup_finished(self) -> None:
        if self._kv_manager is not None:
            for req_id in list(self.running.keys()):
                req = self.running[req_id]
                if RequestStatus.is_finished(req.status):
                    table = self._block_tables.pop(req_id, None)
                    if table is not None:
                        self._kv_manager.free_request(table)
        super()._cleanup_finished()

    def get_stats(self) -> dict:
        stats = super().get_stats()
        if self._kv_manager is not None:
            stats["kv_cache"] = {
                "usage": round(self._kv_manager.usage, 3),
                "free_blocks": self._kv_manager.num_free_blocks,
                "block_size": self._kv_manager.block_size,
                "active_block_tables": len(self._block_tables),
            }
        return stats
