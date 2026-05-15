from __future__ import annotations
"""Thinking-Segment KV Substore — reasoning token KV cache reuse.

Stores KV cache segments keyed by (conversation_id, step_hash) so that
reasoning models (DeepSeek-R1, Qwen3) can reuse thinking KV cache across
multi-turn conversations with shared reasoning steps.

Architecture (Section 3.6 / Delta-6 from whitepaper):
- Key: (conversation_id, step_hash) — identifies a reasoning step uniquely
- Value: KV cache tensors for the thinking segment
- Store: in-memory hot + optional SSD persistence
- Lookup: by conversation_id prefix, then step_hash match
- Eviction: LRU by access time, per-conversation limit
- Compression: optional KV quantization for stored segments
- Persistence: SSD save/load for warm-tier segments

Integration:
- Scheduler detects <think/> tag boundaries
- After </think/>, extracts the thinking KV segment
- Stores it with (conversation_id, hash(thinking_tokens))
- On next turn in same conversation, checks for reusable segments
- If found, injects cached KV and skips re-thinking identical steps
"""

import gc
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ThinkingSegment:
    """A cached thinking segment with its KV data."""
    conversation_id: str
    step_hash: str
    kv_data: Any  # Serialized KV tensors
    num_tokens: int
    thinking_text_hash: str
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)
    access_count: int = 0


@dataclass
class ThinkingSegmentConfig:
    """Configuration for the thinking segment substore."""
    # Maximum segments per conversation
    max_segments_per_conversation: int = 10
    # Maximum total segments across all conversations
    max_total_segments: int = 1000
    # Minimum thinking tokens to cache (skip trivial thinking)
    min_tokens_to_cache: int = 32
    # TTL in seconds for cached segments
    ttl_seconds: float = 3600.0  # 1 hour
    # Enable SSD persistence for thinking segments
    enable_ssd: bool = False
    ssd_cache_dir: str = ""
    # Enable KV compression for stored segments
    enable_compression: bool = False
    # Compression bits per element (4 = 4-bit quantization, 8 = no compression)
    compression_bits: int = 4
    # Compression group size
    compression_group_size: int = 64
    # TTL cleanup interval in seconds (0 = cleanup on every store call)
    ttl_cleanup_interval: float = 60.0


class ThinkingSegmentSubstore:
    """Manages KV cache segments for reasoning model thinking steps.

    Provides:
    - Store: save thinking KV after </think/> with step hash
    - Lookup: find cached thinking by (conversation_id, step_hash)
    - Reuse: inject cached KV to skip re-thinking
    - Eviction: LRU + TTL + per-conversation limits
    - Compression: optional KV quantization for stored segments
    - SSD persistence: save/load thinking segments to disk
    - TTL cleanup: periodic expiry with memory reclamation
    """

    def __init__(self, config: ThinkingSegmentConfig | None = None) -> None:
        self.config = config or ThinkingSegmentConfig()
        # conversation_id → list of ThinkingSegment
        self._segments: dict[str, list[ThinkingSegment]] = {}
        # step_hash → ThinkingSegment (global index for cross-conversation lookup)
        self._hash_index: dict[str, ThinkingSegment] = {}
        self._total_segments = 0
        self._stats = {
            "stored": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "tokens_saved": 0,
            "compressions": 0,
            "ssd_saves": 0,
            "ssd_loads": 0,
            "ttl_cleanups": 0,
        }
        # KV quantizer for compression (lazy init)
        self._quantizer = None
        # SSD cache directory
        if self.config.enable_ssd and self.config.ssd_cache_dir:
            self._ssd_dir = Path(self.config.ssd_cache_dir)
        elif self.config.enable_ssd:
            self._ssd_dir = Path.home() / ".yunshu" / "thinking_segments"
        else:
            self._ssd_dir = None
        # TTL cleanup tracking
        self._last_ttl_cleanup: float = time.monotonic()

    def compute_step_hash(self, thinking_tokens: list[int], context_tokens: list[int]) -> str:
        """Compute a hash for a thinking step.

        Includes both the thinking tokens and the context that preceded them,
        so identical thinking in different contexts gets different hashes.
        """
        h = hashlib.sha256()
        # Hash context prefix (last 128 tokens of context)
        ctx = context_tokens[-128:] if len(context_tokens) > 128 else context_tokens
        for t in ctx:
            h.update(t.to_bytes(4, "little"))
        # Hash thinking content
        for t in thinking_tokens:
            h.update(t.to_bytes(4, "little"))
        return h.hexdigest()[:16]

    def store(
        self,
        conversation_id: str,
        thinking_tokens: list[int],
        context_tokens: list[int],
        kv_data: Any,
    ) -> str | None:
        """Store a thinking segment after reasoning completes.

        Applies KV compression if enabled, then stores the segment.
        If SSD persistence is enabled, also saves to disk.

        Args:
            conversation_id: The conversation identifier.
            thinking_tokens: Token IDs of the thinking content.
            context_tokens: Token IDs of the context before thinking.
            kv_data: The KV cache data for this thinking segment.

        Returns:
            The step_hash if stored, None if skipped (too short, etc.)
        """
        if len(thinking_tokens) < self.config.min_tokens_to_cache:
            return None

        step_hash = self.compute_step_hash(thinking_tokens, context_tokens)

        # Check if already stored
        if step_hash in self._hash_index:
            existing = self._hash_index[step_hash]
            existing.last_accessed = time.monotonic()
            existing.access_count += 1
            return step_hash

        # Evict if at capacity
        self._maybe_evict(conversation_id)

        # Compress KV data if enabled
        compressed_kv = kv_data
        if self.config.enable_compression and kv_data is not None:
            compressed_kv = self._compress_kv(kv_data)

        # Hash of thinking text for deduplication
        text_hash = hashlib.sha256()
        for t in thinking_tokens:
            text_hash.update(t.to_bytes(4, "little"))

        segment = ThinkingSegment(
            conversation_id=conversation_id,
            step_hash=step_hash,
            kv_data=compressed_kv,
            num_tokens=len(thinking_tokens),
            thinking_text_hash=text_hash.hexdigest()[:16],
        )

        self._segments.setdefault(conversation_id, []).append(segment)
        self._hash_index[step_hash] = segment
        self._total_segments += 1
        self._stats["stored"] += 1

        # SSD persistence
        if self.config.enable_ssd and self._ssd_dir is not None:
            self._save_to_ssd(segment)

        # Periodic TTL cleanup
        self._maybe_ttl_cleanup()

        logger.debug(
            f"Stored thinking segment: conv={conversation_id[:8]}... "
            f"hash={step_hash}, tokens={len(thinking_tokens)}"
        )
        return step_hash

    def lookup(
        self,
        conversation_id: str,
        step_hash: str,
    ) -> ThinkingSegment | None:
        """Look up a cached thinking segment.

        Checks in-memory first, then SSD if enabled. Decompresses KV data
        if compression was applied during store.

        Args:
            conversation_id: The conversation identifier.
            step_hash: The hash of the thinking step to look up.

        Returns:
            The ThinkingSegment if found and not expired, None otherwise.
        """
        segment = self._hash_index.get(step_hash)
        if segment is None:
            # Try SSD if enabled
            if self.config.enable_ssd and self._ssd_dir is not None:
                segment = self._load_from_ssd(conversation_id, step_hash)
                if segment is not None:
                    self._segments.setdefault(conversation_id, []).append(segment)
                    self._hash_index[step_hash] = segment
                    self._total_segments += 1
                else:
                    self._stats["misses"] += 1
                    return None
            else:
                self._stats["misses"] += 1
                return None

        # Check TTL
        age = time.monotonic() - segment.created_at
        if age > self.config.ttl_seconds:
            self._remove_segment(segment)
            self._stats["misses"] += 1
            return None

        # Decompress KV data if needed (lazy decompression)
        if self.config.enable_compression and segment.kv_data is not None:
            if isinstance(segment.kv_data, dict) and segment.kv_data.get("_compressed"):
                segment.kv_data = self._decompress_kv(segment.kv_data)

        segment.last_accessed = time.monotonic()
        segment.access_count += 1
        self._stats["hits"] += 1
        self._stats["tokens_saved"] += segment.num_tokens

        return segment

    def lookup_by_context(
        self,
        conversation_id: str,
        context_tokens: list[int],
        thinking_prefix: list[int],
    ) -> ThinkingSegment | None:
        """Find a cached segment by context and thinking prefix.

        Useful for finding similar reasoning steps when exact hash
        isn't known. Matches on conversation_id prefix match and
        thinking prefix similarity.
        """
        conv_segments = self._segments.get(conversation_id, [])
        if not conv_segments:
            return None

        # Try exact prefix hash first
        prefix_hash = self.compute_step_hash(thinking_prefix, context_tokens)
        for seg in conv_segments:
            if seg.step_hash == prefix_hash:
                seg.last_accessed = time.monotonic()
                seg.access_count += 1
                self._stats["hits"] += 1
                return seg

        return None

    def get_conversation_segments(self, conversation_id: str) -> list[ThinkingSegment]:
        """Get all cached segments for a conversation."""
        return list(self._segments.get(conversation_id, []))

    def clear_conversation(self, conversation_id: str) -> int:
        """Remove all segments for a conversation. Returns count removed."""
        segments = self._segments.pop(conversation_id, [])
        count = 0
        for seg in segments:
            self._hash_index.pop(seg.step_hash, None)
            # Reclaim memory from KV data
            seg.kv_data = None
            # Remove SSD file if persisted
            if self.config.enable_ssd and self._ssd_dir is not None:
                self._remove_ssd_file(seg)
            count += 1
        self._total_segments -= count
        return count

    def _maybe_evict(self, conversation_id: str) -> None:
        """Evict segments if at capacity."""
        # Per-conversation limit
        conv_segs = self._segments.get(conversation_id, [])
        while len(conv_segs) >= self.config.max_segments_per_conversation:
            oldest = min(conv_segs, key=lambda s: s.last_accessed)
            self._remove_segment(oldest)
            conv_segs = self._segments.get(conversation_id, [])
            self._stats["evictions"] += 1

        # Global limit
        while self._total_segments >= self.config.max_total_segments:
            # Find globally oldest segment
            all_segs = [s for segs in self._segments.values() for s in segs]
            if not all_segs:
                break
            oldest = min(all_segs, key=lambda s: s.last_accessed)
            self._remove_segment(oldest)
            self._stats["evictions"] += 1

        # TTL cleanup
        now = time.monotonic()
        expired = [
            s for segs in self._segments.values() for s in segs
            if now - s.created_at > self.config.ttl_seconds
        ]
        for seg in expired:
            self._remove_segment(seg)

    def _remove_segment(self, segment: ThinkingSegment) -> None:
        """Remove a single segment from all indices and reclaim memory."""
        self._hash_index.pop(segment.step_hash, None)
        conv_segs = self._segments.get(segment.conversation_id, [])
        if segment in conv_segs:
            conv_segs.remove(segment)
        else:
            # Segment was already removed (e.g., by per-conversation and
            # global eviction targeting the same entry). Avoid undercounting.
            return
        self._total_segments -= 1
        # Reclaim memory from KV data
        if segment.kv_data is not None:
            segment.kv_data = None
        # Remove SSD file if persisted
        if self.config.enable_ssd and self._ssd_dir is not None:
            self._remove_ssd_file(segment)

    # ── KV Compression ──

    def _get_quantizer(self):
        """Lazy-initialize the KV quantizer."""
        if self._quantizer is not None:
            return self._quantizer
        try:
            from yunshu_engine.kv_quantization import KVQuantizer, KVQuantConfig
            config = KVQuantConfig(
                bits=self.config.compression_bits,
                group_size=self.config.compression_group_size,
            )
            self._quantizer = KVQuantizer(config)
        except ImportError:
            logger.warning("kv_quantization module not available — compression disabled")
            self._quantizer = None
        return self._quantizer

    def _compress_kv(self, kv_data: Any) -> Any:
        """Compress KV data using group-wise quantization.

        Returns a dict with compressed data and metadata, or the original
        data if compression is not available.
        """
        quantizer = self._get_quantizer()
        if quantizer is None:
            return kv_data

        try:
            # Convert KV data to list format for quantization
            if isinstance(kv_data, list):
                flat_data = kv_data
            elif hasattr(kv_data, 'tolist'):
                flat_data = kv_data.tolist()
            else:
                flat_data = list(kv_data) if hasattr(kv_data, '__iter__') else [kv_data]

            packed, meta = quantizer.quantize(flat_data)
            self._stats["compressions"] += 1
            return {
                "_compressed": True,
                "packed": packed,
                "metadata": meta,
            }
        except Exception as e:
            logger.debug(f"KV compression failed, storing uncompressed: {e}")
            return kv_data

    def _decompress_kv(self, compressed: dict) -> Any:
        """Decompress KV data from quantized format."""
        quantizer = self._get_quantizer()
        if quantizer is None:
            return compressed

        try:
            packed = compressed.get("packed", b"")
            meta = compressed.get("metadata", {})
            return quantizer.dequantize(packed, meta)
        except Exception as e:
            logger.debug(f"KV decompression failed: {e}")
            return compressed

    # ── SSD Persistence ──

    def _save_to_ssd(self, segment: ThinkingSegment) -> None:
        """Save a thinking segment to SSD.

        Serializes the segment data as JSON (with binary kv_data as hex)
        to the SSD cache directory.
        """
        if self._ssd_dir is None:
            return

        try:
            self._ssd_dir.mkdir(parents=True, exist_ok=True)
            conv_dir = self._ssd_dir / segment.conversation_id
            conv_dir.mkdir(parents=True, exist_ok=True)

            filepath = conv_dir / f"{segment.step_hash}.json"

            # Serialize kv_data
            kv_serialized = self._serialize_kv(segment.kv_data)

            data = {
                "conversation_id": segment.conversation_id,
                "step_hash": segment.step_hash,
                "num_tokens": segment.num_tokens,
                "thinking_text_hash": segment.thinking_text_hash,
                "created_at": segment.created_at,
                "kv_data": kv_serialized,
            }

            with open(filepath, "w") as f:
                json.dump(data, f)

            self._stats["ssd_saves"] += 1
            logger.debug(f"Saved thinking segment to SSD: {filepath}")
        except Exception as e:
            logger.debug(f"SSD save failed for segment {segment.step_hash}: {e}")

    def _load_from_ssd(self, conversation_id: str, step_hash: str) -> ThinkingSegment | None:
        """Load a thinking segment from SSD.

        Returns the segment if found, None otherwise.
        """
        if self._ssd_dir is None:
            return None

        try:
            filepath = self._ssd_dir / conversation_id / f"{step_hash}.json"
            if not filepath.exists():
                return None

            with open(filepath, "r") as f:
                data = json.load(f)

            kv_data = self._deserialize_kv(data.get("kv_data"))

            segment = ThinkingSegment(
                conversation_id=data["conversation_id"],
                step_hash=data["step_hash"],
                kv_data=kv_data,
                num_tokens=data["num_tokens"],
                thinking_text_hash=data["thinking_text_hash"],
                created_at=data.get("created_at", time.monotonic()),
            )

            self._stats["ssd_loads"] += 1
            logger.debug(f"Loaded thinking segment from SSD: {filepath}")
            return segment
        except Exception as e:
            logger.debug(f"SSD load failed for {conversation_id}/{step_hash}: {e}")
            return None

    def _remove_ssd_file(self, segment: ThinkingSegment) -> None:
        """Remove a segment's SSD file."""
        if self._ssd_dir is None:
            return
        try:
            filepath = self._ssd_dir / segment.conversation_id / f"{segment.step_hash}.json"
            if filepath.exists():
                filepath.unlink()
        except Exception:
            logger.debug("SSD file removal failed for segment %s", segment.step_hash, exc_info=True)

    def _serialize_kv(self, kv_data: Any) -> Any:
        """Serialize KV data for JSON storage."""
        if kv_data is None:
            return None
        if isinstance(kv_data, dict):
            # Compressed format
            if kv_data.get("_compressed"):
                packed = kv_data.get("packed", b"")
                return {
                    "_compressed": True,
                    "packed_hex": packed.hex() if isinstance(packed, bytes) else str(packed),
                    "metadata": kv_data.get("metadata", {}),
                }
            return kv_data
        if isinstance(kv_data, bytes):
            return {"_type": "bytes", "hex": kv_data.hex()}
        if isinstance(kv_data, list):
            return kv_data
        if hasattr(kv_data, 'tolist'):
            return kv_data.tolist()
        return str(kv_data)

    def _deserialize_kv(self, serialized: Any) -> Any:
        """Deserialize KV data from JSON storage."""
        if serialized is None:
            return None
        if isinstance(serialized, dict):
            if serialized.get("_compressed"):
                packed_hex = serialized.get("packed_hex", "")
                return {
                    "_compressed": True,
                    "packed": bytes.fromhex(packed_hex) if packed_hex else b"",
                    "metadata": serialized.get("metadata", {}),
                }
            if serialized.get("_type") == "bytes":
                return bytes.fromhex(serialized.get("hex", ""))
        return serialized

    # ── TTL Cleanup ──

    def _maybe_ttl_cleanup(self) -> None:
        """Periodic TTL cleanup with actual memory reclamation.

        Runs at most once per ttl_cleanup_interval seconds. Removes all
        expired segments and forces garbage collection to reclaim memory.
        """
        now = time.monotonic()
        interval = self.config.ttl_cleanup_interval
        if interval > 0 and (now - self._last_ttl_cleanup) < interval:
            return

        self._last_ttl_cleanup = now

        # Find all expired segments
        expired = []
        for conv_id, segments in self._segments.items():
            for seg in segments:
                age = now - seg.created_at
                if age > self.config.ttl_seconds:
                    expired.append(seg)

        if not expired:
            return

        # Remove expired segments
        for seg in expired:
            self._remove_segment(seg)
            self._stats["evictions"] += 1

        self._stats["ttl_cleanups"] += 1

        # Force garbage collection to reclaim memory from freed segments
        collected = gc.collect()

        logger.debug(
            f"TTL cleanup: removed {len(expired)} expired segments, "
            f"gc collected {collected} objects"
        )

    def get_stats(self) -> dict:
        return {
            "total_segments": self._total_segments,
            "conversations_tracked": len(self._segments),
            "stored": self._stats["stored"],
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "tokens_saved": self._stats["tokens_saved"],
            "compressions": self._stats["compressions"],
            "ssd_saves": self._stats["ssd_saves"],
            "ssd_loads": self._stats["ssd_loads"],
            "ttl_cleanups": self._stats["ttl_cleanups"],
            "hit_rate": (
                self._stats["hits"] / (self._stats["hits"] + self._stats["misses"])
                if (self._stats["hits"] + self._stats["misses"]) > 0
                else 0.0
            ),
        }
