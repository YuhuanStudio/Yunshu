from __future__ import annotations
"""Yunshu KV Cache Block Table — per-request logical→physical block mapping.

Each request maintains a BlockTable that maps logical token positions
to physical KVBlock objects in the BlockPool.
"""



from .block import KVBlock


class BlockTable:
    """Maps logical block indices to physical KVBlock objects for one request.

    The block table grows as the request generates more tokens:
    - During prefill: blocks are allocated to cover the full prompt.
    - During decode: a new block is allocated when the current block fills up.
    """

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self._blocks: list[KVBlock] = []  # logical index → physical block
        self.total_tokens: int = 0
        self._last_block_occupancy: int = 0  # tokens used in the last (partial) block

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    def append_block(self, block: KVBlock) -> None:
        """Add a new physical block at the end.

        The new block starts empty — total_tokens is unchanged (previous
        partial block's occupancy is already tracked via _last_block_occupancy).
        """
        self._blocks.append(block)
        self._last_block_occupancy = 0

    def append_blocks(self, blocks: list[KVBlock]) -> None:
        """Append multiple blocks. All but the last appended block are full."""
        if not blocks:
            return
        # All appended blocks except the very last one are full.
        # The last appended block starts empty (occupancy 0).
        full_new = len(blocks) - 1
        self.total_tokens += full_new * self.block_size
        self._blocks.extend(blocks)
        self._last_block_occupancy = 0

    def update_last_block_occupancy(self, occupancy: int) -> None:
        """Update the token occupancy of the last block (incremental)."""
        new_occ = min(occupancy, self.block_size)
        self.total_tokens += new_occ - self._last_block_occupancy
        self._last_block_occupancy = new_occ

    def get_block(self, logical_idx: int) -> KVBlock:
        if logical_idx < 0 or logical_idx >= len(self._blocks):
            raise IndexError(f"Block index {logical_idx} out of range (0..{len(self._blocks)-1})")
        return self._blocks[logical_idx]

    def get_blocks(self) -> list[KVBlock]:
        return list(self._blocks)

    def get_full_blocks(self) -> list[KVBlock]:
        """Return all fully-filled blocks (all but possibly the last)."""
        if len(self._blocks) <= 1:
            return []
        return self._blocks[:-1]

    def fork(self) -> BlockTable:
        """Create a copy of this block table for prefix sharing.

        The physical blocks are shared (not copied). The caller must
        increment ref counts via BlockPool.touch(). When a forked request
        needs to write into a shared block, the caller must use
        BlockPool.cow_block_in_table() to clone it first (COW semantics).
        """
        new_table = BlockTable(self.block_size)
        new_table._blocks = list(self._blocks)
        new_table.total_tokens = self.total_tokens
        new_table._last_block_occupancy = self._last_block_occupancy
        return new_table

    def clear(self) -> list[KVBlock]:
        """Clear and return all blocks for the caller to free."""
        blocks = self._blocks
        self._blocks = []
        self.total_tokens = 0
        self._last_block_occupancy = 0
        return blocks

    def block_id_for_token(self, token_position: int) -> int:
        """Get the physical block ID for a given token position."""
        logical_idx = token_position // self.block_size
        if logical_idx < 0 or logical_idx >= len(self._blocks):
            raise IndexError(
                f"Token position {token_position} maps to logical block "
                f"{logical_idx}, but table has {len(self._blocks)} blocks "
                f"(total_tokens={self.total_tokens})"
            )
        return self._blocks[logical_idx].block_id

    def slot_for_token(self, token_position: int) -> tuple[int, int]:
        """Get (block_id, offset_within_block) for a token position."""
        logical_idx = token_position // self.block_size
        if logical_idx < 0 or logical_idx >= len(self._blocks):
            raise IndexError(
                f"Token position {token_position} maps to logical block "
                f"{logical_idx}, but table has {len(self._blocks)} blocks "
                f"(total_tokens={self.total_tokens})"
            )
        offset = token_position % self.block_size
        return self._blocks[logical_idx].block_id, offset
