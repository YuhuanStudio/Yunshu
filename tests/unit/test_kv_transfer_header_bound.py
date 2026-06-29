"""a grounded disagg KV-transfer wire hunt found the framing core sound
(partial-read/readexactly, bf16 dtype round-trip, compression fallback, loop-binding
all correct). One MEDIUM DoS: read_frame bounds-checked the PAYLOAD length but not
header_len (a !I, up to 4 GiB) — a corrupt/malicious peer connecting to the 0.0.0.0-bound
KVTransferServer could send magic + 0xFFFFFFFF and force a multi-GB readexactly buffer →
memory-pressure SIGABRT on a 36GB Mac. Now header_len is bounded (_MAX_HEADER_SIZE) before
the read, mirroring the existing payload guard.
"""

from __future__ import annotations

import asyncio
import struct

import pytest

from yunshu_engine import (
    kv_transfer as KT,  # noqa: N812  # intentional short module alias
)
from yunshu_engine.kv_transfer import _MAGIC, _MAX_HEADER_SIZE, KVTransferProtocol


class _FakeReader:
    """Feeds a fixed byte string to readexactly; raises if asked for more than supplied
    (so an UNBOUNDED header_len read would surface as an over-read, not a real 4GB alloc)."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    async def readexactly(self, n: int):
        if self._pos + n > len(self._data):
            raise asyncio.IncompleteReadError(self._data[self._pos :], n)
        chunk = self._data[self._pos : self._pos + n]
        self._pos += n
        return chunk


def test_oversized_header_len_rejected_before_read():
    # magic + a header_len just over the cap. The bound must raise ValueError BEFORE the
    # readexactly(header_len) is attempted.
    frame = _MAGIC + struct.pack("!I", _MAX_HEADER_SIZE + 1)
    reader = _FakeReader(frame)
    with pytest.raises(ValueError, match="Header too large"):
        asyncio.run(KVTransferProtocol.read_frame(reader))


def test_max_header_size_is_sane():
    # generous for a small JSON header, but far below the 512MB message cap / GBs of RAM
    assert 64 * 1024 <= _MAX_HEADER_SIZE <= 16 * 1024 * 1024
    assert _MAX_HEADER_SIZE < KT._MAX_MESSAGE_SIZE
