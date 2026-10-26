"""Which load failures say nothing about the file they were reading.

A checkpoint file that failed to load because the process ran out of memory, a system call was
interrupted or a descriptor/disk limit was hit is still a good file: deleting it on that error
throws away a valid cache entry for a condition that passes. Corruption (a torn header, a missing
tensor, bad metadata) is the only reason to quarantine or delete.
"""

from __future__ import annotations

import errno

_TRANSIENT_ERRNOS = frozenset(
    {
        errno.ENOMEM,
        errno.EINTR,
        errno.EAGAIN,
        errno.EMFILE,
        errno.ENFILE,
        errno.ENOSPC,
        errno.EBUSY,
    }
)


def is_transient_load_error(e: BaseException) -> bool:
    """True for out-of-memory, interrupted / exhausted system calls and MLX allocation failures."""
    if isinstance(e, MemoryError):
        return True
    if isinstance(e, OSError):
        return e.errno in _TRANSIENT_ERRNOS
    if isinstance(e, RuntimeError):
        text = str(e).lower()
        return "allocat" in text or "out of memory" in text
    return False
