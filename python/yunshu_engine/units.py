"""Memory units for API output.

Every ``*_gb`` field the engine returns is binary: GB = 1024**3 bytes, the same unit
macOS, ``yunshu doctor`` and the ``YUNSHU_*_GB`` settings use (a 128 GB Mac reads 128.0).
Next to each ``*_gb`` value the exact integer ``*_bytes`` is returned; scripts that need
precision should read the bytes. Prometheus metrics stay in bytes.
"""

from __future__ import annotations

GIB = 1 << 30


def gb(n_bytes: float | int | None, digits: int = 2) -> float | None:
    """Bytes -> binary GB (rounded); None stays None."""
    if n_bytes is None:
        return None
    return round(n_bytes / GIB, digits)


def put_gb(out: dict, key: str, n_bytes: float | int, digits: int = 2) -> None:
    """Set ``out[key + '_gb']`` (binary GB) and ``out[key + '_bytes']`` (exact int)."""
    out[f"{key}_gb"] = round(n_bytes / GIB, digits)
    out[f"{key}_bytes"] = int(n_bytes)
