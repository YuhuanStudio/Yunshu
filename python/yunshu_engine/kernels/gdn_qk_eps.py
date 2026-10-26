"""The epsilon of the Qwen3.5-family GDN q/k RMS normalisation, read from the installed mlx-vlm.

``Qwen3_5GatedDeltaNet._normalize_qk`` is ``mx.fast.rms_norm(x, None, eps)``. mlx-vlm 0.7.8 (the
pinned version) uses ``eps = 1e-6``; mlx-vlm b8bf04c (v0.7.10) uses ``1e-6 / head_dim`` so it equals
mlx-lm's L2 form ``x * rsqrt(sum(x^2) + 1e-6)``. Yunshu's fused Metal kernels
(``gdn_prefill._NORM_SCALE_SOURCE``, the RMS arm of ``omlx.qwen35_gdn_prework``) must carry the
same epsilon as the stock function they replace, or fused prefill != stock decode/verify bits.

The value is therefore never written in a kernel: it is measured once from the installed function
(run on a tiny-magnitude input, where eps decides the result, and matched to the exact candidate),
so a bump of mlx-vlm flips the stock and fused paths together. The L2 arms of the prework kernels
use mlx-lm's ``sum + 1e-6`` and do not depend on this.
"""

from __future__ import annotations

import logging
from functools import cache

import mlx.core as mx

logger = logging.getLogger(__name__)

BASE_EPS = 1e-6  # mlx-vlm <= 0.7.8
HEAD_DIM = 128  # Qwen3.5-family GDN key head width the fused kernels are written for
MARKER = "__QK_RMS_EPS__"


FIXED = "fixed"
PER_HEAD_DIM = "per_head_dim"


def candidates(head_dim: int = HEAD_DIM) -> dict[str, float]:
    return {FIXED: BASE_EPS, PER_HEAD_DIM: BASE_EPS / head_dim}


def _matches(fn, eps: float, head_dim: int) -> bool:
    # A tiny-magnitude input: mean(x^2) is of the order of eps, so a wrong eps changes every output.
    x = mx.full((1, 2, 2, head_dim), 1e-3, dtype=mx.float32) * mx.arange(1, 5).reshape(
        1, 2, 2, 1
    )
    with mx.stream(mx.cpu):
        q, k = fn(x, x)
        inv = head_dim**-0.5
        want_q = (inv**2) * mx.fast.rms_norm(x, None, eps)
        want_k = inv * mx.fast.rms_norm(x, None, eps)
        return bool(
            mx.array_equal(q, want_q).item() and mx.array_equal(k, want_k).item()
        )


def measure(normalize_qk=None, head_dim: int = HEAD_DIM) -> str | None:
    """Which epsilon ``normalize_qk(q, k)`` uses (FIXED or PER_HEAD_DIM), or None if neither matches."""
    if normalize_qk is None:
        from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

        cls = Qwen3_5GatedDeltaNet
        normalize_qk = lambda q, k: cls._normalize_qk(None, q, k)  # noqa: E731
    found = [
        kind
        for kind, eps in candidates(head_dim).items()
        if _matches(normalize_qk, eps, head_dim)
    ]
    return found[0] if len(found) == 1 else None


@cache
def qk_norm_kind() -> str | None:
    """The installed upstream's q/k RMS epsilon form (None: unrecognised)."""
    try:
        return measure()
    except Exception:
        logger.warning(
            "GDN q/k norm epsilon could not be measured from mlx-vlm", exc_info=True
        )
        return None


def qk_norm_eps(head_dim: int = HEAD_DIM) -> float | None:
    kind = qk_norm_kind()
    return None if kind is None else candidates(head_dim)[kind]


def expression(head_dim: str | int = HEAD_DIM, kind: str | None = None) -> str:
    """The epsilon as a Metal float expression. ``head_dim`` is an int (folded on the host exactly
    as upstream's Python does) or a Metal expression such as ``float(DK)`` (an exact float division
    for the power-of-two widths Qwen uses). Unrecognised upstream keeps the pinned fixed value."""
    kind = kind or qk_norm_kind()
    if kind is None:
        logger.warning(
            "GDN q/k norm epsilon not recognised in mlx-vlm; kernels keep %g", BASE_EPS
        )
        kind = FIXED
    if kind == FIXED:
        return f"{BASE_EPS!r}f"
    if isinstance(head_dim, int):
        return f"{BASE_EPS / head_dim!r}f"
    return f"({BASE_EPS!r}f / {head_dim})"


def render(source: str, head_dim: str | int = HEAD_DIM) -> str:
    """``source`` with the epsilon marker replaced by the installed upstream's expression."""
    return source.replace(MARKER, expression(head_dim))
