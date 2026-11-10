# Patches upstream mlx-vlm Qwen3.5-family q/k normalization (MIT); see vendor.json.
"""Qwen3.5 q/k normalization constants shared by stock, prefill and verify.

The reviewed mlx-vlm upgrade changes epsilon from 1e-6 to 1e-6/head_dim
(b8bf04c). Yunshu keeps the pre-upgrade arithmetic as its lossless default:
changing epsilon is a model-output change, not a layout optimization. The
compatibility hook runs before fused kernels compile, and their constants are
still measured from the actual installed function. Standalone measurement also
recognizes upstream's per-head form so unsupported changes fail qualification.
"""

from __future__ import annotations

import logging
from functools import cache

import mlx.core as mx

logger = logging.getLogger(__name__)

BASE_EPS = 1e-6  # lossless pre-upgrade arithmetic
HEAD_DIM = 128  # Qwen3.5-family GDN key head width the fused kernels are written for
MARKER = "__QK_RMS_EPS__"


FIXED = "fixed"
PER_HEAD_DIM = "per_head_dim"


def install_lossless_default() -> bool:
    """Retain v0.7.8 q/k arithmetic across the reviewed dependency upgrade.

    b8bf04c changes the normalization itself, not just its implementation.
    Taking that new epsilon silently changes the default model's outputs and
    exposed APC/spec invariance failures with MLX 0.32.4 on M3. Keep the old
    default in stock, prefill and verify alike; fused kernels measure this
    installed function before compiling their matching constants.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        reviewed = version("mlx-vlm") in ("0.7.10", "0.7.11", "0.7.12")
    except PackageNotFoundError:
        return False
    if not reviewed:
        return False
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

    cls = Qwen3_5GatedDeltaNet
    if getattr(cls, "_yunshu_lossless_qk_default", False):
        return False
    if getattr(cls._normalize_qk, "_yunshu_norm_scale", False):
        raise RuntimeError("install lossless q/k arithmetic before fused normalization")

    def normalize(self, q, k):
        inv = k.shape[-1] ** -0.5
        return (inv**2) * mx.fast.rms_norm(q, None, BASE_EPS), inv * mx.fast.rms_norm(
            k, None, BASE_EPS
        )

    cls._normalize_qk = normalize
    cls._yunshu_lossless_qk_default = True
    qk_norm_kind.cache_clear()
    return True


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
    """The active q/k RMS epsilon form (None: unrecognised)."""
    try:
        install_lossless_default()
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
