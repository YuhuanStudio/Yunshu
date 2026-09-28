# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-family MTP verify kernels vendored from oMLX.

Source: https://github.com/jundot/omlx (Apache-2.0), ``omlx/patches/``; per-file
source commits are in ``vendor.json`` (``just vendor-check`` reports upstream
changes). The modules are copied unchanged except for imports that pointed at
other oMLX packages (``_sync_and_clear_cache`` and ``is_nax_available`` below)
and the flattened ``mlx_vlm_mtp/qwen35_verify_linear.py`` import. Upstream credits in
the files (MTPLX, dflash-mlx, Splash — all Apache-2.0) are kept as written.

They monkey-patch mlx-vlm's Qwen3.5 target-verify forward (the multi-token
forward MTP runs to check drafted tokens):

- ``gdn_prework`` + ``gdn_replay``: fused GatedDeltaNet prework and a deferred
  replay commit instead of storing every per-step recurrent state. Bit-exact.
- ``sdpa_split``: chunked causal attention for verify rows instead of one SDPA
  call per row. Bit-exact.
- ``verify_qmm`` (+ ``verify_linear`` routing): one-weight-pass quantized
  matmuls for 4-8 verify rows. Not bit-exact (tail-ULP differences), so it is
  armed only when ``fast`` is requested.

Measured on Qwen3.8-27B (docs/research/runs/2026-09-28-matrix/): upstream MTP
~57 tok/s; exact kernels ~65; all kernels ~78-84 on code.
"""

from __future__ import annotations

import contextlib
import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_STATE = {"applied": None, "fast": False, "row_exact": False}


def _sync_and_clear_cache(stream=None):
    """Synchronize in-flight GPU work before clearing the Metal buffer cache."""
    with contextlib.suppress(RuntimeError):
        mx.synchronize()
    mx.clear_cache()


_NAX_CACHE: dict = {}


def is_nax_available() -> bool:
    """True on M5-class GPUs (tensor unit) with an mlx wheel that ships NAX kernels.

    Pure-Python mirror of oMLX ``qwen35_prefill.fast`` fallback detection:
    macOS >= 26.2, GPU architecture ``applegpu_g{N}{s}`` with N >= 17 (18 for
    ``p`` parts), and ``affine_qmm_t_nax`` present in the installed metallib.
    """
    if "value" in _NAX_CACHE:
        return _NAX_CACHE["value"]
    import platform
    import re
    from pathlib import Path

    ok = False
    try:
        release = tuple(int(x) for x in platform.mac_ver()[0].split(".")[:2])
        arch = str(mx.device_info().get("architecture", ""))
        match = re.fullmatch(r"applegpu_g(\d+)([a-z])", arch)
        if release >= (26, 2) and match is not None:
            gen, suffix = int(match.group(1)), match.group(2)
            ok = gen >= (18 if suffix == "p" else 17)
        if ok:
            lib = Path(mx.__file__).parent / "lib" / "mlx.metallib"
            if lib.is_file():
                ok = b"affine_qmm_t_nax" in lib.read_bytes()
    except Exception:
        ok = False
    _NAX_CACHE["value"] = ok
    return ok


def pack_projections(model) -> int:
    """Repack eligible 4-bit dense Qwen3.5 projections for the M5 tensor unit.

    Returns the number of packed layers (0 when unsupported). Must run on the
    MLX executor thread after weights are loaded.
    """
    from . import qwen35_packed_linear

    if not qwen35_packed_linear.enabled(model):
        return 0
    return int(qwen35_packed_linear.pack_model(model) or 0)


def apply(fast: bool = False, row_exact: bool = False) -> dict:
    """Install the verify kernels once.

    ``fast`` arms the non-exact matmuls. ``row_exact`` arms upstream's
    row-exact mode instead: every multi-row verify projection runs one-row
    decode arithmetic per row (``row_exact_qmv``) and verify attention keeps
    each row on its own one-row SDPA plan, so verify rows equal serial decode
    with stock kernels (no change to the decode path).
    """
    _STATE["fast"] = bool(fast) and not row_exact
    _STATE["row_exact"] = bool(row_exact)
    if _STATE["applied"] is not None:
        return _STATE["applied"]
    import mlx_vlm.speculative.mtp as mtp

    from . import (
        qwen35_gdn_prework,
        qwen35_gdn_verify_fused,
        qwen35_verify_linear,
        qwen35_verify_qmm,
        qwen35_verify_sdpa_split,
    )

    applied = {}
    for name, fn in (
        ("verify_linear", qwen35_verify_linear.apply),
        ("gdn_prework", qwen35_gdn_prework.apply_qwen35_gdn_prework_patch),
        ("sdpa_split", qwen35_verify_sdpa_split.apply_qwen35_verify_sdpa_split_patch),
        ("verify_qmm", qwen35_verify_qmm.apply_verify_qmm_patch),
        ("gdn_replay", qwen35_gdn_verify_fused.apply_arrays_cache_replay_patch),
    ):
        try:
            result = fn()
            applied[name] = result is not False
        except Exception:
            logger.warning("oMLX verify kernel %s not applied", name, exc_info=True)
            applied[name] = False

    import mlx_vlm.speculative.dflash as dflash

    def arm(module, name):
        original = getattr(module, name, None)
        if original is None or getattr(original, "_yunshu_armed", False):
            return

        def armed(*args, **kwargs):
            if _STATE["row_exact"]:
                qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
            else:
                qwen35_verify_qmm.set_verify_qmm_armed(_STATE["fast"])
            try:
                return original(*args, **kwargs)
            finally:
                qwen35_verify_qmm.set_verify_qmm_armed(False)

        armed._yunshu_armed = True
        setattr(module, name, armed)

    # MTP and DFlash each call their verify entry by module-level name.
    arm(mtp, "_mtp_verify_target")
    arm(dflash, "_dflash_verify_greedy")
    arm(dflash, "_dflash_verify")

    _STATE["applied"] = applied
    logger.info(
        "Qwen MTP verify kernels: %s (fast=%s, row_exact=%s)",
        applied,
        _STATE["fast"],
        _STATE["row_exact"],
    )
    return applied
