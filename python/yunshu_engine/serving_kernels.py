"""The served kernel stack for a loaded VLM model.

``install`` is the kernel-setup block VLMEngine._build_batch_runner runs for a loaded model: the
Qwen4 lane (configure_lane), or the Qwen3.5-family verify kernels (omlx, row-invariant lane
projections, NAX prefill, BQ32 attention, chunked GDN, split-invariant prefill, the optional ANE FFN
split) with the drafter kernels that follow them. VLMEngine and the M0 round-driver golden harness
(scripts/research/round_driver_golden.py) both call it, so the harness exercises the configuration
that is served. The code moved here unchanged; only its inputs became parameters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import settings
from .qwen4_family import is_qwen4_family

# Log records keep the logger name they had inside VLMEngine.
logger = logging.getLogger("yunshu_engine.vlm_engine")


def install(
    model: Any,
    config: dict,
    *,
    drafter: Any,
    draft_kind: str,
    spec_family: bool,
    model_path: Any,
    tokenizer: Any,
    block: int | None = None,
    driver_lanes: int = 0,
) -> tuple[dict | None, int | None]:
    """Install the served kernels on a loaded ``model``; return ``(kernels, block)``.

    ``kernels`` is None when this family and drafter combination installs nothing. ``driver_lanes``
    is the lane count VLMEngine converted before the drafter loaded (a second convert then finds
    them in place). ``block`` is the MTP block size; the Qwen4 lane and the drafter may change it.
    """
    from .blocks import linear as lane_linear
    from .kernels import attn_prefill, gdn_prefill, nax_prefill, sdpa_chunk

    lm = model.language_model
    qwen4 = is_qwen4_family(config)
    kernels = None
    # M1-M4 only; bounds the unfused head-dim-256 prefill score matrix
    sdpa_chunk.install()
    # Spec off on a spec family still runs the lane's plain-decode arithmetic
    # (invariant kernels), so spec on == spec off token-for-token.
    if qwen4:
        from .qwen4_mtp import configure_lane

        kernels, block = configure_lane(lm, drafter, block)
    if not qwen4 and (drafter is not None or spec_family):
        from .kernels.omlx import apply as apply_verify_kernels

        kernels = apply_verify_kernels()
        # Lossless MTP decode: decode and verify share row-invariant kernels,
        # so speculative output == this engine's plain decode. On Qwen3.8-27B
        # (M5 Max) invariant + NAX-packed at block 6 decodes code/prose/json
        # at 88.6/59.9/67.3 tok/s vs 57-67/50-53/58-62 for row-exact kernels
        # at block 3 (the removed row-exact alternative) and
        # 83.8/59.9/66.7 for the non-exact fast verify, with parity on every
        # task (measured on Qwen3.8-27B, 2026-09-28).
        # DFlash verifies through the same target forward, so it gets the
        # same guarantee.
        from .kernels.batch_invariant import install as install_invariant
        from .kernels.batch_invariant import set_active

        # Target projections run TensorFold's row-invariant lane matmul
        # (M5-class tensor ops; sg8 elsewhere). 27B, in-process vs the
        # NAX packed kernel: MTP / DFlash2 decode +0..+17% at 1K-131K
        # with identical tokens, and 0.6 GiB less memory (no repacked copy).
        from .kernels.tensorfold.lane_qmm import ready

        if ready():
            from .blocks import linear as lane_linear

            # (a second convert finds the driver's lanes already in place)
            kernels["lane_linear"] = (
                lane_linear.convert(lm)["converted"] or driver_lanes
            )
            if settings.get("YUNSHU_PREFILL_MATMUL") == "stock":
                lane_linear.set_stock_rows(lane_linear.STOCK_MIN_ROWS)
                # every spec-family size: nax_prefill.eligible() checks each call's shape (was 27B-only)
                if kernels["lane_linear"] and nax_prefill.enable():
                    nax_prefill.warmup(
                        {
                            module.bits
                            for _, module in lm.named_modules()
                            if isinstance(module, lane_linear.LaneLinear)
                        }
                    )
                    kernels["nax_prefill"] = nax_prefill.arithmetic_id()
            kernels["prefill_matmul"] = lane_linear.prefill_kernel_id()
        if settings.get("YUNSHU_PREFILL_GDN") == "chunked":
            from .kernels import gdn_prefill

            gdn_prefill.install()
        if settings.get("YUNSHU_PREFILL_ATTENTION") == "tile":
            # Bit-identical to stock MLX attention, so no APC key change.
            kernels["attn_prefill"] = (
                attn_prefill.arithmetic_id() if attn_prefill.enable() else None
            )
        kernels["invariant"] = install_invariant(lm, model=model, packed=False)
        # Every prompt piece uses the same arithmetic, while decode and
        # speculative verify retain their own invariant kernels. Only the
        # jointly qualified 27B device/ABI/weight geometry can enter it. The KV precision is
        # not part of it: prefill and the prefix cache stay bf16 whatever the decode rows hold.
        split_ready = bool(
            kernels.get("nax_prefill")
            and kernels.get("attn_prefill")
            and settings.get("YUNSHU_PREFILL_GDN") == "chunked"
            and all(
                module.output_dims < lane_linear.NARROW
                or module.output_dims >= 100_000
                or nax_prefill.split_eligible(
                    64,
                    module.input_dims,
                    module.output_dims,
                    module.bits,
                    module.group_size,
                    module.tiled,
                )
                for _, module in lm.named_modules()
                if isinstance(module, lane_linear.LaneLinear)
            )
        )
        # Flash-Next shared gate: row-invariant f32 dot, so a prefill tail == cold prefill.
        from .kernels import sgate_rowwise

        kernels["sgate_rowwise"] = sgate_rowwise.install(lm) or None
        gdn_prefill.configure_split(split_ready)
        if split_ready:
            gdn_prefill.install_split_forward(lm)
            kernels["split_prefill"] = "split64-v1"
            logger.info("Split-invariant prefill engaged: NAX/BQ32/GDN, alignment=64")
        if settings.get("YUNSHU_ANE_FFN_SPLIT") == "splash" and split_ready:
            from . import ane_ffn_runtime

            runtime = ane_ffn_runtime.install(
                lm,
                Path(
                    settings.get("YUNSHU_ANE_FFN_CACHE")
                    or Path.home() / ".yunshu" / "ane_cache" / "ffn"
                ),
                str(model_path),
                share=settings.get("YUNSHU_ANE_FFN_SHARE"),
                minimum_rows=settings.get("YUNSHU_ANE_FFN_MIN_ROWS"),
            )
            if runtime and settings.get("YUNSHU_ANE_FFN_TRACE"):
                runtime.trace = True
                runtime.split.profile(True)
            kernels["ane_ffn_split"] = (
                ane_ffn_runtime.kernel_suffix() if runtime else None
            )
        # The runner turns them on for spec and lone AR target steps.
        set_active(False)
        if draft_kind == "dflash" and drafter is not None:
            # Reduced candidate vocabulary for the DFlash lattice (draft_vocab.logits_subset).
            from .draft_vocab import DRAFT_VOCAB as DFLASH_KEEP
            from .draft_vocab import install as _install_dflash_vocab

            assert model is not None
            found = _install_dflash_vocab(drafter, lm, DFLASH_KEEP, tokenizer)
            kernels["draft_vocab"] = DFLASH_KEEP if found else None
        if draft_kind == "mtp" and drafter is not None:
            # Copy-row budget for the driver's MTP verify, fused
            # residual+norm layers and a reduced draft vocabulary.
            from . import mtp_lane
            from .draft_vocab import DRAFT_VOCAB
            from .draft_vocab import install as install_draft_vocab
            from .kernels import lane_layers

            kernels["copy_rows"] = mtp_lane.set_copy_rows(
                settings.get("YUNSHU_SPEC_COPY_ROWS"),
                mtp_lane.verify_max_rows(bool(kernels.get("lane_linear")), lm),
            )
            kernels["lane_layers"] = lane_layers.install()
            found = install_draft_vocab(drafter, lm, DRAFT_VOCAB, tokenizer)
            kernels["draft_vocab"] = DRAFT_VOCAB if found else None
        # Invariant kernels cost little per extra verify row and peak at 6
        # for MTP. A DFlash drafter proposes a whole block in one forward
        # and upstream adapts the depth to acceptance under this ceiling,
        # so the ceiling is the block it was trained on.
        if block is None and drafter is not None:
            if draft_kind == "dflash":
                block = int(getattr(drafter.config, "block_size", 0)) or None
            else:
                block = 6

    return kernels, block
