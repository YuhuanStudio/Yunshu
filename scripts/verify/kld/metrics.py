"""Teacher-forced KL-divergence metrics from top-k logprob records (CPU, numpy only).

A record is one window: ``ids [N, K]`` int32 and ``lp [N, K]`` float32 (the K most probable next tokens at
every position, descending, normalized log-probabilities of the full distribution), ``next_lp [N]`` (the
log-probability of the token that actually follows, NaN at the last position) and ``tokens [N]`` (the
window's input ids). KL(ref || arm) is taken over the reference's K tokens plus one remainder bucket; a
reference token missing from the arm's top-K gets the arm's smallest listed probability (an upper bound on
its true value, so a missing token never makes the arm look worse than it is).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEPTH_EDGES = (0, 1024, 2048, 4096, 8192, 16384, 32768, 1 << 30)


@dataclass
class Window:
    ids: np.ndarray  # [N, K] int32
    lp: np.ndarray  # [N, K] float32
    next_lp: np.ndarray  # [N] float32
    tokens: np.ndarray  # [N] int32

    def check(self, expect: int | None = None) -> None:
        n = self.ids.shape[0]
        if (
            self.ids.shape != self.lp.shape
            or self.next_lp.shape != (n,)
            or self.tokens.shape != (n,)
        ):
            raise ValueError("inconsistent record shapes")
        if expect is not None and abs(n - expect) > 2:
            raise ValueError(f"window has {n} positions, expected about {expect}")
        if not np.isfinite(self.lp).all():
            raise ValueError("non-finite top-k logprobs")


def kld_per_position(ref: Window, arm: Window, block: int = 4096) -> np.ndarray:
    """KL(ref || arm) at each position, ``[N]`` (positions both windows cover)."""
    n = min(ref.ids.shape[0], arm.ids.shape[0])
    return np.concatenate(
        [
            _kld_block(
                ref.ids[i : min(i + block, n)],
                ref.lp[i : min(i + block, n)],
                arm.ids[i : min(i + block, n)],
                arm.lp[i : min(i + block, n)],
            )
            for i in range(0, n, block)
        ]
    )


def _kld_block(ref_ids, ref_lp, arm_ids, arm_lp) -> np.ndarray:
    p = np.exp(ref_lp.astype(np.float64))  # [n, K]
    q_arm = np.exp(arm_lp.astype(np.float64))
    floor = q_arm.min(axis=1, keepdims=True)
    # arm probability of each reference token: its listed value, else the floor
    match = ref_ids[:, :, None] == arm_ids[:, None, :]  # [n, K, K]
    listed = (match * q_arm[:, None, :]).sum(axis=2)
    present = match.any(axis=2)
    q = np.where(present, listed, floor)
    p_rest = np.clip(1.0 - p.sum(axis=1), 0.0, None)
    q_rest = np.clip(1.0 - q.sum(axis=1), 1e-9, None)
    kl = (p * np.log(np.clip(p, 1e-12, None) / np.clip(q, 1e-12, None))).sum(axis=1)
    kl += np.where(
        p_rest > 1e-12, p_rest * np.log(np.clip(p_rest, 1e-12, None) / q_rest), 0.0
    )
    return np.clip(kl, 0.0, None)


def top1_agreement(ref: Window, arm: Window) -> np.ndarray:
    n = min(ref.ids.shape[0], arm.ids.shape[0])
    return (ref.ids[:n, 0] == arm.ids[:n, 0]).astype(np.float64)


def top5_overlap(ref: Window, arm: Window) -> np.ndarray:
    n = min(ref.ids.shape[0], arm.ids.shape[0])
    r, a = ref.ids[:n, :5], arm.ids[:n, :5]
    return (r[:, :, None] == a[:, None, :]).any(axis=2).mean(axis=1)


def perplexity(window: Window) -> float:
    lp = window.next_lp[:-1]
    return float(math.exp(-np.nanmean(lp)))


def depth_curve(per_position: np.ndarray) -> dict[str, float]:
    """Mean of a per-position series by context depth bucket (position index = context depth)."""
    out = {}
    for lo, hi in zip(DEPTH_EDGES[:-1], DEPTH_EDGES[1:]):
        chunk = per_position[lo:hi]
        if len(chunk):
            out[f"{lo}-{hi if hi < 1 << 29 else 'end'}"] = float(chunk.mean())
    return out


def summarize(ref: dict[str, Window], arm: dict[str, Window]) -> dict:
    """Metrics of ``arm`` against ``ref`` over every window both cover; fails closed on a missing window."""
    missing = sorted(set(ref) - set(arm))
    if missing:
        raise RuntimeError(f"arm lacks windows: {missing[:5]}")
    kl, t1, t5 = [], [], []
    per_depth: dict[str, list[float]] = {}
    ppl_ref, ppl_arm = [], []
    for wid in sorted(ref):
        k = kld_per_position(ref[wid], arm[wid])
        kl.append(k)
        t1.append(top1_agreement(ref[wid], arm[wid]))
        t5.append(top5_overlap(ref[wid], arm[wid]))
        for bucket, value in depth_curve(k).items():
            per_depth.setdefault(bucket, []).append(value)
        ppl_ref.append(perplexity(ref[wid]))
        ppl_arm.append(perplexity(arm[wid]))
    all_kl = np.concatenate(kl)
    return dict(
        windows=len(ref),
        positions=int(all_kl.size),
        kld_mean=float(all_kl.mean()),
        kld_p99=float(np.quantile(all_kl, 0.99)),
        top1=float(np.concatenate(t1).mean()),
        top5=float(np.concatenate(t5).mean()),
        ppl_ref=float(np.mean(ppl_ref)),
        ppl_arm=float(np.mean(ppl_arm)),
        delta_ppl=float(np.mean(ppl_arm) - np.mean(ppl_ref)),
        kld_by_depth={b: float(np.mean(v)) for b, v in per_depth.items()},
        window_kld_mean=[float(k.mean()) for k in kl],
    )


def verdict(
    ref_vs_base: dict, ref_vs_cand: dict, eps: float, top1_noise: float
) -> tuple[bool, list[str]]:
    """PASS when cand is no further from the reference than base: KLD <= base x 1.1 + eps, top-1 not lower
    beyond the noise (both measured against the reference)."""
    reasons = []
    limit = ref_vs_base["kld_mean"] * 1.1 + eps
    if ref_vs_cand["kld_mean"] > limit:
        reasons.append(
            f"KLD(ref||cand) {ref_vs_cand['kld_mean']:.5f} > {limit:.5f} (base {ref_vs_base['kld_mean']:.5f} x 1.1 + eps {eps})"
        )
    if ref_vs_cand["top1"] < ref_vs_base["top1"] - top1_noise:
        reasons.append(
            f"top-1 vs ref {ref_vs_cand['top1']:.4f} < base {ref_vs_base['top1']:.4f} - noise {top1_noise}"
        )
    return not reasons, reasons
