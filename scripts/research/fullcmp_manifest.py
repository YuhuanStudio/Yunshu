#!/usr/bin/env python3
"""Designed-grid expected-cell manifest for the full competitor comparison (FULLCMP_SPEC, 2026-11-03).

Replaces the full cross product (209K rows) with a designed grid (lead decision 2026-11-03,
target <= 48 GPU hours including 3 reps). Every planned cell is one JSONL row, generated before
any GPU work. Batches (each audited before the next starts):

  B1  core cross, 27B 4-bit: ctx {1K,8K,32K,64K,128K} x conc {1,8} x kind {code,prose} x out 512
      metrics: cold TTFT, decode, warm, immediate follow-up, follow-up decode, memory
  B2  one-factor sweeps at the centre (27B 4-bit, centre = 8K, conc 1, code, out 512, thinking off)
  B3  secondary models: Flash-Next, Qwen3.5-9B, 27B 8-bit and bf16 (subsets from the lead)
  B4  behaviour checks, once per engine (7 checks, 1 rep)
  B5  agents: Claude Code, Codex, opencode x each engine x the agentbench task set, 3 reps
  B6  soak: 30 min once for Yunshu and once for the best competitor (TBD, counted once)

Row status (never dropped, never silently zero):
  runnable                     a run in the evidence set (cmp-1102, snapshot014) covers this cell
  unverified (census pending)  not established yet; still costed as if supported (the batch totals
                               report runnable and unverified hours separately)
  unsupported (reason)         certain: text-only server for images, or no checkpoint (Ollama)

Cost model (est_gpu_min per row, all REPS included, minutes):
  wall_s(cell) = cold + warm + follow-up + ... (see wall_s); each prefill term is
      conc * tokens / PREFILL_TPS(engine, ctx)      (distinct prompts, one-at-a-time prefill)
      out * slowdown(engine, conc) / DECODE_TPS(engine, ctx)
  est_gpu_min = REPS * wall_s / 60 + share of the process boot (READY_S x REPS per process group)

Every measured constant is in ENGINE_PERF with the file it came from. Engines without a
measurement use the Yunshu curve and are flagged 'assumed'. Assumed scalars are named ASSUMED.

Usage:
  uv run python scripts/research/fullcmp_manifest.py            # write manifest + print summary
  uv run python scripts/research/fullcmp_manifest.py --out X    # other output path
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_OUT = Path(
    "/Volumes/P5Plus/yunshu-build/competitors/compsync/fullcmp-1103/manifest.jsonl"
)
ROOT = Path(__file__).resolve().parents[2]
AGENTIC = ROOT / "scripts/research/agentic"

REPS = 3
BEHAVIOUR_REPS = 1
SOAK_REPS = 1
SOAK_MIN = 30.0
TARGET_GPU_H = 48.0
# ASSUMED scalars (each one is named in the summary so the lead can change it).
AGENT_RUN_MIN = (
    12.0  # agentbench shard budget, PERF_TREND 2026-10-23 (62 runs, 12-min budget)
)
BEHAVIOUR_MIN = 3.0  # per engine, the 7 behaviour checks plus one boot
WARM_TTFT_S = 0.5  # per warm request; cmp-1102 warm TTFT 0.13-0.25 s at 32K
FOLLOWUP_NEW_TOKENS = 1024  # new user turn appended to the previous reply
THINK_TOKENS = 1024  # extra reasoning tokens with thinking on
IDLE_S = 5.0  # spec: follow-up after 5 s idle
T10_TURNS = 10  # spec: 10-turn growing agent session
RESTART_RESTORE_FRAC = 0.25  # disk-tier restore as a fraction of cold prefill
IMAGE_TOKENS = {"none": 0, "img1": 1024, "img4": 4096}  # visual tokens per image
MIXED_SHORT_N = 7  # spec: one 128K request plus 7 x 1K requests
MIXED_SHORT_TOKENS = 1024
MIXED_LONG_CTX = 131072

PRIMARY = "q27-4bit"
MODELS = {
    PRIMARY: "Qwen3.8-27B 4-bit (primary)",
    "q27-8bit": "Qwen3.8-27B 8-bit",
    "q27-bf16": "Qwen3.8-27B bf16",
    "flashnext-oq3e": "Qwen3.8-Flash-Next oQ3e-mtp",
    "qwen35-9b-4bit": "Qwen3.5-9B-MLX-4bit (tier-1 family)",
}
FLASH = "flashnext-oq3e"
Q9 = "qwen35-9b-4bit"
Q8 = "q27-8bit"
BF16 = "q27-bf16"

# ASSUMED model-size factors on the 27B 4-bit curve (memory-bound decode, compute-bound prefill).
# Flash-Next and Qwen3.5-9B use the 27B curve: conservative (both are smaller or MoE).
MODEL_FACTOR = {
    PRIMARY: (1.0, 1.0),  # (prefill factor, decode factor)
    Q8: (0.8, 0.55),
    BF16: (0.5, 0.3),
    FLASH: (1.0, 1.0),
    Q9: (1.0, 1.0),
}

CTX_TOKENS = {
    "1K": 1024,
    "4K": 4096,
    "8K": 8192,
    "16K": 16384,
    "32K": 32768,
    "64K": 65536,
    "128K": 131072,
    "256K": 262144,
}
CORE_CTX = ["1K", "8K", "32K", "64K", "128K"]
CORE_CONC = [1, 8]
CORE_KINDS = ["code", "prose"]
CENTER = dict(
    ctx="8K",
    conc=1,
    kind="code",
    out=512,
    turns="single",
    thinking="off",
    media="none",
    sampling="greedy",
)

# ---------------------------------------------------------------- measured constants
# Yunshu curve (27B 4-bit, cold prefill tok/s and decode tok/s, per ctx). Source: verify run
# snapshot014-released-v2-13824db790a2, cells snapshot.yunshu-new-{d1k,d8k,d32k,d64k-*,d128k-*}-r*.jsonl,
# 'decode' part, cached == 0, median over reps. prefill = pt / ttft_s. Yunshu 0.1.4 @ 8c099150
# (main 0f2960047 is newer; PERF_TREND 2026-10-23 shows 128K code decode 90 tok/s on newer main).
YUNSHU_CURVE = {
    "prefill": {1024: 836.0, 8192: 1027.0, 32768: 946.0, 65536: 825.0, 131072: 656.0},
    "decode": {1024: 78.2, 8192: 81.8, 32768: 100.7, 65536: 79.5, 131072: 57.4},
}

ENGINE_PERF = {
    "yunshu": dict(
        full=True,
        prefill={1024: 836.0, 8192: 1027.0, 32768: 946.0, 65536: 825.0, 131072: 656.0},
        decode={1024: 78.2, 8192: 81.8, 32768: 100.7, 65536: 79.5, 131072: 57.4},
        # snapshot014 yunshu-new c2c4 'conc' part, per_req_dec medians (32K, out 2048) vs c1 32K decode.
        # 17.1 tok/s at c2 and 12.1 at c4 -> 100.7/17.1, 100.7/12.1.
        slow={1: 1.0, 2: 5.9, 4: 8.3},
        ready_s=6.1,  # snapshot014 yunshu-new ready_s median (n=20)
        source="snapshot014-released-v2 yunshu-new (0.1.4 @8c099150): decode/cold curves, c2c4 'conc', ready_s",
        assumed=[],
    ),
    "mlx-lm-server": dict(
        full=True,
        prefill={1024: 869.0, 8192: 936.0, 32768: 838.0, 65536: 716.0, 131072: 577.0},
        decode={1024: 29.8, 8192: 28.6, 32768: 26.1, 65536: 23.0, 131072: 19.4},
        slow={1: 1.0, 2: 1.19, 4: 1.60},  # snapshot014 mlxlm c2c4: 26.1/22.0, 26.1/16.3
        ready_s=2.0,
        source="snapshot014-released-v2 mlxlm (mlx-lm 0.32.0, stock AR, no spec mode engaged)",
        assumed=[
            "AR decode measured (no spec mode engaged), not the best lossless mode"
        ],
    ),
    "omlx-stable": dict(
        full=True,  # proxy: oMLX 0.7.0 measured in snapshot014 (stable 0.7.2 not run)
        prefill={1024: 877.0, 8192: 1004.0, 32768: 910.0, 65536: 794.0, 131072: 651.0},
        decode={1024: 71.7, 8192: 79.6, 32768: 73.5, 65536: 58.0, 131072: 52.9},
        slow={1: 1.0, 2: 1.97, 4: 3.45},  # snapshot014 omlx c2c4: 73.5/37.4, 73.5/21.3
        ready_s=2.0,
        source="PROXY snapshot014-released-v2 omlx (oMLX 0.7.0); stable 0.7.2 not measured",
        assumed=["curve is a 0.7.0 proxy", "slowdown from the 0.7.0 proxy"],
    ),
    "omlx-rc": dict(
        full=False,
        prefill={
            1024: 772.0,
            32768: 886.0,
        },  # cmp-1102 omlx-latest (v0.7.3rc1): mean(654,890), 886
        decode={1024: 64.8, 32768: 71.0},  # cmp-1102 omlx-latest: mean(70.8,58.8), 71.0
        slow=None,
        ready_s=3.3,  # cmp-1102 omlx-latest ready_s median
        source="cmp-1102 base/omlx-latest-decode-{1024-code,1024-prose,32768-code}-r0.jsonl (v0.7.3rc1)",
        assumed=["slowdown from the Yunshu curve"],
    ),
    "mlx-serve": dict(
        full=False,
        prefill={
            1024: 922.5,
            32768: 751.0,
        },  # cmp-1102 base/mlx-serve-decode-* (v26.10.3)
        decode={1024: 70.0, 32768: 139.8},
        slow=None,
        ready_s=8.6,  # cmp-1102 mlx-serve ready_s median
        source="cmp-1102 base/mlx-serve-decode-{1024-code,1024-prose,32768-code}-r0.jsonl (v26.10.3)",
        assumed=["slowdown from the Yunshu curve"],
    ),
    "splash-latest": dict(
        full=False,
        prefill={
            1024: 1192.5,
            32768: 1065.0,
        },  # cmp-1102 base/splash-latest-decode-* (1.4.1)
        decode={1024: 80.2, 32768: 109.4},
        slow=None,
        ready_s=75.9,  # cmp-1102 splash ready_s median (75.9, 15.7, 79.0)
        source="cmp-1102 base/splash-latest-decode-{1024-code,1024-prose,32768-code}-r0.jsonl (1.4.1)",
        assumed=["slowdown from the Yunshu curve"],
    ),
    "tf-1.2.0": dict(
        full=False,
        prefill={1024: 962.0, 32768: 854.0},  # cmp-1102 tf-1.2.0 (prose 1K, code 32K)
        decode={1024: 84.0, 32768: 112.0},
        slow=None,
        ready_s=17.5,  # cmp-1102 tf-1.2.0 ready_s median (14.6, 20.4)
        source="cmp-1102 tf-1.2.0/tf-native-decode-{1024-prose,32768-code}-r0.jsonl (v1.2.0)",
        assumed=["slowdown from the Yunshu curve", "1K code not run (prose only)"],
    ),
    "tf-1.1.8": dict(
        full=False,
        prefill={
            1024: 959.5,
            32768: 847.0,
        },  # cmp-1102 tf-1.1.8 (mean of code/prose 1K, code 32K)
        decode={
            1024: 99.4,
            32768: 237.1,
        },  # 32K code 237 tok/s is high (draft acceptance on code)
        slow=None,
        ready_s=8.5,  # cmp-1102 tf-1.1.8 ready_s
        source="cmp-1102 tf-1.1.8/tf-native-decode-*-r0.jsonl (v1.1.8)",
        assumed=["slowdown from the Yunshu curve"],
    ),
    # No measurement anywhere: the Yunshu curve, flagged assumed.
    "llama-cpp": dict(
        full=False,
        prefill={},
        decode={},
        slow=None,
        ready_s=6.1,
        source="ASSUMED: no run (GGUF not built); Yunshu curve",
        assumed=["no measurement; Yunshu curve", "ready_s from Yunshu"],
    ),
    "lmstudio": dict(
        full=False,
        prefill={},
        decode={},
        slow=None,
        ready_s=6.1,
        source="ASSUMED: no run (headless lms, same checkpoint not verified); Yunshu curve",
        assumed=["no measurement; Yunshu curve", "ready_s from Yunshu"],
    ),
    "mlx-vlm-server": dict(
        full=False,
        prefill={},
        decode={},
        slow=None,
        ready_s=6.1,
        source="ASSUMED: no run (mlx-vlm server); Yunshu curve",
        assumed=["no measurement; Yunshu curve", "ready_s from Yunshu"],
    ),
    "kapitsa": dict(
        full=False,
        prefill={},
        decode={},
        slow=None,
        ready_s=6.1,
        source="ASSUMED: no run (Kapitsa 0.1.39 main); Yunshu curve",
        assumed=["no measurement; Yunshu curve", "ready_s from Yunshu"],
    ),
    "ollama": dict(
        full=False,
        prefill={},
        decode={},
        slow=None,
        ready_s=6.1,
        source="n/a: unsupported, no cost",
        assumed=[],
    ),
}


def _loglin(points, ctx):
    """Linear in log2(ctx) between measured points; the last segment extrapolates beyond it."""
    xs = sorted(points)
    if ctx <= xs[0]:
        return points[xs[0]]
    a, b = xs[-2], xs[-1]
    for lo, hi in zip(xs[:-1], xs[1:], strict=True):
        if ctx <= hi:
            a, b = lo, hi
            break
    t = (math.log2(ctx) - math.log2(a)) / (math.log2(b) - math.log2(a))
    return points[a] + t * (points[b] - points[a])


def _curve_value(perf, key, ctx):
    """Engine value at ctx. Full curves interpolate their points. Partial engines pass exactly
    through their measured anchors, interpolate between them, and outside them keep the Yunshu
    shape scaled to the nearest anchor (no measurement there, so the shape is the assumption)."""
    anchors = perf[key]
    if perf["full"] or not anchors:
        if not anchors:
            return _loglin(YUNSHU_CURVE[key], ctx)
        return _loglin(anchors, ctx)
    xs = sorted(anchors)
    if xs[0] <= ctx <= xs[-1]:
        return _loglin(anchors, ctx)
    a = xs[0] if ctx < xs[0] else xs[-1]
    return anchors[a] / _loglin(YUNSHU_CURVE[key], a) * _loglin(YUNSHU_CURVE[key], ctx)


def prefill_tps(eng, ctx_tokens, model=PRIMARY):
    return (
        _curve_value(ENGINE_PERF[eng], "prefill", ctx_tokens) * MODEL_FACTOR[model][0]
    )


def decode_tps(eng, ctx_tokens, model=PRIMARY):
    return _curve_value(ENGINE_PERF[eng], "decode", ctx_tokens) * MODEL_FACTOR[model][1]


def slowdown(eng, conc):
    """Per-request decode slowdown at concurrency conc (1.0 at conc 1)."""
    s = ENGINE_PERF[eng]["slow"] or ENGINE_PERF["yunshu"]["slow"]
    if conc in s:
        return s[conc]
    exp = math.log(s[4] / s[2]) / math.log(2)
    return s[4] * (conc / 4.0) ** exp


# ---------------------------------------------------------------- engines and status
R, U, X = "runnable", "unverified (census pending)", "unsupported"
ENGINES = {
    "yunshu": dict(
        label="Yunshu main 0f2960047 (0.1.7.dev0)",
        version="0.1.7.dev0 @ 0f2960047 (curve from 0.1.4 snapshot014)",
        supported=True,
        text_only=False,
        ctx_run={"1K", "8K", "32K", "64K", "128K"},
        conc_run={1, 2, 4},
        models_run={PRIMARY},
        agents_api={
            "claude": "Anthropic",
            "codex": "Responses",
            "opencode": "OpenAI chat",
        },
    ),
    "tf-1.2.0": dict(
        label="TensorFold 1.2.0 (tf-native)",
        version="v1.2.0",
        supported=True,
        text_only=False,
        ctx_run={"1K", "32K"},
        conc_run={1},
        models_run={PRIMARY},
        agents_api={},
    ),
    "tf-1.1.8": dict(
        label="TensorFold 1.1.8 (tf-native, lone loop)",
        version="v1.1.8",
        supported=True,
        text_only=False,
        ctx_run={"1K", "32K"},
        conc_run={1},
        models_run={PRIMARY},
        agents_api={},
    ),
    "splash-latest": dict(
        label="Splash latest",
        version="1.4.1 (latest to confirm)",
        supported=True,
        text_only=False,
        ctx_run={"1K", "32K"},
        conc_run={1},
        models_run={PRIMARY},
        agents_api={},
    ),
    "omlx-stable": dict(
        label="oMLX latest stable",
        version="0.7.2 stable (not run; 0.7.0 proxy costed)",
        supported=True,
        text_only=False,
        ctx_run=set(),
        conc_run=set(),
        models_run=set(),
        agents_api={},
    ),
    "omlx-rc": dict(
        label="oMLX latest rc",
        version="v0.7.3rc1",
        supported=True,
        text_only=False,
        ctx_run={"1K", "32K"},
        conc_run={1},
        models_run={PRIMARY},
        agents_api={},
    ),
    "mlx-serve": dict(
        label="mlx-serve latest",
        version="v26.10.3",
        supported=True,
        text_only=False,
        ctx_run={"1K", "32K"},
        conc_run={1},
        models_run={PRIMARY},
        agents_api={},
    ),
    "mlx-lm-server": dict(
        label="mlx_lm.server (stock, latest PyPI)",
        version="mlx-lm 0.32.0 measured; PyPI latest to confirm",
        supported=True,
        text_only=True,
        ctx_run={"1K", "8K", "32K", "64K", "128K"},
        conc_run={1, 2, 4},
        models_run={PRIMARY},
        agents_api={},
    ),
    "mlx-vlm-server": dict(
        label="mlx-vlm server (stock, latest PyPI)",
        version="census pending",
        supported=True,
        text_only=False,
        ctx_run=set(),
        conc_run=set(),
        models_run=set(),
        agents_api={},
    ),
    "llama-cpp": dict(
        label="llama.cpp llama-server (Metal, Q4_K_M GGUF of the same base)",
        version="b11848 (latest release to confirm; GGUF not built)",
        supported=True,
        text_only=False,
        ctx_run=set(),
        conc_run=set(),
        models_run=set(),
        agents_api={},
    ),
    "lmstudio": dict(
        label="LM Studio (headless lms, same MLX checkpoint)",
        version="census pending",
        supported=True,
        text_only=False,
        ctx_run=set(),
        conc_run=set(),
        models_run=set(),
        agents_api={},
    ),
    "ollama": dict(
        label="Ollama",
        version="census pending",
        supported=False,
        text_only=True,
        ctx_run=set(),
        conc_run=set(),
        models_run=set(),
        agents_api={},
        reason=(
            "n/a: Ollama serves GGUF via llama.cpp and cannot run the same MLX checkpoint "
            "headless (the GGUF baseline is the llama-cpp row)"
        ),
    ),
    "kapitsa": dict(
        label="Kapitsa main (CMake Release)",
        version="0.1.39 main 8b190f5",
        supported=True,
        text_only=False,
        ctx_run=set(),
        conc_run=set(),
        models_run=set(),
        agents_api={},
    ),
}
ENGINE_IDS = list(ENGINES)
AGENTS = ["claude", "codex", "opencode"]
AGENT_NEEDS = {
    "claude": "Anthropic Messages API",
    "codex": "Responses API",
    "opencode": "OpenAI-compatible chat API",
}
BEHAVIOUR_CHECKS = [
    "cancel_latency",
    "stop_sequences",
    "tool_validity",
    "json_validity",
    "logprobs",
    "anthropic_api",
    "responses_api",
]


def cell_status(eng, spec):
    """(status, reason) for one decode cell. spec keys: model ctx conc kind out turns thinking media sampling."""
    e = ENGINES[eng]
    if not e["supported"]:
        return X, e["reason"]
    if spec["media"] != "none" and e["text_only"]:
        return X, "text-only server: no image input path"
    if spec["model"] not in e["models_run"]:
        return U, f"{MODELS[spec['model']]} not run on {eng} (census pending)"
    if spec["ctx"] not in e["ctx_run"]:
        return U, f"ctx {spec['ctx']} not run on {eng} (driver or census pending)"
    if spec["conc"] not in e["conc_run"]:
        return U, f"concurrency {spec['conc']} not run on {eng} (census pending)"
    if spec["kind"] not in CORE_KINDS:
        return U, f"prompt kind {spec['kind']} not run (census pending)"
    if spec["out"] != 512:
        return U, f"output length {spec['out']} not run (census pending)"
    if spec["turns"] != "single":
        return (
            U,
            f"turn mode {spec['turns']} not run (follow-up/idle/10-turn/restart/mixed)",
        )
    if spec["thinking"] != "off":
        return U, "thinking on not run (census pending)"
    if spec["media"] != "none":
        return U, "image input not run on this engine (VLM capability census pending)"
    if spec["sampling"] != "greedy":
        return U, "generation_config sampling not run (census pending)"
    return (
        R,
        f"measured on a run in the evidence set (ctx {spec['ctx']}, conc {spec['conc']})",
    )


# ---------------------------------------------------------------- cells
def spec(**kw):
    base = dict(
        model=PRIMARY,
        ctx="8K",
        conc=1,
        kind="code",
        out=512,
        turns="single",
        thinking="off",
        media="none",
        sampling="greedy",
        phases=("cold", "warm", "fu"),
    )
    base.update(kw)
    return base


def core_specs():
    for ctx in CORE_CTX:
        for conc in CORE_CONC:
            for kind in CORE_KINDS:
                yield "core", spec(ctx=ctx, conc=conc, kind=kind)


def sweep_specs():
    for ctx in ("8K", "32K"):
        for conc in (2, 4, 16):
            yield "conc", spec(ctx=ctx, conc=conc)
    for ctx in ("4K", "16K", "256K"):
        yield "ctx", spec(ctx=ctx)
    for ctx in ("1K", "32K"):
        for out in (128, 2048, 8192):
            yield "out", spec(ctx=ctx, out=out)
    for kind in ("tool-chat", "cjk", "json-schema"):
        yield "kind", spec(kind=kind)
    yield "thinking", spec(thinking="on")
    for ctx in ("1K", "32K"):
        for conc in (1, 8):
            yield "sampling", spec(ctx=ctx, conc=conc, sampling="genconfig")
    for ctx in ("8K", "32K", "128K"):
        yield "idle5", spec(ctx=ctx, turns="idle5", phases=("cold", "warm", "fu_idle"))
    yield "restart", spec(ctx="32K", turns="restart", phases=("restart",))
    yield "t10", spec(ctx="8K", turns="t10", phases=("cold", "t10"))
    yield "mixed", spec(ctx="128K", conc=8, turns="mixed", phases=("mixed",))
    for media in ("img1", "img4"):
        yield "image", spec(ctx="1K", media=media, phases=("cold",))


def secondary_specs():
    for ctx in ("1K", "32K", "128K"):
        for conc in (1, 8):
            yield "secondary", spec(model=FLASH, ctx=ctx, conc=conc)
    for ctx in ("1K", "32K"):
        yield "secondary", spec(model=Q9, ctx=ctx)
        yield "secondary", spec(model=Q8, ctx=ctx)
        yield "secondary", spec(model=BF16, ctx=ctx)


BATCH_OF_SWEEP = {"core": "B1", "secondary": "B3"}
BATCH_NAMES = {
    "B1": "B1 core cross, 27B 4-bit",
    "B2": "B2 one-factor sweeps (27B 4-bit)",
    "B3": "B3 secondary models (Flash-Next, Qwen3.5-9B, 27B 8-bit/bf16)",
    "B4": "B4 behaviour checks (once per engine)",
    "B5": "B5 agents (Claude Code, Codex, opencode x agentbench)",
    "B6": "B6 soak (30 min)",
}


def wall_s(eng, model, ctx_tokens, conc, out, think, img_tokens, phases):
    """Wall seconds for one rep of one cell (sum of the phases the cell measures)."""
    pf = prefill_tps(eng, ctx_tokens, model)
    dec = decode_tps(eng, ctx_tokens, model)
    s = slowdown(eng, conc)
    o = out + (THINK_TOKENS if think else 0)
    t = 0.0
    if "cold" in phases:
        t += conc * (ctx_tokens + img_tokens) / pf + o * s / dec
    if "warm" in phases:
        t += conc * WARM_TTFT_S
    if "fu" in phases or "fu_idle" in phases:
        t += conc * (out + FOLLOWUP_NEW_TOKENS) / pf + out * s / dec
        if "fu_idle" in phases:
            t += IDLE_S
    if "t10" in phases:
        t += (T10_TURNS - 1) * (conc * (out + FOLLOWUP_NEW_TOKENS) / pf + out * s / dec)
    if "restart" in phases:
        t += RESTART_RESTORE_FRAC * ctx_tokens / pf + o * s / dec
    if "mixed" in phases:
        short_pf = prefill_tps(eng, 1024, model)
        t += ctx_tokens / pf + MIXED_SHORT_N * MIXED_SHORT_TOKENS / short_pf
        t += out * slowdown(eng, conc) / decode_tps(eng, 1024, model)
    return t


def _cell_row(eng, sweep, sp, batch):
    ctx_tokens = CTX_TOKENS[sp["ctx"]]
    status, why = cell_status(eng, sp)
    cid = (
        f"{sweep}.{eng}.{sp['model']}.{sp['ctx']}.c{sp['conc']}.{sp['kind']}.o{sp['out']}."
        f"{sp['turns']}.th-{sp['thinking']}.{sp['media']}.{sp['sampling']}"
    )
    work = (
        wall_s(
            eng,
            sp["model"],
            ctx_tokens,
            sp["conc"],
            sp["out"],
            sp["thinking"] == "on",
            IMAGE_TOKENS[sp["media"]],
            sp["phases"],
        )
        if status != X
        else 0.0
    )
    return {
        "id": cid,
        "batch": batch,
        "sweep": sweep,
        "table": "decode",
        "engine": eng,
        "engine_version": ENGINES[eng]["version"],
        "model": sp["model"],
        "ctx": sp["ctx"],
        "ctx_tokens": ctx_tokens,
        "conc": sp["conc"],
        "prompt_kind": sp["kind"],
        "out_len": sp["out"],
        "turns": sp["turns"],
        "phases": list(sp["phases"]),
        "thinking": sp["thinking"],
        "media": sp["media"],
        "sampling": sp["sampling"],
        "status": status,
        "reason": why,
        "reps": REPS,
        "_work_min": REPS * work / 60.0,
        "_group": (
            batch,
            eng,
            sp["model"],
            sp["ctx"],
            sp["turns"] if sp["turns"] in ("restart",) else "proc",
        ),
    }


def build_rows(engines=None, with_agents=True, agent_tasks_list=None):
    """Return (rows, meta). Deterministic: same args -> same rows in the same order."""
    engines = list(engines or ENGINE_IDS)
    rows = []
    for eng in engines:
        for sweep, sp in (
            list(core_specs()) + list(sweep_specs()) + list(secondary_specs())
        ):
            batch = BATCH_OF_SWEEP.get(sweep, "B2")
            rows.append(_cell_row(eng, sweep, sp, batch))
    for eng in engines:
        for check in BEHAVIOUR_CHECKS:
            st, why = (
                (X, ENGINES[eng]["reason"])
                if not ENGINES[eng]["supported"]
                else (U, f"behaviour check {check} not run (census pending)")
            )
            rows.append(
                {
                    "id": f"behaviour.{eng}.{check}",
                    "batch": "B4",
                    "sweep": "behaviour",
                    "table": "behaviour",
                    "engine": eng,
                    "engine_version": ENGINES[eng]["version"],
                    "model": PRIMARY,
                    "check": check,
                    "status": st,
                    "reason": why,
                    "reps": BEHAVIOUR_REPS,
                    "_work_min": BEHAVIOUR_MIN / len(BEHAVIOUR_CHECKS) * BEHAVIOUR_REPS,
                    "_group": None,
                }
            )
    tasks = (
        agent_tasks()
        if (with_agents and agent_tasks_list is None)
        else (agent_tasks_list or [])
    )
    for eng in engines:
        for agent in AGENTS:
            for task in tasks:
                rows.append(_agent_row(eng, agent, task))
    for name, label in (
        ("yunshu", "Yunshu main 0f2960047"),
        ("best-competitor (TBD)", "best competitor (TBD, counted once)"),
    ):
        st, why = (
            (U, "30-min soak on main not run in this grid")
            if name == "yunshu"
            else (U, "best competitor not chosen yet (TBD); counted once")
        )
        rows.append(
            {
                "id": f"soak.{name}",
                "batch": "B6",
                "sweep": "soak",
                "table": "soak",
                "engine": name,
                "engine_version": label,
                "model": PRIMARY,
                "status": st,
                "reason": why,
                "reps": SOAK_REPS,
                "_work_min": SOAK_MIN * SOAK_REPS,
                "_group": None,
            }
        )
    # process boots: each (batch, engine, model, ctx) group starts one server per rep, shared by its rows
    groups = defaultdict(list)
    for r in rows:
        if r.get("_group") and r["status"] != X:
            groups[r["_group"]].append(r)
    for g, members in groups.items():
        eng = g[1]
        boot = ENGINE_PERF[eng]["ready_s"] * REPS / 60.0
        for r in members:
            r["_work_min"] += boot / len(members)
    for r in rows:
        work = r.pop("_work_min")
        r.pop("_group", None)
        r["est_gpu_min"] = 0.0 if r["status"] == X else round(work, 3)
    return rows, _meta()


def _meta():
    return {
        "constants": {
            "REPS": REPS,
            "AGENT_RUN_MIN": AGENT_RUN_MIN,
            "BEHAVIOUR_MIN": BEHAVIOUR_MIN,
            "WARM_TTFT_S": WARM_TTFT_S,
            "FOLLOWUP_NEW_TOKENS": FOLLOWUP_NEW_TOKENS,
            "THINK_TOKENS": THINK_TOKENS,
            "IDLE_S": IDLE_S,
            "T10_TURNS": T10_TURNS,
            "RESTART_RESTORE_FRAC": RESTART_RESTORE_FRAC,
            "IMAGE_TOKENS": IMAGE_TOKENS,
            "MIXED_SHORT_N": MIXED_SHORT_N,
            "MIXED_SHORT_TOKENS": MIXED_SHORT_TOKENS,
            "SOAK_MIN": SOAK_MIN,
            "TARGET_GPU_H": TARGET_GPU_H,
            "MODEL_FACTOR": MODEL_FACTOR,
        }
    }


def agent_tasks():
    """Task ids from scripts/research/agentic: custom task dirs + polyglot slugs."""
    custom = sorted(
        d.name for d in (AGENTIC / "tasks").iterdir() if (d / "task.json").exists()
    )
    tree = ast.parse((AGENTIC / "tasks.py").read_text())
    slugs = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "POLYGLOT" for t in node.targets
        ):
            slugs = [ast.literal_eval(v) for v in node.value.elts]
    return custom + [f"polyglot-{s}" for s in slugs]


def _agent_row(eng, agent, task):
    e = ENGINES[eng]
    if not e["supported"]:
        st, why = X, e["reason"]
    else:
        st, why = (
            U,
            f"{agent} needs {AGENT_NEEDS[agent]}; not verified on {eng} (census pending)",
        )
    return {
        "id": f"agent.{agent}.{eng}.{task}",
        "batch": "B5",
        "sweep": "agent",
        "table": "agent",
        "engine": eng,
        "engine_version": e["version"],
        "model": PRIMARY,
        "agent": agent,
        "task": task,
        "status": st,
        "reason": why,
        "reps": REPS,
        "_work_min": AGENT_RUN_MIN * REPS,
        "_group": None,
    }


# ---------------------------------------------------------------- summary
def _h(rows, status=None):
    return (
        sum(r["est_gpu_min"] for r in rows if status is None or r["status"] == status)
        / 60.0
    )


def summarize(rows, out=sys.stdout):
    w = out.write
    c = Counter(r["status"] for r in rows)
    w(
        f"Designed-grid manifest: {len(rows)} rows  runnable {c[R]}  unverified {c[U]}  unsupported {c[X]}\n\n"
    )
    w(
        "Per engine (rows; runnable / unverified / unsupported; GPU h runnable | unverified | total)\n"
    )
    w(
        "| engine | rows | runnable | unverified | unsupported | h runnable | h unverified | h total |\n"
    )
    w("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for eng in ENGINE_IDS + ["best-competitor (TBD)"]:
        rs = [r for r in rows if r["engine"] == eng]
        if not rs:
            continue
        cc = Counter(r["status"] for r in rs)
        hr, hu = _h(rs, R), _h(rs, U)
        w(
            f"| {eng} | {len(rs)} | {cc[R]} | {cc[U]} | {cc[X]} | {hr:.2f} | {hu:.2f} | {hr + hu:.2f} |\n"
        )
    w("\nPer batch (rows; GPU h runnable | unverified (potential) | total)\n")
    w(
        "| batch | rows | runnable | unverified | unsupported | h runnable | h unverified | h total |\n"
    )
    w("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    tot_r = tot_u = 0.0
    for b in sorted(BATCH_NAMES):
        rs = [r for r in rows if r["batch"] == b]
        cc = Counter(r["status"] for r in rs)
        hr, hu = _h(rs, R), _h(rs, U)
        tot_r += hr
        tot_u += hu
        w(
            f"| {BATCH_NAMES[b]} | {len(rs)} | {cc[R]} | {cc[U]} | {cc[X]} | {hr:.2f} | {hu:.2f} | {hr + hu:.2f} |\n"
        )
    w(
        f"\nGRAND TOTAL: runnable {tot_r:.2f} h; unverified (census pending) {tot_u:.2f} h; "
        f"total as if supported {tot_r + tot_u:.2f} h vs target {TARGET_GPU_H:.0f} h "
        f"({'WITHIN' if tot_r + tot_u <= TARGET_GPU_H else 'OVER by %.2f h' % (tot_r + tot_u - TARGET_GPU_H)})\n"
    )
    return tot_r, tot_u


CUT_ORDER = [
    (
        "B5 agents on engines with no measurement (kapitsa, lmstudio, llama-cpp, mlx-vlm-server)",
        lambda r: (
            r["batch"] == "B5"
            and r["engine"] in ("kapitsa", "lmstudio", "llama-cpp", "mlx-vlm-server")
        ),
    ),
    (
        "B3 Qwen3.5-9B and Flash-Next, all engines",
        lambda r: r["batch"] == "B3" and r["model"] in (Q9, FLASH),
    ),
    ("B3 27B 8-bit and bf16, all engines", lambda r: r["batch"] == "B3"),
    ("B2 256K ctx sweep", lambda r: r["sweep"] == "ctx" and r["ctx"] == "256K"),
    ("B2 conc 16 sweeps", lambda r: r["sweep"] == "conc" and r["conc"] == 16),
    ("B2 out 8192 sweeps", lambda r: r["sweep"] == "out" and r["out_len"] == 8192),
    (
        "B5 opencode on the remaining engines",
        lambda r: r["batch"] == "B5" and r.get("agent") == "opencode",
    ),
    (
        "B5 codex on the remaining engines",
        lambda r: r["batch"] == "B5" and r.get("agent") == "codex",
    ),
    (
        "B1 conc 8 at 64K/128K on non-Yunshu engines",
        lambda r: (
            r["batch"] == "B1"
            and r["conc"] == 8
            and r["ctx"] in ("64K", "128K")
            and r["engine"] != "yunshu"
        ),
    ),
    (
        "B1 conc 8 at 1K-32K on non-Yunshu engines",
        lambda r: r["batch"] == "B1" and r["conc"] == 8 and r["engine"] != "yunshu",
    ),
    (
        "B5 claude on the remaining engines",
        lambda r: r["batch"] == "B5" and r.get("agent") == "claude",
    ),
]


def cut_plan(rows):
    """Cumulative drop plan in CUT_ORDER; each row is dropped at most once. [(label, rows, h_cut, h_left)]."""
    live = [r for r in rows if r["status"] != X]
    left = sum(r["est_gpu_min"] for r in live) / 60.0
    dropped = set()
    plan = []
    for label, pred in CUT_ORDER:
        hit = [i for i, r in enumerate(live) if i not in dropped and pred(r)]
        if not hit:
            continue
        h = sum(live[i]["est_gpu_min"] for i in hit) / 60.0
        dropped.update(hit)
        left -= h
        plan.append((label, len(hit), h, left))
    return plan


def top_cells(rows, n=8):
    live = [r for r in rows if r["status"] != X and r["table"] == "decode"]
    live.sort(key=lambda r: -r["est_gpu_min"])
    return live[:n]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--summary", type=Path, default=None, help="also write the summary text here"
    )
    args = ap.parse_args(argv)
    rows, meta = build_rows()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    buf = []

    class Tee:
        def write(self, s):
            buf.append(s)
            sys.stdout.write(s)

    summarize(rows, out=Tee())
    tot = _h(rows, R) + _h(rows, U)
    if tot > TARGET_GPU_H:
        head = "\nOver target. Cumulative drop plan (lowest value first; each row dropped once):\n"
        head += "| cut | rows | hours cut | hours left |\n|---|---:|---:|---:|\n"
        for label, n, h, left in cut_plan(rows):
            head += f"| {label} | {n} | {h:.2f} | {left:.2f} |\n"
        sys.stdout.write(head)
        buf.append(head)
    top = "\nMost expensive decode cells (per row, reps included):\n"
    for r in top_cells(rows):
        top += f"  {r['est_gpu_min'] / 60:.2f} h  {r['id']}\n"
    sys.stdout.write(top)
    buf.append(top)
    sys.stdout.write(f"\nwrote {args.out} ({len(rows)} rows)\n")
    if args.summary:
        args.summary.write_text("".join(buf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
