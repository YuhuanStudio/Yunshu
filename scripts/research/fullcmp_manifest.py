#!/usr/bin/env python3
"""Expected-cell manifest for the full competitor comparison (user-approved scope, 2026-11-03).

Scope (replaces the designed grid; target <= 48 GPU hours including 3 reps, NO soak):
  engines   12 measured/planned engines + ollama (unsupported, zero cost, reason kept)
  B1        core cross, 27B 4-bit, conc 1: ctx {1K,8K,32K,64K,128K} x kind {code,prose} x out 512
            (cold TTFT, decode, warm, immediate follow-up, follow-up decode, memory)
  B2        main sweeps, 27B 4-bit: conc {2,4,8} at 8K/32K code; ctx {4K,16K} new (5 others are B1 cells);
            out {128,2048} at 1K/32K; tool-chat/CJK/JSON-schema at 8K; thinking on at 8K; genconfig at
            1K/32K; idle 5 s at 8K/32K/128K; restart-with-disk-cache at 32K; mixed (128K + 7 x 1K) once;
            images 1 and 4 at 1K
  B4        behaviour checks, once per engine (7 checks, 1 rep, 3 min per engine)
  B5        agents: Claude Code only, on Yunshu + ONE placeholder competitor, every agentbench task x 3 reps
  DROPPED   256K, conc 16, out 8192, secondary models (Flash-Next, Qwen3.5, 8-bit, bf16), soak, Codex, opencode

Row status (never silently dropped):
  unverified            costed; census gives no positive or negative evidence for this cell on this engine
  unsupported (reason)  certain from the census (text-only server for images, no JSON-schema, no
                        Anthropic/Responses route, no disk prefix tier) or no checkpoint (Ollama); zero cost
  cut (reason)          removed by a cut from CUT_ORDER to fit the target; zero cost in the totals, est kept

Cost model (est_gpu_min per row, REPS included, minutes), fitted to measured cells (2026-11-03):
  wall_s(cell) = sum over the phases the cell measures of
      cold     conc * (ctx + img) / PREFILL_TPS + out_eff * SLOW(conc) / DECODE_TPS
      warm     conc * WARM_TTFT_S + out_eff * SLOW / DECODE_TPS
      fu       conc * (out + FOLLOWUP) / PREFILL_TPS + out * SLOW / DECODE_TPS
      ...      (see wall_s)
  est_gpu_min = REPS * (wall_s + ready_s + OVERHEAD_S) / 60
  Each cell is one tfbench invocation, so each row pays its own server start (ready_s) and the
  memory idle + shutdown (OVERHEAD_S). The measured record is scripts/research/fullcmp_cell_timing.json
  (per-number provenance); the 7 complete yunshu B1 cells reproduce it within about 1 s.
  B5 agent rows still share one server per (engine, agent) group.

Every measured constant is in ENGINE_PERF with its source. Engines without a measurement use the Yunshu
curve and are flagged in 'assumed'. Concurrency slowdown: see SLOW_POINTS and slowdown().

Usage:
  uv run python scripts/research/fullcmp_manifest.py            # write manifest + summary
  uv run python scripts/research/fullcmp_manifest.py --out X --summary Y
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
AGENTIC = (
    ROOT / "scripts/research/agentic"
)  # agentbench: tasks.py + tasks/ (the task set)

TARGET_GPU_H = 48.0
REPS = 2  # 2 reps per cell; fullcmp_reps.py adds rep 3 only where the rule says need_rep3
BEHAVIOUR_REPS = 1
BEHAVIOUR_MIN = 3.0  # per engine: the 7 checks plus one boot (ASSUMED budget)
# ASSUMED per-run cap for one Claude Code agentbench run. PERF_TREND 2026-10-23 ran 62 runs in
# 12-minute budget shards; the per-run average was not recorded, so this is an upper bound.
AGENT_RUN_MIN = 12.0
_TIMING_PATH = Path(__file__).with_name("fullcmp_cell_timing.json")
TIMING = json.loads(_TIMING_PATH.read_text())
_PLAN = TIMING["yunshu"]["planning"]
WARM_TTFT_S = float(_PLAN["warm_ttft_s"]["value"])  # measured max + margin (B1 cells)
FOLLOWUP_NEW_TOKENS = 1024  # new user turn appended to the previous reply
THINK_TOKENS = 1024  # extra reasoning tokens with thinking on
IDLE_S = 5.0  # spec: follow-up after 5 s idle
RESTART_RESTORE_FRAC = 0.25  # disk-tier restore as a fraction of cold prefill (ASSUMED)
IMAGE_TOKENS = {
    "none": 0,
    "img1": 1024,
    "img4": 4096,
}  # visual tokens per image (ASSUMED)
MIXED_SHORT_N = 7  # spec: one 128K request plus 7 x 1K requests
MIXED_SHORT_TOKENS = 1024
MIXED_LONG_CTX = 131072
# ASSUMED fallback for engines without a concurrency measurement (user model):
# per-request decode slowdown = 1 + FALLBACK_SLOPE * (conc - 1).
FALLBACK_SLOPE = 0.15

PRIMARY = "q27-4bit"
PRIMARY_NAME = "Qwen3.8-27B 4-bit (Yunshu: oQ4e-mtp; llama-cpp: UD-Q4_K_M GGUF)"

CTX_TOKENS = {
    "1K": 1024,
    "4K": 4096,
    "8K": 8192,
    "16K": 16384,
    "32K": 32768,
    "64K": 65536,
    "128K": 131072,
}
CORE_CTX = ["1K", "8K", "32K", "64K", "128K"]
CORE_KINDS = ["code", "prose"]
# Request families of the B2 kind/thinking/image cells. tfbench runs them (request construction,
# reply validation); fullcmp_audit's family_validity guard requires every such row to be valid.
FAMILY_KINDS = ("tool-chat", "cjk", "json-schema")
IMAGE_COUNT = {"none": 0, "img1": 1, "img4": 4}


def is_family_cell(kind, thinking, media) -> bool:
    """True for a cell whose reply is validated: a request family, thinking on, or image input."""
    return kind in FAMILY_KINDS or thinking == "on" or (media or "none") != "none"

# ---------------------------------------------------------------- measured constants
# Yunshu curve (27B 4-bit, cold prefill tok/s and decode tok/s per ctx). Measured by the B1 cells of
# job 1103-220358 (0f2960047, 2026-11-03; replaced the 0.1.4 snapshot014 curve). The record and its
# per-number provenance are scripts/research/fullcmp_cell_timing.json (planning = worst kind per ctx).
OVERHEAD_S = float(TIMING["overhead_s"]["value"])  # memory idle 30 s + shutdown, per cell
YUNSHU_CURVE = {
    "prefill": {int(k): float(v["value"]) for k, v in _PLAN["prefill_tps"].items() if k[0].isdigit()},
    "decode": {int(k): float(v["value"]) for k, v in _PLAN["decode_tps"].items() if k[0].isdigit()},
}

# Concurrency slowdown: per-request decode time multiplier at conc N vs single request.
# Measured (Yunshu default path, not the round driver). Single-request reference for each ctx:
#   8K and 64K: the snapshot014 curve scaled by 72/100.7 (same-era 32K single from PERF_TREND 2026-10-02).
#   32K: 72 tok/s (PERF_TREND 2026-10-02, single 32K ctx bench 68-76, midpoint).
# Per-request means (PERF_TREND 2026-10-05 'off' arm, the default path at the time):
#   8K c2 25.4/24.9 -> 2.33x; 8K c4 21-23 (22) -> 2.66x; 32K c2 22.5/27.1 -> 2.90x;
#   32K c4 17.6/12.3/19.1/17.6 (16.65) -> 4.32x; 64K c2 17.4/18.0 -> 3.2x.
# c8: PERF_TREND 2026-10-01 probe_concurrency, single 91.1 tok/s, 8-way aggregate 167.0 -> 20.9 per request -> 4.36x
#     (ctx not stated; applied to all ctx).
# The snapshot014 5.9x / 8.3x (c2/c4) is NOT used: it divides a 32K single decode by a 2048-token c2/c4
# run of a different path and does not match the same-era default-path measurements above.
YUNSHU_SLOW_POINTS = {
    2: {8192: 2.33, 32768: 2.90, 65536: 3.2},
    4: {8192: 2.66, 32768: 4.32},
    8: {None: 4.36},
}
# Competitor concurrency: only the snapshot014 measurements at 32K (mlx-lm and oMLX 0.7.0). Same harness
# as the suspect Yunshu c2/c4 numbers, so flagged. Other engines use the fallback.
MLXLM_SLOW_POINTS = {2: {None: 1.19}, 4: {None: 1.60}}
OMLX_SLOW_POINTS = {2: {None: 1.97}, 4: {None: 3.45}}

TF120 = {"prefill": {1024: 962.0, 32768: 854.0}, "decode": {1024: 84.0, 32768: 112.0}}
TF118 = {"prefill": {1024: 959.5, 32768: 847.0}, "decode": {1024: 99.4, 32768: 237.1}}
SPLASH141 = {
    "prefill": {1024: 1192.5, 32768: 1065.0},
    "decode": {1024: 80.2, 32768: 109.4},
}
OMLX_RC = {"prefill": {1024: 772.0, 32768: 886.0}, "decode": {1024: 64.8, 32768: 71.0}}
MLXSERVE = {
    "prefill": {1024: 922.5, 32768: 751.0},
    "decode": {1024: 70.0, 32768: 139.8},
}
MLXLM_CURVE = {
    "prefill": {1024: 869.0, 8192: 936.0, 32768: 838.0, 65536: 716.0, 131072: 577.0},
    "decode": {1024: 29.8, 8192: 28.6, 32768: 26.1, 65536: 23.0, 131072: 19.4},
}
NO_MEASUREMENT = {"prefill": {}, "decode": {}}

PLACEHOLDER = "best-competitor (TBD)"

ENGINE_PERF = {
    "yunshu": dict(
        full=True,
        prefill=YUNSHU_CURVE["prefill"],
        decode=YUNSHU_CURVE["decode"],
        slow=YUNSHU_SLOW_POINTS,
        ready_s=float(_PLAN["ready_s_max"]["value"]),  # max measured ready in the B1 cells
        source="B1 cells, job 1103-220358 (0f2960047; fullcmp_cell_timing.json); "
        "concurrency from PERF_TREND 2026-10-01/05 (default path)",
        assumed=[
            "prefill/decode curve from 1K/32K/64K/128K B1 cells; 4K/8K/16K interpolated",
            "concurrency slowdown is not re-measured on 0f2960047 (PERF_TREND 2026-10-01/05)",
        ],
    ),
    "tf-1.2.1": dict(
        full=False,
        **TF120,
        slow=None,
        ready_s=17.5,
        source="cmp-1102 tf-1.2.0 constants (1.2.1 not measured)",
        assumed=[
            "1.2.1 not measured: 1.2.0 constants",
            "1K code not run in cmp-1102 (prose only)",
        ],
    ),
    "tf-1.1.8": dict(
        full=False,
        **TF118,
        slow="serial",
        ready_s=8.5,
        source="cmp-1102 tf-1.1.8 (v1.1.8)",
        assumed=[
            "census: 'lone loop' build, requests serialized (slowdown = conc)",
            "32K decode 237 tok/s is code-only (draft acceptance)",
        ],
    ),
    "splash-1.5.0": dict(
        full=False,
        **SPLASH141,
        slow=None,
        ready_s=75.9,
        source="cmp-1102 splash 1.4.1 constants (1.5.0 not measured); ready_s 75.9 is the 1.4.1 median",
        assumed=[
            "1.5.0 not measured: 1.4.1 constants",
            "ready_s from 1.4.1 (cold-start heavy)",
        ],
    ),
    "omlx-0.7.3": dict(
        full=False,
        **OMLX_RC,
        slow=OMLX_SLOW_POINTS,
        ready_s=3.3,
        source="cmp-1102 omlx v0.7.3rc1 constants (stable 0.7.3 not measured); slowdown snapshot014 0.7.0",
        assumed=[
            "stable 0.7.3 not measured: rc1 constants",
            "slowdown from oMLX 0.7.0 (snapshot014)",
        ],
    ),
    "omlx-0.7.3rc1": dict(
        full=False,
        **OMLX_RC,
        slow=OMLX_SLOW_POINTS,
        ready_s=3.3,
        source="cmp-1102 omlx-latest (v0.7.3rc1) constants; slowdown snapshot014 0.7.0",
        assumed=["slowdown from oMLX 0.7.0 (snapshot014)"],
    ),
    "mlx-serve-26.11.1": dict(
        full=False,
        **MLXSERVE,
        slow=None,
        ready_s=8.6,
        source="cmp-1102 mlx-serve v26.10.3 constants (26.11.1 not measured)",
        assumed=[
            "26.11.1 not measured: 26.10.3 constants",
            "MTP is default on Metal: not modelled separately",
        ],
    ),
    "mlx-lm-server": dict(
        full=True,
        **MLXLM_CURVE,
        slow=MLXLM_SLOW_POINTS,
        ready_s=2.0,
        source="snapshot014-released-v2 mlxlm (mlx-lm 0.32.0, stock AR); current PyPI 0.32.2",
        assumed=[
            "AR decode measured (no spec mode engaged)",
            "0.32.0 constants, 0.32.2 current",
        ],
    ),
    "mlx-vlm-server": dict(
        full=False,
        **NO_MEASUREMENT,
        slow=None,
        ready_s=6.1,
        source="ASSUMED: no run (mlx-vlm 0.7.11 server); Yunshu curve",
        assumed=["no measurement; Yunshu curve", "ready_s from Yunshu"],
    ),
    "llama-cpp-b12061": dict(
        full=False,
        **NO_MEASUREMENT,
        slow=None,
        ready_s=6.1,
        source="ASSUMED: no run (b12061, UD-Q4_K_M GGUF); Yunshu curve",
        assumed=[
            "no measurement; Yunshu curve",
            "GGUF UD-Q4_K_M is not the MLX checkpoint",
        ],
    ),
    "lmstudio": dict(
        full=False,
        **NO_MEASUREMENT,
        slow=None,
        ready_s=6.1,
        source="ASSUMED: no run (headless lms, not verified); Yunshu curve",
        assumed=["no measurement; Yunshu curve", "ready_s from Yunshu"],
    ),
    "kapitsa-0.2.26": dict(
        full=False,
        **NO_MEASUREMENT,
        slow=None,
        ready_s=6.1,
        source="ASSUMED: no run (kapitsa 0.2.26); Yunshu curve",
        assumed=["no measurement; Yunshu curve", "ready_s from Yunshu"],
    ),
    "ollama": dict(
        full=False,
        **NO_MEASUREMENT,
        slow=None,
        ready_s=0.0,
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
    if not anchors:
        return _loglin(YUNSHU_CURVE[key], ctx)
    if perf["full"]:
        return _loglin(anchors, ctx)
    xs = sorted(anchors)
    if xs[0] <= ctx <= xs[-1]:
        return _loglin(anchors, ctx)
    a = xs[0] if ctx < xs[0] else xs[-1]
    return anchors[a] / _loglin(YUNSHU_CURVE[key], a) * _loglin(YUNSHU_CURVE[key], ctx)


def prefill_tps(eng, ctx_tokens):
    return _curve_value(ENGINE_PERF[eng], "prefill", ctx_tokens)


def decode_tps(eng, ctx_tokens):
    return _curve_value(ENGINE_PERF[eng], "decode", ctx_tokens)


def slowdown(eng, ctx_tokens, conc):
    """Per-request decode time multiplier at concurrency conc (1.0 at conc 1).

    'serial' engines run requests one at a time (slowdown = conc). Measured points are used when
    present (a None ctx key is a ctx-independent value, otherwise the nearest measured ctx). Anything
    else uses the ASSUMED fallback 1 + FALLBACK_SLOPE * (conc - 1)."""
    if conc <= 1:
        return 1.0
    s = ENGINE_PERF[eng]["slow"]
    if s == "serial":
        return float(conc)
    if s and conc in s:
        vals = s[conc]
        if None in vals:
            return vals[None]
        near = min(vals, key=lambda c: abs(math.log2(c) - math.log2(ctx_tokens)))
        return vals[near]
    return 1.0 + FALLBACK_SLOPE * (conc - 1)


# ---------------------------------------------------------------- engines and status
UNV, UNS, CUT = "unverified", "unsupported", "cut"
ENGINES = {
    "yunshu": dict(
        label="Yunshu main 0f2960047 (0.1.7.dev0)",
        version="0.1.7.dev0 @ 0f2960047 (curve from 0.1.4 snapshot014, provisional)",
        supported=True,
        caps=dict(
            images=True,
            json_schema=True,
            anthropic=True,
            responses=True,
            disk_cache=True,
            tools=True,
            concurrency=True,
        ),
    ),
    "tf-1.2.1": dict(
        label="TensorFold 1.2.1 (tf-native, latest)",
        version="1.2.1",
        supported=True,
        caps=dict(anthropic=True, responses=True),
    ),
    "tf-1.1.8": dict(
        label="TensorFold 1.1.8 (tf-native, best release row; lone loop)",
        version="1.1.8",
        supported=True,
        caps=dict(anthropic=True, responses=True, concurrency=False),
    ),
    "splash-1.5.0": dict(
        label="Splash 1.5.0",
        version="1.5.0 (arm64-macos26)",
        supported=True,
        caps=dict(anthropic=True, responses=True),
    ),
    "omlx-0.7.3": dict(
        label="oMLX 0.7.3 (stable)",
        version="0.7.3 (wheel cp313)",
        supported=True,
        caps=dict(anthropic=True, responses=True),
    ),
    "omlx-0.7.3rc1": dict(
        label="oMLX 0.7.3rc1",
        version="0.7.3rc1 (wheel cp313)",
        supported=True,
        caps=dict(anthropic=True, responses=True),
    ),
    "mlx-serve-26.11.1": dict(
        label="mlx-serve 26.11.1",
        version="26.11.1",
        supported=True,
        caps=dict(anthropic=True, responses=True),
    ),
    "mlx-lm-server": dict(
        label="mlx_lm.server (stock)",
        version="0.32.2 (PyPI); 0.32.0 measured",
        supported=True,
        caps=dict(
            images=False,
            json_schema=False,
            anthropic=False,
            responses=False,
            disk_cache=False,
        ),
    ),
    "mlx-vlm-server": dict(
        label="mlx-vlm server (stock)",
        version="0.7.11",
        supported=True,
        caps=dict(images=True, anthropic=True, responses=True),
    ),
    "llama-cpp-b12061": dict(
        label="llama.cpp llama-server b12061 (Metal, UD-Q4_K_M GGUF)",
        version="b12061",
        supported=True,
        caps=dict(anthropic=True, responses=True),
    ),
    "lmstudio": dict(
        label="LM Studio (headless lms)",
        version="0.4.28-2 (not installed; headless unverified)",
        supported=True,
        caps=dict(),
    ),
    "kapitsa-0.2.26": dict(
        label="Kapitsa 0.2.26",
        version="0.2.26",
        supported=True,
        caps=dict(anthropic=True),
    ),
    "ollama": dict(
        label="Ollama 0.40.13",
        version="0.40.13",
        supported=False,
        caps=dict(),
        reason=(
            "n/a: Ollama imports GGUF only (ollama create); it cannot load the MLX checkpoint "
            "headless. Its GGUF path is llama.cpp's runtime (the llama-cpp row)"
        ),
    ),
    PLACEHOLDER: dict(
        label="Best competitor (placeholder, chosen from B1)",
        version="TBD",
        supported=True,
        caps=dict(),
    ),
}
ENGINE_IDS = [e for e in ENGINES if e != PLACEHOLDER]
AGENT_ENGINES = ["yunshu", PLACEHOLDER]
AGENTS = ["claude"]  # Claude Code only (scope 2026-11-03)
BEHAVIOUR_CHECKS = [
    "cancel_latency",
    "stop_sequences",
    "tool_validity",
    "json_validity",
    "logprobs",
    "anthropic_api",
    "responses_api",
]
BATCH_NAMES = {
    "B1": "B1 core cross, 27B 4-bit, conc 1",
    "B2": "B2 main sweeps, 27B 4-bit",
    "B4": "B4 behaviour checks (once per engine)",
    "B5": "B5 agents (Claude Code x agentbench, Yunshu + 1 placeholder)",
}
TOP_ENGINES = {"yunshu", "tf-1.2.1", "splash-1.5.0", "mlx-serve-26.11.1"}


def cell_status(eng, sp):
    """(status, reason) for one decode cell from the census capability flags.

    A flag that is False is a certain unsupported; None or absent is unverified and costed."""
    e = ENGINES[eng]
    if not e["supported"]:
        return UNS, e["reason"]
    caps = e["caps"]
    if sp["media"] != "none" and caps.get("images") is False:
        return UNS, "text-only server: no image input path (census images false)"
    if sp["kind"] == "json-schema" and caps.get("json_schema") is False:
        return UNS, "no JSON-schema constrained output (census json_schema false)"
    if sp["turns"] == "restart" and caps.get("disk_cache") is False:
        return UNS, "no disk prefix-cache tier (census disk_prefix_cache false)"
    if sp["conc"] > 1 and caps.get("concurrency") is False:
        return (
            UNV,
            f"serial engine (census lone loop): conc {sp['conc']} measured as serial",
        )
    if sp["media"] != "none":
        return UNV, "image input unverified (census: VLM path not confirmed)"
    return UNV, "census support not confirmed; cell costed"


# ---------------------------------------------------------------- cells
def spec(**kw):
    base = dict(
        sweep="core",
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


def b1_specs():
    for ctx in CORE_CTX:
        for kind in CORE_KINDS:
            yield "B1", spec(sweep="core", ctx=ctx, kind=kind)


def b2_specs():
    for ctx in ("8K", "32K"):
        for conc in (2, 4, 8):
            yield "B2", spec(sweep="conc", ctx=ctx, conc=conc, phases=("cold",))
    for conc in (2, 4, 8):
        # same prompt on every concurrent stream (prefix sharing between streams), next to the
        # distinct-prompt conc cells above
        yield "B2", spec(sweep="conc-same", ctx="8K", conc=conc, phases=("cold",))
    for ctx in ("4K", "16K"):
        yield "B2", spec(sweep="ctx", ctx=ctx)
    for ctx in ("1K", "32K"):
        for out in (128, 2048):
            yield "B2", spec(sweep="out", ctx=ctx, out=out)
    for kind in ("tool-chat", "cjk", "json-schema"):
        yield "B2", spec(sweep="kind", ctx="8K", kind=kind, phases=("cold",))
    yield "B2", spec(sweep="thinking", ctx="8K", thinking="on", phases=("cold",))
    for ctx in ("1K", "32K"):
        yield (
            "B2",
            spec(sweep="sampling", ctx=ctx, sampling="genconfig", phases=("cold",)),
        )
    for ctx in ("8K", "32K", "128K"):
        yield (
            "B2",
            spec(sweep="idle5", ctx=ctx, turns="idle5", phases=("cold", "fu_idle")),
        )
    yield (
        "B2",
        spec(sweep="restart", ctx="32K", turns="restart", phases=("cold", "restart")),
    )
    yield (
        "B2",
        spec(sweep="mixed", ctx="128K", conc=8, turns="mixed", phases=("mixed",)),
    )
    for media in ("img1", "img4"):
        yield "B2", spec(sweep="image", ctx="1K", media=media, phases=("cold",))


def _cell_row(eng, batch, sp):
    ctx_tokens = CTX_TOKENS[sp["ctx"]]
    status, why = cell_status(eng, sp)
    cid = (
        f"{batch}.{sp['sweep']}.{eng}.{sp['ctx']}.c{sp['conc']}.{sp['kind']}.o{sp['out']}."
        f"{sp['turns']}.th-{sp['thinking']}.{sp['media']}.{sp['sampling']}"
    )
    work = 0.0
    if status != UNS:
        work = (
            REPS
            * wall_s(
                eng,
                ctx_tokens,
                sp["conc"],
                sp["out"],
                sp["thinking"] == "on",
                IMAGE_TOKENS[sp["media"]],
                sp["phases"],
            )
            / 60.0
        )
    return {
        "id": cid,
        "batch": batch,
        "sweep": sp["sweep"],
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
        "_work_min": work,
        "_group": (batch, eng, "restart" if sp["turns"] == "restart" else "main"),
    }


def wall_s(eng, ctx_tokens, conc, out, think, img_tokens, phases):
    """Wall seconds for one rep of one cell (sum of the phases the cell measures)."""
    pf = prefill_tps(eng, ctx_tokens)
    dec = decode_tps(eng, ctx_tokens)
    s = slowdown(eng, ctx_tokens, conc)
    o = out + (THINK_TOKENS if think else 0)
    t = 0.0
    if "cold" in phases:
        t += conc * (ctx_tokens + img_tokens) / pf + o * s / dec
    if "warm" in phases:
        # the warm reply decodes too (measured warm total = warm TTFT + the 512-token reply)
        t += conc * WARM_TTFT_S + o * s / dec
    if "fu" in phases or "fu_idle" in phases:
        t += conc * (out + FOLLOWUP_NEW_TOKENS) / pf + out * s / dec
        if "fu_idle" in phases:
            t += IDLE_S
    if "restart" in phases:
        t += RESTART_RESTORE_FRAC * ctx_tokens / pf + o * s / dec
    if "mixed" in phases:
        short_pf = prefill_tps(eng, 1024)
        t += ctx_tokens / pf + MIXED_SHORT_N * MIXED_SHORT_TOKENS / short_pf
        t += out * s / decode_tps(eng, 1024)
    return t


def _behaviour_row(eng, check):
    e = ENGINES[eng]
    caps = e["caps"]
    st, why = UNV, f"behaviour check {check} not run; census pending"
    if not e["supported"]:
        st, why = UNS, e["reason"]
    elif check == "anthropic_api" and caps.get("anthropic") is False:
        st, why = UNS, "no Anthropic Messages route (census)"
    elif check == "responses_api" and caps.get("responses") is False:
        st, why = UNS, "no Responses route (census)"
    elif check == "json_validity" and caps.get("json_schema") is False:
        st, why = UNS, "no JSON-schema output (census)"
    return {
        "id": f"B4.behaviour.{eng}.{check}",
        "batch": "B4",
        "sweep": "behaviour",
        "table": "behaviour",
        "engine": eng,
        "engine_version": e["version"],
        "model": PRIMARY,
        "check": check,
        "status": st,
        "reason": why,
        "reps": BEHAVIOUR_REPS,
        "_work_min": BEHAVIOUR_MIN / len(BEHAVIOUR_CHECKS) * BEHAVIOUR_REPS
        if st != UNS
        else 0.0,
        "_group": None,
    }


def agent_tasks():
    """Task ids from scripts/research/agentic: custom task dirs + polyglot slugs (the agentbench set)."""
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
        st, why = UNS, e["reason"]
    elif agent == "claude" and e["caps"].get("anthropic") is False:
        st, why = UNS, "Claude Code needs the Anthropic Messages route (census: none)"
    else:
        st, why = UNV, f"{agent} agent run; census pending"
    return {
        "id": f"B5.agent.{agent}.{eng}.{task}",
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
        "_work_min": AGENT_RUN_MIN * REPS if st != UNS else 0.0,
        "_group": ("B5", eng, "agent"),
    }


def build_rows(engines=None, tasks=None):
    """Return (rows, meta). Deterministic: same inputs -> same rows in the same order."""
    engines = list(engines or ENGINE_IDS)
    rows = []
    for eng in engines:
        for batch, sp in list(b1_specs()) + list(b2_specs()):
            rows.append(_cell_row(eng, batch, sp))
    for eng in engines:
        for check in BEHAVIOUR_CHECKS:
            rows.append(_behaviour_row(eng, check))
    task_ids = agent_tasks() if tasks is None else tasks
    for eng in AGENT_ENGINES:
        for agent in AGENTS:
            for task in task_ids:
                rows.append(_agent_row(eng, agent, task))
    # server boots: one server per (batch, engine, server kind) group and rep, shared by its rows
    groups = defaultdict(list)
    for r in rows:
        if r.get("_group") and r["status"] != UNS:
            groups[r["_group"]].append(r)
    for g, members in groups.items():
        eng = g[1]
        if eng not in ENGINE_PERF or ENGINE_PERF[eng]["ready_s"] <= 0:
            continue
        if g[0] != "B5":
            # a cell is one tfbench invocation: it starts its own server (fresh process), then the
            # memory idle and shutdown (OVERHEAD_S). Nothing is shared between cells.
            boot = (ENGINE_PERF[eng]["ready_s"] + OVERHEAD_S) * REPS / 60.0
            for r in members:
                r["_work_min"] += boot
            continue
        boot = ENGINE_PERF[eng]["ready_s"] * REPS / 60.0
        for r in members:
            r["_work_min"] += boot / len(members)
    for r in rows:
        work = r.pop("_work_min")
        r.pop("_group", None)
        r["est_gpu_min"] = 0.0 if r["status"] == UNS else round(work, 3)
    return rows, _meta()


# ---------------------------------------------------------------- smoke manifest
# A small pre-flight matrix (not the approved scope): yunshu only, 1K prompts, each cell at two prompt
# variants so the audit's >= 2 distinct prompts per (model, ctx, kind) guard can pass. The real matrix
# rows never carry prompt_variant; variant 0 keeps its prompt_id unsuffixed.
SMOKE_BATCH = "SMOKE"
SMOKE_ENGINE = "yunshu"
SMOKE_VARIANTS = (0, 1)
SMOKE_REPS = 1
SMOKE_OUT = Path("/Volumes/Micron/yunshu-fullcmp/smoke-manifest.jsonl")


def smoke_specs():
    """1K code/prose out 64 cold+warm+turn2, conc 2 (code/prose) and one profile rep (code)."""
    for kind in CORE_KINDS:
        yield spec(
            sweep="smoke-core",
            ctx="1K",
            kind=kind,
            out=64,
            phases=("cold", "warm", "turn2"),
        )
    for kind in CORE_KINDS:
        yield spec(
            sweep="smoke-conc", ctx="1K", kind=kind, out=64, conc=2, phases=("conc",)
        )
    yield spec(sweep="smoke-profile", ctx="1K", kind="code", out=64, phases=("cold",))


def build_smoke_rows():
    """One row per (spec, variant). Ids end in .v<N>; every smoke row carries prompt_variant."""
    rows = []
    for variant in SMOKE_VARIANTS:
        for sp in smoke_specs():
            row = _cell_row(SMOKE_ENGINE, SMOKE_BATCH, sp)
            row["id"] = f"{row['id']}.v{variant}"
            row["reps"] = SMOKE_REPS
            row["prompt_variant"] = variant
            row["est_gpu_min"] = 0.0
            rows.append(row)
    return rows


def _meta():
    return {
        "constants": {
            "REPS": REPS,
            "BEHAVIOUR_MIN": BEHAVIOUR_MIN,
            "AGENT_RUN_MIN": AGENT_RUN_MIN,
            "WARM_TTFT_S": WARM_TTFT_S,
            "FOLLOWUP_NEW_TOKENS": FOLLOWUP_NEW_TOKENS,
            "THINK_TOKENS": THINK_TOKENS,
            "IDLE_S": IDLE_S,
            "RESTART_RESTORE_FRAC": RESTART_RESTORE_FRAC,
            "IMAGE_TOKENS": IMAGE_TOKENS,
            "MIXED_SHORT_N": MIXED_SHORT_N,
            "MIXED_SHORT_TOKENS": MIXED_SHORT_TOKENS,
            "FALLBACK_SLOPE": FALLBACK_SLOPE,
            "TARGET_GPU_H": TARGET_GPU_H,
        }
    }


# ---------------------------------------------------------------- totals, cuts, summary
def planned_h(rows):
    """GPU hours of rows still planned (unverified), i.e. not unsupported and not cut."""
    return sum(r["est_gpu_min"] for r in rows if r["status"] == UNV) / 60.0


def _h(rows, batch=None, engine=None):
    return (
        sum(
            r["est_gpu_min"]
            for r in rows
            if r["status"] == UNV
            and (batch is None or r["batch"] == batch)
            and (engine is None or r["engine"] == engine)
        )
        / 60.0
    )


# Cuts, applied in this order only while the planned total is over TARGET_GPU_H (each category whole).
CUT_ORDER = [
    (
        "(1) conc 8 sweep at 32K on engines without concurrency support (census: tf-1.1.8 lone loop)",
        lambda r: (
            r["sweep"] == "conc"
            and r["ctx"] == "32K"
            and r["conc"] == 8
            and ENGINES[r["engine"]]["caps"].get("concurrency") is False
        ),
    ),
    (
        "(2) out 2048 at 32K on non-top engines (top = yunshu, tf-1.2.1, splash-1.5.0, mlx-serve)",
        lambda r: (
            r["sweep"] == "out"
            and r["ctx"] == "32K"
            and r["out_len"] == 2048
            and r["engine"] not in TOP_ENGINES
        ),
    ),
    (
        "(3) image cells (img1, img4 at 1K)",
        lambda r: r["sweep"] == "image",
    ),
]


def apply_cuts(rows, out=sys.stdout):
    """Apply CUT_ORDER until the planned total fits. Prints the total after each cut.
    Returns the list of (label, rows_cut, hours_cut, total_after)."""
    applied = []
    total = planned_h(rows)
    out.write(
        f"planned total before cuts: {total:.2f} h (target {TARGET_GPU_H:.0f} h)\n"
    )
    if total <= TARGET_GPU_H:
        out.write("fits without cuts: no cut applied\n")
        return applied
    for label, pred in CUT_ORDER:
        if planned_h(rows) <= TARGET_GPU_H:
            break
        n = 0
        h = 0.0
        for r in rows:
            if r["status"] == UNV and pred(r):
                h += r["est_gpu_min"] / 60.0
                r["reason"] = f"cut to fit {TARGET_GPU_H:.0f} h: {label}"
                r["status"] = CUT
                n += 1
        total = planned_h(rows)
        applied.append((label, n, h, total))
        out.write(f"cut {label}: {n} rows, {h:.2f} h; total now {total:.2f} h\n")
    if planned_h(rows) > TARGET_GPU_H:
        left = planned_h(rows) - TARGET_GPU_H
        out.write(
            f"STILL OVER by {left:.2f} h after all cuts. Ranked proposal (largest planned groups):\n"
        )
        groups = defaultdict(float)
        for r in rows:
            if r["status"] == UNV:
                groups[(r["batch"], r["sweep"], r["engine"])] += r["est_gpu_min"] / 60.0
        for (b, sw, eng), h in sorted(groups.items(), key=lambda kv: -kv[1])[:8]:
            out.write(f"  {h:6.2f} h  {b} {sw} {eng}\n")
    return applied


def summarize(rows, out=sys.stdout):
    w = out.write
    c = Counter(r["status"] for r in rows)
    w(
        f"Full comparison manifest: {len(rows)} rows  unverified {c[UNV]}  "
        f"unsupported {c[UNS]}  cut {c[CUT]}\n\n"
    )
    w(
        "GPU hours by engine and batch (planned rows only; reps and server boots included)\n"
    )
    w("| engine | B1 | B2 | B4 | B5 | total h | rows unv/unsup/cut |\n")
    w("|---|---:|---:|---:|---:|---:|---:|\n")
    for eng in ENGINE_IDS + [PLACEHOLDER]:
        rs = [r for r in rows if r["engine"] == eng]
        if not rs:
            continue
        cc = Counter(r["status"] for r in rs)
        w(
            f"| {eng} | {_h(rs, 'B1'):.2f} | {_h(rs, 'B2'):.2f} | {_h(rs, 'B4'):.2f} | "
            f"{_h(rs, 'B5'):.2f} | {_h(rs):.2f} | {cc[UNV]}/{cc[UNS]}/{cc[CUT]} |\n"
        )
    w("\nPer batch\n| batch | rows | h planned |\n|---|---:|---:|\n")
    for b in BATCH_NAMES:
        rs = [r for r in rows if r["batch"] == b]
        w(f"| {BATCH_NAMES[b]} | {len(rs)} | {_h(rs):.2f} |\n")
    tot = planned_h(rows)
    w(
        f"\nGRAND TOTAL planned {tot:.2f} h vs target {TARGET_GPU_H:.0f} h "
        f"({'WITHIN' if tot <= TARGET_GPU_H else 'OVER by %.2f h' % (tot - TARGET_GPU_H)})\n"
    )
    return tot


def constants_block():
    lines = ["\nConstants and assumptions that matter\n"]
    lines.append(
        f"REPS {REPS}; AGENT_RUN_MIN {AGENT_RUN_MIN} (ASSUMED per-run cap; B5 is the largest batch); "
        f"FALLBACK_SLOPE {FALLBACK_SLOPE} (ASSUMED: 1 + slope*(N-1) for unmeasured concurrency)\n"
    )
    lines.append(
        "Yunshu slowdown (measured, default path): "
        + ", ".join(
            f"c{c} {ctx}: {v}"
            for c, d in YUNSHU_SLOW_POINTS.items()
            for ctx, v in d.items()
        )
        + "\n"
    )
    for eng in ENGINE_IDS:
        p = ENGINE_PERF[eng]
        if eng == "ollama":
            continue
        lines.append(f"- {eng}: {p['source']}; " + "; ".join(p["assumed"]) + "\n")
    return "".join(lines)


def top_cells(rows, n=8):
    live = [r for r in rows if r["status"] == UNV and r["table"] == "decode"]
    live.sort(key=lambda r: -r["est_gpu_min"])
    return live[:n]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--summary", type=Path, default=None, help="also write the summary text here"
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="write the smoke manifest (yunshu, 1K, two prompt variants) instead of the real matrix",
    )
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = SMOKE_OUT if args.smoke else DEFAULT_OUT
    if args.smoke:
        rows = build_smoke_rows()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
        sys.stdout.write(f"wrote {args.out} ({len(rows)} smoke rows)\n")
        return 0
    rows, _meta_unused = build_rows()
    buf = []

    class Tee:
        def write(self, s):
            buf.append(s)
            sys.stdout.write(s)

    apply_cuts(rows, out=Tee())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    summarize(rows, out=Tee())
    Tee().write(constants_block())
    top = "\nMost expensive decode cells (per row, reps included):\n"
    for r in top_cells(rows):
        top += f"  {r['est_gpu_min'] / 60:.2f} h  {r['id']}\n"
    Tee().write(top)
    sys.stdout.write(f"\nwrote {args.out} ({len(rows)} rows)\n")
    if args.summary:
        args.summary.write_text("".join(buf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
