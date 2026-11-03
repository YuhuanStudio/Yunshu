#!/usr/bin/env python3
"""Expected-cell manifest for the full competitor comparison (FULLCMP_SPEC, 2026-11-03).

Generates, before any GPU work, every cell of the comparison as one JSONL row:
engine x model x concurrency x context x prompt kind x output length x turns x cache state
x thinking x media x sampling, plus the agent table (agent x engine x agentbench task).

Rows that cannot run are kept, never dropped: status is one of
  runnable                     verified by a cmp-1102 run or a driver that exists for it
  unsupported (reason)         the engine cannot do it (structural: text-only server, n/a headless)
  unverified (census pending)  not yet established either way; never counted as supported

Only combinations that are impossible by construction are PRUNED (not generated). Each
pruning rule has an id, is listed in PRUNE_RULES, and its rejected-candidate count is
printed in the summary. Engine-level gaps are never pruned; they become status rows.

Cost model (est_gpu_min, per cell, all REPS included):
  work_s = prefill_s + decode_s + idle_s
  prefill_s = prefill_tokens / PREFILL_TOK_S        (conservative: cmp-1102 measured Yunshu
              980-1066 tok/s at 4-32K cold and Kapitsa 484-541 tok/s; 500 is below both)
  decode_s  = decode_tokens * DECODE_S_PER_TOK * (1 + conc / 8)
              (DECODE_S_PER_TOK = 1/30 s, a planning value slower than the measured 68-190 tok/s
              for Yunshu at 32K and close to the 29 ms/token 27B roofline at 1K; batched requests
              are charged a mild slowdown)
  est_gpu_min = (REPS * work_s + startup_share_s) / 60, startup_share_s = STARTUP_S split over the
              cells of one process group (full STARTUP_S for cold_process cells, which restart).
Agent rows: AGENT_RUN_MIN per run, REPS runs. Unsupported rows cost 0. Unverified rows carry the
potential cost, and the summary reports runnable and unverified hours separately.

Usage:
  uv run python scripts/research/fullcmp_manifest.py            # write manifest + print summary
  uv run python scripts/research/fullcmp_manifest.py --out X    # other output path
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_OUT = Path(
    "/Volumes/P5Plus/yunshu-build/competitors/compsync/fullcmp-1103/manifest.jsonl"
)
ROOT = Path(__file__).resolve().parents[2]
AGENTIC = ROOT / "scripts/research/agentic"

REPS = 3
PREFILL_TOK_S = 500.0
DECODE_S_PER_TOK = 1.0 / 30.0
STARTUP_S = 90.0
FOLLOWUP_TOKENS = 1024
THINK_TOKENS = 1024
IMAGE_TOKENS = {"none": 0, "img1": 1024, "img4": 4096}
MIXED_SHORT_REQUESTS = 7
MIXED_SHORT_TOKENS = 1024
IDLE_5S = 5.0
AGENT_RUN_MIN = 10.0

RUNNABLE = "runnable"
UNVERIFIED = "unverified (census pending)"
UNSUPPORTED = "unsupported"

CTX = [
    ("1K", 1024),
    ("4K", 4096),
    ("8K", 8192),
    ("16K", 16384),
    ("32K", 32768),
    ("64K", 65536),
    ("128K", 131072),
    ("256K", 262144),
]
CTX_TOKENS = dict(CTX)
OUTS = [128, 512, 2048, 8192]
KINDS = ["code", "prose", "tool-chat", "cjk", "json-schema"]
PROMPTS = [(k, i) for k in KINDS for i in (0, 1)]  # >= 2 prompts per kind
CONC_VARIANTS = [(1, "single")] + [
    (c, v) for c in (2, 4, 8, 16) for v in ("same", "distinct")
]
TURN_CACHE = [
    ("single", "cold_process"),
    ("single", "cold_prompt"),
    ("single", "warm_repeat"),
    ("single", "prefix_hit"),
    ("single", "restart_disk"),
    ("t2_immediate", "follow_up"),
    ("t2_idle5s", "follow_up"),
    ("t10", "follow_up"),
]
TURNS_N = {"single": 1, "t2_immediate": 2, "t2_idle5s": 2, "t10": 10}
THINKING = ["off", "on"]
MEDIA = ["none", "img1", "img4"]
SAMPLING = ["greedy", "genconfig"]
SPOT_8192 = {(131072, "code"), (131072, "prose"), (262144, "code")}  # the 3 spot cells

PRUNE_RULES = {
    "P1": "warm_repeat / prefix_hit are not a dimension of distinct-prompt concurrency "
    "(distinct prompts never hit the prefix cache of each other; only same-prompt and "
    "single requests get warm/prefix states)",
    "P2": "cold_process and restart_disk exist only for a single request in a fresh process "
    "(a concurrent group is one process start; conc>1 cold_process duplicates cold_prompt)",
    "P3": "multi-turn sessions (2-turn immediate, 2-turn idle 5 s, 10-turn) run only at conc 1 "
    "and at conc 8 distinct prompts (session cost bound)",
    "P4": "JSON-schema kind needs output <= 2048 tokens (constrained output at 8192 is not a "
    "real agent workload and dominates the grid)",
    "P5": "8192-token outputs only at ctx <= 32K, plus 3 spot cells (128K code, 128K prose, 256K "
    "code; conc 1, cold prompt, thinking off)",
    "P6": "multi-turn sessions use output 512 or 2048 only (no 8192 turns)",
    "P7": "media (1 and 4 images) runs at conc 1, single turn, greedy, output 512, thinking off, "
    "ctx 1K/8K/32K, code and prose prompt 0 only; a separate family, not a product of all dims",
    "P8": "generation_config sampling runs only for ctx 1K/32K, conc 1 and 8 distinct, output 512, "
    "cold prompt, thinking off (the agent-facing subset from the spec)",
    "P9": "8-bit and bf16 run only the subset ctx 1K/32K, conc 1 and 8 distinct, prompt 0 of each "
    "kind, output 512, cold prompt, thinking off",
    "P10": "Flash-Next and the Qwen3.5 tier-1 model run the core grid only (all ctx and kinds, "
    "output 512/2048, conc 1 and 8 distinct; single turn; cold/warm/prefix states); full cross "
    "product is for the primary 27B 4-bit model",
    "P11": "mixed load (1 x 128K + 7 x 1K) is one row per engine at ctx 128K, prose prompt 0, "
    "output 512; the ctx dimension does not multiply it",
    "P12": "agent rows are one table (agent x engine x agentbench task), not multiplied by the "
    "decode dimensions",
}

# ---------------------------------------------------------------- engines and models
MODELS = {
    "q27-4bit": "Qwen3.8-27B 4-bit (primary)",
    "q27-8bit": "Qwen3.8-27B 8-bit",
    "q27-bf16": "Qwen3.8-27B bf16",
    "flashnext-oq3e": "Qwen3.8-Flash-Next oQ3e-mtp",
    "qwen35-9b-4bit": "Qwen3.5-9B-MLX-4bit (tier-1 family)",
}
PRIMARY = "q27-4bit"
SUBSET_BITS = ["q27-8bit", "q27-bf16"]
SECONDARY = ["flashnext-oq3e", "qwen35-9b-4bit"]

R, U, X = RUNNABLE, UNVERIFIED, UNSUPPORTED
CMP_CTX = {"1K", "32K", "128K"}  # ctx with a cmp-1102 driver for the six cmp engines
YUNSHU_CTX = {"1K", "4K", "16K", "32K", "128K"}


def _feat(**over):
    base = {
        "code_prose": (R, "cmp-1102 driver + r0 (code/prose exact snapshot prompts)"),
        "cjk": (U, "no CJK run yet"),
        "tools": (U, "tool-chat prompts not run in cmp-1102"),
        "json": (U, "JSON-schema constrained output not run in cmp-1102"),
        "media": (U, "image input not run"),
        "thinking": (U, "thinking on/off switch not run"),
        "restart_disk": (U, "no restart-with-disk-tier run"),
        "conc": (U, "concurrency not run in cmp-1102"),
        "ctx256": (U, "256K support not claimed in docs (census pending)"),
        "genconfig": (U, "generation_config sampling not run"),
        "multiturn": (R, "2-turn harness (long32k.py turn2) in cmp-1102"),
        "warm": (R, "warm exact-repeat in cmp-1102 (long32k.py)"),
        "out512": (R, "512-token greedy decode in cmp-1102"),
        "out_other": (U, "output length other than 512 not measured in cmp-1102"),
    }
    base.update(over)
    return base


def _unrun_feat(why):
    return {k: (U, why) for k in _feat()}


CMP_FEAT = _feat()
YUNSHU_FEAT = _feat(
    cjk=(U, "CJK prompts not run yet"),
    tools=(
        R,
        "agent tool-call path present (scripts/research/agentic); validity still per-cell",
    ),
    json=(R, "constrained decoding wired (CLAUDE.md); validity still per-cell"),
    thinking=(R, "reasoning split + Qwen3.8 template (CLAUDE.md)"),
    restart_disk=(R, "SSD prefix tier (CLAUDE.md; yunshu_kv)"),
    out_other=(U, "output lengths other than 512 not measured in cmp-1102"),
)

ENGINES = {
    "yunshu": dict(
        label="Yunshu main (sha recorded at run)",
        version="current local main",
        ctx=YUNSHU_CTX,
        feat=YUNSHU_FEAT,
        models={
            PRIMARY: R,
            "q27-8bit": U,
            "q27-bf16": U,
            "flashnext-oq3e": R,
            "qwen35-9b-4bit": R,
        },
        agents={"claude": R, "codex": R, "opencode": R},
    ),
    "tf-1.2.0": dict(
        label="TensorFold 1.2.0 (tf-native)",
        version="1.2.0",
        ctx=CMP_CTX,
        feat=CMP_FEAT,
        models={PRIMARY: R},
        agents={},
    ),
    "tf-1.1.8": dict(
        label="TensorFold 1.1.8 (tf-native, lone loop)",
        version="1.1.8",
        ctx=CMP_CTX,
        feat=CMP_FEAT,
        models={PRIMARY: R},
        agents={},
    ),
    "splash-latest": dict(
        label="Splash latest",
        version="1.4.1 in cmp-1102; latest to confirm",
        ctx=CMP_CTX,
        feat=CMP_FEAT,
        models={PRIMARY: R},
        agents={},
    ),
    "omlx-stable": dict(
        label="oMLX latest stable",
        version="stable to confirm (v0.7.2 seen in cmp-1102 notes)",
        ctx=CMP_CTX,
        feat=_unrun_feat("oMLX stable not run in cmp-1102 (rc only)"),
        models={PRIMARY: U},
        agents={},
    ),
    "omlx-rc": dict(
        label="oMLX latest rc",
        version="v0.7.3rc1",
        ctx=CMP_CTX,
        feat=CMP_FEAT,
        models={PRIMARY: R},
        agents={},
    ),
    "mlx-serve": dict(
        label="mlx-serve latest",
        version="26.10.2 (+26.10.3-pre to confirm)",
        ctx=CMP_CTX,
        feat=CMP_FEAT,
        models={PRIMARY: R},
        agents={},
    ),
    "mlx-lm-server": dict(
        label="mlx_lm.server (stock, latest PyPI)",
        version="latest PyPI (census pending)",
        ctx=CMP_CTX,
        feat=_unrun_feat("mlx_lm.server not run in cmp-1102"),
        feat_override={
            "media": (X, "mlx_lm.server is text-only (no image input path)")
        },
        models={PRIMARY: U},
        agents={},
    ),
    "mlx-vlm-server": dict(
        label="mlx-vlm server (stock, latest PyPI)",
        version="latest PyPI (census pending)",
        ctx=CMP_CTX,
        feat=_unrun_feat("mlx-vlm server not run in cmp-1102"),
        models={PRIMARY: U},
        agents={},
    ),
    "llama-cpp": dict(
        label="llama.cpp llama-server (Metal, Q4_K_M GGUF of same base)",
        version="b11848 in cmp-1102; latest release to confirm",
        ctx=CMP_CTX,
        feat=_unrun_feat("llama-server not run in cmp-1102; GGUF not built"),
        models={PRIMARY: U},
        agents={},
    ),
    "lmstudio": dict(
        label="LM Studio (headless lms, same MLX checkpoint)",
        version="census pending",
        ctx=CMP_CTX,
        feat=_unrun_feat("headless lms with the same checkpoint not verified"),
        models={PRIMARY: U},
        agents={},
    ),
    "ollama": dict(
        label="Ollama",
        version="census pending",
        ctx=CMP_CTX,
        feat=_unrun_feat("n/a"),
        models={m: X for m in MODELS},
        model_reason=(
            "n/a: Ollama serves GGUF via llama.cpp and cannot run the same MLX "
            "checkpoint headless (the GGUF baseline is the llama-cpp row)"
        ),
        agents={},
    ),
    "kapitsa": dict(
        label="Kapitsa main (CMake Release)",
        version="0.1.39 main 8b190f5",
        ctx=CMP_CTX,
        feat=CMP_FEAT,
        models={PRIMARY: R},
        agents={},
    ),
}
ENGINE_IDS = list(ENGINES)
AGENTS = ["claude", "codex", "opencode"]
AGENT_NEEDS = {
    "claude": "Anthropic Messages API",
    "codex": "Responses API",
    "opencode": "OpenAI-compatible chat API",
}


def model_status(eng_id, model):
    e = ENGINES[eng_id]
    st = e["models"].get(model, U)
    if st == X:
        return X, e.get("model_reason", "unsupported model for this engine")
    if st == R:
        return R, "verified by cmp-1102 or the engine's tier/support record"
    return U, f"{model} on {eng_id} not run yet"


def resolve(
    eng_id,
    model,
    ctx_label,
    variant,
    conc,
    kind,
    out,
    turns,
    cache,
    thinking,
    media,
    sampling,
):
    """Return (status, reason) for one decode cell; first non-runnable check wins."""
    e = ENGINES[eng_id]
    st, why = model_status(eng_id, model)
    if st == X:
        return st, why
    feat = dict(e["feat"])
    feat.update(e.get("feat_override", {}))
    checks = []
    ctx_gap = None
    if ctx_label == "256K":
        checks.append("ctx256")
    elif ctx_label not in e["ctx"]:
        ctx_gap = f"ctx {ctx_label} not in the cmp-1102 driver set for {eng_id} (census pending)"
    if conc > 1 or variant == "mixed":
        checks.append("conc")
    if kind == "tool-chat":
        checks.append("tools")
    elif kind == "json-schema":
        checks.append("json")
    elif kind == "cjk":
        checks.append("cjk")
    else:
        checks.append("code_prose")
    if out != 512:
        checks.append("out_other")
    else:
        checks.append("out512")
    if turns != "single":
        checks.append("multiturn")
    if cache in ("warm_repeat", "prefix_hit"):
        checks.append("warm")
    if cache == "restart_disk":
        checks.append("restart_disk")
    if thinking == "on":
        checks.append("thinking")
    if media != "none":
        checks.append("media")
    if sampling == "genconfig":
        checks.append("genconfig")
    for key in checks:  # structural X (e.g. text-only media) outranks an unrun model
        s, r = feat[key]
        if s == X:
            return s, r
    if st != R:
        return st, why
    if ctx_gap:
        return U, ctx_gap
    for key in checks:
        s, r = feat[key]
        if s != R:
            return s, r
    return R, "all checks verified for this engine and model"


def est_gpu_min(
    variant,
    conc,
    ctx_tokens,
    out,
    turns,
    cache,
    thinking,
    media,
    sampling,
    shared_startup=True,
):
    """Return the est_gpu_min (with REPS, startup share excluded; caller adds startup)."""
    n = TURNS_N[turns]
    if variant == "mixed":
        pre = ctx_tokens + MIXED_SHORT_REQUESTS * MIXED_SHORT_TOKENS
        prefill_s = pre / PREFILL_TOK_S
    elif cache == "warm_repeat":
        prefill_s = 0.5
    elif cache == "prefix_hit":
        prefill_s = FOLLOWUP_TOKENS * conc / PREFILL_TOK_S
    elif cache == "restart_disk":
        prefill_s = ctx_tokens * 0.25 * conc / PREFILL_TOK_S
    else:  # cold_process, cold_prompt, follow_up (turn 1 is cold)
        prefill_s = ctx_tokens * conc / PREFILL_TOK_S
    if n > 1:
        prefill_s += (n - 1) * FOLLOWUP_TOKENS * conc / PREFILL_TOK_S
    prefill_s += IMAGE_TOKENS[media] * conc / PREFILL_TOK_S
    dec_tokens = out * n + (THINK_TOKENS * n if thinking == "on" else 0)
    decode_s = dec_tokens * DECODE_S_PER_TOK * (1 + conc / 8)
    idle_s = IDLE_5S * (n - 1) if turns == "t2_idle5s" else 0.0
    return REPS * (prefill_s + decode_s + idle_s) / 60.0


def _decode_cell(
    eng,
    model,
    ctx_label,
    variant,
    conc,
    kind,
    pidx,
    out,
    turns,
    cache,
    thinking,
    media,
    sampling,
    family,
):
    ctx_tokens = CTX_TOKENS.get(ctx_label, 131072)
    st, why = resolve(
        eng,
        model,
        ctx_label,
        variant,
        conc,
        kind,
        out,
        turns,
        cache,
        thinking,
        media,
        sampling,
    )
    cid = (
        f"{eng}.{model}.{ctx_label}.c{conc}-{variant}.{kind}{pidx}.o{out}.{turns}."
        f"{cache}.th-{thinking}.{media}.{sampling}"
    )
    return {
        "id": cid,
        "table": "decode",
        "family": family,
        "engine": eng,
        "engine_version": ENGINES[eng]["version"],
        "model": model,
        "ctx": ctx_label,
        "ctx_tokens": ctx_tokens,
        "conc": conc,
        "variant": variant,
        "prompt_kind": kind,
        "prompt_idx": pidx,
        "out_len": out,
        "turns": turns,
        "cache_state": cache,
        "thinking": thinking,
        "media": media,
        "sampling": sampling,
        "status": st,
        "reason": why,
        "_work": est_gpu_min(
            variant, conc, ctx_tokens, out, turns, cache, thinking, media, sampling
        ),
        "_cold_process": cache == "cold_process",
        "_group": (eng, model, ctx_label, variant, conc),
    }


def _prune_ok(ctx_label, conc, variant, kind, out, turns, cache, stats):
    """Apply P1-P6 to a primary-grid candidate; return True when the cell is kept."""
    if cache in ("warm_repeat", "prefix_hit") and variant == "distinct":
        stats["P1"] += 1
        return False
    if cache in ("cold_process", "restart_disk") and conc != 1:
        stats["P2"] += 1
        return False
    if turns != "single" and not (conc == 1 or (conc == 8 and variant == "distinct")):
        stats["P3"] += 1
        return False
    if kind == "json-schema" and out > 2048:
        stats["P4"] += 1
        return False
    if out == 8192 and CTX_TOKENS[ctx_label] > 32768:
        if (
            (ctx_label, kind)
            not in {("128K", "code"), ("128K", "prose"), ("256K", "code")}
            or conc != 1
            or variant != "single"
            or cache != "cold_prompt"
        ):
            stats["P5"] += 1
            return False
    if turns != "single" and out == 8192:
        stats["P6"] += 1
        return False
    return True


def _primary_grid(eng, model, stats):
    for ctx_label, _ in CTX:
        for conc, variant in CONC_VARIANTS:
            combos = TURN_CACHE
            for kind, pidx in PROMPTS:
                for out in OUTS:
                    for thinking in THINKING:
                        for turns, cache in combos:
                            if not _prune_ok(
                                ctx_label, conc, variant, kind, out, turns, cache, stats
                            ):
                                continue
                            yield _decode_cell(
                                eng,
                                model,
                                ctx_label,
                                variant,
                                conc,
                                kind,
                                pidx,
                                out,
                                turns,
                                cache,
                                thinking,
                                "none",
                                "greedy",
                                "core",
                            )


def _secondary_grid(eng, model):
    for ctx_label, _ in CTX:
        for conc, variant in [(1, "single"), (8, "distinct")]:
            caches = (
                [
                    ("single", "cold_prompt"),
                    ("single", "warm_repeat"),
                    ("single", "prefix_hit"),
                ]
                if variant == "single"
                else [("single", "cold_prompt")]
            )
            for kind, pidx in PROMPTS:
                for out in (512, 2048):
                    if kind == "json-schema" and out > 2048:
                        continue
                    for turns, cache in caches:
                        yield _decode_cell(
                            eng,
                            model,
                            ctx_label,
                            variant,
                            conc,
                            kind,
                            pidx,
                            out,
                            turns,
                            cache,
                            "off",
                            "none",
                            "greedy",
                            "secondary-core",
                        )


def _subset_bits(eng, model):
    for ctx_label in ("1K", "32K"):
        for conc, variant in [(1, "single"), (8, "distinct")]:
            for kind, pidx in [(k, 0) for k in KINDS]:
                yield _decode_cell(
                    eng,
                    model,
                    ctx_label,
                    variant,
                    conc,
                    kind,
                    pidx,
                    512,
                    "single",
                    "cold_prompt",
                    "off",
                    "none",
                    "greedy",
                    "bits-subset",
                )


def _media(eng):
    for ctx_label in ("1K", "8K", "32K"):
        for kind in ("code", "prose"):
            for media in ("img1", "img4"):
                yield _decode_cell(
                    eng,
                    PRIMARY,
                    ctx_label,
                    "single",
                    1,
                    kind,
                    0,
                    512,
                    "single",
                    "cold_prompt",
                    "off",
                    media,
                    "greedy",
                    "media",
                )


def _sampling(eng):
    for ctx_label in ("1K", "32K"):
        for conc, variant in [(1, "single"), (8, "distinct")]:
            for kind, pidx in PROMPTS:
                yield _decode_cell(
                    eng,
                    PRIMARY,
                    ctx_label,
                    variant,
                    conc,
                    kind,
                    pidx,
                    512,
                    "single",
                    "cold_prompt",
                    "off",
                    "none",
                    "genconfig",
                    "sampling-subset",
                )


def _mixed(eng):
    yield _decode_cell(
        eng,
        PRIMARY,
        "128K",
        "mixed",
        8,
        "prose",
        0,
        512,
        "single",
        "cold_prompt",
        "off",
        "none",
        "greedy",
        "mixed-load",
    )


def _agent_cells(eng, tasks):
    for agent in AGENTS:
        for task in tasks:
            st, why = _agent_status(eng, agent)
            work = AGENT_RUN_MIN * REPS
            cid = f"agent.{agent}.{eng}.{task}"
            yield {
                "id": cid,
                "table": "agent",
                "family": "agent",
                "engine": eng,
                "engine_version": ENGINES[eng]["version"],
                "model": PRIMARY,
                "agent": agent,
                "task": task,
                "status": st,
                "reason": why,
                "_work": work,
                "_cold_process": False,
                "_group": None,
            }


def _agent_status(eng, agent):
    e = ENGINES[eng]
    st = e["models"].get(PRIMARY, U)
    if st == X:
        return X, e.get("model_reason", "unsupported")
    if st != R:
        return U, f"no run of {eng} for {PRIMARY} yet (census pending)"
    s = e["agents"].get(agent, U)
    if s == R:
        return R, f"{agent} against {AGENT_NEEDS[agent]} verified (census)"
    return (
        U,
        f"{agent} needs {AGENT_NEEDS[agent]}; not verified for {eng} (census pending)",
    )


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


def batch_of(r):
    if r["table"] == "agent":
        return "B5 agents (last)"
    if (
        r["model"] == PRIMARY
        and r["conc"] in (1, 8)
        and r["out_len"] in (512, 2048)
        and r["media"] == "none"
        and r["sampling"] == "greedy"
        and r["turns"] == "single"
    ):
        return "B1 27B-4bit conc 1/8 x all ctx x 512/2048 (importance first)"
    if (
        r["model"] == PRIMARY
        and r["turns"] == "single"
        and r["media"] == "none"
        and r["sampling"] == "greedy"
    ):
        return "B2 27B-4bit remaining single-turn (conc 2/4/16, out 128/8192)"
    if r["model"] == PRIMARY:
        return "B3 27B-4bit multi-turn, media, generation_config, mixed load"
    return "B4 secondary: Flash-Next, Qwen3.5-9B, 8-bit/bf16 subset"


def build_rows(engines=None, models=None, ctx_labels=None, with_agents=True):
    """Return (rows, prune_counts). Deterministic: same args -> same rows in same order."""
    engines = list(engines or ENGINE_IDS)
    models = list(models or MODELS)
    stats = Counter()
    rows = []
    tasks = agent_tasks() if with_agents else []
    for eng in engines:
        for model in models:
            if model == PRIMARY:
                gen = list(_primary_grid(eng, model, stats))
                gen += list(_media(eng)) + list(_sampling(eng)) + list(_mixed(eng))
            elif model in SUBSET_BITS:
                gen = list(_subset_bits(eng, model))
            else:
                gen = list(_secondary_grid(eng, model))
            if ctx_labels is not None:
                gen = [r for r in gen if r["ctx"] in ctx_labels]
            rows.extend(gen)
        if with_agents:
            rows.extend(_agent_cells(eng, tasks))
    # startup shares: each process group pays STARTUP_S once, split over its cells
    groups = defaultdict(int)
    for r in rows:
        if (
            r["table"] == "decode"
            and not r["_cold_process"]
            and r["status"] != UNSUPPORTED
        ):
            groups[r["_group"]] += 1
    for r in rows:
        work = r.pop("_work")
        cold = r.pop("_cold_process")
        group = r.pop("_group")
        if r["status"] == UNSUPPORTED:
            r["est_gpu_min"] = 0.0
        else:
            if r["table"] == "agent":
                startup = 0.0
            elif cold:
                startup = STARTUP_S
            else:
                startup = STARTUP_S / groups[group]
            r["est_gpu_min"] = round(work + startup / 60.0, 3)
        r["batch"] = batch_of(r)
    return rows, stats


def _pct(a, b):
    return f"{100.0 * a / b:.0f}%" if b else "-"


def summarize(rows, stats, out=sys.stdout):
    w = out.write
    total = len(rows)
    by_status = Counter(r["status"] for r in rows)
    w(f"Expected-cell manifest: {total} rows\n")
    w(
        f"  runnable {by_status[RUNNABLE]}  unverified {by_status[UNVERIFIED]}  "
        f"unsupported {by_status[UNSUPPORTED]}\n\n"
    )
    w(
        "Per engine (rows; runnable / unverified / unsupported; est GPU h runnable, unverified)\n"
    )
    w(
        "| engine | rows | runnable | unverified | unsupported | est h runnable | est h unverified |\n"
    )
    w("|---|---:|---:|---:|---:|---:|---:|\n")
    for eng in ENGINE_IDS:
        rs = [r for r in rows if r["engine"] == eng]
        c = Counter(r["status"] for r in rs)
        h_run = sum(r["est_gpu_min"] for r in rs if r["status"] == RUNNABLE) / 60
        h_unv = sum(r["est_gpu_min"] for r in rs if r["status"] == UNVERIFIED) / 60
        w(
            f"| {eng} | {len(rs)} | {c[RUNNABLE]} | {c[UNVERIFIED]} | {c[UNSUPPORTED]} | "
            f"{h_run:.1f} | {h_unv:.1f} |\n"
        )
    w("\nPer dimension (runnable / unverified / unsupported counts)\n")
    dec = [r for r in rows if r["table"] == "decode"]
    for dim in (
        "model",
        "ctx",
        "conc",
        "variant",
        "prompt_kind",
        "out_len",
        "turns",
        "cache_state",
        "thinking",
        "media",
        "sampling",
    ):
        vals = sorted({str(r[dim]) for r in dec})
        parts = []
        for v in vals:
            rs = [r for r in dec if str(r[dim]) == v]
            c = Counter(r["status"] for r in rs)
            parts.append(f"{v}: {c[RUNNABLE]}/{c[UNVERIFIED]}/{c[UNSUPPORTED]}")
        w(f"  {dim}: " + "; ".join(parts) + "\n")
    ag = [r for r in rows if r["table"] == "agent"]
    c = Counter(r["status"] for r in ag)
    w(
        f"  agent rows: {len(ag)} (runnable {c[RUNNABLE]}, unverified {c[UNVERIFIED]}, "
        f"unsupported {c[UNSUPPORTED]})\n"
    )
    w(
        "\nPruned (impossible by construction, never generated). P1-P6 show rejected candidates "
        "on the primary grid; P7-P12 are family definitions (structural, no candidate count).\n"
    )
    for rid in sorted(PRUNE_RULES):
        count = str(stats[rid]) if rid in stats else "structural"
        w(f"  {rid}: {count}  {PRUNE_RULES[rid]}\n")
    run_h = sum(r["est_gpu_min"] for r in rows if r["status"] == RUNNABLE) / 60
    unv_h = sum(r["est_gpu_min"] for r in rows if r["status"] == UNVERIFIED) / 60
    w(
        f"\nGPU hours (est, reps x{REPS} included): runnable {run_h:.1f} h; "
        f"unverified potential {unv_h:.1f} h; total if all run {run_h + unv_h:.1f} h\n"
    )
    w("\nProposed batch order (importance first; each batch audited before the next)\n")
    w("| batch | rows | runnable h | unverified potential h |\n|---|---:|---:|---:|\n")
    for b in sorted({r["batch"] for r in rows}):
        rs = [r for r in rows if r["batch"] == b]
        rh = sum(r["est_gpu_min"] for r in rs if r["status"] == RUNNABLE) / 60
        uh = sum(r["est_gpu_min"] for r in rs if r["status"] == UNVERIFIED) / 60
        w(f"| {b} | {len(rs)} | {rh:.1f} | {uh:.1f} |\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    rows, stats = build_rows()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    summarize(rows, stats)
    print(f"\nwrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
