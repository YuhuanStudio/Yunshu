"""Flash-Next (qwen4_exp) cold prefill attribution on the real server path.

  fnprefill_profile.py --model DIR --ctx 8192 32768 --out OUT.json     (driver, one gpuq job)
  fnprefill_profile.py --serve ...                                     (internal: profiled server)
  fnprefill_profile.py --serve-plain ...                               (internal: unprofiled, chunk wall only)

Synchronized, nested spans (exclusive time = elapsed minus nested spans) around the qwen4_exp classes and the
quantized projections the prefill actually runs. Profiled walls are perturbed by the synchronization; the plain
server reports each chunk's true wall and the host gap between chunks, and the report scales nothing: it prints
both. Aggregates carry shapes, flops and weight bytes only, never prompts or outputs.
"""

from __future__ import annotations

import collections
import functools
import json
import re
import sys
import time
from pathlib import Path

TFLOPS_REF = 65.0  # bf16 tensor-op reference used by coldttft_categories.py
GBPS_REF = 522.0  # lead-measured M5 Max stream bandwidth
PREFIX = "FNPREFILL_PROFILE "


def category(key: str) -> str:
    """Report category of a span key (CPU only)."""
    head = key.split("@")[0].split(":")[0]
    parent = key.split("@")[1].split(":")[0] if "@" in key else ""
    if key.startswith("gemm"):
        n = int(re.search(r":N(\d+)", key).group(1)) if ":N" in key else 0
        if n >= 100_000:
            return "lm_head"
        if parent.startswith("moe.shared"):
            return "shared expert (projections)"
        if n == 512 and parent.startswith("moe"):
            return "router (gate gemm)"
        if parent.startswith("hyper_connection"):
            return "hyper-connection (gemm)"
        if parent.startswith("ple"):
            return "PLE (gemm)"
        if parent.startswith("gdn"):
            return "GDN projections"
        if parent.startswith(("attention", "qsa")):
            return "attention projections"
        return "other gemm"
    table = {
        "moe.experts": "expert gather_qmm",
        "moe.remainder": "router softmax/top-k/combine",
        "moe.shared_expert": "shared expert (swiglu)",
        "gdn.core": "GDN core",
        "gdn.conv": "GDN conv",
        "gdn.norm_gated": "GDN gated norm",
        "gdn.remainder": "GDN remainder",
        "attention.core": "attention core (QSA/SDPA)",
        "attention.remainder": "attention remainder",
        "qsa.indexer": "QSA indexer",
        "ple.ssd_read": "PLE SSD read",
        "ple.lookup": "PLE lookup",
        "ple.layer": "PLE remainder",
        "hyper_connection": "hyper-connection remainder",
        "norm": "norms",
        "embed": "embedding",
    }
    for prefix, name in table.items():
        if head == prefix or head.startswith(prefix + "."):
            return name
    return "other (" + head + ")"


def aggregate(lines: list[dict]) -> dict:
    """Per-category ms per chunk, share, achieved TFLOP/s and GB/s from FNPREFILL profile rows."""
    chunks = [r for r in lines if r.get("method") == "prefill-chunk" and r.get("rows")]
    if not chunks:
        raise RuntimeError("no profiled prefill chunks")
    cats: dict[str, dict] = collections.defaultdict(
        lambda: dict(excl_s=0.0, flops=0.0, bytes=0.0, count=0)
    )
    for r in chunks:
        for key, v in r["spans"].items():
            c = cats[category(key)]
            c["excl_s"] += v["exclusive_s"]
            c["flops"] += v.get("flops", 0.0)
            c["bytes"] += v.get("bytes", 0.0)
            c["count"] += v["count"]
    n = len(chunks)
    total = sum(c["excl_s"] for c in cats.values())
    wall = sum(r["wall_s"] for r in chunks)
    rows = []
    for name, c in sorted(cats.items(), key=lambda p: -p[1]["excl_s"]):
        s = c["excl_s"]
        rows.append(
            dict(
                category=name,
                ms_per_chunk=round(1000 * s / n, 2),
                share=round(s / total, 4) if total else 0.0,
                tflops=round(c["flops"] / s / 1e12, 2) if s and c["flops"] else None,
                gbps=round(c["bytes"] / s / 1e9, 1) if s and c["bytes"] else None,
                calls_per_chunk=round(c["count"] / n, 1),
            )
        )
    return dict(
        chunks=n,
        profiled_wall_ms_per_chunk=round(1000 * wall / n, 1),
        attributed_ms_per_chunk=round(1000 * total / n, 1),
        unattributed_ms_per_chunk=round(1000 * (wall - total) / n, 1),
        categories=rows,
    )


def install(*, plain: bool = False) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_vlm.generate.ar import PromptProcessingBatch

    active = False
    installed = False
    stack: list[list] = []
    names: list[str] = []
    totals: dict = collections.defaultdict(lambda: [0.0, 0, 0.0, 0.0, 0.0])
    experts_m: list[tuple[int, int]] = []

    def evaluate(value):
        arrays = [v for _, v in tree_flatten(value) if isinstance(v, mx.array)]
        if arrays:
            mx.eval(*arrays)
        mx.synchronize()

    def nbytes(module):
        return sum(v.nbytes for _, v in tree_flatten(module.parameters()))

    def wrap(fn, label, meta=None, name=None):
        @functools.wraps(fn)
        def call(*args, **kwargs):
            if not active:
                return fn(*args, **kwargs)
            plain_args = tuple(a for a in args if not isinstance(a, nn.Module))
            try:
                evaluate((plain_args, kwargs))
            except ValueError:  # inside a compiled function
                return fn(*args, **kwargs)
            key = label(*args, **kwargs) if callable(label) else label
            if "@" not in key and key.startswith("gemm") and names:
                key += "@" + names[-1]
            frame = [0.0]
            stack.append(frame)
            names.append(name or key)
            start = time.perf_counter()
            result = None
            try:
                result = fn(*args, **kwargs)
                evaluate(result)
            finally:
                elapsed = time.perf_counter() - start
                names.pop()
                stack.pop()
                if stack:
                    stack[-1][0] += elapsed
                row = totals[key]
                row[0] += max(0.0, elapsed - frame[0])
                row[1] += 1
                row[2] += elapsed
                if meta is not None and result is not None:
                    try:
                        flops, moved = meta(args, kwargs, result)
                        row[3] += flops
                        row[4] += moved
                    except Exception:  # noqa: BLE001 - metadata never breaks a span
                        pass
            return result

        return call

    def gemm_key(module, x, *rest, **kw):
        rows = 1
        for n in x.shape[:-1]:
            rows *= n
        out = module.weight.shape[0]
        bits = getattr(module, "bits", 16)
        return f"gemm:M{rows}:K{x.shape[-1]}:N{out}:b{bits}"

    def gemm_meta(args, kwargs, result):
        module, x = args[0], args[1]
        rows = 1
        for n in x.shape[:-1]:
            rows *= n
        return 2.0 * rows * x.shape[-1] * module.weight.shape[0], float(nbytes(module))

    def experts_meta(args, kwargs, result):
        module, x, inds = args[0], args[1], args[2]
        tokens = 1
        for n in x.shape[:-1]:
            tokens *= n
        k = inds.shape[-1]
        flat = inds.reshape(-1).tolist()
        unique = len(set(flat))
        experts_m.append((tokens * k, unique))
        total_experts = module.gate_proj.weight.shape[0]
        per_expert = nbytes(module) / total_experts
        d = x.shape[-1]
        inter = module.gate_proj.weight.shape[1]
        flops = 2.0 * tokens * k * 3 * d * inter
        return flops, per_expert * unique

    def hooks():
        nonlocal installed
        if installed:
            return
        installed = True
        from mlx_vlm.models.qwen3_5 import language as l3
        from mlx_vlm.models.qwen3_5_moe import language as lm3
        from mlx_vlm.models.qwen4_exp import language as l4
        from mlx_vlm.models.qwen4_exp import ple_storage

        from yunshu_engine.kernels.lane_linear import LaneLinear

        for cls in (nn.QuantizedLinear, nn.Linear, LaneLinear):
            cls.__call__ = wrap(cls.__call__, gemm_key, gemm_meta)
        mx.fast.rms_norm = wrap(mx.fast.rms_norm, "norm.rms")
        l4.Qwen4ExpRMSNorm.__call__ = wrap(
            l4.Qwen4ExpRMSNorm.__call__, "norm.qwen4", name="norm"
        )
        l4.Qwen4ExpRMSNormGated.__call__ = wrap(
            l4.Qwen4ExpRMSNormGated.__call__, "gdn.norm_gated"
        )
        nn.Conv1d.__call__ = wrap(nn.Conv1d.__call__, "gdn.conv")
        l3.Qwen3_5GatedDeltaNet.__call__ = wrap(
            l3.Qwen3_5GatedDeltaNet.__call__, "gdn.remainder"
        )
        l3.gated_delta_update = wrap(l3.gated_delta_update, "gdn.core")
        l4.Qwen4ExpAttention.__call__ = wrap(
            l4.Qwen4ExpAttention.__call__, "attention.remainder"
        )
        l4.dispatch_qsa_attention = wrap(l4.dispatch_qsa_attention, "attention.core")
        for meth in ("select_from_projected", "build_mask"):
            setattr(
                l4.Qwen4ExpQSAIndexer,
                meth,
                wrap(
                    getattr(l4.Qwen4ExpQSAIndexer, meth),
                    "qsa.indexer." + meth,
                    name="qsa.indexer",
                ),
            )
        lm3.Qwen3_5MoeSparseMoeBlock.__call__ = wrap(
            lm3.Qwen3_5MoeSparseMoeBlock.__call__, "moe.remainder"
        )
        lm3.Qwen3_5MoeMLP.__call__ = wrap(
            lm3.Qwen3_5MoeMLP.__call__, "moe.shared_expert", name="moe.shared_expert"
        )
        from mlx_vlm.models.switch_layers import SwitchGLU

        SwitchGLU.__call__ = wrap(
            SwitchGLU.__call__, "moe.experts", experts_meta, name="moe.experts"
        )
        l4.Qwen4ExpGatedResidual.__call__ = wrap(
            l4.Qwen4ExpGatedResidual.__call__, "hyper_connection"
        )
        l4.Qwen4ExpPLELayer.__call__ = wrap(l4.Qwen4ExpPLELayer.__call__, "ple.layer")
        l4.Qwen4ExpNGramEmbedding.__call__ = wrap(
            l4.Qwen4ExpNGramEmbedding.__call__, "ple.lookup", name="ple.lookup"
        )
        for cls_name in ("QuantizedMMapNGramEmbedding",):
            cls = getattr(ple_storage, cls_name, None)
            if cls is not None and hasattr(cls, "_read_rows"):
                cls._read_rows = wrap(cls._read_rows, "ple.ssd_read")
        nn.Embedding.__call__ = wrap(nn.Embedding.__call__, "embed")

    original = PromptProcessingBatch.prompt_step
    last_end = [None]

    def step(self):
        nonlocal active
        if not plain:
            hooks()
        mx.synchronize()
        totals.clear()
        experts_m.clear()
        start = time.perf_counter()
        gap = None if last_end[0] is None else start - last_end[0]
        processed = self._processed_prompt_columns
        active = not plain
        try:
            result = original(self)
            mx.synchronize()
        finally:
            active = False
        wall = time.perf_counter() - start
        last_end[0] = time.perf_counter()
        ms = [m for m, _ in experts_m]
        uniq = [u for _, u in experts_m]
        print(
            PREFIX
            + json.dumps(
                dict(
                    method="prefill-chunk",
                    start=processed,
                    rows=result,
                    wall_s=wall,
                    host_gap_since_prev_s=gap,
                    plain=plain,
                    expert_rows_per_layer=(sum(ms) / len(ms) if ms else None),
                    experts_touched_per_layer=(sum(uniq) / len(uniq) if uniq else None),
                    spans={
                        k: dict(
                            exclusive_s=v[0],
                            count=v[1],
                            inclusive_s=v[2],
                            flops=v[3],
                            bytes=v[4],
                        )
                        for k, v in totals.items()
                    },
                )
            ),
            flush=True,
        )
        return result

    PromptProcessingBatch.prompt_step = step


def parse_log(text: str) -> list[dict]:
    return [
        json.loads(line.split(PREFIX, 1)[1])
        for line in text.splitlines()
        if PREFIX in line
    ]


def driver(args) -> dict:
    import subprocess

    import tfbench as t

    popen = subprocess.Popen

    def launch(command, *a, **kw):
        if command and Path(command[0]).name == "yunshu":
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--serve-plain" if args.plain else "--serve",
                *command[1:],
            ]
        return popen(command, *a, **kw)

    subprocess.Popen = launch
    out: dict = dict(model=args.model, plain=args.plain, runs={})
    srv = t.Srv(
        "yunshu",
        {
            "YUNSHU_VLM_DRAFT": "off",
            "YUNSHU_VLM_APC_DISK": "0",
            "YUNSHU_VLM_APC_MEMORY_GB": "0",
            "YUNSHU_NEXT_TURN_PREFILL": "off",
            "YUNSHU_CONSOLE": "0",
            **dict(kv.split("=", 1) for kv in args.env),
        },
        "fnprefill-plain" if args.plain else "fnprefill",
        model=args.model,
    )
    try:
        t.send(srv.url, t.req(srv.model, "Say hi.", 8))
        for ctx in args.ctx:
            prompt = t.load_prompt(f"prose-{ctx}")
            start = srv.log.stat().st_size
            row = t.send(srv.url, t.req(srv.model, prompt, 4), timeout=1800)
            text = srv.log.read_bytes()[start:].decode(errors="replace")
            lines = parse_log(text)
            out["runs"][str(ctx)] = dict(
                ttft_s=row["ttft_s"],
                prompt_tokens=row["pt"],
                cached=row["cached"],
                chunks=lines if args.plain else None,
                report=None if args.plain else aggregate(lines),
                chunk_wall_s=[round(r["wall_s"], 4) for r in lines],
                host_gap_s=[
                    None
                    if r["host_gap_since_prev_s"] is None
                    else round(r["host_gap_since_prev_s"], 4)
                    for r in lines
                ],
                expert_rows_per_layer=[r.get("expert_rows_per_layer") for r in lines][
                    :3
                ],
                experts_touched_per_layer=[
                    r.get("experts_touched_per_layer") for r in lines
                ][:3],
            )
            print(
                json.dumps(
                    {
                        ctx: {
                            k: v
                            for k, v in out["runs"][str(ctx)].items()
                            if k != "chunks"
                        }
                    }
                ),
                flush=True,
            )
    finally:
        srv.kill()
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] in (["--serve"], ["--serve-plain"]):
        mode = argv.pop(0)
        install(plain=mode == "--serve-plain")
        from yunshu_cli import main as cli

        sys.argv = [sys.argv[0], *argv]
        cli()
        return 0
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--ctx", type=int, nargs="+", default=[8192, 32768])
    p.add_argument("--plain", action="store_true")
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    result = driver(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
