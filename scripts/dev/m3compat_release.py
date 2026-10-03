"""Same-checkpoint release receipts; execute GPU arms only through gpuq."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

ART = Path("/Volumes/P5Plus/yunshu-build/codex/m3compat2")
ROOT = Path(__file__).resolve().parents[2]
MODEL = "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
DRAFT = "/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2"


def server(args):
    import mlx.core as mx
    import mlx.nn as nn

    from yunshu_engine.kernels import batch_invariant, ragged_attention
    from yunshu_engine.kernels.tensorfold import lane_qmm
    from yunshu_engine.vlm_batch_runner import VLMBatchRunner

    counts = {}
    modules = {}
    original_linear = batch_invariant.invariant_linear

    def tracked(linear, x, fallback):
        def fallback_receipt(layer, value):
            if isinstance(layer, nn.Linear) and value.ndim == 3 and value.shape[1] > 1:
                key = str(
                    (
                        str(layer.weight.dtype),
                        tuple(layer.weight.shape),
                        tuple(value.shape),
                    )
                )
                counts[key] = counts.get(key, 0) + 1
            return fallback(layer, value)

        return original_linear(linear, x, fallback_receipt)

    batch_invariant.invariant_linear = tracked
    original = VLMBatchRunner.iter_tokens

    def tokens(self, *a, **kw):
        if not modules:
            for _, layer in self.model.language_model.named_modules():
                kind = type(layer).__name__
                modules[kind] = modules.get(kind, 0) + 1
            print("m3compat_modules " + json.dumps(modules), flush=True)
        ids = []
        before = dict(counts)
        try:
            for token in original(self, *a, **kw):
                ids.append(int(token))
                yield token
        finally:
            delta = {
                k: v - before.get(k, 0)
                for k, v in counts.items()
                if v != before.get(k, 0)
            }
            print(
                "m3compat_receipt " + json.dumps({"ids": ids, "fp_exact_rows": delta}),
                flush=True,
            )

    VLMBatchRunner.iter_tokens = tokens
    probes = {
        "device": mx.device_info(),
        "lane_ready": lane_qmm.ready()
        if hasattr(lane_qmm, "ready")
        else "main-compile-only",
    }
    if hasattr(ragged_attention, "tile_ready"):
        probes["tile_ready"] = ragged_attention.tile_ready()
    print("m3compat_probes " + json.dumps(probes), flush=True)
    sys.argv = ["yunshu", "serve", "-m", args.model, "--port", str(args.port)]
    runpy.run_module("yunshu_cli", run_name="__main__")


def read_receipts(log, expected):
    receipts = [
        json.loads(line.split(" ", 1)[1])
        for line in log.splitlines()
        if line.startswith("m3compat_receipt ")
    ]
    if len(receipts) != expected or not receipts[-1]["ids"]:
        raise RuntimeError("missing or empty committed-token receipt")
    return receipts[-1]


def cases(smoke, smoke_tokens=24):
    sys.path.insert(0, str(ROOT / "scripts/research"))
    import tfbench as tf

    if smoke:
        return [
            (
                "smoke",
                tf.req("", "Write a Python function to sum integers.", smoke_tokens),
            )
        ]
    result = [
        (f"{kind}-{ctx}", tf.req("", tf.load_prompt(f"{kind}-{ctx}"), 256))
        for ctx in (1024, 8192, 32768)
        for kind in ("prose", "code")
    ]
    for parent, name in (
        (tf.BODIES, "0002-req.json"),
        (tf.BODIES, "0004-req.json"),
        (tf.BODIES, "0006-req.json"),
        (tf.BODIES2, "0003-req.json"),
    ):
        body = json.loads((parent / name).read_text())
        body.update(temperature=0, seed=1234)
        result.append((parent.parent.name + "/" + name, body))
    return result


def arm(args):
    sys.path.insert(0, str(ROOT / "scripts/research"))
    import tfbench as tf

    source = ART / "python" if args.arm == "main" else ROOT / "python"
    tf.OUT = ART / "servers"
    tf.YUNSHU_SRC = str(source)
    # Srv handles startup, isolated HOME, mode validation and owned process cleanup.
    original_popen = tf.subprocess.Popen

    def popen(cmd, **kw):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--server",
            "--model",
            args.model,
            "--port",
            cmd[-1],
        ]
        kw["env"].update(
            TMPDIR=str(ART / "scratch"),
            UV_CACHE_DIR="/Volumes/P5Plus/yunshu-test-cache/uv",
        )
        return original_popen(cmd, **kw)

    tf.subprocess.Popen = popen
    env = {
        "YUNSHU_AUTH_DISABLED": "1",
        "YUNSHU_VLM_DRAFT": DRAFT if args.mode == "dflash" else "mtp",
        "YUNSHU_VLM_APC_MEMORY_GB": "0",
        "YUNSHU_VLM_APC_DISK": "0",
    }
    s = tf.Srv(
        "yunshu",
        env,
        f"m3compat-{args.arm}-{args.mode}-r{args.rep}-{args.suffix}",
        args.model,
    )
    rows = []
    try:
        for _ in range(2):
            tf.send(s.url, tf.req(s.model, "Say hi.", 24))
        for name, body in cases(args.smoke, args.smoke_tokens):
            body["model"] = s.model
            row = tf.send(s.url, body)
            row.pop("_text")
            log = s.log.read_text(errors="replace")
            receipt = read_receipts(log, 3 + len(rows))
            row.update(cell=name, **receipt)
            row["token_sha256"] = hashlib.sha256(
                json.dumps(row["ids"]).encode()
            ).hexdigest()
            rows.append(row)
            print(
                json.dumps(
                    {
                        "cell": name,
                        "dec_tps": row["dec_tps"],
                        "sha": row["token_sha256"],
                    }
                ),
                flush=True,
            )
        s.verify_spec_mode()
        log = s.log.read_text(errors="replace")
        probes = [
            json.loads(l.split(" ", 1)[1])
            for l in log.splitlines()
            if l.startswith("m3compat_probes ")
        ][-1]
        modules = [
            json.loads(line.split(" ", 1)[1])
            for line in log.splitlines()
            if line.startswith("m3compat_modules ")
        ][-1]
        result = dict(
            complete=True,
            modules=modules,
            arm=args.arm,
            mode=args.mode,
            rep=args.rep,
            model=args.model,
            source=str(source),
            engaged_mode=s.engaged_spec_mode,
            probes=probes,
            rows=rows,
            log=str(s.log),
        )
        args.out.write_text(json.dumps(result, indent=2))
    finally:
        s.kill()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-tokens", type=int, default=24)
    p.add_argument("--arm", choices=["main", "compat"], default="compat")
    p.add_argument("--mode", choices=["dflash", "mtp"], default="mtp")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--port", type=int, default=18998)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--suffix", default="a")
    p.add_argument("--out", type=Path, default=ART / "smoke.json")
    a = p.parse_args()
    if a.server:
        server(a)
    elif a.dry_run:
        assert (Path(a.model) / "config.json").is_file()
        print(
            json.dumps(
                {"complete": "dry-run", "cells": [name for name, _ in cases(a.smoke)]}
            )
        )
    else:
        (ART / "scratch").mkdir(exist_ok=True)
        arm(a)


if __name__ == "__main__":
    main()
