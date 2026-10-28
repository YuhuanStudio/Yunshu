"""Teacher-forced capture for the yv ``kld`` stage: one cold prefill per corpus window, top-K logprobs at every
position, from the engine of whatever source tree PYTHONPATH names.

  capture.py run --arm base|cand|ref --model DIR --cache-dir D --shard k/n [--env K=V ...] --out F.jsonl
  capture.py serve ARM <yunshu serve args>        (internal: the server process with the capture hooks)

Hooks (installed before the CLI starts, so any tree is measured with its own code): the final hidden state of
every prefill forward goes through the model's own ``lm_head`` in blocks, and the top-K ids / logprobs, the
logprob of the token that follows, and the input ids are kept per request. When the request's prompt phase
ends (``PromptProcessingBatch.generate``) the window is written to ``D/<window id>.npz``. The ``ref`` arm
replaces the QSA attention with dense fp32 attention over the same selected keys (slow, exact), for
Flash-Next (qwen4_exp); other architectures have no reference mode and fail closed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

K = 32
BLOCK = 128
H, KV, D, TOPK, QBLOCK = 24, 2, 256, 512, 4


def reference_attend(
    queries, keys, values, block_indices, query_ends, scale, rows_per_chunk=64
):
    """Dense fp32 attention over each row's selected blocks and tail block: ``[1, 24, T, 256]``."""
    import mlx.core as mx

    rows = int(queries.shape[2])
    bi = mx.sort(block_indices.astype(mx.int32), axis=-1).reshape(rows, TOPK)
    qe = query_ends.astype(mx.int32).reshape(rows)
    kf, vf, qf = (
        keys[0].astype(mx.float32),
        values[0].astype(mx.float32),
        queries[0].astype(mx.float32),
    )
    offs = mx.arange(QBLOCK, dtype=mx.int32)
    tail_offs = mx.arange(QBLOCK - 1, dtype=mx.int32)
    outs = []
    for r0 in range(0, rows, rows_per_chunk):
        r1 = min(rows, r0 + rows_per_chunk)
        b, e = bi[r0:r1], qe[r0:r1]
        ids_sel = (b[:, :, None] * QBLOCK + offs).reshape(r1 - r0, -1)
        valid_sel = mx.repeat(b >= 0, QBLOCK, axis=1)
        ids_tail = ((e // QBLOCK) * QBLOCK)[:, None] + tail_offs
        ids = mx.concatenate([mx.maximum(ids_sel, 0), ids_tail], axis=1)
        valid = mx.concatenate([valid_sel, ids_tail < e[:, None]], axis=1)
        heads = []
        for g in range(KV):
            k_c, v_c = kf[g][ids], vf[g][ids]
            q_c = qf[12 * g : 12 * g + 12, r0:r1].transpose(1, 0, 2)
            s = mx.where(
                valid[:, None, :],
                mx.matmul(q_c, k_c.transpose(0, 2, 1)) * scale,
                -mx.inf,
            )
            heads.append(mx.matmul(mx.softmax(s, axis=-1), v_c))
        o = mx.concatenate(heads, axis=1)
        mx.eval(o)
        outs.append(o)
    return mx.concatenate(outs, axis=0).transpose(1, 0, 2)[None].astype(queries.dtype)


def _eligible(queries, keys, values, block_indices, kw) -> bool:
    import mlx.core as mx

    return bool(
        kw.get("causal")
        and not kw.get("allow_sparse_decode")
        and isinstance(keys, mx.array)
        and queries.ndim == 4
        and queries.shape[0] == 1
        and queries.shape[1] == H
        and queries.shape[2] > 1
        and queries.shape[3] == D
        and keys.shape[:2] == (1, KV)
        and keys.shape[3] == D
        and block_indices.shape[-1] == TOPK
        and kw.get("block_size") == QBLOCK
    )


def install(arm: str) -> None:
    import mlx.core as mx
    from mlx_vlm.generate.ar import PromptProcessingBatch
    from mlx_vlm.models.qwen4_exp import language as l4

    outdir = Path(os.environ["KLD_CACHE_DIR"])
    expect_file = outdir / "_current.json"
    state: dict = {
        "lm": None,
        "tokens": [],
        "ids": [],
        "lp": [],
        "next": [],
        "carry": None,
        "active": False,
    }

    if arm == "ref":
        original = l4.dispatch_qsa_attention

        def dispatch(queries, keys, values, block_indices, query_ends, **kw):
            if _eligible(queries, keys, values, block_indices, kw):
                return reference_attend(
                    queries, keys, values, block_indices, query_ends, float(kw["scale"])
                )
            return original(queries, keys, values, block_indices, query_ends, **kw)

        l4.dispatch_qsa_attention = dispatch
    elif arm == "cand":
        pass  # the candidate's own flags arrive through the environment

    lm_call = l4.LanguageModel.__call__

    def language_model_call(self, *args, **kwargs):
        state["lm"] = self
        return lm_call(self, *args, **kwargs)

    l4.LanguageModel.__call__ = language_model_call
    model_call = l4.Qwen4ExpModel.__call__

    def record(inputs, hidden):
        lm = state["lm"]
        head = lm.lm_head
        tokens = inputs[0]
        rows = int(tokens.shape[0])
        tok_np = [int(x) for x in tokens.tolist()]
        for start in range(0, rows, BLOCK):
            stop = min(rows, start + BLOCK)
            logits = head(hidden[:, start:stop])[0].astype(mx.float32)  # [b, V]
            lse = mx.logsumexp(logits, axis=-1, keepdims=True)
            vocab = logits.shape[-1]
            idx = mx.argpartition(logits, kth=vocab - K, axis=-1)[:, vocab - K :]
            vals = mx.take_along_axis(logits, idx, axis=-1)
            order = mx.argsort(-vals, axis=-1)
            idx, vals = (
                mx.take_along_axis(idx, order, axis=-1),
                mx.take_along_axis(vals, order, axis=-1),
            )
            lp = vals - lse
            # logprob of the token that follows each position inside this block (the last one waits)
            nxt = mx.array(tok_np[start + 1 : stop] + [0], dtype=mx.int32)
            chosen = mx.take_along_axis(logits, nxt[:, None], axis=-1)[:, 0] - lse[:, 0]
            mx.eval(idx, lp, chosen, logits[-1:], lse[-1:])
            if (
                state["carry"] is not None
            ):  # the previous block's last position: its next token is our first
                first = int(tok_np[start])
                state["next"][-1] = float(
                    (state["carry"][0][first] - state["carry"][1]).item()
                )
                state["carry"] = None
            state["ids"].append(idx.astype(mx.int32))
            state["lp"].append(lp)
            state["next"].extend(float(x) for x in chosen.tolist()[:-1])
            state["next"].append(float("nan"))
            state["carry"] = (logits[-1], lse[-1, 0])
        state["tokens"].extend(tok_np)

    def qwen4_model_call(self, inputs, *args, **kwargs):
        out = model_call(self, inputs, *args, **kwargs)
        if state["lm"] is not None and inputs.shape[0] == 1 and expect_file.exists():
            state["active"] = True
            record(inputs, out)
        return out

    l4.Qwen4ExpModel.__call__ = qwen4_model_call

    generate = PromptProcessingBatch.generate

    def finish(self, *args, **kwargs):
        result = generate(self, *args, **kwargs)
        if state["active"]:
            import numpy as np

            wid = json.loads(expect_file.read_text())["wid"]
            ids = np.concatenate([np.array(a) for a in state["ids"]], axis=0)
            lp = np.concatenate([np.array(a) for a in state["lp"]], axis=0).astype(
                np.float32
            )
            tmp = outdir / f"{wid}.tmp.npz"
            np.savez(
                tmp,
                ids=ids.astype(np.int32),
                lp=lp,
                next_lp=np.array(state["next"], dtype=np.float32),
                tokens=np.array(state["tokens"], dtype=np.int32),
            )
            tmp.rename(outdir / f"{wid}.npz")
            expect_file.unlink()
            state.update(tokens=[], ids=[], lp=[], next=[], carry=None, active=False)
        return result

    PromptProcessingBatch.generate = finish


def ask(url: str, model: str, text: str) -> tuple[int, float]:
    """One completion of one token; (prompt tokens, wall seconds)."""
    import urllib.error
    import urllib.request

    body = dict(model=model, prompt=text, max_tokens=1, temperature=0)
    req = urllib.request.Request(
        url + "/v1/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer k"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            reply = json.load(r)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"/v1/completions {exc.code}: {exc.read()[:300]!r}") from exc
    return int((reply.get("usage") or {})["prompt_tokens"]), time.perf_counter() - t0


def run(args) -> None:
    import subprocess

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research"))
    import tfbench as t

    from . import corpus

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["KLD_CACHE_DIR"] = str(cache)
    k, n = (int(x) for x in args.shard.split("/"))
    windows = corpus.load_texts()[k::n]
    todo = [w for w in windows if not (cache / f"{w['id']}.npz").exists()]
    popen = subprocess.Popen

    def launch(command, *a, **kw):
        if command and Path(command[0]).name == "yunshu":
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "serve",
                args.arm,
                *command[1:],
            ]
        return popen(command, *a, **kw)

    out = Path(args.out)
    rows = []
    if todo:
        subprocess.Popen = launch
        env = {
            "YUNSHU_VLM_DRAFT": "off",
            "YUNSHU_VLM_APC_DISK": "0",
            "YUNSHU_VLM_APC_MEMORY_GB": "0",
            "YUNSHU_NEXT_TURN_PREFILL": "off",
            "YUNSHU_CONSOLE": "0",
            "KLD_CACHE_DIR": str(cache),
            **dict(kv.split("=", 1) for kv in args.env),
        }
        srv = t.Srv("yunshu", env, f"kld-{args.arm}-{k}of{n}", model=args.model)
        try:
            t.send(srv.url, t.req(srv.model, "Say hi.", 8))
            for w in todo:
                (cache / "_current.json").write_text(json.dumps(dict(wid=w["id"])))
                tokens, wall = ask(srv.url, srv.model, w["text"])
                if not (cache / f"{w['id']}.npz").exists():
                    raise RuntimeError(
                        f"{w['id']}: the capture hook wrote no window (prompt_tokens={tokens})"
                    )
                rows.append(
                    dict(id=w["id"], prompt_tokens=tokens, wall_s=round(wall, 2))
                )
                print(json.dumps(rows[-1]), flush=True)
        finally:
            srv.kill()
            (cache / "_current.json").unlink(missing_ok=True)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        fh.write(
            json.dumps(dict(complete=True, windows=len(windows), captured=len(rows)))
            + "\n"
        )


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["serve"]:
        argv.pop(0)
        arm = argv.pop(0)
        install(arm)
        from yunshu_cli import main as cli

        sys.argv = [sys.argv[0], *argv]
        cli()
        return 0
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=["base", "cand", "ref"], required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--cache-dir", required=True)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--env", action="append", default=[])
    r.add_argument("--out", required=True)
    a = p.parse_args(argv)
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
