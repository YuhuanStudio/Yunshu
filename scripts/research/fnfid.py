"""Long-context next-token fidelity of Flash-Next prefill attention: main vs candidate vs an fp32 reference.

  fnfid.py run --arm main|cand|ref --model DIR --out arm.jsonl      (one server per arm, one gpuq job)
  fnfid.py report main.jsonl cand.jsonl ref.jsonl                   (CPU)

Every prompt is prefilled cold and one token is requested with the top-20 logprobs of the last position.
Arms: main = upstream's indexed-sparse kernel; cand = the tensor-unit kernel (YUNSHU_QSA_SPARSE_PREFILL);
ref = dense fp32 attention over the same selected keys (slow, chunked). The report gives KL(ref || arm)
over the reference's top-20 tokens plus a tail bucket, and top-1 / top-5 agreement with the reference.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

H, KV, D, TOPK, BLOCK = 24, 2, 256, 512, 4


# ---- CPU: metrics -------------------------------------------------------------------------------
def dist(top: list[dict]) -> dict[str, float]:
    """{token: probability} of a top-k logprob list."""
    return {t["token"]: math.exp(t["logprob"]) for t in top}


def kl_to_reference(ref: dict[str, float], arm: dict[str, float]) -> float:
    """KL(ref || arm) over the reference's tokens plus one tail bucket.

    A reference token missing from the arm's top-k gets the arm's smallest listed probability (an upper
    bound on what it can have), so a missing token can only make the arm look closer by at most that."""
    floor = min(arm.values()) if arm else 1e-12
    total = 0.0
    q_listed = 0.0
    p_listed = 0.0
    for token, p in ref.items():
        q = arm.get(token, floor)
        q_listed += q
        p_listed += p
        total += p * math.log(max(p, 1e-12) / max(q, 1e-12))
    p_tail, q_tail = max(1.0 - p_listed, 0.0), max(1.0 - q_listed, 1e-12)
    if p_tail > 1e-12:
        total += p_tail * math.log(p_tail / q_tail)
    return total


def agreement(ref_top: list[dict], arm_top: list[dict]) -> tuple[bool, float]:
    """(top-1 equal, overlap of the two top-5 sets)."""
    r = [t["token"] for t in ref_top[:5]]
    a = [t["token"] for t in arm_top[:5]]
    return r[0] == a[0], len(set(r) & set(a)) / 5.0


def report(main_rows: list[dict], cand_rows: list[dict], ref_rows: list[dict]) -> dict:
    by = lambda rows: {r["id"]: r for r in rows if "top" in r}  # noqa: E731
    m, c, f = by(main_rows), by(cand_rows), by(ref_rows)
    ids = sorted(set(m) & set(c) & set(f))
    if not ids:
        raise RuntimeError("no prompt answered by all three arms")
    out = {"prompts": len(ids), "groups": {}}
    for group in ["all", *sorted({m[i]["group"] for i in ids})]:
        sel = [i for i in ids if group == "all" or m[i]["group"] == group]
        stats = {}
        for name, rows in (("main", m), ("cand", c)):
            kls = [
                kl_to_reference(dist(f[i]["top"]), dist(rows[i]["top"])) for i in sel
            ]
            ag = [agreement(f[i]["top"], rows[i]["top"]) for i in sel]
            stats[name] = dict(
                kl_mean=sum(kls) / len(kls),
                kl_median=sorted(kls)[len(kls) // 2],
                top1=sum(a[0] for a in ag) / len(ag),
                top5=sum(a[1] for a in ag) / len(ag),
            )
        diff = [
            kl_to_reference(dist(f[i]["top"]), dist(c[i]["top"]))
            - kl_to_reference(dist(f[i]["top"]), dist(m[i]["top"]))
            for i in sel
        ]
        mean = sum(diff) / len(diff)
        var = sum((d - mean) ** 2 for d in diff) / max(len(diff) - 1, 1)
        stats["kl_cand_minus_main"] = dict(
            mean=mean, stderr=math.sqrt(var / len(diff)), n=len(sel)
        )
        stats["cand_no_worse"] = bool(mean <= 2 * math.sqrt(var / len(diff)) + 1e-6)
        out["groups"][group] = stats
    return out


# ---- GPU: the three arms ------------------------------------------------------------------------
def reference_attend(
    queries, keys, values, block_indices, query_ends, scale, rows_per_chunk=64
):
    """Dense fp32 attention over each row's selected blocks and tail block: ``[1, 24, T, 256]``."""
    import mlx.core as mx

    rows = int(queries.shape[2])
    bi = mx.sort(block_indices.astype(mx.int32), axis=-1).reshape(rows, TOPK)
    qe = query_ends.astype(mx.int32).reshape(rows)
    kf = keys[0].astype(mx.float32)  # [KV, L, D]
    vf = values[0].astype(mx.float32)
    qf = queries[0].astype(mx.float32)  # [H, T, D]
    offs = mx.arange(BLOCK, dtype=mx.int32)
    tail_offs = mx.arange(BLOCK - 1, dtype=mx.int32)
    outs = []
    for r0 in range(0, rows, rows_per_chunk):
        r1 = min(rows, r0 + rows_per_chunk)
        b, e = bi[r0:r1], qe[r0:r1]
        ids_sel = (b[:, :, None] * BLOCK + offs).reshape(r1 - r0, -1)
        valid_sel = mx.repeat(b >= 0, BLOCK, axis=1)
        tail_start = (e // BLOCK) * BLOCK
        ids_tail = tail_start[:, None] + tail_offs
        valid_tail = ids_tail < e[:, None]
        ids = mx.concatenate([mx.maximum(ids_sel, 0), ids_tail], axis=1)
        valid = mx.concatenate([valid_sel, valid_tail], axis=1)
        head_outs = []
        for g in range(KV):
            k_c = kf[g][ids]  # [c, K, D]
            v_c = vf[g][ids]
            q_c = qf[12 * g : 12 * g + 12, r0:r1].transpose(1, 0, 2)  # [c, 12, D]
            s = mx.matmul(q_c, k_c.transpose(0, 2, 1)) * scale
            s = mx.where(valid[:, None, :], s, -mx.inf)
            p = mx.softmax(s, axis=-1)
            head_outs.append(mx.matmul(p, v_c))  # [c, 12, D]
        o = mx.concatenate(head_outs, axis=1)  # [c, 24, D]
        mx.eval(o)
        outs.append(o)
    out = mx.concatenate(outs, axis=0).transpose(1, 0, 2)[None]  # [1, 24, T, D]
    return out.astype(queries.dtype)


def install_arm(arm: str) -> None:
    from yunshu_engine.kernels import qsa_sparse_prefill as qsp

    if arm == "cand":
        qsp.install()
    elif arm == "ref":
        from mlx_vlm.models.qwen4_exp import language as lang

        original = lang.dispatch_qsa_attention

        def dispatch(queries, keys, values, block_indices, query_ends, **kw):
            if (
                kw.get("causal")
                and not kw.get("allow_sparse_decode")
                and qsp.eligible(queries, keys, values, block_indices, kw["block_size"])
            ):
                return reference_attend(
                    queries, keys, values, block_indices, query_ends, float(kw["scale"])
                )
            return original(queries, keys, values, block_indices, query_ends, **kw)

        lang.dispatch_qsa_attention = dispatch


def prompts() -> list[dict]:
    """~60 cold prompts: prose/code at ~8K and ~32K tokens plus recorded agent-client requests."""
    import tfbench as t

    out = []
    for ntok in (8000, 32000):
        for salt in range(1, 26):
            kind = "prose" if salt % 2 else "code"
            out.append(
                dict(
                    id=f"{kind}-{ntok}-{salt}",
                    group=f"{kind}{ntok // 1000}k",
                    body=dict(
                        messages=[
                            {"role": "user", "content": t.make_prompt(kind, ntok, salt)}
                        ]
                    ),
                )
            )
    root = Path(
        "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/docs/research/runs/2026-09-30-agtraffic/artifacts"
    )
    for f in sorted(root.glob("cap-*/bodies/*-req.json")):
        body = json.loads(f.read_text())
        if len(f.read_text()) < 12000:
            continue
        keep = {k: body[k] for k in ("messages", "tools") if k in body}
        out.append(
            dict(
                id=f"agent-{f.parent.parent.name}-{f.name[:4]}",
                group="agent",
                body=keep,
            )
        )
    return out


def ask(url: str, body: dict) -> tuple[float, list[dict], dict]:
    """One non-streamed request: (wall s, top logprobs of the first generated token, usage)."""
    import time
    import urllib.request

    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer k"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        reply = json.load(r)
    content = (reply["choices"][0].get("logprobs") or {}).get("content") or []
    top = content[0].get("top_logprobs") if content else None
    return time.perf_counter() - t0, top or [], reply.get("usage") or {}


def run(args) -> None:
    import subprocess

    import tfbench as t

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

    subprocess.Popen = launch
    srv = t.Srv(
        "yunshu",
        {
            "YUNSHU_VLM_DRAFT": "off",
            "YUNSHU_VLM_APC_DISK": "0",
            "YUNSHU_VLM_APC_MEMORY_GB": "0",
            "YUNSHU_NEXT_TURN_PREFILL": "off",
            "YUNSHU_CONSOLE": "0",
        },
        f"fnfid-{args.arm}",
        model=args.model,
    )
    done = set()
    out = Path(args.out)
    if out.exists():
        done = {
            json.loads(line)["id"]
            for line in out.read_text().splitlines()
            if '"top"' in line
        }
    try:
        t.send(srv.url, t.req(srv.model, "Say hi.", 8))
        k, n = (int(x) for x in args.shard.split("/"))
        for p in prompts()[k::n][: args.limit or None]:
            if p["id"] in done:
                continue
            body = dict(
                model=srv.model,
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=20,
                chat_template_kwargs={"enable_thinking": False},
                **p["body"],
            )
            wall, first, usage = ask(srv.url, body)
            if not first:
                raise RuntimeError(f"{p['id']}: no top logprobs in the reply")
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            if cached:
                raise RuntimeError(
                    f"{p['id']}: a cold prefill was served from the cache ({cached} tokens)"
                )
            rec = dict(
                id=p["id"],
                group=p["group"],
                pt=usage["prompt_tokens"],
                wall_s=round(wall, 2),
                top=first,
            )
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(p["id"], rec["pt"], rec["wall_s"], flush=True)
    finally:
        srv.kill()
    with out.open("a") as fh:
        fh.write(json.dumps(dict(complete=True)) + "\n")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["serve"]:
        argv.pop(0)
        arm = argv.pop(0)
        install_arm(arm)
        from yunshu_cli import main as cli

        sys.argv = [sys.argv[0], *argv]
        cli()
        return 0
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=["main", "cand", "ref"], required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--shard", default="0/1", help="k/n: every n-th prompt from k")
    q = sub.add_parser("report")
    q.add_argument(
        "files", nargs=3, help="per arm: a file or a comma list of shard files"
    )
    a = p.parse_args(argv)
    if a.cmd == "run":
        run(a)
        return 0
    rows = [
        [json.loads(line) for line in Path(f).read_text().splitlines()] for f in a.files
    ]
    print(json.dumps(report(*rows), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
