"""Fair Yunshu vs TensorFold battery: one server session per call.

  tfbench.py --engine yunshu|tf-old|tf-new --part decode|conc|agent --rep N --out F.jsonl [--only-ctx 1024]
Every record is one HTTP request; a part that does not finish writes no 'part_done' record (fail closed).
"""

import argparse
import concurrent.futures as cf
import contextlib
import hashlib
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

M = "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
D = "/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2"
TF = {
    "tf-old": "/Volumes/P5Plus/yunshu-test-envs/tensorfold-0.3.6.1/bin/tensorfold",
    "tf-new": "/Volumes/P5Plus/yunshu-test-envs/tensorfold-0.6.1/bin/tensorfold",
}
YUNSHU_BIN = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/yunshu"
YUNSHU_SRC = os.environ.get("TFB_YUNSHU_SRC", "")
ROOT = Path("/Users/yuhuan/Documents/YuhuanStudio/Yunshu")
RUNS = ROOT / "docs/research/runs/2026-09-30-agtraffic/artifacts"
BODIES = RUNS / "cap-opencode-fix-cart-discount-r1/bodies"
BODIES2 = RUNS / "cap-opencode-polyglot-wordy-r1/bodies"
WORK = Path("/Volumes/P5Plus/yunshu-build/tfnew")


def free_port():
    for p in range(18990, 19000):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                continue
        return p
    raise RuntimeError("no port")


class Srv:
    def __init__(self, engine, extra_env, tag):
        self.engine, self.port = engine, free_port()
        self.home = WORK / "home" / tag
        shutil.rmtree(self.home, ignore_errors=True)
        self.home.mkdir(parents=True)
        self.log = WORK / "out" / f"server-{tag}.log"
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("ANTHROPIC_", "OPENAI_", "CLAUDE", "CODEX", "YUNSHU_"))
        }
        env.update(
            HOME=str(self.home), HF_HUB_OFFLINE="1", NO_PROXY="127.0.0.1", **extra_env
        )
        if engine == "yunshu":
            if YUNSHU_SRC:
                env["PYTHONPATH"] = YUNSHU_SRC
            cmd = [YUNSHU_BIN, "serve", "-m", M, "--port", str(self.port)]
        else:
            cmd = [
                TF[engine],
                "serve",
                M,
                "--port",
                str(self.port),
                "--drafter",
                D,
                "--snapshot-dir",
                str(self.home / "snap"),
                "--no-update-check",
            ]
        self.cmd = cmd
        self.proc = subprocess.Popen(
            cmd,
            stdout=open(self.log, "ab"),
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self.url = f"http://127.0.0.1:{self.port}"
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited early")
            try:
                with urllib.request.urlopen(self.url + "/v1/models", timeout=3) as r:
                    self.model = json.load(r)["data"][0]["id"]
                    self.ready_s = time.time() - t0
                    return
            except Exception:
                time.sleep(2)
        self.kill()
        raise RuntimeError("not ready")

    def kill(self):
        if self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                self.proc.wait(30)


def send(url, body, timeout=900):
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer k"},
    )
    t0 = time.perf_counter()
    tf = None
    content, reasoning, tools = [], [], {}
    usage = finish = xy = None
    done = False
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            p = line[5:].strip()
            if p == b"[DONE]":
                done = True
                break
            d = json.loads(p)
            if d.get("error"):
                raise RuntimeError(f"SSE error {d['error']}")
            if d.get("usage"):
                usage = d["usage"]
            if d.get("x_yunshu"):
                xy = d["x_yunshu"]
            for ch in d.get("choices") or []:
                dl = ch.get("delta") or {}
                got = False
                if dl.get("content"):
                    content.append(dl["content"])
                    got = True
                if dl.get("reasoning_content"):
                    reasoning.append(dl["reasoning_content"])
                    got = True
                for f in dl.get("tool_calls") or []:
                    c = tools.setdefault(
                        f.get("index", 0), {"name": "", "arguments": ""}
                    )
                    fn = f.get("function") or {}
                    c["name"] += fn.get("name") or ""
                    c["arguments"] += fn.get("arguments") or ""
                    got = True
                if got and tf is None:
                    tf = time.perf_counter()
                finish = ch.get("finish_reason") or finish
    t1 = time.perf_counter()
    if not done or finish is None or usage is None:
        raise RuntimeError("incomplete stream")
    ct = usage["completion_tokens"]
    c = "".join(content)
    rs = "".join(reasoning)
    sha = hashlib.sha256(
        json.dumps(
            [c, rs, [tools[i] for i in sorted(tools)], finish], ensure_ascii=False
        ).encode()
    ).hexdigest()[:16]
    return dict(
        ttft_s=round((tf or t1) - t0, 3),
        total_s=round(t1 - t0, 3),
        ct=ct,
        pt=usage["prompt_tokens"],
        cached=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        dec_tps=round((ct - 1) / (t1 - tf), 1) if tf and ct > 1 and t1 > tf else None,
        finish=finish,
        sha=sha,
        reasoning_chars=len(rs),
        head=(c or rs)[:80],
        _text=c,
        xy={
            k: xy.get(k)
            for k in (
                "speculative",
                "decode_tps",
                "prefill_tps",
                "ttft_ms",
                "decode_ms",
            )
        }
        if xy
        else None,
    )


def corpus(kind):
    pats = {
        "prose": [
            "reference/vllm/docs/**/*.md",
            "reference/sglang/docs/**/*.md",
            "reference/llama.cpp/docs/**/*.md",
            "docs/**/*.md",
        ],
        "code": [
            "reference/mlx-lm/mlx_lm/**/*.py",
            "reference/mlx-vlm/mlx_vlm/**/*.py",
            "python/**/*.py",
        ],
    }[kind]
    files = []
    for p in pats:
        files += sorted(ROOT.glob(p))
    files = [
        f
        for f in files
        if f.is_file() and 2000 < f.stat().st_size < 200000 and "research" not in str(f)
    ]
    random.Random(11).shuffle(files)
    return files


def load_prompt(name):
    return (WORK / "prompts" / f"{name}.txt").read_text()


def make_prompt(kind, ntok, salt):
    files = corpus(kind)
    cpt = 4.0 if kind == "prose" else 3.2
    need = int(ntok * cpt)
    buf = []
    n = 0
    i = salt * 7
    while n < need:
        f = files[i % len(files)]
        i += 1
        t = f.read_text(errors="ignore")[: need - n]
        buf.append(f"\n### {f.name}\n{t}")
        n += len(t)
    ask = {
        "prose": "\n\n---\nWrite a long, detailed essay in your own words that explains the main ideas of the material above. Do not use lists. Keep writing at length.",
        "code": "\n\n---\nWrite a thorough code review of the code above: for every module quote the key functions in full, then propose a rewritten version of each. Keep writing at length.",
    }[kind]
    return "".join(buf) + ask


def req(model, text, mt, seed=None, extra=None, temp=0):
    b = dict(
        model=model,
        messages=[{"role": "user", "content": text}],
        max_tokens=mt,
        temperature=temp,
        chat_template_kwargs={"enable_thinking": False},
    )
    if seed is not None:
        b["seed"] = seed
    if extra:
        b.update(extra)
    return b


def emit(out, **kw):
    out.write(json.dumps(kw) + "\n")
    out.flush()


def part_decode(s, out, a):
    ctxs = [a.only_ctx] if a.only_ctx else [1024, 8192, 32768]
    for ctx in ctxs:
        for kind in ("prose", "code"):
            salt = ctx // 1024 + (3 if kind == "code" else 0)
            text = load_prompt(f"{kind}-{ctx}")
            reply = ""
            for phase in ("cold", "warm", "turn2"):
                b = req(s.model, text, 256)
                if phase == "turn2":
                    b["messages"] += [
                        {"role": "assistant", "content": reply},
                        {
                            "role": "user",
                            "content": "Continue with the next part, at the same length.",
                        },
                    ]
                r = send(s.url, b)
                if phase == "cold":
                    reply = r["_text"]
                r["text"] = r.pop("_text")
                emit(out, part="decode", ctx=ctx, kind=kind, phase=phase, **r)
            if a.engine != "yunshu" and ctx == 1024:
                r = send(s.url, req(s.model, text, 256, extra={"draft": False}))
                r["text"] = r.pop("_text")
                emit(out, part="decode", ctx=ctx, kind=kind, phase="specoff", **r)


def part_conc(s, out, a):
    for n in (2, 4, 8):
        for trial in range(2):
            texts = [
                load_prompt(f"conc-{'prose' if (i + trial) % 2 == 0 else 'code'}-{i}")
                for i in range(n)
            ]
            t0 = time.perf_counter()
            with cf.ThreadPoolExecutor(n) as ex:
                rs = list(ex.map(lambda t: send(s.url, req(s.model, t, 256)), texts))
            for r in rs:
                r.pop("_text")
            wall = time.perf_counter() - t0
            tot = sum(r["ct"] for r in rs)
            emit(
                out,
                part="conc",
                n=n,
                trial=trial,
                wall_s=round(wall, 3),
                total_tokens=tot,
                agg_tps=round(tot / wall, 1),
                per_req_dec=[r["dec_tps"] for r in rs],
                ttfts=[r["ttft_s"] for r in rs],
                shas=[r["sha"] for r in rs],
            )


def part_agent(s, out, a):
    files = [
        BODIES / "0002-req.json",
        BODIES / "0004-req.json",
        BODIES / "0006-req.json",
        BODIES2 / "0003-req.json",
    ]
    title = json.loads((BODIES / "0001-req.json").read_text())
    for f in files:
        body = json.loads(f.read_text())
        body["model"] = s.model
        body["seed"] = 1234
        for i in range(3):
            th = fut = None
            if i == 0:
                th = cf.ThreadPoolExecutor(1)
                fut = th.submit(send, s.url, dict(title, model=s.model, seed=1234))
                time.sleep(0.3)
            r = send(s.url, body)
            if th:
                fut.result()
                th.shutdown()
            r.pop("_text")
            emit(
                out,
                part="agent",
                body=f.parent.parent.name[-22:] + "/" + f.name,
                i=i,
                **r,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--part", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-ctx", type=int)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    extra_env = dict(kv.split("=", 1) for kv in a.env)
    s = Srv(a.engine, extra_env, f"{a.engine}-{a.part}-{a.rep}{a.tag}")
    try:
        with open(a.out, "a") as out:
            emit(
                out,
                part="session",
                engine=a.engine,
                rep=a.rep,
                ready_s=round(s.ready_s, 1),
                cmd=s.cmd,
                env=extra_env,
                tag=a.tag,
            )
            for _ in range(2):
                send(s.url, req(s.model, "Say hi.", 24))
            if a.part == "ca":
                part_conc(s, out, a)
                part_agent(s, out, a)
            else:
                {"decode": part_decode}[a.part](s, out, a)
            emit(out, part="part_done", engine=a.engine, which=a.part, rep=a.rep)
    finally:
        s.kill()


if __name__ == "__main__":
    main()
