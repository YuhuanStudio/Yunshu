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
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev"))
from gpuq_contention import was_contended  # noqa: E402

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
# Server homes and logs go here (prompts stay in WORK); lets reruns keep their data apart.
OUT = Path(os.environ.get("TFB_OUT", str(WORK)))


def free_port():
    for p in range(18990, int(os.environ.get("TFB_PORT_LAST", "18999")) + 1):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                continue
        return p
    raise RuntimeError("no port")


def spec_request(engine, extra_env):
    """Make the comparison independent of drafter discovery under isolated HOME."""
    if engine != "yunshu":
        return "dflash", dict(extra_env)
    env = dict(extra_env)
    override = env.setdefault("YUNSHU_VLM_DRAFT", D)
    mode = {"mtp": "mtp", "force-mtp": "mtp", "off": "off", "none": "off"}.get(
        override.lower(), "dflash"
    )
    return mode, env


def engaged_spec_mode(engine, log):
    """Read the initialized runner, not the earlier selection/fallback message."""
    if engine == "yunshu":
        modes = re.findall(r"VLM batch runner: [^\n]*?draft=(dflash|mtp|off)\b", log)
        return modes[-1] if modes else None
    if re.search(r"\[tensorfold\] drafter [^\n]*DFlash[^\n]*block=", log):
        return "dflash"
    return None


class Srv:
    def __init__(self, engine, extra_env, tag, model=None):
        self.engine, self.port = engine, free_port()
        self.requested_spec_mode, extra_env = spec_request(engine, extra_env)
        self.extra_env = extra_env
        self.home = OUT / "home" / tag
        shutil.rmtree(self.home, ignore_errors=True)
        self.home.mkdir(parents=True)
        self.log = OUT / "out" / f"server-{tag}.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
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
            cmd = [YUNSHU_BIN, "serve", "-m", model or M, "--port", str(self.port)]
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
        with open(self.log, "wb") as server_log:
            self.proc = subprocess.Popen(
                cmd,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        self.url = f"http://127.0.0.1:{self.port}"
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.proc.poll() is not None:
                self.kill()
                raise RuntimeError("server exited early")
            if "FATAL:" in self.log.read_text(errors="replace"):
                self.kill()
                raise RuntimeError(f"server startup failed; see {self.log}")
            try:
                with urllib.request.urlopen(self.url + "/v1/models", timeout=3) as r:
                    self.model = json.load(r)["data"][0]["id"]
                    self.ready_s = time.time() - t0
                    break
            except Exception:
                time.sleep(2)
        else:
            self.kill()
            raise RuntimeError("not ready")
        try:
            self.verify_spec_mode()
        except Exception:
            self.kill()
            raise

    def verify_spec_mode(self):
        self.engaged_spec_mode = engaged_spec_mode(
            self.engine, self.log.read_text(errors="replace")
        )
        if self.engaged_spec_mode != self.requested_spec_mode:
            raise RuntimeError(
                f"requested spec={self.requested_spec_mode}, "
                f"engaged={self.engaged_spec_mode}; see {self.log}"
            )

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
    kw["device"] = os.environ.get("GPUQ_DEVICE", "m5")
    kw["contended"] = was_contended()
    out.write(json.dumps(kw) + "\n")
    out.flush()


# Appended to long decode prompts: a plain "keep writing at length" ask ends by itself around
# 1.5K tokens on Qwen3.8-27B, and a decode cell needs a reply that reaches its token budget.
LONG_ASK = (
    "\n\nRequirement: the answer must be extremely long, at least 5000 words. "
    "Do not conclude, summarize or stop early; keep going section after section."
)


def check_decode_len(r, n, what):
    """A decode cell measures exactly n tokens; a short reply is an error (a request that died
    mid-stream must not count as a result)."""
    if r.get("finish") != "length" or r.get("ct") != n:
        raise RuntimeError(
            f"{what}: finish={r.get('finish')} ct={r.get('ct')} (want length/{n})"
        )


def ngram_repeat(text, n=4):
    """Share of word n-grams that repeat an earlier n-gram (0 = none, ->1 = a loop)."""
    w = text.split()
    if len(w) <= n:
        return 0.0
    grams = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
    return round(1 - len(set(grams)) / len(grams), 4)


def part_decode(s, out, a):
    n_dec = int(a.decode_tokens)
    ctxs = [512] if a.smoke else a.only_ctx or [1024, 8192, 32768]
    for ctx in ctxs:
        for kind in a.only_kind or ("prose", "code"):
            text = (
                "Write a short example and explain it."
                if a.smoke
                else load_prompt(f"{kind}-{ctx}") + (LONG_ASK if a.long_ask else "")
            )
            reply = ""
            for phase in ("cold", "warm", "turn2"):
                want = (a.turn2_tokens or n_dec) if phase == "turn2" else n_dec
                b = req(s.model, text, 16 if a.smoke else want)
                if phase == "turn2":
                    b["messages"] += [
                        {"role": "assistant", "content": reply},
                        {
                            "role": "user",
                            "content": "Continue with the next part, at the same length.",
                        },
                    ]
                r = send(s.url, b)
                if not a.smoke:
                    check_decode_len(r, want, f"decode {kind}-{ctx} {phase}")
                if phase == "cold":
                    reply = r["_text"]
                r["text"] = r.pop("_text")
                r["rep4"] = ngram_repeat(r["text"])
                emit(out, part="decode", ctx=ctx, kind=kind, phase=phase, **r)
            if a.engine != "yunshu" and ctx == 1024:
                r = send(s.url, req(s.model, text, n_dec, extra={"draft": False}))
                r["text"] = r.pop("_text")
                emit(out, part="decode", ctx=ctx, kind=kind, phase="specoff", **r)


NEEDLE_NAMES = [
    "Aldrin", "Borealis", "Calypso", "Dunmore", "Elmstead", "Fairhaven", "Glenrock",
    "Highmarsh", "Ironwood", "Juniper", "Kestrel", "Larkspur", "Mirefield", "Northgate",
    "Oakhollow", "Pinecrest", "Quillon", "Redwater", "Stonebridge", "Thornfield",
]  # fmt: skip
NEEDLES_PER_CTX = 10


def needle_items(ctx, n=NEEDLES_PER_CTX):
    """Deterministic key-value items for one context: (station name, 6-digit code)."""
    rnd = random.Random(7000 + ctx)
    names = rnd.sample(NEEDLE_NAMES, n)
    return [(nm, f"{rnd.randint(100000, 999999)}") for nm in names]


def needle_haystack(base, ctx, items):
    """Splice one needle sentence per item into `base` (the corpus without its final ask) at
    evenly spread depths (line boundaries), then trim the tail so the length stays the same."""
    sents = [
        f"\nRecord note: the passcode for station {nm} is {code}.\n"
        for nm, code in items
    ]
    out, last = [], 0
    for i, snt in enumerate(sents):
        pos = int(len(base) * (i + 0.5) / len(sents))
        nl = base.find("\n", pos)
        pos = len(base) if nl < 0 else nl
        out.append(base[last:pos])
        out.append(snt)
        last = pos
    out.append(base[last:])
    text = "".join(out)
    return text[: len(base) - 400]  # the question + template need headroom


def needle_question(name):
    return (
        f"\n\n---\nIn the text above, what is the passcode for station {name}? "
        "Answer with the six-digit number only."
    )


def part_needle(s, out, a):
    ctxs = [512] if a.smoke else a.only_ctx or [32768, 65536, 131072]
    for ctx in ctxs:
        if a.smoke:
            base, items = "Some filler text.\n" * 40, needle_items(ctx, 2)
        else:
            full = load_prompt(f"prose-{ctx}")
            base = full[: full.rfind("\n\n---\n")]
            items = needle_items(ctx)
        hay = needle_haystack(base, ctx, items)
        for i, (nm, code) in enumerate(items):
            r = send(s.url, req(s.model, hay + needle_question(nm), 16))
            ans = r.pop("_text")
            emit(
                out,
                part="needle",
                ctx=ctx,
                item=i,
                name=nm,
                expect=code,
                answer=ans[:80],
                correct=code in ans,
                **{k: r[k] for k in ("ttft_s", "pt", "cached", "finish", "ct")},
            )


def part_conc32(s, out, a):
    """Two sub-agents at once: each has a warm cached prefix (32K) and sends a ~2K new turn,
    asking for a 1K reply. Reports TTFT and per-request decode."""
    ctx, n_new = (512, 16) if a.smoke else (32768, int(a.decode_tokens or 1024))
    kinds = ("prose", "code")
    prefixes = {
        k: ("Hello. " * 80 if a.smoke else load_prompt(f"{k}-{ctx}")) for k in kinds
    }
    for k in kinds:  # warm each prefix
        send(s.url, req(s.model, prefixes[k], 8))
    for trial in range(1 if a.smoke else 2):

        def turn(k):
            other = "code" if k == "prose" else "prose"
            src = "Extra." * 20 if a.smoke else load_prompt(f"{other}-8192")
            piece = src[trial * 1500 : trial * 1500 + 6500]
            txt = prefixes[k] + "\n\nNew material:\n" + piece + "\nContinue at length."
            return send(s.url, req(s.model, txt, n_new))

        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(2) as ex:
            rs = list(ex.map(turn, kinds))
        wall = time.perf_counter() - t0
        for r in rs:
            if not a.smoke:
                check_decode_len(r, n_new, f"conc32 trial {trial}")
            r.pop("_text")
        emit(
            out,
            part="conc32",
            trial=trial,
            wall_s=round(wall, 2),
            ttfts=[r["ttft_s"] for r in rs],
            per_req_dec=[r["dec_tps"] for r in rs],
            cached=[r["cached"] for r in rs],
            pts=[r["pt"] for r in rs],
            cts=[r["ct"] for r in rs],
            shas=[r["sha"] for r in rs],
        )


def part_conc(s, out, a):
    for n in (2,) if a.smoke else (2, 4, 8):
        for trial in range(1 if a.smoke else 2):
            texts = [
                "Say hello."
                if a.smoke
                else load_prompt(
                    f"conc-{'prose' if (i + trial) % 2 == 0 else 'code'}-{i}"
                )
                for i in range(n)
            ]
            t0 = time.perf_counter()
            with cf.ThreadPoolExecutor(n) as ex:
                rs = list(
                    ex.map(
                        lambda t: send(s.url, req(s.model, t, 16 if a.smoke else 256)),
                        texts,
                    )
                )
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
    for f in files[:1] if a.smoke else files:
        body = json.loads(f.read_text())
        body["model"] = s.model
        body["seed"] = 1234
        if a.smoke:
            body["max_tokens"] = 16
            title["max_tokens"] = 16
        for i in range(1 if a.smoke else 3):
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


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--part", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-ctx", type=int, action="append")
    ap.add_argument("--only-kind", action="append", choices=["prose", "code"])
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", default=M)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--long-ask",
        action="store_true",
        help="append LONG_ASK to decode prompts so replies reach --decode-tokens",
    )
    ap.add_argument(
        "--turn2-tokens",
        type=int,
        default=0,
        help="reply length of the follow-up (turn2) request; 0 = --decode-tokens",
    )
    ap.add_argument(
        "--decode-tokens",
        type=int,
        default=256,
        help="reply length of decode cells; a cell that does not end finish=length at exactly N is an error",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = list(sys.argv[1:] if argv is None else argv)
    # Tags are filename suffixes and often begin with '-'. Keep registered
    # options as options, but bind a tag value before argparse classifies it.
    i = 0
    while i + 1 < len(args):
        if args[i] == "--tag" and args[i + 1] not in ap._option_string_actions:
            args[i : i + 2] = ["--tag=" + args[i + 1]]
        i += 1
    return ap.parse_args(args)


def main():
    a = parse_args()
    extra_env = dict(kv.split("=", 1) for kv in a.env)
    if a.part not in ("decode", "conc", "agent", "ca", "needle", "conc32"):
        raise ValueError("supported parts: decode, conc, agent, ca, needle, conc32")
    if a.dry_run:
        if a.part == "decode" and not a.smoke:
            for ctx in a.only_ctx or [1024, 8192, 32768]:
                for kind in a.only_kind or ("prose", "code"):
                    load_prompt(f"{kind}-{ctx}")
        elif a.part in ("conc", "agent", "ca"):
            for directory in (BODIES, BODIES2):
                if not list(directory.glob("*-req.json")):
                    raise FileNotFoundError(directory)
        if not (Path(a.model) / "config.json").is_file():
            raise FileNotFoundError(a.model)
        mode, selected = spec_request(a.engine, extra_env)
        print(
            json.dumps(
                {
                    "complete": "dry-run",
                    "requested_spec_mode": mode,
                    "env": selected,
                    "model": a.model,
                }
            )
        )
        return
    s = Srv(a.engine, extra_env, f"{a.engine}-{a.part}-{a.rep}{a.tag}", model=a.model)
    try:
        with open(a.out, "a") as out:
            emit(
                out,
                part="session",
                engine=a.engine,
                rep=a.rep,
                ready_s=round(s.ready_s, 1),
                cmd=s.cmd,
                env=s.extra_env,
                requested_spec_mode=s.requested_spec_mode,
                engaged_spec_mode=s.engaged_spec_mode,
                tag=a.tag,
            )
            for _ in range(2):
                send(s.url, req(s.model, "Say hi.", 24))
            if a.part == "ca":
                part_conc(s, out, a)
                part_agent(s, out, a)
            else:
                {
                    "decode": part_decode,
                    "conc": part_conc,
                    "agent": part_agent,
                    "needle": part_needle,
                    "conc32": part_conc32,
                }[a.part](s, out, a)
            s.verify_spec_mode()
            emit(
                out,
                part="part_done",
                engine=a.engine,
                which=a.part,
                rep=a.rep,
                complete=True,
            )
    finally:
        s.kill()


if __name__ == "__main__":
    main()
