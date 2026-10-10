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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_engines  # noqa: E402
from gpuq_contention import was_contended  # noqa: E402

M = "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
D = "/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2"
TF = {
    "tf-old": "/Volumes/P5Plus/yunshu-test-envs/tensorfold-0.3.6.1/bin/tensorfold",
    "tf-new": "/Volumes/P5Plus/yunshu-test-envs/tensorfold-0.6.1/bin/tensorfold",
}
YUNSHU_BIN = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/yunshu"
YUNSHU_SRC = os.environ.get("TFB_YUNSHU_SRC", "")


def own_venv_bin(src: str, name: str) -> str | None:
    """A worktree that carries its own .venv (marker file `.yv-own-venv`, for A/Bs of a
    dependency upgrade) is served from that venv; every other tree uses the shared one."""
    if not src:
        return None
    tree = os.path.dirname(src.rstrip("/"))
    if not os.path.exists(os.path.join(tree, ".yv-own-venv")):
        return None
    path = os.path.join(tree, ".venv", "bin", name)
    return path if os.path.exists(path) else None


ROOT = Path("/Users/yuhuan/Documents/YuhuanStudio/Yunshu")
RUNS = ROOT / "docs/research/runs/2026-09-30-agtraffic/artifacts"
BODIES = RUNS / "cap-opencode-fix-cart-discount-r1/bodies"
BODIES2 = RUNS / "cap-opencode-polyglot-wordy-r1/bodies"
WORK = Path(os.environ.get("TFB_WORK", "/Volumes/P5Plus/yunshu-build/tfnew"))
# Server homes and logs go here (prompts stay in WORK); lets reruns keep their data apart.
OUT = Path(os.environ.get("TFB_OUT", str(WORK)))


def free_port(wait_s=600.0, interval_s=5.0, sleep=None, clock=None):
    """Paused gpuq jobs retain ports; wait for the bounded shared pool."""
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    deadline = clock() + wait_s
    while True:
        for p in range(18990, int(os.environ.get("TFB_PORT_LAST", "18999")) + 1):
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", p))
                except OSError:
                    continue
            return p
        left = deadline - clock()
        if left <= 0:
            raise RuntimeError("no port after bounded wait")
        print(f"[tfbench] port pool busy; waiting ({left:.0f}s left)", flush=True)
        sleep(min(interval_s, left))


def owns_listener(pid, port):
    """A ready endpoint must belong to our process group before inference."""
    result = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    try:
        group = os.getpgid(pid)
        return any(os.getpgid(int(owner)) == group for owner in result.stdout.split())
    except (ProcessLookupError, ValueError):
        return False


def start_server(engine, extra_env, tag, **kwargs):
    """Retry bind races only; model/protocol failures remain fail-fast."""
    for attempt in range(5):
        try:
            return Srv(engine, extra_env, tag, **kwargs)
        except RuntimeError:
            log = OUT / "out" / f"server-{tag}.log"
            text = log.read_text(errors="replace") if log.exists() else ""
            if not any(
                marker in text.lower()
                for marker in (
                    "address already in use",
                    "errno 48",
                    "error while attempting to bind",
                )
            ):
                raise
            log.rename(log.with_suffix(f".bind-attempt{attempt}.log"))
            if attempt == 4:
                raise
            print(f"port bind race; retrying our {engine} instance", flush=True)
            time.sleep(1)
    raise RuntimeError("port pool remained unavailable")


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


def ready_request(url, extra_env):
    """Readiness probe; the server may require the key the arm configured."""
    token = extra_env.get("YUNSHU_AUTH_TOKEN", "k")
    return urllib.request.Request(
        url + "/v1/models", headers={"Authorization": "Bearer " + token}
    )


def warm_lazy_engine(engine, url, model):
    # /v1/models is discovery-only for oMLX. Load the target/drafter before
    # checking the engaged mode, and keep this startup request out of timing.
    if engine in ("omlx", "llamacpp"):
        send(url, req(model, "Write a short example and explain it.", 16), timeout=240)


class Srv:
    def __init__(
        self, engine, extra_env, tag, model=None, ctx_tokens=140000, parallel=1
    ):
        self.engine, self.port = engine, free_port()
        self.requested_spec_mode, extra_env = spec_request(engine, extra_env)
        self.launch = None
        self.probe = None
        if bench_engines.is_new_engine(engine):
            self.requested_spec_mode = bench_engines.ENGINES[engine].expected_mode
        self.extra_env = extra_env
        self.home = OUT / "home" / tag
        shutil.rmtree(self.home, ignore_errors=True)
        self.home.mkdir(parents=True)
        self.log = OUT / "out" / f"server-{tag}.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        env = bench_engines.scrubbed_env(os.environ)
        env.update(
            HOME=str(self.home), HF_HUB_OFFLINE="1", NO_PROXY="127.0.0.1", **extra_env
        )
        if bench_engines.is_new_engine(engine):
            self.launch = bench_engines.build_launch(
                engine, self.port, self.home, os.environ, ctx_tokens, parallel
            )
            env = dict(self.launch.env, **extra_env)
            for path, text in self.launch.files.items():
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(text)
            for link, target in self.launch.links.items():
                Path(link).parent.mkdir(parents=True, exist_ok=True)
                if Path(link).is_symlink() or Path(link).exists():
                    Path(link).unlink()
                os.symlink(target, link)
            cmd = list(self.launch.cmd)
            if bench_engines.ENGINES[engine].kind == "yunshu":
                if not YUNSHU_SRC:
                    raise RuntimeError(
                        "TFB_YUNSHU_SRC (pinned tree python/) is required"
                    )
                env["PYTHONPATH"] = YUNSHU_SRC
                cmd[0] = own_venv_bin(YUNSHU_SRC, "yunshu") or cmd[0]
        elif engine == "yunshu":
            if YUNSHU_SRC:
                env["PYTHONPATH"] = YUNSHU_SRC
            cmd = [
                own_venv_bin(YUNSHU_SRC, "yunshu") or YUNSHU_BIN,
                "serve",
                "-m",
                model or M,
                "--port",
                str(self.port),
            ]
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
        baseline_used = (
            host_used_bytes()
        )  # before launch: weights/mmap count in the delta
        with open(self.log, "wb") as server_log:
            self.proc = subprocess.Popen(
                cmd,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        self.url = f"http://127.0.0.1:{self.port}"
        self.mem = MemSampler(
            getattr(self.proc, "pid", None), baseline_bytes=baseline_used
        )
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.proc.poll() is not None:
                self.kill()
                raise RuntimeError("server exited early")
            if "FATAL:" in self.log.read_text(errors="replace"):
                self.kill()
                raise RuntimeError(f"server startup failed; see {self.log}")
            try:
                if not owns_listener(self.proc.pid, self.port):
                    time.sleep(0.2)
                    continue
                request = ready_request(self.url, self.extra_env)
                with urllib.request.urlopen(request, timeout=3) as r:
                    self.model = json.load(r)["data"][0]["id"]
                    self.ready_s = time.time() - t0
                    break
            except Exception:
                time.sleep(2)
        else:
            self.kill()
            raise RuntimeError("not ready")
        try:
            warm_lazy_engine(self.engine, self.url, self.model)
            self.verify_spec_mode()
        except Exception:
            self.kill()
            raise

    def probe_text(self):
        path = self.launch.probe if self.launch else None
        if not path:
            return None
        try:
            with urllib.request.urlopen(self.url + path, timeout=10) as r:
                return r.read().decode("utf-8", "replace")[:20000]
        except Exception as e:  # noqa: BLE001
            return f"probe failed: {e!r}"

    def verify_spec_mode(self):
        log = self.log.read_text(errors="replace")
        if self.engine in bench_engines.ENGINES:
            # registry engines: evidence from the log (and a status probe) must match the expected mode
            self.probe = self.probe_text()
            self.engaged_spec_mode = bench_engines.detect_mode(
                self.engine, log, self.probe
            )
            if os.environ.get("TFB_SKIP_ENGAGED") == "1":
                self.engaged_spec_mode = self.engaged_spec_mode or "unchecked"
                return
            bench_engines.check_engaged(self.engine, self.engaged_spec_mode)
            return
        self.engaged_spec_mode = engaged_spec_mode(self.engine, log)
        if os.environ.get("TFB_SKIP_ENGAGED") == "1":
            # old releases (v0.1.0) have no speculative decoding and no engagement marker
            self.engaged_spec_mode = self.engaged_spec_mode or "unchecked"
            return
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
        # The server's SSD prefix cache (up to ~8 GB per 128K cell) is dead once the server is:
        # the next server of this tag starts from an emptied home anyway. Left behind, finished
        # yv runs held 362 GB of it on P5Plus (2026-10-07).
        if self.proc.poll() is not None:
            for sub in (".yunshu/cache", "omlx-ssd", "mtplx-cache", "omlx-base/cache"):
                shutil.rmtree(self.home / sub, ignore_errors=True)
        if getattr(self, "mem", None):
            self.mem.stop()


def send(url, body, timeout=600):
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
    nchunks = 0
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
                if got:
                    nchunks += 1
                    if nchunks % 256 == 0:  # life sign for gpuq's no-output stall check
                        print(
                            f"  streamed {nchunks} chunks, {time.perf_counter() - t0:.0f}s",
                            flush=True,
                        )
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
        energy=(xy or {}).get("energy"),
        joules_per_token=((xy or {}).get("energy") or {})
        .get("decode", {})
        .get("joules_per_token"),
        gpu_watts_mean=((xy or {}).get("energy") or {})
        .get("decode", {})
        .get("gpu_watts_mean"),
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
                "energy",
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


def decode_prompt(kind, ctx, long_ask):
    text = load_prompt(f"{kind}-{ctx}")
    if os.environ.get("TFB_EXACT_PROMPTS") == "1":
        from snapshot_prompts import assert_chat_prompt

        assert_chat_prompt(text, ctx)
        if long_ask and LONG_ASK not in text:
            raise AssertionError("exact prompt lacks LONG_ASK")
        return text
    return text + (LONG_ASK if long_ask else "")


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


META: dict = {}  # engine / version / sha / flags / drafter / engaged spec mode / checkpoint, on every row


def host_used_bytes():
    """Host used memory (active+wired+compressor); None when vm_stat is unavailable."""
    try:
        from process_memory import system_used_bytes

        return system_used_bytes()
    except Exception:  # noqa: BLE001 - a baseline miss disables system-delta, never fakes it
        return None


class MemSampler:
    """Server memory, sampled every 2 s from server start.

    Headline numbers use memory_method="system-delta": host used memory (vm_stat active + wired +
    compressor) minus a baseline taken BEFORE the server launched. Unlike per-process phys_footprint it
    counts mmap'd file-backed weights, so engines that load weights differently are comparable. The
    process-tree phys_footprint / RSS stay as secondary fields. Without a baseline the method falls
    back to "process-footprint" and the parity board does not compare it."""

    def __init__(self, pid, every=2.0, baseline_bytes=None, used_fn=None):
        import threading

        self.pid, self.every = pid, every
        self.baseline_bytes = baseline_bytes
        self._used_fn = used_fn or host_used_bytes
        self.peak_gib = 0.0
        self.peak_rss_gib = 0.0
        self.last_rss_gib = None
        self.last_gib = None
        self.peak_footprint_gib = 0.0
        self.last_footprint_gib = None
        self.peak_delta_gib = 0.0
        self.last_delta_gib = None
        self.n = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        if pid:
            self._t.start()

    @property
    def method(self):
        return "system-delta" if self.baseline_bytes else "process-footprint"

    def sample(self):
        try:
            from process_memory import process_tree_memory

            m = process_tree_memory(self.pid)
            g = round(m["physical_footprint_sum_bytes"] / 2**30, 3)
            self.last_footprint_gib = g
            self.peak_footprint_gib = max(self.peak_footprint_gib, g)
            rss = m.get("rss_sum_bytes")
            if rss is not None:
                self.last_rss_gib = round(rss / 2**30, 3)
                self.peak_rss_gib = max(self.peak_rss_gib, self.last_rss_gib)
            head = g
            if self.baseline_bytes:
                used = self._used_fn()
                if used is not None:
                    head = round((used - self.baseline_bytes) / 2**30, 3)
                    self.last_delta_gib = head
                    self.peak_delta_gib = max(self.peak_delta_gib, head)
            self.last_gib = head
            self.peak_gib = max(self.peak_gib, head)
            self.n += 1
            return head
        except Exception:  # noqa: BLE001 - a sampling miss is not a result
            return None

    def _run(self):
        while not self._stop.wait(self.every):
            self.sample()

    def stop(self):
        self._stop.set()


def emit(out, **kw):
    kw = {**META, **kw}
    print(
        json.dumps({k: v for k, v in kw.items() if k not in ("text", "cmd", "env")})[
            :300
        ],
        flush=True,
    )
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


ALL_PHASES = ("cold", "warm", "turn2", "specoff")


def parse_phases(text):
    """Selected decode phases; turn2 replays the cold reply, so it needs cold."""
    phases = [p.strip() for p in str(text).split(",") if p.strip()]
    bad = [p for p in phases if p not in ALL_PHASES]
    if bad or not phases:
        raise SystemExit(
            f"--phases: unknown or empty selection {text!r} (use {ALL_PHASES})"
        )
    if "turn2" in phases and "cold" not in phases:
        raise SystemExit("--phases: turn2 needs cold (it continues the cold reply)")
    return set(phases)


def part_decode(s, out, a):
    phases = parse_phases(a.phases)
    n_dec = int(a.decode_tokens)
    ctxs = [512] if a.smoke else a.only_ctx or [1024, 8192, 32768]
    for ctx in ctxs:
        for kind in a.only_kind or ("prose", "code"):
            text = (
                "Write a short example and explain it."
                if a.smoke
                else decode_prompt(kind, ctx, a.long_ask)
            )
            reply = ""
            for phase in [p for p in ("cold", "warm", "turn2") if p in phases]:
                want = (a.turn2_tokens or n_dec) if phase == "turn2" else n_dec
                b = req(
                    s.model,
                    text,
                    16 if a.smoke else want,
                    extra=json.loads(a.request_extra) if a.request_extra else None,
                )
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
                emit(
                    out,
                    part="decode",
                    ctx=ctx,
                    kind=kind,
                    phase=phase,
                    reference_prompt_tokens=ctx
                    if os.environ.get("TFB_EXACT_PROMPTS") == "1" and not a.smoke
                    else None,
                    **r,
                )
            if (
                a.engine in bench_engines.TF_ENGINES
                and ctx == 1024
                and os.environ.get("TFB_EXACT_PROMPTS") != "1"
                and "specoff" in phases
            ):
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
        lo, hi = (int(x) for x in (a.items or f"0:{len(items)}").split(":"))
        for i, (nm, code) in enumerate(items):
            if not lo <= i < hi:
                continue
            text = hay + needle_question(nm)
            if not a.smoke and os.environ.get("TFB_EXACT_PROMPTS") == "1":
                from snapshot_prompts import assert_chat_prompt, exact_chat_prompt

                text = exact_chat_prompt(hay * 2, ctx, needle_question(nm))
                assert_chat_prompt(text, ctx)
            r = send(s.url, req(s.model, text, 16))
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
    """Two sub-agents at once, each a growing agentic conversation: a warm 32K prefix, then three
    rounds that each append ~2K of new text after the previous reply (the usual agent pattern)
    and ask for a 1K reply. Reports TTFT, cached tokens and per-request decode per round."""
    ctx, n_new = (512, 16) if a.smoke else (32768, 1024)
    kinds = ("prose", "code")
    prefixes = {
        k: ("Hello. " * 80 if a.smoke else load_prompt(f"{k}-{ctx}")) for k in kinds
    }
    for k in kinds:  # warm each prefix
        send(s.url, req(s.model, prefixes[k], 8))
    msgs = {k: [{"role": "user", "content": prefixes[k]}] for k in kinds}
    for rnd in range(1 if a.smoke else 3):

        def turn(k, rnd=rnd):
            other = "code" if k == "prose" else "prose"
            src = "Extra." * 20 if a.smoke else load_prompt(f"{other}-8192")
            piece = src[rnd * 6500 : rnd * 6500 + 6500]
            if rnd == 0:
                msgs[k][0]["content"] += "\n\nNew material:\n" + piece + "\nContinue."
            else:
                msgs[k].append(
                    {
                        "role": "user",
                        "content": "More material:\n" + piece + "\nContinue.",
                    }
                )
            b = req(s.model, "", n_new)
            b["messages"] = list(msgs[k])
            r = send(s.url, b)
            msgs[k].append({"role": "assistant", "content": r["_text"]})
            return r

        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(2) as ex:
            rs = list(ex.map(turn, kinds))
        wall = time.perf_counter() - t0
        for r in rs:
            if not a.smoke:
                check_decode_len(r, n_new, f"conc32 round {rnd}")
            r.pop("_text")
        emit(
            out,
            part="conc32",
            trial=rnd,
            wall_s=round(wall, 2),
            ttfts=[r["ttft_s"] for r in rs],
            per_req_dec=[r["dec_tps"] for r in rs],
            cached=[r["cached"] for r in rs],
            pts=[r["pt"] for r in rs],
            cts=[r["ct"] for r in rs],
            shas=[r["sha"] for r in rs],
        )


def part_conc(s, out, a):
    ns = [int(x) for x in (a.conc_ns or "2,4,8").split(",")]
    for n in (2,) if a.smoke else ns:
        for trial in range(1 if a.smoke else a.conc_trials):
            texts = [
                "Say hello."
                if a.smoke
                else load_prompt(
                    f"conc-{n}-{trial}-{'prose' if (i + trial) % 2 == 0 else 'code'}-{i}"
                    if os.environ.get("TFB_EXACT_PROMPTS") == "1"
                    else f"conc-{'prose' if (i + trial) % 2 == 0 else 'code'}-{i}"
                )
                for i in range(n)
            ]
            if not a.smoke and os.environ.get("TFB_EXACT_PROMPTS") == "1":
                from snapshot_prompts import assert_chat_prompt

                for text in texts:
                    assert_chat_prompt(text, 32768)
            t0 = time.perf_counter()
            with cf.ThreadPoolExecutor(n) as ex:
                rs = list(
                    ex.map(
                        lambda t: send(
                            s.url, req(s.model, t, 16 if a.smoke else a.conc_tokens)
                        ),
                        texts,
                    )
                )
            for r in rs:
                if not a.smoke:
                    check_decode_len(r, a.conc_tokens, f"conc n={n} trial={trial}")
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
                cached=[r["cached"] for r in rs],
                pts=[r["pt"] for r in rs],
                cts=[r["ct"] for r in rs],
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
        "--items", default="", help="needle item range a:b (one job per slice)"
    )
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
    ap.add_argument(
        "--conc-ns", default="", help="concurrency levels, e.g. 2,4 (default 2,4,8)"
    )
    ap.add_argument("--conc-trials", type=int, default=2)
    ap.add_argument("--conc-tokens", type=int, default=256)
    ap.add_argument(
        "--ctx-tokens",
        type=int,
        default=140000,
        help="context window to start engines that need one (llama.cpp -c)",
    )
    ap.add_argument(
        "--parallel", type=int, default=1, help="server slots (llama.cpp -np)"
    )
    ap.add_argument(
        "--idle-s",
        type=float,
        default=30.0,
        help="pause before the idle footprint sample",
    )
    ap.add_argument(
        "--request-extra",
        default="",
        help="JSON object merged into every decode request body (e.g. penalties)",
    )
    ap.add_argument(
        "--phases",
        default=",".join(ALL_PHASES),
        help="decode phases to run (comma list of cold,warm,turn2,specoff); the checks stay "
        "fail-closed for every selected phase",
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
                    decode_prompt(kind, ctx, a.long_ask)
        elif a.part in ("conc", "agent", "ca"):
            for directory in (BODIES, BODIES2):
                if not list(directory.glob("*-req.json")):
                    raise FileNotFoundError(directory)
        if a.engine in bench_engines.ENGINES:
            gone = bench_engines.missing_paths(a.engine)
            if gone:
                raise FileNotFoundError(f"{a.engine}: missing {gone}")
        elif not (Path(a.model) / "config.json").is_file():
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
    if (
        a.engine in bench_engines.ENGINES
        and bench_engines.ENGINES[a.engine].kind == "yunshu"
    ):
        expected = os.environ.get("TFB_EXPECT_YUNSHU_SHA")
        if expected and bench_engines.tree_version(YUNSHU_SRC)[1] != expected:
            raise AssertionError(
                "Yunshu pinned tree does not match expected release SHA"
            )
    s = start_server(
        a.engine,
        extra_env,
        f"{a.engine}-{a.part}-{a.rep}{a.tag}",
        model=a.model,
        ctx_tokens=a.ctx_tokens,
        parallel=a.parallel,
    )
    try:
        with open(a.out, "a") as out:
            if a.engine in bench_engines.ENGINES:
                version, sha = bench_engines.engine_version(a.engine, YUNSHU_SRC)
                META.update(
                    bench_engines.meta_row(
                        a.engine,
                        version=version,
                        git_sha=sha,
                        engaged=s.engaged_spec_mode,
                        flags=(s.launch.flags if s.launch else {"drafter": D}),
                    )
                )
                META["snapshot_rep"] = a.rep
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
                engine_probe=getattr(s, "probe", None),
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
            mem = getattr(s, "mem", None)
            if mem is not None:
                time.sleep(a.idle_s)
                idle = mem.sample()
                peak = max(mem.peak_gib, idle or 0.0)
                if not peak or idle is None:
                    raise RuntimeError("memory sampling produced no data")
                emit(
                    out,
                    part="memory",
                    peak_gib=peak,
                    idle_gib=idle,
                    memory_method=mem.method,
                    baseline_used_gib=round((mem.baseline_bytes or 0) / 2**30, 3),
                    peak_footprint_gib=mem.peak_footprint_gib,
                    idle_footprint_gib=mem.last_footprint_gib,
                    peak_rss_gib=mem.peak_rss_gib,
                    idle_rss_gib=mem.last_rss_gib,
                    samples=mem.n,
                    idle_after_s=a.idle_s,
                    engine_probe=s.probe_text(),
                )
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
