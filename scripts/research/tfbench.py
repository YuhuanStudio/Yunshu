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

import jsonschema

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_engines  # noqa: E402
import fullcmp_manifest as fm  # noqa: E402  (request families: FAMILY_KINDS, IMAGE_COUNT)
import fullcmp_attribution  # noqa: E402  (layer-1 attribution schema)
import fullcmp_engaged as fe  # noqa: E402  (engaged spec mode from evidence, never the request)
import fullcmp_provenance as fp  # noqa: E402
import fullcmp_quiet as fq  # noqa: E402  (quiet-machine snapshots per cell, FULLCMP_SPEC)
from gpuq_contention import was_contended  # noqa: E402

# Identifies this harness in every request row (fullcmp audit: provenance_identity).
HARNESS_ID = "tfbench/fullcmp-1"

M = "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
D = "/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2"
TF = {
    "tf-old": "/Volumes/P5Plus/yunshu-test-envs/tensorfold-0.3.6.1/bin/tensorfold",
    "tf-new": "/Volumes/P5Plus/yunshu-test-envs/tensorfold-0.6.1/bin/tensorfold",
}
is_yunshu_command = bench_engines.is_yunshu_command
yunshu_args = bench_engines.yunshu_args
DEFAULT_YUNSHU_BIN = "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/yunshu"
YUNSHU_BIN = DEFAULT_YUNSHU_BIN
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


def server_home(out: Path, tag: str, environ) -> Path:
    """HOME of a server. In a full-comparison job (TFB_FULLCMP=1, TFB_JOB_LABEL set by the
    driver) every server of the job uses the job's one directory: the audit's fairness control
    is one isolated HOME per gpuq job, and the job's cells run their servers back to back. Other
    runs keep one directory per server tag."""
    label = environ.get("TFB_JOB_LABEL", "")
    if environ.get("TFB_FULLCMP") == "1" and label:
        return out / "home" / label
    return out / "home" / tag


def ephemeral_port():
    """An OS-assigned loopback port (never the shared 18990-18999 pool)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def parse_lsof_pids(text):
    """PIDs from `lsof -t` output (one per line); blank or junk lines are ignored."""
    return sorted({int(x) for x in text.split() if x.isdigit()})


def listening_pids(port, run=subprocess.run):
    """PIDs holding a LISTEN socket on the TCP port, named by lsof. Read-only."""
    result = run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return parse_lsof_pids(result.stdout or "")


def assert_port_free(port, run=subprocess.run):
    """Fail closed when a cell's port is already bound: name the holder, never kill it."""
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return
        except OSError:
            pass
    pids = listening_pids(port, run=run)
    holder = ", ".join(str(p) for p in pids) or "unknown (lsof named no pid)"
    raise RuntimeError(
        f"port {port} is already bound by pid {holder}; refusing to start a cell on it "
        f"(check: lsof -nP -iTCP:{port} -sTCP:LISTEN). Nothing was killed."
    )


def free_port(wait_s=600.0, interval_s=5.0, sleep=None, clock=None):
    """Paused gpuq jobs retain ports; wait for the bounded shared pool. Stub runs (TFB_STUB_SERVER,
    tests only) take an ephemeral port so they never touch the pool."""
    if os.environ.get("TFB_STUB_SERVER"):
        return ephemeral_port()
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


def _hidden_size(path):
    try:
        cfg = json.loads((Path(path) / "config.json").read_text())
    except (OSError, ValueError):
        return None
    for c in (cfg, cfg.get("text_config") or {}):
        if c.get("hidden_size"):
            return int(c["hidden_size"])
    return None


def default_drafter(model):
    """The 27B DFlash2 drafter only for a target it fits (same hidden size); any other model
    (Flash-Next: 2560) drafts with its own MTP head. Forcing the 27B drafter on it failed
    every Flash-Next yv run at server start (2026-11-02)."""
    if model is None:
        return D
    want = _hidden_size(model)
    return D if want is not None and want == _hidden_size(D) else "mtp"


def spec_request(engine, extra_env, model=None):
    """Make the comparison independent of drafter discovery under isolated HOME."""
    if engine != "yunshu":
        return "dflash", dict(extra_env)
    env = dict(extra_env)
    # Benchmarks own only their API port; never auto-start the console on8100.
    env.setdefault("YUNSHU_CONSOLE", "0")
    override = env.setdefault("YUNSHU_VLM_DRAFT", default_drafter(model))
    mode = {"mtp": "mtp", "force-mtp": "mtp", "off": "off", "none": "off"}.get(
        override.lower(), "dflash"
    )
    return mode, env


def engaged_spec_mode_line(engine, log):
    """(engaged mode, the log line it was parsed from), read from the initialized runner, not
    the earlier selection/fallback message. (None, None) when the log shows no mode."""
    if engine == "yunshu":
        found = list(
            re.finditer(r"VLM batch runner: [^\n]*?draft=(dflash|mtp|off)\b", log)
        )
        if not found:
            return None, None
        m = found[-1]
        start = log.rfind("\n", 0, m.start()) + 1
        end = log.find("\n", m.end())
        return m.group(1), log[start : end if end != -1 else len(log)]
    if engine.startswith("omlx-"):
        # omlx prints this once the DFlash drafter is attached as a batched drafter (the
        # high-confidence pattern of fullcmp_audit_manifest.EXPECTED_MODE_PROBE)
        m = re.search(
            r"DFlash enabled for \S+ as a batched drafter, draft=\S+[^\n]*", log
        )
        if m:
            start = log.rfind("\n", 0, m.start()) + 1
            end = log.find("\n", m.end())
            return "dflash", log[start : end if end != -1 else len(log)]
    m = re.search(r"\[tensorfold\] drafter [^\n]*DFlash[^\n]*block=[^\n]*", log)
    if m:
        return "dflash", m.group(0)
    if engine.startswith("tf-"):
        # TensorFold 1.2.x serving line (bench_engines.detect_mode reads the same evidence)
        m = re.search(r"drafts: DFlash2\b[^\n]*", log)
        if m:
            start = log.rfind("\n", 0, m.start()) + 1
            end = log.find("\n", m.end())
            return "dflash", log[start : end if end != -1 else len(log)]
        # TensorFold 1.1.8 (2026-11-03 smoke): "Qwen3.8-27B: ... DFlash2 drafts for one of them"
        # with the serving line "drafts: on" (bench_engines.detect_mode reads the same pair)
        m = re.search(r"DFlash2 drafts for [^\n]*", log)
        if m and re.search(r"drafts: on\b", log):
            start = log.rfind("\n", 0, m.start()) + 1
            end = log.find("\n", m.end())
            return "dflash", log[start : end if end != -1 else len(log)]
    if engine.startswith("splash"):
        # Splash 1.5.0 (2026-11-03 smoke): printed once the DFlash2 draft is loaded and wired into
        # the decode loop (the /status probe is only corroboration)
        m = re.search(r"Decode cycles verify every row of the draft's block[^\n]*", log)
        if m:
            start = log.rfind("\n", 0, m.start()) + 1
            end = log.find("\n", m.end())
            return "dflash", log[start : end if end != -1 else len(log)]
    return None, None


def engaged_spec_mode(engine, log):
    return engaged_spec_mode_line(engine, log)[0]


# Warnings that mean a tuned kernel path silently turned off for this model.
_DISABLED_FAST_PATHS = (re.compile(r"ragged KV off: [^\n]*"),)


def disabled_fast_path(log):
    """The first log line saying a tuned fast path is off, or None."""
    for pattern in _DISABLED_FAST_PATHS:
        if m := pattern.search(log):
            return m.group(0)
    return None


def ready_request(url, extra_env):
    """Readiness probe; the server may require the key the arm configured."""
    token = extra_env.get("YUNSHU_AUTH_TOKEN", "k")
    return urllib.request.Request(
        url + "/v1/models", headers={"Authorization": "Bearer " + token}
    )


def warm_lazy_engine(engine, url, model):
    # /v1/models is discovery-only for lazy servers. Load the target/drafter before
    # checking the engaged mode, and keep this startup request out of timing.
    if engine in (
        "omlx",
        "llamacpp",
        "omlx-latest",
        "omlx-0.7.3",
        "omlx-0.7.3rc1",
        "llamacpp-mtp",
        "llamacpp-dflash",
        "fn-omlx-latest",
        "fn-llamacpp-mtp",
        "ollama-mlx",
    ):
        send(url, req(model, "Write a short example and explain it.", 16), timeout=240)


def wait_startup_ready(proc, log, pattern, timeout=120):
    """Some engines advertise HTTP before their background accelerator compiler is ready."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("server exited during accelerator startup")
        if re.search(pattern, log.read_text(errors="replace")):
            return
        time.sleep(0.2)
    raise RuntimeError(f"accelerator startup not ready: {pattern}")


def stub_server_cmd(engine, port, log, stub_dir, version=None) -> list[str]:
    """TEST-ONLY: argv of the fullcmp stub server that replaces a real engine launch when
    TFB_STUB_SERVER=<dir> is set. The stub is <dir>/fullcmp_stub_server.py (the tests point it at
    tests/unit), so no engine is started. Its engine lines go to the server log; version is the
    version it reports on /v1/yunshu/status (the pinned version for yunshu)."""
    cmd = [
        sys.executable,
        # TFB_STUB_SCRIPT picks another stub file in the same dir (tests: the request-family stub)
        str(Path(stub_dir) / os.environ.get("TFB_STUB_SCRIPT", "fullcmp_stub_server.py")),
        "--engine",
        engine,
        "--port",
        str(port),
        "--log",
        str(log),
    ]
    if version:
        cmd += ["--version", str(version)]
    return cmd


class Srv:
    @staticmethod
    def _is_yunshu(engine):
        return engine == "yunshu" or (
            bench_engines.is_new_engine(engine)
            and bench_engines.ENGINES[engine].kind == "yunshu"
        )

    def __init__(
        self,
        engine,
        extra_env,
        tag,
        model=None,
        ctx_tokens=140000,
        parallel=1,
        reuse_home=False,
    ):
        # Fullcmp: a Yunshu server runs only from the pinned tree and venv, or not at all.
        # Checked before any port, directory or process is created.
        self.pin = None
        self.observe = None  # fullcmp Yunshu: True (profile) or False (timed); see observe_state
        if fp.fullcmp_mode(os.environ) and self._is_yunshu(engine):
            self.pin = fp.require_yunshu_pin(os.environ)
            self.observe = fp.observe_state(os.environ, True)
        self.engine, self.port = engine, free_port()
        assert_port_free(self.port)  # fail closed before any dir or process is created
        self.tag, self.model_arg = tag, model
        self.ctx_tokens, self.parallel = ctx_tokens, parallel
        self.requested_spec_mode, extra_env = spec_request(
            engine, extra_env, model or M
        )
        self.launch = None
        self.probe = None
        if bench_engines.is_new_engine(engine):
            self.requested_spec_mode = bench_engines.ENGINES[engine].expected_mode
        self.extra_env = extra_env
        self.home = server_home(OUT, tag, os.environ)
        if not reuse_home:  # a restart keeps HOME: the disk prefix tier lives there
            shutil.rmtree(self.home, ignore_errors=True)
        self.home.mkdir(parents=True, exist_ok=True)
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
                own_py = own_venv_bin(YUNSHU_SRC, "python")
                if own_py and bench_engines.is_yunshu_command(cmd):
                    cmd = [
                        own_py,
                        "-m",
                        bench_engines.YUNSHU_MODULE,
                        *bench_engines.yunshu_args(cmd),
                    ]
        elif engine == "yunshu":
            if YUNSHU_SRC:
                env["PYTHONPATH"] = YUNSHU_SRC
            own_py = own_venv_bin(YUNSHU_SRC, "python")
            if own_py:
                entry = [own_py, "-m", bench_engines.YUNSHU_MODULE]
            elif YUNSHU_BIN != DEFAULT_YUNSHU_BIN:
                entry = [YUNSHU_BIN]  # explicit launcher override (probes set it)
            else:
                entry = bench_engines.yunshu_entry(os.environ)
            cmd = [
                *entry,
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
        # A watched engine always runs in observe mode; TFB_OBSERVE=0 (or an explicit
        # YUNSHU_OBSERVE via --env) keeps a pure timing run free of telemetry.
        is_yunshu = self._is_yunshu(engine)
        if fp.fullcmp_mode(os.environ) and is_yunshu:
            # Fullcmp: the server env carries exactly what observe_state decided (no leftovers,
            # no --env override); the header records the value actually passed.
            if "YUNSHU_OBSERVE" in extra_env:
                raise fp.ProvenanceError("fullcmp: YUNSHU_OBSERVE must not come from --env")
            env.pop("YUNSHU_OBSERVE", None)
            if self.observe:
                env["YUNSHU_OBSERVE"] = "1"
        elif is_yunshu and os.environ.get("TFB_OBSERVE", "1") != "0":
            env.setdefault("YUNSHU_OBSERVE", "1")
        self.yunshu_observe_env = env.get("YUNSHU_OBSERVE")
        if self.pin is not None:
            # Pinned: `<pinned venv python> -m yunshu_cli serve ...` from the pinned tree only.
            if not bench_engines.is_yunshu_command(cmd):
                raise fp.ProvenanceError(
                    f"pinned launch is not a yunshu command: {cmd[:2]}"
                )
            cmd = fp.pinned_yunshu_cmd(cmd, self.pin, bench_engines.yunshu_args)
            env["PYTHONPATH"] = self.pin["src"]
        self.cmd = cmd
        baseline_used = (
            host_used_bytes()
        )  # before launch: weights/mmap count in the delta
        stub_dir = os.environ.get("TFB_STUB_SERVER")
        if stub_dir:
            # TEST-ONLY (fullcmp dry run): the server is tests/unit/fullcmp_stub_server.py on
            # this port, writing its engine lines to this server's log. Launch, readiness and
            # every check after it stay the same code path.
            cmd = stub_server_cmd(
                self.engine,
                self.port,
                self.log,
                stub_dir,
                version=(self.pin or {}).get("version"),
            )
            self.cmd = cmd
            self.log.write_bytes(
                b""
            )  # as the real launch does: the log exists before readiness
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
        else:
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
        # the log offset after readiness: a match before it is startup evidence (loaded only)
        self.ready_log_offset = self.log.stat().st_size if self.log.exists() else 0
        try:
            startup = getattr(bench_engines.ENGINES.get(self.engine), "extra", {})
            if pattern := startup.get("startup_ready_regex"):
                wait_startup_ready(
                    self.proc, self.log, pattern, startup.get("startup_timeout_s", 120)
                )
                self.ready_s = time.time() - t0
            warm_lazy_engine(self.engine, self.url, self.model)
            if self.pin is not None:
                self.verify_pinned()
            self.verify_spec_mode()
        except Exception:
            self.kill()
            raise

    def verify_pinned(self):
        """The running server reports the pinned version and runs from the pinned python."""
        req = urllib.request.Request(
            self.url + "/v1/yunshu/status",
            headers={
                "Authorization": "Bearer "
                + self.extra_env.get("YUNSHU_AUTH_TOKEN", "k")
            },
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            status = json.load(r)
        ps = subprocess.run(
            ["ps", "-o", "command=", "-p", str(self.proc.pid)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        fp.verify_pinned_status(status, self.pin)
        if os.environ.get("TFB_STUB_SERVER"):
            # TEST-ONLY: the live process is the stub, so its command line cannot be the pinned
            # python; every other pin check (version, binary, paths) still runs.
            return
        fp.verify_pinned_cmdline(ps.stdout, self.pin)

    def probe_text(self):
        path = self.launch.probe if self.launch else None
        if not path:
            return None
        try:
            with urllib.request.urlopen(self.url + path, timeout=10) as r:
                return r.read().decode("utf-8", "replace")[:20000]
        except Exception as e:  # noqa: BLE001
            return f"probe failed: {e!r}"

    def verify_spec_mode(self, final=False, rows=(), probe_response=None, concurrent=(), concurrent_probe=None):
        log = self.log.read_text(errors="replace")
        if fp.fullcmp_mode(os.environ) and self._is_yunshu(self.engine) and final and concurrent and not rows:
            # a Yunshu concurrent cell: the same derived verdict the audit re-derives (fe.concurrent_check), not the
            # runner line (the header and the audit must agree on the mode and the line)
            ev, problems = fe.concurrent_check(fe.load_table(), self.engine, self.requested_spec_mode, log,
                                               concurrent, getattr(self, "ready_log_offset", None))
            self.engaged_spec_mode = ev["engaged_mode"]
            self.engaged_log_line = ev["matched_line"]
            self.engaged_evidence_kind = ev["evidence_kind"]
            self.engaged_evidence_source = ev["source"]
            self.engaged_label = ev.get("label")
            if problems:
                raise RuntimeError("concurrent evidence: " + "; ".join(problems) + f"; see {self.log}")
            return
        if fp.fullcmp_mode(os.environ) and not self._is_yunshu(self.engine):
            # Fullcmp: the engaged mode comes from the evidence table only (fullcmp_engaged); the
            # request is never copied into the header. At startup only the load lines exist (a
            # startup line proves 'loaded'), so only the final check (after the requests, with the
            # cell's records) may refuse a cell.
            self.probe = self.probe_text()
            self._verify_fullcmp_evidence(
                log, final=final, rows=rows, probe_response=probe_response, concurrent=concurrent,
                concurrent_probe=concurrent_probe
            )
            return
        # The engaged-mode line the audit checks (fullcmp header engaged_log_line).
        self.engaged_log_line = engaged_spec_mode_line(
            "yunshu" if self._is_yunshu(self.engine) else self.engine, log
        )[1]
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
        if self.engine == "yunshu" and (off := disabled_fast_path(log)):
            # the lane's spec on == spec off rests on these kernels; a run without them
            # measures a different engine (2026-10-20: ragged KV off broke identity)
            raise RuntimeError(f"{off}; see {self.log}")

    def _verify_fullcmp_evidence(self, log, final=False, rows=(), probe_response=None, concurrent=(), concurrent_probe=None):
        table = fe.load_table()  # None when the table is missing or malformed: no evidence at all
        ready = getattr(self, "ready_log_offset", None)
        if concurrent and not rows:
            # concurrent cells: the same shared verdict the audit uses (fe.concurrent_check)
            ev, problems = fe.concurrent_check(table, self.engine, self.requested_spec_mode, log, concurrent, ready,
                                               probe_responses=concurrent_probe)
            if problems and final:
                text = self.log.read_text(errors="replace") if self.log.exists() else ""
                shown = [ln for ln in text.splitlines() if re.search(r"done|chat:|MTP\[", ln)][:10]
                raise RuntimeError("concurrent evidence: " + "; ".join(problems) + f"; see {self.log}"
                                   + "; first done lines: " + " | ".join(shown))
        else:
            ev = fe.evaluate(
                self.engine,
                self.requested_spec_mode,
                log,
                rows,
                table,
                probe_text=self.probe,
                ready_offset=ready,
                probe_response=probe_response,
            )
        self.engaged_spec_mode = ev["engaged_mode"]
        self.engaged_log_line = ev["matched_line"]
        self.engaged_evidence_kind = ev["evidence_kind"]
        self.engaged_evidence_source = ev["source"]
        self.engaged_label = ev.get("label")
        self.engaged_per_row = (ev.get("stats") or {}).get("per_row", [])  # reported with the base mode (e.g. 'dflash (kflash tree)')
        derived_ar = bool(concurrent) and not rows and self.engaged_spec_mode == "ar"
        if final and self.engaged_spec_mode != self.requested_spec_mode and not derived_ar:
            raise RuntimeError(
                f"requested spec={self.requested_spec_mode}, engaged={self.engaged_spec_mode} "
                f"(evidence: {self.engaged_evidence_kind}, table: {'ok' if table else 'missing'}); "
                f"see {self.log}"
            )

    def kill(self, keep_disk=False):
        if self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                self.proc.wait(30)
        # The server's SSD prefix cache (up to ~8 GB per 128K cell) is dead once the server is:
        # the next server of this tag starts from an emptied home anyway. Left behind, finished
        # yv runs held 362 GB of it on P5Plus (2026-10-07). A restart cell stops its first server
        # with keep_disk=True: the relaunched server reads that cache from the same HOME.
        if self.proc.poll() is not None and not keep_disk:
            for sub in (".yunshu/cache", "omlx-ssd", "mtplx-cache", "omlx-base/cache"):
                shutil.rmtree(self.home / sub, ignore_errors=True)
        if getattr(self, "mem", None):
            self.mem.stop()


# Every server a run started (the restart phase replaces the live one): main() kills all of them.
_LIVE_SERVERS: list = []
# The first raw chunk that carried a usage object (the server's own usage shape, as sent).
_USAGE_SAMPLE: list = [None]


# Full bytes of each engine log that a restart truncated during the current cell (restart_server).
_LOG_PREFIX = []


def server_log_path(out):
    """The cell's engine-log slice: <cell>-rN.server.log beside <cell>-rN.jsonl (or -prof.jsonl)."""
    text = str(out)
    stem = text[: -len(".jsonl")] if text.endswith(".jsonl") else text
    return stem + ".server.log"


def job_log_suffix() -> str:
    """A suffix unique to this job/chunk, so two chunks of one cell never share an engine log file
    (wire-1106d: mixed-0 of p1 and p2 both wrote server-omlx-0.7.3-mixed-0.log; p2 overwrote p1's evidence)."""
    job = os.environ.get("GPUQ_JOB_ID") or os.environ.get("TFB_RUN_ID") or ""
    chunk = os.environ.get("TFB_CHUNK") or ""
    parts = [p for p in (job, chunk) if p]
    return ("-" + "-".join(re.sub(r"[^A-Za-z0-9._-]", "_", p) for p in parts)) if parts else ""


def cell_engine_log(log_path, end=None, prefix=()):
    """This cell's engine log. Each launch truncates its log (open 'wb'), so the cell's slice is the
    whole server log from byte 0: the engaged-mode evidence is printed at startup, before readiness
    (Splash prints its drafter line there and logs nothing per request), so a slice that began at
    readiness was empty for Splash (2026-11-04). prefix: bytes of logs a restart truncated."""
    return cell_log_slice(log_path, 0, end, prefix=prefix)


def cell_log_slice(log_path, start, end=None, prefix=()):
    """Bytes of one cell's engine log: the shared log from offset start to end (None: its end),
    after the full bytes of any log a restart in this cell truncated (prefix, in order)."""
    log_path = Path(log_path)
    data = log_path.read_bytes() if log_path.exists() else b""  # no log: an empty slice (audit fails)
    if not prefix:
        return data[start:end]
    pieces = [*prefix, data]
    pieces[0] = pieces[0][start:]
    pieces[-1] = pieces[-1] if end is None else pieces[-1][:end]
    return b"".join(pieces)


def restart_server(s):
    """Stop the live server (its disk prefix tier is kept) and launch the same engine again on the
    same HOME: a fresh process and port, the same cache directory (restart-with-disk-cache)."""
    log = getattr(s, "log", None)  # a stub server has no engine log
    if log is not None and log.exists():
        _LOG_PREFIX.append(log.read_bytes())  # the relaunch truncates this log: keep its bytes
    s.kill(keep_disk=True)
    if os.environ.get("TFB_STUB_SERVER") and os.environ.get(
        "TFB_STUB_WIPE_HOME_ON_RESTART"
    ) == "1":
        # TEST-ONLY negative case: a restart on a wiped HOME must lose the disk tier (the audit fails)
        shutil.rmtree(s.home, ignore_errors=True)
    fresh = start_server(
        s.engine,
        s.extra_env,
        s.tag,
        model=s.model_arg,
        ctx_tokens=s.ctx_tokens,
        parallel=s.parallel,
        reuse_home=True,
    )
    _LIVE_SERVERS.append(fresh)
    return fresh


CACHED_UNAVAILABLE = (
    "usage has no cached-token count (prompt_tokens_details.cached_tokens or usage.cached_tokens)"
)


def cached_tokens(usage):
    """Keep absent cache accounting unknown; zero is a measured cache miss. Two server shapes
    carry the count: OpenAI's prompt_tokens_details.cached_tokens and a top-level usage.cached_tokens."""
    details = usage.get("prompt_tokens_details") or {}
    value = details.get("cached_tokens")
    if value is None:
        value = usage.get("cached_tokens")
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= usage["prompt_tokens"]
    ):
        raise ValueError(f"invalid cached_tokens: {value!r}")
    return value


def concurrent_prompt(index, trial, ctx, variant=0):
    """Independent exact-size requests; salt at the front prevents shared-prefix priming."""
    from snapshot_prompts import exact_chat_prompt

    kind = "prose" if (index + trial) % 2 == 0 else "code"
    source = load_variant_prompt(f"{kind}-32768", variant)
    return exact_chat_prompt(f"Request {trial}:{index}.\n" + source, ctx, LONG_ASK)


def prompt_digest(messages):
    """sha256 of the request messages exactly as sent (canonical JSON). Two engines that got the same
    digest were given the same prompt text; gaptable.py's summary fails closed when they differ."""
    return hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


_ACTIVE_TELEMETRY = (
    None  # the cell's fullcmp Sampler while a cell runs (marks go to it)
)
_REQ_SEQ = [0]


def _mark(ev, **fields):
    """Request event for the cell's telemetry sampler (no-op when no sampler is running)."""
    if _ACTIVE_TELEMETRY is not None:
        _ACTIVE_TELEMETRY.mark(ev, **fields)


def send(
    url, body, timeout=600, *, capture_timestamps=False, trace_id=None, sample=False
):
    _progress()
    try:
        return _send_impl(
            url, body, timeout, capture_timestamps=capture_timestamps,
            trace_id=trace_id, sample=sample,
        )
    finally:
        _progress()


def _send_impl(
    url, body, timeout=600, *, capture_timestamps=False, trace_id=None, sample=False
):
    # sample=True marks a real benchmark request: only it may fill --log-usage-sample. Warm-up and
    # readiness requests (short "Say hi" style probes) must not, or the sample is not the benchmark.
    prompt_sha = prompt_digest(body.get("messages"))
    max_tokens, temperature = body.get("max_tokens"), body.get("temperature")
    stop = list(body.get("stop") or [])
    body = dict(body, stream=True, stream_options={"include_usage": True})
    headers = {"Content-Type": "application/json", "Authorization": "Bearer k"}
    if trace_id is not None:
        headers["X-TTFT-Probe-ID"] = trace_id
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        headers,
    )
    _REQ_SEQ[0] += 1
    rid = str(_REQ_SEQ[0])
    t0 = time.perf_counter()
    ew_receive = time.time()  # layer-1 attribution: client wall clock (epoch s)
    ew_first = ew_last = None
    tf = None
    content, reasoning, tools = [], [], {}
    lp_tokens, lps = [], []
    usage = finish = xy = None
    done = False
    nchunks = 0
    _mark("request_start", req=rid)
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
                if sample and _USAGE_SAMPLE[0] is None:
                    _USAGE_SAMPLE[0] = d  # the final usage chunk of the first benchmark request
            if d.get("x_yunshu"):
                xy = d["x_yunshu"]
            elif isinstance(d.get("usage"), dict) and d["usage"].get("x_yunshu"):
                # the gateway puts x_yunshu inside the final usage object (x_yunshu.py: usage["x_yunshu"]),
                # not at the top level, on the include_usage chunk
                xy = d["usage"]["x_yunshu"]
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
                    ew_first = time.time()
                    _mark("first_token", req=rid, chunks=nchunks)
                if got:
                    ew_last = time.time()
                for e in (ch.get("logprobs") or {}).get("content") or []:
                    lp_tokens.append(e.get("token"))
                    lps.append(e.get("logprob"))
                finish = ch.get("finish_reason") or finish
    t1 = time.perf_counter()
    ew_end = time.time()
    if not done or finish is None or usage is None:
        raise RuntimeError("incomplete stream")
    ct = usage["completion_tokens"]
    _mark("end", req=rid, chunks=nchunks, completion_tokens=ct)
    c = "".join(content)
    rs = "".join(reasoning)
    sha = hashlib.sha256(
        json.dumps(
            [c, rs, [tools[i] for i in sorted(tools)], finish], ensure_ascii=False
        ).encode()
    ).hexdigest()[:16]
    result = dict(
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
        prompt_tokens=usage["prompt_tokens"],
        # The audit's name (prompt_ids_sha) and gaptable's (prompt_sha256) carry the same value:
        # sha256 of the request messages as sent. Servers report counts, not token ids.
        prompt_ids_sha=prompt_sha,
        prompt_sha256=prompt_sha,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=body.get("top_p"),
        top_k=body.get("top_k"),
        stop=stop,
        harness=HARNESS_ID,
        cached=cached_tokens(usage),
        cached_unavailable=None if cached_tokens(usage) is not None else CACHED_UNAVAILABLE,
        dec_tps=round((ct - 1) / (t1 - tf), 1) if tf and ct > 1 and t1 > tf else None,
        finish=finish,
        sha=sha,
        reasoning_chars=len(rs),
        lp_tokens=lp_tokens or None,
        lps=lps or None,
        head=(c or rs)[:80],
        _text=c,
        xy={
            k: xy.get(k)
            for k in (
                "speculative",
                "cache",
                "latency",
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
    result["_tools"] = [tools[i] for i in sorted(tools)]
    result["_image_tokens"] = (usage.get("prompt_tokens_details") or {}).get("image_tokens")
    result["_attr_client"] = {
        "t_receive": ew_receive,
        "t_first_token": ew_first,
        "t_last_token": ew_last,
        "t_response_end": ew_end,
        "cache_hit_tokens": result["cached"],
    }
    result["_xy_speculative"] = (xy or {}).get("speculative")
    result["_xy_latency"] = (xy or {}).get("latency")
    if capture_timestamps:
        result["_sent_at_s"] = t0
        result["_first_at_s"] = tf
    return result


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


# Smoke-only prompt variants (scripts/research/fullcmp_make_variants.py writes them). Variant 0 is the
# real corpus prompt in WORK/prompts; variant N is <name>-v<N>.txt in this directory.
PROMPT_VARIANT_DIR = Path(
    os.environ.get(
        "TFB_PROMPT_VARIANT_DIR", "/Volumes/Micron/yunshu-fullcmp/smoke-prompts"
    )
)


def prompt_path(name, variant=0):
    if variant == 0:
        return WORK / "prompts" / f"{name}.txt"
    return PROMPT_VARIANT_DIR / f"{name}-v{variant}.txt"


def load_prompt(name):
    return (WORK / "prompts" / f"{name}.txt").read_text()


def load_variant_prompt(name, variant=0):
    """Variant 0 is load_prompt (the real corpus); variant N reads the smoke-only directory."""
    if variant == 0:
        return load_prompt(name)
    return (PROMPT_VARIANT_DIR / f"{name}-v{variant}.txt").read_text()


def decode_prompt_file(kind, ctx, smoke=False, variant=0):
    """The prompt source a decode cell reads (recorded as prompt_file on every decode row)."""
    return "inline:smoke" if smoke else str(prompt_path(f"{kind}-{ctx}", variant))


def prompt_file_sha256(kind, ctx, smoke=False, variant=0):
    """sha256 of the prompt FILE bytes a decode cell read (the bytes prompt_files.sha256.txt hashes).
    None for a smoke cell (inline prompt). prompt_sha256 is the rendered request, not the file."""
    if smoke:
        return None
    return hashlib.sha256(prompt_path(f"{kind}-{ctx}", variant).read_bytes()).hexdigest()


def decode_prompt(kind, ctx, long_ask, variant=0):
    text = load_variant_prompt(f"{kind}-{ctx}", variant)
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


# ---- request families (B2 kind / thinking / image cells)
# Built on the real prose corpus plus a family ask; variant N changes only the ask, so two variants
# give two distinct prompts (the audit's >= 2 prompts per (model, ctx, kind) guard).
FAMILY_KINDS_LOCAL = fm.FAMILY_KINDS
FAMILY_SYSTEM = (
    "You are a coding agent. Use the provided tools when a file or shell command is needed. "
    "Reply with a tool call when one is needed."
)
FAMILY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run one shell command in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]
FAMILY_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
                "required": ["name", "count"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "items"],
    "additionalProperties": False,
}
# thinking / image cells on a plain kind: variant 0 is the corpus prompt, variant 1 adds one line
GENERIC_ASKS = ("", "Keep the answer short.")
FAMILY_ASKS = {
    "tool-chat": [
        "Read the file notes/todo.md with a tool call.",
        "List the files under src/ with the shell tool.",
    ],
    "cjk": [
        "請用中文逐段說明上面材料的主要概念，每段至少一句。",
        "請用繁體中文總結上面材料的重點，並保留專有名詞。",
    ],
    "json-schema": [
        "Summarise the material above as JSON with a title and a list of items (name, count).",
        "Extract three terms from the material above as JSON with a title and a list of items.",
    ],
}
# A valid 1x1 PNG, sent as the image_url data URI of each image part.
IMAGE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def is_family_cell(kind, thinking, media):
    return fm.is_family_cell(kind, thinking, media)


def family_apply(body, kind, variant, thinking, media):
    """Turn a plain decode body into the family's request, in place: the agent system prompt and
    tool schemas (tool-chat), a JSON-schema response_format (json-schema), the Qwen chat-template
    thinking switch (thinking on), and image_url content parts (img1 / img4)."""
    text = body["messages"][-1]["content"]
    asks = FAMILY_ASKS.get(kind, GENERIC_ASKS)
    ask = asks[variant % len(asks)]
    if ask:
        text = f"{text}\n\n---\n{ask}"
    msgs = [{"role": "user", "content": text}]
    if kind == "tool-chat":
        msgs.insert(0, {"role": "system", "content": FAMILY_SYSTEM})
        body["tools"] = FAMILY_TOOLS
        body["tool_choice"] = "auto"
    if kind == "json-schema":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "summary", "schema": FAMILY_SCHEMA, "strict": True},
        }
    n = fm.IMAGE_COUNT.get(media, 0)
    if n:
        url = "data:image/png;base64," + IMAGE_PNG_B64
        msgs[-1]["content"] = [{"type": "text", "text": text}] + [
            {"type": "image_url", "image_url": {"url": url}} for _ in range(n)
        ]
    body["messages"] = msgs
    body["chat_template_kwargs"] = {"enable_thinking": thinking == "on"}
    return body


def family_problems(kind, r, thinking, media):
    """Reasons a family reply is invalid (empty list = valid), judged from the client's stream.
    Tool calls must name a declared tool with JSON arguments that satisfy its schema; a json-schema
    reply must be JSON that validates; a cjk reply must be mostly CJK text; thinking on needs the
    reasoning split (reasoning_content) and no reasoning in the content; image input must be
    accepted and reported as image tokens."""
    probs = []
    content = r.get("_text") or ""
    if kind == "tool-chat":
        calls = r.get("tool_calls") or []
        if not calls:
            probs.append("no tool_calls in reply")
        declared = {t["function"]["name"]: t["function"] for t in FAMILY_TOOLS}
        for c in calls:
            fn = declared.get(c.get("name"))
            if fn is None:
                probs.append(f"undeclared tool {c.get('name')!r}")
                continue
            try:
                args = json.loads(c.get("arguments") or "")
            except ValueError:
                probs.append(f"tool {c['name']} arguments are not JSON")
                continue
            try:
                jsonschema.validate(args, fn["parameters"])
            except jsonschema.ValidationError as e:
                probs.append(f"tool {c['name']} arguments: {e.message}")
    if kind == "json-schema":
        try:
            obj = json.loads(content)
        except ValueError:
            probs.append("reply is not JSON")
        else:
            try:
                jsonschema.validate(obj, FAMILY_SCHEMA)
            except jsonschema.ValidationError as e:
                probs.append(f"reply does not match the schema: {e.message}")
    if kind == "cjk":
        chars = [ch for ch in content if not ch.isspace()]
        cjk = sum(1 for ch in chars if "一" <= ch <= "鿿")
        if not chars or cjk * 2 < len(chars) or "�" in content:
            probs.append(f"reply is not CJK text ({cjk}/{len(chars)} CJK characters)")
    if "<think>" in content or "</think>" in content:
        probs.append("reasoning leaked into the content")
    if thinking == "on" and not r.get("reasoning_chars"):
        probs.append("thinking on but no reasoning split (reasoning_content empty)")
    if fm.IMAGE_COUNT.get(media, 0) and not r.get("image_tokens"):
        probs.append("image input not accepted or no image tokens reported")
    return probs


def rehearsal_prefix(rendered, marker="<|im_start|>user"):
    """The next turn's shared prefix: the rendered conversation up to its last user header."""
    cut = rendered.rfind(marker)
    if cut <= 0:
        raise ValueError(f"marker {marker!r} not found in the rendered next turn")
    return rendered[:cut]


def _post_json(url, path, body, timeout=600):
    req = urllib.request.Request(
        url + path,
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer k"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def rehearsal_count(metrics_text, outcome):
    """Value of yunshu_next_turn_rehearsals_total{outcome=...} (0 when absent)."""
    for line in metrics_text.splitlines():
        if line.startswith("yunshu_next_turn_rehearsals_total{") and (
            f'outcome="{outcome}"' in line
        ):
            return int(float(line.rsplit(" ", 1)[1]))
    return 0


def completed_rehearsals(metrics_text):
    return rehearsal_count(metrics_text, "completed")


def check_rehearsal_engaged(s, a):
    """Fail closed: a server run with YUNSHU_NEXT_TURN_PREFILL=1 and a think time must have
    completed at least one rehearsal, or the A/B would measure a feature that never ran."""
    if str(s.extra_env.get("YUNSHU_NEXT_TURN_PREFILL", "")).lower() in (
        "",
        "off",
        "boundary",  # the no-rehearsal reference
        "0",
        "false",
    ):
        return None
    req = urllib.request.Request(
        s.url + "/metrics", headers={"Authorization": "Bearer k"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        done = completed_rehearsals(r.read().decode())
    if (a.think_s or getattr(a, "require_engaged", False)) and done < 1:
        raise RuntimeError(
            "YUNSHU_NEXT_TURN_PREFILL=1 but /metrics shows 0 completed rehearsals: "
            "the feature never engaged (not noise, broken run)"
        )
    return done


def rehearse(s, b):
    """Prefill the assistant reply the way the next turn will (opt-in pilot, --rehearse)."""
    kw = b.get("chat_template_kwargs")
    rendered = _post_json(
        s.url,
        "/apply-template",
        {"model": s.model, "messages": b["messages"], "chat_template_kwargs": kw},
    )["prompt"]
    prefix = rehearsal_prefix(rendered)
    t0 = time.perf_counter()
    r = _post_json(
        s.url,
        "/v1/completions",
        {"model": s.model, "prompt": prefix, "max_tokens": 1, "temperature": 0},
    )
    return {
        "rehearsal_s": round(time.perf_counter() - t0, 3),
        "rehearsal_usage": r.get("usage"),
    }


META: dict = {}  # engine / version / sha / flags / drafter / engaged spec mode / checkpoint, on every row


def host_used_bytes():
    """Host used memory (active+wired+compressor); None when vm_stat is unavailable."""
    try:
        from process_memory import system_used_bytes

        return system_used_bytes()
    except Exception:  # noqa: BLE001 - a baseline miss disables system-delta, never fakes it
        return None


def memory_peak_or_raise(mem, idle):
    """Peak for the memory part. "No data" means no successful sample at all. A legitimate 0.0 GiB
    delta (a tiny server that grew nothing against its baseline) is data, not a miss (regression,
    e2e 2026-11-03: the old `not peak` test failed the stub run on a real 0.0 delta)."""
    if idle is None or mem.n == 0:
        raise RuntimeError("memory sampling produced no data")
    return max(mem.peak_gib, idle)


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


def check_decode_len(r, n, what, allow_eos=False):
    """A decode cell measures exactly n tokens; a short reply is an error (a request that died
    mid-stream must not count as a result). Identity cells (allow_eos) accept the model's own end of
    reply: finish=stop with 1..n tokens; the paired comparison then requires both arms to stop at the
    same token (same sha and count), so an early stop can never hide a divergence."""
    if allow_eos and r.get("finish") == "stop" and 1 <= (r.get("ct") or 0) <= n:
        return
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
    bad = [p for p in phases if p not in ALL_PHASES + ("restart",)]
    if bad or not phases:
        raise SystemExit(
            f"--phases: unknown or empty selection {text!r} (use {ALL_PHASES + ('restart',)})"
        )
    if "turn2" in phases and "cold" not in phases:
        raise SystemExit("--phases: turn2 needs cold (it continues the cold reply)")
    return set(phases)


def part_corpus(s, out, a):
    """Cold decode of many distinct prompts on one server (averages out greedy-divergence content).

    A reply that stops early is recorded with ``short`` set instead of aborting the job, so the
    analysis counts only complete cells and reports how many were dropped."""
    n_dec = int(a.decode_tokens)
    shorts: list[str] = []
    for path in [Path(x) for x in a.prompt_files.split(",") if x]:
        r = send(s.url, req(s.model, path.read_text(), n_dec))
        r["text"] = r.pop("_text")
        short = r.get("finish") != "length" or r.get("ct") != n_dec
        emit(out, part="corpus", prompt=path.stem, phase="cold", short=short, **r)
        shorts.append(path.stem) if short else None
    if shorts:  # fail closed: a short reply is data loss, the job must not end complete
        raise RuntimeError(f"corpus: {len(shorts)} short replies: {shorts}")


def attribution_fields(s, r):
    """Layer-1 fields of one decode row (FULLCMP_SPEC "Attribution data"). Measured client-side
    values are present; server-side stages and counters the engine does not expose are
    'unavailable (reason)' from the engine's schema (fullcmp_attribution.ENGINE_OVERRIDES); every
    other field stays absent so the audit's attribution_fields guard reports it MISSING. Nothing
    is estimated. Only in fullcmp mode (TFB_FULLCMP=1), where an unknown engine is an error."""
    client = r.pop("_attr_client", {}) or {}
    spec = r.pop("_xy_speculative", None) or {}
    lat = r.pop("_xy_latency", None) or {}
    attr = {k: v for k, v in client.items() if v is not None}
    attr["engine_log_path"] = str(s.log)
    for f in ("rounds", "drafted", "accepted"):
        if isinstance(spec.get(f), (int, float)):
            attr[f] = int(spec[f])
    unavailable = {}
    if os.environ.get("TFB_FULLCMP") == "1":
        # The schema is keyed by manifest id (yunshu), the server runs as the bench id (yunshu-main).
        eng = fullcmp_attribution.engine_attr(
            os.environ.get("TFB_ATTR_ENGINE") or bench_engines.manifest_id(s.engine)
        )
        unavailable = dict(eng.unavailable)
        if eng.engine == "yunshu":
            _yunshu_server_fields(attr, unavailable, client, spec, lat)
    r["attr"] = attr
    r["attr_unavailable"] = unavailable
    return r


# Yunshu milestone name -> layer-1 field (x_yunshu.latency.milestones_ms, ms since gateway arrival).
YUNSHU_MILESTONE_FIELDS = {
    "t_template": "template_end",
    "t_admit": "engine_admit",
    "t_cache_lookup": "apc_end",
}


def _yunshu_server_fields(attr, unavailable, client, spec, lat):
    """Map Yunshu's exposed server stages onto layer-1 fields. A milestone is epoch seconds as
    client t_receive + ms (the gateway clock is the request's arrival, so the loopback offset is
    the only error). A milestone the request did not reach gets a per-row reason."""
    ms = lat.get("milestones_ms") if isinstance(lat, dict) else None
    ms = ms if isinstance(ms, dict) else {}
    t_recv = client.get("t_receive")
    for field, key in YUNSHU_MILESTONE_FIELDS.items():
        v = ms.get(key)
        if isinstance(v, (int, float)) and t_recv is not None:
            attr[field] = t_recv + float(v) / 1000.0
        else:
            unavailable[field] = f"milestone {key} not reached by this request"
    if "rounds" not in attr:
        unavailable["rounds"] = (
            "no speculative mode for this request (x_yunshu.speculative absent)"
            if not spec
            else "x_yunshu.speculative has no rounds (zero verify rounds)"
        )
    copy = spec.get("copy") if isinstance(spec, dict) else None
    if isinstance(copy, dict) and isinstance(copy.get("rounds"), (int, float)):
        attr["copy_rounds"] = int(copy["rounds"])


GENCONFIG_KEYS = ("temperature", "top_p", "top_k")


def genconfig_sampling(model_dir) -> dict:
    """Sampler fields of the model's generation_config.json (the model card's defaults). Fails
    closed: a missing file or no temperature/top_p is an error, never a silent greedy run."""
    path = Path(str(model_dir)) / "generation_config.json"
    try:
        cfg = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--sampling genconfig: cannot read {path}: {exc}")
    picked = {k: cfg[k] for k in GENCONFIG_KEYS if k in cfg}
    if "temperature" not in picked or "top_p" not in picked:
        raise SystemExit(f"--sampling genconfig: {path} has no temperature/top_p")
    return picked


def part_decode(s, out, a):
    phases = parse_phases(a.phases)
    samp = (
        genconfig_sampling(a.model)
        if getattr(a, "sampling", "greedy") == "genconfig"
        else {}
    )
    n_dec = int(a.decode_tokens)
    ctxs = [512] if a.smoke else a.only_ctx or [1024, 8192, 32768]
    thinking, media = getattr(a, "thinking", "off"), getattr(a, "media", "none")
    for ctx in ctxs:
        for kind in a.only_kind or ("prose", "code"):
            family = is_family_cell(kind, thinking, media)
            variant = getattr(a, "prompt_variant", 0)
            # family cells build on the real corpus (variant 0); their variants differ in the ask
            # only, so the audit's distinct-prompt guard still sees two prompts
            base_kind = "prose" if kind in FAMILY_KINDS_LOCAL else kind
            base_variant = 0 if family else variant
            text = (
                "Write a short example and explain it."
                if a.smoke
                else decode_prompt(base_kind, ctx, a.long_ask, base_variant)
            )
            reply = ""
            order = [p for p in ("cold", "warm", "turn2", "restart") if p in phases]
            if (
                "warm" in order
            ):  # repeated warm hits separate a one-time cost from a per-hit one
                at = order.index("warm")
                order[at : at + 1] = ["warm"] * max(1, int(a.warm_reps))
            for phase in order:
                want = (a.turn2_tokens or n_dec) if phase == "turn2" else n_dec
                if phase == "restart":
                    # the cold prompt again, on a relaunched server that reads the same HOME: a
                    # disk-tier hit is the only way the prefix can be cached (the cell's proof)
                    s = restart_server(s)
                b = req(
                    s.model,
                    text,
                    16 if a.smoke else want,
                    extra={
                        **(json.loads(a.request_extra) if a.request_extra else {}),
                        **({"logprobs": True} if a.logprobs else {}),
                        **samp,
                    }
                    or None,
                )
                if phase == "turn2" and not getattr(a, "allow_eos", False):
                    # "Continue at the same length" is a weak ask: Flash-Next ends its follow-up after
                    # ~150 tokens, and a speed cell needs its full budget. The follow-up cell measures
                    # TTFT and decode rate, not where the model would stop.
                    b["ignore_eos"] = True
                if phase == "turn2":
                    b["messages"] += [
                        {"role": "assistant", "content": reply},
                        {
                            "role": "user",
                            "content": "Continue with the next part, at the same length.",
                        },
                    ]
                extra_rec = {}
                if phase == "turn2" and a.think_s:
                    time.sleep(
                        a.think_s
                    )  # the user reads the reply before the follow-up
                if phase == "turn2" and a.rehearse:
                    extra_rec = rehearse(s, b)
                if family and phase == "cold":
                    family_apply(b, kind, variant, thinking, media)
                raw = send(s.url, b, sample=True)
                tool_calls = raw.pop("_tools", [])
                image_tokens = raw.pop("_image_tokens", None)
                r = attribution_fields(s, raw)
                r.update(extra_rec)
                if family:
                    r.update(
                        family=kind,
                        thinking=thinking,
                        media=media,
                        image_parts=fm.IMAGE_COUNT.get(media, 0),
                        image_tokens=image_tokens,
                        tool_calls=[
                            {"name": c.get("name"), "arguments": c.get("arguments")}
                            for c in tool_calls
                        ],
                    )
                    probs = family_problems(kind, r, thinking, media)
                    r.update(valid=not probs, invalid_reason="; ".join(probs) or None)
                if not a.smoke and not family:
                    check_decode_len(
                        r,
                        want,
                        f"decode {kind}-{ctx} {phase}",
                        getattr(a, "allow_eos", False),
                    )
                if phase == "cold":
                    reply = r["_text"]
                r["text"] = r.pop("_text")
                r["rep4"] = ngram_repeat(r["text"])
                if phase == "turn2":
                    done = check_rehearsal_engaged(s, a)
                    if done is not None:
                        r["rehearsals_completed"] = done
                emit(
                    out,
                    part="decode",
                    ctx=ctx,
                    kind=kind,
                    phase=phase,
                    prompt_file=decode_prompt_file(base_kind, ctx, a.smoke, base_variant),
                    prompt_file_sha256=prompt_file_sha256(
                        base_kind, ctx, a.smoke, base_variant
                    ),
                    cell=cell_name(getattr(a, "out", ""), getattr(a, "rep", 0)),
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
                r = attribution_fields(
                    s, send(s.url, req(s.model, text, n_dec, extra={"draft": False}))
                )
                r["text"] = r.pop("_text")
                emit(out, part="decode", ctx=ctx, kind=kind, phase="specoff", **r)


def arrival_offsets(window_s, n):
    """n arrival times spread uniformly over the rehearsal window, plus one after it (control)."""
    step = window_s / n
    return [round(step * (i + 0.5), 3) for i in range(n)] + [round(window_s * 3, 3)]


def part_arrive(s, out, a):
    """A request arriving while a next-turn rehearsal runs (YUNSHU_NEXT_TURN_PREFILL).

    One shared prompt is primed once; every trial asks a variant of it (cached prefix, so a trial
    costs one decode), waits its offset after the reply and sends either a diverging request
    (--arrive-mode diverge: another session, a tiny prompt) or the conversation's follow-up
    (adopt). TTFT of that request against the offset gives the wait a rehearsal can cost
    (diverge) or the time it saves (adopt)."""
    ctx = (a.only_ctx or [32768])[0]
    kind = (a.only_kind or ["code"])[0]
    base = load_prompt(f"{kind}-{ctx}") + LONG_ASK
    n_dec = int(a.decode_tokens)
    send(s.url, req(s.model, base, 16))  # prime: the 32K prefix enters the cache
    baseline = send(s.url, req(s.model, "Say hi.", 8))["ttft_s"]
    emit(
        out,
        part="arrive",
        ctx=ctx,
        kind=kind,
        mode=a.arrive_mode,
        role="baseline",
        ttft_s=baseline,
    )
    for k, off in enumerate(arrival_offsets(a.arrive_window, a.arrive_n)):
        text = base + f"\nVariant {k}: answer in a different style."
        r = send(s.url, req(s.model, text, n_dec))
        check_decode_len(r, n_dec, f"arrive {ctx} trial {k}")
        reply = r["_text"]
        time.sleep(off)
        if a.arrive_mode == "diverge":
            b = req(s.model, "Say hi.", 8)
        else:
            b = req(s.model, text, 16)
            b["messages"] += [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": "Continue with the next part, at the same length.",
                },
            ]
        probe = send(s.url, b)
        rec = {
            "offset_s": off,
            "ttft_s": probe["ttft_s"],
            "cached": probe.get("cached"),
            "pt": probe.get("pt"),
        }
        if a.arrive_mode == "diverge":
            rec["wait_s"] = round(probe["ttft_s"] - baseline, 3)
        emit(
            out,
            part="arrive",
            ctx=ctx,
            kind=kind,
            mode=a.arrive_mode,
            role="trial",
            trial=k,
            **rec,
        )
        time.sleep(a.arrive_settle)
    a.require_engaged = True
    done = check_rehearsal_engaged(s, a)
    metrics_req = urllib.request.Request(
        s.url + "/metrics", headers={"Authorization": "Bearer k"}
    )
    with urllib.request.urlopen(metrics_req, timeout=30) as r:
        preempted = rehearsal_count(r.read().decode(), "preempted")
    emit(
        out,
        part="arrive",
        ctx=ctx,
        kind=kind,
        mode=a.arrive_mode,
        role="summary",
        rehearsals_completed=done,
        preempted=preempted,
    )


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


# The old "Continue." prompt naturally stopped at 993/1024 on base 03b86abf.
# Keep the length gate strict; request enough concrete work on every growing turn.
CONC32_TASK = (
    "Write a complete Python implementation of a persistent task scheduler with "
    "SQLite storage, retries, dependency graphs, cancellation, and recovery. "
    "Write at least 400 lines of implementation and tests. Output code only, "
    "with one statement per line; do not summarize or conclude early. "
    "Start a fresh implementation for this turn using the material above."
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
                msgs[k][0]["content"] += (
                    "\n\nNew material:\n" + piece + "\n" + CONC32_TASK
                )
            else:
                msgs[k].append(
                    {
                        "role": "user",
                        "content": "More material:\n" + piece + "\n" + CONC32_TASK,
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


def conc_aggregates(decs):
    """Per-request decode summary and fairness of one conc record (lead ruling 2026-11-03). Any
    missing or non-positive rate makes every summary None, so the audit fails the record."""
    import statistics

    ok = bool(decs) and all(isinstance(d, (int, float)) and d > 0 for d in decs)
    if not ok:
        return {"dec_min": None, "dec_max": None, "dec_median": None, "fair_max_min": None}
    return {
        "dec_min": min(decs),
        "dec_max": max(decs),
        "dec_median": statistics.median(decs),
        "fair_max_min": round(max(decs) / min(decs), 3),
    }


def part_promote(s, out, a):
    """A long reply A starts while a shorter reply B runs; B ends first and A then decodes alone.

    Solo A is the reference; the concurrent A must produce the same text, and its tail should
    run on the speculative lane (the server log shows "Promoted request")."""
    n_a = int(a.decode_tokens)
    ctx = (a.only_ctx or [1024])[0]
    kind_a = os.environ.get("TFB_PROMOTE_KIND", "code")
    text_a = decode_prompt(kind_a, ctx, False)
    text_b = decode_prompt("prose" if kind_a == "code" else "code", ctx, False)

    def send_a():
        return send(s.url, req(s.model, text_a, n_a))

    solo = send_a()
    emit(out, part="promote", phase="solo", **{**solo, "text": solo["_text"]})
    solo.pop("_text")
    with cf.ThreadPoolExecutor(2) as ex:
        # B starts first and owns the lane; A arrives while B decodes (so it joins the shared
        # batch) and is the only request left once B ends.
        fb = ex.submit(lambda: send(s.url, req(s.model, text_b, 200)))
        time.sleep(float(os.environ.get("TFB_PROMOTE_DELAY_S", "0.8")))
        fa = ex.submit(send_a)
        rb, ra = fb.result(), fa.result()
    emit(out, part="promote", phase="conc_B", **{**rb, "text": rb["_text"]})
    emit(out, part="promote", phase="conc_A", **{**ra, "text": ra["_text"]})


def part_forced(s, out, a):
    """Same-text decoding: continue the other engine's output in windows of WINDOW tokens.

    TFB_FORCED_TEXT is a decode JSONL record file holding that engine's cold text for the
    cell; each window sends prompt + the text's first k tokens as an assistant prefill and
    decodes WINDOW tokens greedily, so round statistics compare on identical text (until the
    continuation diverges; ``match`` is the length of the common prefix in characters)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    window = int(os.environ.get("TFB_WINDOW", "160"))
    ctx, kind = (a.only_ctx or [1024])[0], (a.only_kind or ["prose"])[0]
    text = None
    for line in Path(os.environ["TFB_FORCED_TEXT"]).read_text().splitlines():
        d = json.loads(line)
        if d.get("part") == "decode" and d.get("phase") == "cold":
            text = d["text"]
    ids = tok.encode(text)
    prompt = decode_prompt(kind, ctx, False)
    for k in range(0, min(len(ids), 6 * window), window):
        prefix = tok.decode(ids[:k]) if k else ""
        b = req(s.model, prompt, window)
        if prefix:
            b["messages"] += [{"role": "assistant", "content": prefix}]
        r = send(s.url, b)
        want = tok.decode(ids[k : k + window])
        got = r["_text"]
        n = next(
            (i for i, (x, y) in enumerate(zip(got, want, strict=False)) if x != y),
            min(len(got), len(want)),
        )
        r.pop("_text")
        emit(
            out,
            part="forced",
            ctx=ctx,
            kind=kind,
            start_tok=k,
            match_chars=n,
            want_chars=len(want),
            **r,
        )


def request_records(rs):
    """The per-request list of a conc / mixed record (schema: fullcmp requests). spec is the request's own
    speculative counters from its response (x_yunshu.speculative) when the engine sends them, otherwise
    'unavailable (reason)': the evidence correlation fills it at finalize, per request."""
    out = []
    for i, r in enumerate(rs):
        xy = r.pop("_xy_speculative", None) or {}
        if isinstance(xy.get("accepted"), (int, float)):
            spec = {"accepted": int(xy["accepted"]), "drafted": xy.get("drafted"), "rounds": xy.get("rounds")}
        else:
            spec = "unavailable (response carries no per-request speculative counters)"
        out.append({
            "rid": i,
            "pt": r["pt"],
            "ct": r["ct"],
            "cached": r["cached"],
            "ttft_s": r["ttft_s"],
            "dec_tps": r["dec_tps"],
            "spec": spec,
        })
    return out


def part_conc(s, out, a):
    ns = [int(x) for x in (a.conc_ns or "2,4,8").split(",")]
    for n in (2,) if a.smoke else ns:
        for trial in range(1 if a.smoke else a.conc_trials):
            same = bool(getattr(a, "conc_same", False))
            texts = [
                "Say hello."
                if a.smoke
                else concurrent_prompt(
                    0 if same else i, trial, a.conc_ctx, getattr(a, "prompt_variant", 0)
                )
                for i in range(n)
            ]
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
            requests = request_records(rs)
            wall = time.perf_counter() - t0
            tot = sum(r["ct"] for r in rs)
            decs = [r["dec_tps"] for r in rs]
            emit(
                out,
                part="conc",
                n=n,
                trial=trial,
                # microseconds: agg_tps is total/wall of the unrounded wall, so a millisecond
                # rounding fails the audit's agg_tps check on a fast engine (CPU stub: wall ~ 8 ms)
                wall_s=round(wall, 6),
                total_tokens=tot,
                agg_tps=round(tot / wall, 1),
                **conc_aggregates(decs),
                per_req_dec=decs,
                ttfts=[r["ttft_s"] for r in rs],
                cached=[r["cached"] for r in rs],
                pts=[r["pt"] for r in rs],
                cts=[r["ct"] for r in rs],
                shas=[r["sha"] for r in rs],
                requests=requests,
                # per-request provenance for the fullcmp audit (one entry per request, in order)
                total_s=[r["total_s"] for r in rs],
                finish=[r["finish"] for r in rs],
                prompt_ids_sha=[r["prompt_ids_sha"] for r in rs],
                # the same for every request of this record (conc prompts are a code/prose mix)
                ctx=a.conc_ctx,
                kind="mixed",
                prompt_file="concurrent_prompt(prose|code, index+trial parity)"
                + (
                    f" variant={getattr(a, 'prompt_variant', 0)}"
                    if getattr(a, "prompt_variant", 0)
                    else ""
                ),
                max_tokens=16 if a.smoke else a.conc_tokens,
                temperature=rs[0]["temperature"],
                stop=rs[0]["stop"],
                same_prompt=same,
                harness=HARNESS_ID,
            )


def part_mixed(s, out, a):
    """Mixed load: one long request (--mixed-long-ctx) and --mixed-short-n short ones started
    together, one trial per conc trial. Request 0 is the long one; the rest are short. The
    aggregates (dec_min/max/median, fair_max_min) summarise the SHORT requests' decode rates;
    agg_tps is all tokens over the wall time."""
    long_ctx = 512 if a.smoke else a.mixed_long_ctx
    short_n = 2 if a.smoke else a.mixed_short_n
    short_ctx = 256 if a.smoke else a.mixed_short_ctx
    tokens = 16 if a.smoke else a.conc_tokens
    variant = getattr(a, "prompt_variant", 0)
    for trial in range(1 if a.smoke else a.conc_trials):
        long_text = (
            "Say hello." if a.smoke else load_variant_prompt(f"code-{long_ctx}", variant)
        )
        texts = [long_text] + [
            "Say hello."
            if a.smoke
            else concurrent_prompt(i, trial, short_ctx, variant)
            for i in range(short_n)
        ]
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(len(texts)) as ex:
            rs = list(
                ex.map(
                    lambda t: send(s.url, req(s.model, t, tokens)),
                    texts,
                )
            )
        wall = time.perf_counter() - t0
        for r in rs[1:]:
            if not a.smoke:
                check_decode_len(r, tokens, f"mixed short trial={trial}")
        if not a.smoke:
            check_decode_len(rs[0], tokens, f"mixed long trial={trial}")
        for r in rs:
            r.pop("_text")
        requests = request_records(rs)
        tot = sum(r["ct"] for r in rs)
        short_decs = [r["dec_tps"] for r in rs[1:]]
        emit(
            out,
            part="mixed",
            trial=trial,
            n=len(rs),
            roles=["long"] + ["short"] * short_n,
            wall_s=round(wall, 6),
            total_tokens=tot,
            agg_tps=round(tot / wall, 1),
            long_ttft_s=rs[0]["ttft_s"],
            # summary of the short requests only (the fairness question of the mixed cell)
            **conc_aggregates(short_decs),
            per_req_dec=[r["dec_tps"] for r in rs],
            ttfts=[r["ttft_s"] for r in rs],
            cached=[r["cached"] for r in rs],
            pts=[r["pt"] for r in rs],
            cts=[r["ct"] for r in rs],
            shas=[r["sha"] for r in rs],
            requests=requests,
            total_s=[r["total_s"] for r in rs],
            finish=[r["finish"] for r in rs],
            prompt_ids_sha=[r["prompt_ids_sha"] for r in rs],
            ctx=long_ctx,
            short_ctx=short_ctx,
            kind="mixed",
            max_tokens=tokens,
            temperature=rs[0]["temperature"],
            stop=rs[0]["stop"],
            harness=HARNESS_ID,
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
    ap.add_argument(
        "--only-kind",
        action="append",
        choices=["prose", "code", *fm.FAMILY_KINDS],
    )
    ap.add_argument(
        "--thinking",
        choices=("off", "on"),
        default="off",
        help="decode part: Qwen chat-template thinking switch (on = reasoning split required)",
    )
    ap.add_argument(
        "--media",
        choices=tuple(fm.IMAGE_COUNT),
        default="none",
        help="decode part: image parts per request (img1 / img4)",
    )
    ap.add_argument(
        "--prompt-variant",
        type=int,
        default=0,
        help="0 = the real corpus prompts; N > 0 = smoke-only variant N (TFB_PROMPT_VARIANT_DIR)",
    )
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
    ap.add_argument("--arrive-mode", choices=("diverge", "adopt"), default="diverge")
    ap.add_argument(
        "--arrive-n", type=int, default=8, help="arrival offsets in the window"
    )
    ap.add_argument(
        "--arrive-window", type=float, default=3.0, help="rehearsal window, seconds"
    )
    ap.add_argument(
        "--arrive-settle", type=float, default=8.0, help="seconds after each probe"
    )
    ap.add_argument(
        "--think-s",
        type=float,
        default=0.0,
        help="seconds between the first reply and the follow-up (the user's reading time)",
    )
    ap.add_argument(
        "--rehearse",
        action="store_true",
        help="before turn2, prefill the rendered assistant reply (next-turn rehearsal pilot)",
    )
    ap.add_argument(
        "--conc-ns", default="", help="concurrency levels, e.g. 2,4 (default 2,4,8)"
    )
    ap.add_argument(
        "--conc-same",
        action="store_true",
        help="conc part: every concurrent request sends the same prompt (record same_prompt=true)",
    )
    ap.add_argument(
        "--mixed-long-ctx",
        type=int,
        default=131072,
        help="mixed part: rendered tokens of the one long request",
    )
    ap.add_argument(
        "--mixed-short-n",
        type=int,
        default=7,
        help="mixed part: number of short requests sent with the long one",
    )
    ap.add_argument(
        "--mixed-short-ctx",
        type=int,
        default=1024,
        help="mixed part: rendered tokens of each short request",
    )
    ap.add_argument(
        "--log-usage-sample",
        action="store_true",
        help="keep the first raw usage chunk the server sent and write it to the session "
        "header as usage_sample (the GPU smoke's usage-shape evidence)",
    )
    ap.add_argument("--conc-trials", type=int, default=2)
    ap.add_argument("--conc-tokens", type=int, default=256)
    ap.add_argument(
        "--conc-ctx", type=int, default=32768, help="rendered concurrent prompt tokens"
    )
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
        "--logprobs",
        action="store_true",
        help="decode part: request chosen-token logprobs (identity evidence compares them)",
    )
    ap.add_argument(
        "--allow-eos",
        action="store_true",
        help="decode part: accept a reply the model ends itself (finish=stop); identity cells compare both arms' streams",
    )
    ap.add_argument(
        "--request-extra",
        default="",
        help="JSON object merged into every decode request body (e.g. penalties)",
    )
    ap.add_argument(
        "--prompt-files",
        default="",
        help="corpus part: comma list of prompt files, one cold request each",
    )
    ap.add_argument(
        "--phases",
        default=",".join(ALL_PHASES),
        help="decode phases to run (comma list of cold,warm,turn2,specoff,restart); the checks stay "
        "fail-closed for every selected phase",
    )
    ap.add_argument(
        "--warm-reps",
        type=int,
        default=1,
        help="warm (full APC hit) requests per prompt; each is recorded",
    )
    ap.add_argument(
        "--sampling",
        choices=("greedy", "genconfig"),
        default="greedy",
        help="decode part: greedy (temperature 0) or genconfig (temperature/top_p/top_k from the "
        "model's generation_config.json, sent explicitly)",
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


# Census key map lives in fullcmp_provenance (one map for header and dry-run gate).
FULLCMP_CENSUS_KEY = fp.FULLCMP_CENSUS_KEY


def cell_name(out_path, rep):
    """<cell>-rN.jsonl -> <cell> (the audit's cell id)."""
    stem = os.path.basename(out_path)
    stem = stem[: -len(".jsonl")] if stem.endswith(".jsonl") else stem
    suffix = f"-r{rep}"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def fullcmp_header(
    s,
    a,
    environ,
    *,
    census,
    job_start_s,
    swap_start,
    telemetry_path,
    quiet_pre=None,
    job_id=None,
):
    """Session-header fields the fullcmp audit requires beyond the bench_engines meta row.
    Every value comes from the pin, the census, the model config, the launched binary or the
    gpuq job environment; a value that cannot be found raises (fail closed). quiet_pre is the
    host-CPU snapshot taken before the server started (fullcmp_quiet.snapshot)."""
    # the weights the launch loads (bench_engines is the one source): llama-cpp runs the GGUF, kapitsa
    # and TensorFold their own checkpoints; every other engine serves the Jundot oQ4e-mtp checkpoint
    model_key = bench_engines.WEIGHTS_CENSUS_KEY.get(a.engine) or fp.census_model_id(census, a.model)
    version, git_sha = bench_engines.engine_version(a.engine, YUNSHU_SRC)
    census_key = fp.census_key_for(a.engine)
    return dict(
        run_id=environ["TFB_RUN_ID"],
        model=model_key,
        cell=cell_name(a.out, a.rep),
        group=environ["TFB_GROUP"],
        version=version,
        git_sha=git_sha,
        release_date=fp.release_date(census, census_key, version, environ),
        binary_sha256=fp.binary_sha256(s.cmd[0]),
        python=(s.pin or {}).get("python"),
        spec_mode_expected=s.requested_spec_mode,
        engaged_log_line=s.engaged_log_line,
        engaged_evidence_kind=getattr(s, "engaged_evidence_kind", None),
        engaged_evidence_source=getattr(s, "engaged_evidence_source", None),
        server_log_ready_offset=getattr(s, "ready_log_offset", None),
        weights=(
            bench_engines.gguf_weights_record(a.engine)
            if bench_engines.launched_gguf_files(a.engine)
            else fp.weights_record(
                bench_engines.served_model_dir(a.engine, a.model), census, model_key
            )
        ),
        weights_note=bench_engines.weights_note(a.engine),
        fresh_process=True,
        # the harness snapshot the driver submitted from (fullcmp_run.snapshot_harness); the live
        # worktree is never a job source, so dirty is false by construction
        harness_commit=environ.get("TFB_HARNESS_COMMIT") or None,
        harness_tree_sha=environ.get("TFB_HARNESS_TREE") or None,
        harness_dirty=False,
        home=str(s.home),
        job_id=job_id or fp.resolve_job_id(environ),
        job_start_s=job_start_s,
        swap_start=swap_start,
        telemetry_path=telemetry_path,
        quiet_pre=quiet_pre,
        engine_log_path=str(s.log),
        server_flags={"argv": [str(x) for x in s.cmd]},
        observe=s.observe,
        yunshu_observe=s.yunshu_observe_env,
        profile=environ.get("TFB_PROFILE") == "1",
    )


def start_cell_telemetry(pid, port, out_path, engine, environ, *, required):
    """Sampler beside the server for this cell; its series path is returned for the record.
    Required (fullcmp): a sampler that does not start raises. Optional: warn and continue."""
    global _ACTIVE_TELEMETRY
    tdir = environ.get("TFB_TELEMETRY_DIR") or (
        fp.DEFAULT_TELEMETRY_DIR if required else ""
    )
    if not tdir:
        return None
    path = fp.telemetry_out(out_path, engine, tdir)
    try:
        import fullcmp_telemetry as tm

        os.makedirs(os.path.dirname(path), exist_ok=True)
        sampler = tm.Sampler(pid, port, path)
        sampler.start()
    except Exception as exc:  # noqa: BLE001 - classified below
        if required:
            raise fp.ProvenanceError(f"telemetry sampler did not start: {exc}") from exc
        print(f"telemetry: optional sampler not started: {exc}", flush=True)
        return None
    _ACTIVE_TELEMETRY = sampler
    return path


def stop_cell_telemetry():
    global _ACTIVE_TELEMETRY
    sampler, _ACTIVE_TELEMETRY = _ACTIVE_TELEMETRY, None
    if sampler is not None:
        sampler.stop()


# Engines whose per-request draft stats only a non-streaming response carries (the probe, see
# fullcmp_engaged response_field_probe). mlx-vlm 0.7.11: openai.py:2759 (non-streaming chat).
PROBE_ENGINES = {"mlx-vlm-server": True}


def probe_request(s, text="Say hi."):
    """One non-streaming chat request on the live server; returns the response JSON (timings)."""
    body = dict(req(s.model, text, 64))
    body["stream"] = False
    return _post_json(s.url, "/v1/chat/completions", body)


def engaged_end_fields(s) -> dict:
    """The session-row fields written at finalize from the FINAL engaged verdict. The top-level
    spec_mode is the verified engaged mode: META carries the startup value (before the evidence
    of the measured requests), so it would still read 'unverified' for an engaged cell."""
    return {
        "engaged_spec_mode": s.engaged_spec_mode,
        "spec_mode": s.engaged_spec_mode,
        "engaged_log_line": s.engaged_log_line,
        "engaged_evidence_kind": getattr(s, "engaged_evidence_kind", None),
        "engaged_evidence_source": getattr(s, "engaged_evidence_source", None),
        "engaged_label": getattr(s, "engaged_label", None),
    }


def read_decode_rows(path):
    """The cell's request rows (part == decode) as written, for the final engaged-mode evidence."""
    rows = []
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("part") == "decode":
                rows.append(r)
    return rows


def finalize_session(path, end_fields):
    """Rewrite the session row (line 1) with the end-of-cell fields: job end, swap at end.
    The file is replaced atomically, so a crash leaves the start-of-cell row or nothing."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    row = json.loads(lines[0])
    if row.get("part") != "session":
        raise RuntimeError(f"{path}: first row is not the session header")
    row.update(end_fields)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("\n".join([json.dumps(row), *lines[1:]]) + "\n")
    os.replace(tmp, path)


QUIET_REFUSED_EXIT = 3  # the driver retries a quiet refusal in the same slot (fullcmp_run)


def quiet_refused(snapshot) -> bool:
    """True when a per-cell quiet snapshot failed the gate (fullcmp_quiet.check's own verdict)."""
    return isinstance(snapshot, dict) and snapshot.get("ok") is False


HEARTBEAT_S = 30.0  # a heartbeat line every 30 s: a long phase never leaves the job log silent
HEARTBEAT_STOP_S = 600.0  # no main-thread progress this long: stop beating, so the stall watchdog fires
_PROGRESS = {"phase": "start", "t": time.time(), "t0": time.time()}


def _progress(phase: str | None = None) -> None:
    """Main-thread progress: every request completion and every phase change calls this."""
    if phase is not None:
        _PROGRESS["phase"] = phase
    _PROGRESS["t"] = time.time()


def _start_heartbeat(interval: float = HEARTBEAT_S, stop_after: float = HEARTBEAT_STOP_S) -> None:
    """Daemon thread: one stdout line per interval while the main thread makes progress: phase name,
    elapsed time and the age of the last progress. A hung main thread stops the beats after stop_after
    seconds without progress, so gpuq's stall watchdog fires instead of a daemon hiding the hang."""
    import threading

    _PROGRESS["t0"] = time.time()

    def beat():
        while True:
            time.sleep(interval)
            now = time.time()
            idle = now - _PROGRESS["t"]
            if idle > stop_after:
                return  # silence on purpose: the watchdog must see a hang
            print(
                f"heartbeat {now - _PROGRESS['t0']:.0f}s phase={_PROGRESS['phase']} "
                f"last_progress={idle:.0f}s",
                flush=True,
            )

    threading.Thread(target=beat, name="tfbench-heartbeat", daemon=True).start()


def main():
    _start_heartbeat()
    a = parse_args()
    _progress(f"part-{a.part}")
    extra_env = dict(kv.split("=", 1) for kv in a.env)
    if a.part not in (
        "decode",
        "conc",
        "agent",
        "ca",
        "needle",
        "conc32",
        "arrive",
        "promote",
        "forced",
        "corpus",
        "mixed",
    ):
        raise ValueError(
            "supported parts: decode, conc, agent, ca, needle, conc32, arrive, promote, forced, corpus, mixed"
        )
    if a.dry_run:
        if a.part == "decode" and not a.smoke:
            for ctx in a.only_ctx or [1024, 8192, 32768]:
                for kind in a.only_kind or ("prose", "code"):
                    decode_prompt(
                        kind, ctx, a.long_ask, getattr(a, "prompt_variant", 0)
                    )
        elif a.part == "conc":
            concurrent_prompt(0, 0, a.conc_ctx, getattr(a, "prompt_variant", 0))
        elif a.part in ("agent", "ca"):
            for directory in (BODIES, BODIES2):
                if not list(directory.glob("*-req.json")):
                    raise FileNotFoundError(directory)
        if a.engine in bench_engines.ENGINES:
            gone = bench_engines.missing_paths(a.engine)
            if gone:
                raise FileNotFoundError(f"{a.engine}: missing {gone}")
        elif not (Path(a.model) / "config.json").is_file():
            raise FileNotFoundError(a.model)
        mode, selected = spec_request(a.engine, extra_env, a.model)
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
    fullcmp = fp.fullcmp_mode(os.environ)
    census = fc_header = end_fields = job_id = None
    telem_path = None
    if fullcmp:
        # Fail closed before anything runs: identifiers, the gpuq job, a quiet GPU, a fresh file.
        job_id = fp.check_required_env(os.environ)
        if was_contended():
            raise fp.ProvenanceError("GPU contended before the cell started (fullcmp)")
        if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
            raise fp.ProvenanceError(
                f"{a.out} exists; a fullcmp cell writes a fresh file"
            )
        census = fp.load_json(fp.CENSUS_PATH)
    # Host CPU before the server exists (foreign holders only; the harness tree is excluded).
    quiet_pre = fq.snapshot() if fullcmp else None
    if quiet_refused(quiet_pre):
        # The host is busy before this cell: refuse BEFORE the server starts and before the raw file is
        # created, so the driver can rerun this cell in the same slot (exit 3 = quiet refusal).
        print(f"QUIET REFUSED before {a.out}: {quiet_pre.get('total_foreign_pct')}% foreign, "
              f"holders {[h.get('name') for h in quiet_pre.get('holders', [])[:5]]}", flush=True)
        sys.exit(QUIET_REFUSED_EXIT)
    job_start_s = time.time()
    swap_start = fp.swap_snapshot() if fullcmp else None
    s = start_server(
        a.engine,
        extra_env,
        f"{a.engine}-{a.part}-{a.rep}{a.tag}{job_log_suffix()}",
        model=a.model,
        ctx_tokens=a.ctx_tokens,
        parallel=a.parallel,
    )
    # The engine log is shared by every cell of a job (same name): this cell's slice is the byte
    # range from here to its end, written beside the cell file (server_log_path).
    first_log = s.log
    log_start = 0  # the launch truncated this log: the cell's evidence includes the startup lines
    _LOG_PREFIX.clear()
    try:
        with open(a.out, "a") as out:
            try:
                if fullcmp or os.environ.get("TFB_TELEMETRY_DIR"):
                    telem_path = start_cell_telemetry(
                        s.proc.pid,
                        s.port,
                        a.out,
                        a.engine,
                        os.environ,
                        required=fullcmp,
                    )
                if fullcmp:
                    fc_header = fullcmp_header(
                        s,
                        a,
                        os.environ,
                        census=census,
                        job_start_s=job_start_s,
                        swap_start=swap_start,
                        telemetry_path=telem_path,
                        quiet_pre=quiet_pre,
                        job_id=job_id,
                    )
            except Exception as exc:
                if fullcmp:
                    emit(
                        out,
                        part="error",
                        stage="start",
                        engine=a.engine,
                        rep=a.rep,
                        error=str(exc),
                    )
                raise
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
                server_log=first_log.name,
                server_log_start=log_start,
                server_log_slice=Path(server_log_path(a.out)).name,
                **(fc_header or {}),
            )
            for _ in range(2):
                send(s.url, req(s.model, "Say hi.", 24))
            if a.part == "ca":
                part_conc(s, out, a)
                part_agent(s, out, a)
            else:
                {
                    "decode": part_decode,
                    "arrive": part_arrive,
                    "conc": part_conc,
                    "agent": part_agent,
                    "needle": part_needle,
                    "conc32": part_conc32,
                    "promote": part_promote,
                    "forced": part_forced,
                    "corpus": part_corpus,
                    "mixed": part_mixed,
                }[a.part](s, out, a)
            if _LIVE_SERVERS:
                s = _LIVE_SERVERS[-1]  # a restart cell measured on its relaunched server
            probe_response = None
            conc_probe = None
            conc_records = fe.read_concurrent_records(a.out)
            if fullcmp and a.engine in PROBE_ENGINES and conc_records:
                # the concurrent probe: the same N requests, non-streaming, at once (the streamed concurrent
                # requests carry no draft stats). Not a table number: part=probe_conc rows, probe=true.
                n = sum(len(r.get("requests") or []) for r in conc_records)
                # the cell's prompts: one shared prompt for a --conc-same cell, else the conc part's prompts
                same = bool(getattr(a, "conc_same", False))
                texts = [
                    concurrent_prompt(0 if same else i, 0, a.conc_ctx, getattr(a, "prompt_variant", 0))
                    for i in range(n)
                ]
                with cf.ThreadPoolExecutor(max(1, n)) as ex:
                    _progress("conc-probe")
                    conc_probe = list(ex.map(lambda t: probe_request(s, t), texts))
                for rid, resp in enumerate(conc_probe):
                    emit(out, part="probe_conc", engine=a.engine, rep=a.rep, rid=rid,
                         probe=True, timings=(resp or {}).get("timings"),
                         probe_fields=fe.probe_fields(resp))
            if fullcmp and a.engine in PROBE_ENGINES:
                # One non-streaming request after the measured parts (not a table number): the serving
                # path returns per-request draft stats only in a non-streaming response.
                probe_response = probe_request(s)
                emit(
                    out,
                    part="probe",
                    engine=a.engine,
                    rep=a.rep,
                    probe=True,
                    timings=probe_response.get("timings"),
                    probe_fields=fe.probe_fields(probe_response),
                    # the streamed rows' decode speed, from the server's own completed-request lines
                    completed_decode_tok_s=fe.completed_decode_tok_s(
                        s.log.read_text(errors="replace") if s.log.exists() else ""
                    ),
                )
            measured = read_decode_rows(a.out)
            try:
                s.verify_spec_mode(
                    final=True,
                    rows=measured,
                    probe_response=probe_response,
                    concurrent=fe.read_concurrent_records(a.out),
                    concurrent_probe=conc_probe,
                )
            except Exception:
                # a refused cell keeps its evidence: the slice is written before the error propagates
                log_end_fail = s.log.stat().st_size if s.log.exists() else 0
                Path(server_log_path(a.out)).write_bytes(
                    cell_engine_log(s.log, log_end_fail, prefix=_LOG_PREFIX)
                )
                raise
            if fullcmp and a.engine != "yunshu" and getattr(s, "engaged_per_row", None):
                # attribution per measured request, from the matched evidence (written after the final check)
                for attr in fe.attr_rows_from(s.engaged_per_row, measured):
                    emit(out, part="attr", engine=a.engine, rep=a.rep, **attr)
            conc_records = fe.read_concurrent_records(a.out)
            if fullcmp and conc_records:
                # attribution per concurrent request (rid, record part), from the shared verdict
                for attr in fe.concurrent_attr_rows(
                    fe.load_table(), a.engine, s.requested_spec_mode,
                    s.log.read_text(errors="replace"), conc_records, getattr(s, "ready_log_offset", None),
                    probe_responses=conc_probe,
                ):
                    emit(out, part="attr", engine=a.engine, rep=a.rep, **attr)
            mem = getattr(s, "mem", None)
            if mem is not None:
                time.sleep(a.idle_s)
                idle = mem.sample()
                peak = memory_peak_or_raise(mem, idle)
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
            stop_cell_telemetry()
            # The cell's engine-log slice, written now: the next cell of the job reuses the name.
            log_end = s.log.stat().st_size if s.log.exists() else 0
            Path(server_log_path(a.out)).write_bytes(
                cell_engine_log(s.log, log_end, prefix=_LOG_PREFIX)
            )
            if fullcmp:
                # The GPU must have stayed exclusive to this cell until its last request.
                if was_contended():
                    raise fp.ProvenanceError("GPU contended during the cell (fullcmp)")
                end_fields = {
                    "server_log_end": log_end,
                    "job_end_s": time.time(),
                    "gpu_exclusive": True,
                    "contended": False,
                    "swap_end": fp.swap_snapshot(),
                    # Server still up (a descendant of this process, so not counted as foreign).
                    "quiet_post": fq.snapshot(),
                    # the raw usage chunk of the first request (GPU smoke evidence, per engine)
                    "usage_sample": _USAGE_SAMPLE[0] if a.log_usage_sample else None,
                    **engaged_end_fields(s),
                }
            emit(
                out,
                part="part_done",
                engine=a.engine,
                which=a.part,
                rep=a.rep,
                complete=True,
            )
        finalize_session(a.out, end_fields or {"server_log_end": log_end})
    except Exception as exc:
        if fullcmp:
            with open(a.out, "a") as out:
                emit(
                    out,
                    part="error",
                    stage="run",
                    engine=a.engine,
                    rep=a.rep,
                    error=str(exc),
                )
        raise
    finally:
        stop_cell_telemetry()
        for live in _LIVE_SERVERS:  # a restart cell's relaunched server, then the first one
            live.kill()
        s.kill()


if __name__ == "__main__":
    if os.environ.get("TFB_STUB_SERVER"):
        # TEST-ONLY: a stub run must not outlive the wrapper that started it. When the parent is
        # gone (a test timed out and killed the shell), this process exits; its stub then sees its
        # own parent gone and exits too (fullcmp_stub_server.watch_parent).
        import threading

        _parent = os.getppid()

        def _exit_with_parent():
            while True:
                time.sleep(1)
                if os.getppid() != _parent or _parent == 1:
                    os._exit(0)

        threading.Thread(target=_exit_with_parent, daemon=True).start()
    main()
