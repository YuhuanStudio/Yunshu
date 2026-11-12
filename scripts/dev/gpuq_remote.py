"""M3 transport. All remote writes stay in the configured Yunshu checkout."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path

BUILD = "/Volumes/P5Plus/yunshu-build"
MODEL_ROOTS = ("/Volumes/P5Plus/models", "/Volumes/Micron/models")
# The M3 is the user's laptop: only these small correctness checkpoints may be copied
# there (user 2026-10-06: "不要傳一堆模型 也不要污染我的筆電與空間"). Anything else fails
# the job before any transfer; extend this list deliberately, never per job.
M3_MODELS = frozenset(
    {
        "Qwen3.5-0.8B-MLX-bf16",
        "Qwen3.5-2B-MLX-bf16",
        "Qwen3.5-9B-MLX-4bit",
        "Qwen2.5-3B-Instruct-4bit",
        # one smallest checkpoint per modality, so speech / ASR / OCR / image generation /
        # embeddings are verified on a real server (user 2026-10-06), ~17 GB together
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        "Qwen3-ASR-1.7B-bf16",
        "GLM-OCR-bf16",
        "Z-Image-Turbo-MLX-4bit",
        "Qwen3-Embedding-0.6B",
        # EmbeddingGemma 2 (1.5 GB): text / image / audio / video embeddings, next to the Qwen3
        # recipe above (last-token pooling vs mean pooling are different code paths)
        "embeddinggemma-2",
        # input-omni (audio + image + video in): the smallest Gemma 4 with an audio tower,
        # 3.3 GB 4-bit, so audio / image / video input and the Realtime voice cascade
        # (with the ASR + TTS above) run on a real server (user 2026-10-06). No speech-out
        # omni checkpoint is small enough: Qwen3-Omni is 20 GB and stays on the M5.
        "gemma-4-e2b-it-4bit",
        "whisper-large-v3-mlx",  # /v1/audio/translations: only Whisper translates (2.9 GB)
    }
)


def kill_tagged_script(job_id):
    """Shell that SIGKILLs every process carrying GPUQ_JOB_ID=<id> in its environment.
    A job that starts a server in its own session escapes the job's process group:
    2026-10-06 a 3B server ran 1.5 h on the laptop after its sweep job ended."""
    tag = shlex.quote("GPUQ_JOB_ID=" + job_id)
    return (
        "ps eww -ax -o pid=,command= | grep -F -- "
        + tag
        + " | grep -v grep | awk '{print $1}' | xargs kill -9 2>/dev/null; true"
    )


IDLE_SCRIPT = (
    "ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}'"
)


M3_MIN_FREE_GIB = 20


def dir_bytes(path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def space_needed(models: dict, states: list, size=dir_bytes) -> int:
    """Bytes of the checkpoints the laptop does not have yet (`states`: "have"/"new" per model)."""
    pairs = zip(models, states)  # noqa: B905 - gpuq runs on the system Python 3.9
    return sum(size(src) for src, st in pairs if st != "have")


def space_ok(free_bytes: int, need_bytes: int) -> bool:
    return free_bytes - need_bytes >= M3_MIN_FREE_GIB * 2**30


def read_remote_rc(cmd, run=None, sleep=time.sleep, waits=(5, 15, 30)):
    """The job's exit code from the laptop. ssh connection failures (exit 255, timeouts) are
    retried: on 2026-10-07 three M3 jobs whose checks had passed and whose outputs were collected
    failed on a banner timeout while reading this one file. A missing rc file is not retried."""
    for wait in (*waits, None):
        try:
            return int(
                (run or subprocess.check_output)(cmd, text=True, timeout=20).strip()
            )
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 255 or wait is None:
                raise
        except subprocess.TimeoutExpired:
            if wait is None:
                raise
        sleep(wait)
    raise AssertionError("unreachable")


def user_idle_s(env, run=subprocess.run):
    """Seconds since the laptop's owner last touched keyboard or mouse; None when unknown.
    The laptop is borrowed: M3 work must not compete with its owner, so the lane waits until
    the owner has been away a while."""
    host, key, _ = config(env)
    try:
        out = run(
            [
                "ssh",
                "-i",
                key,
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                host,
                IDLE_SCRIPT,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return int(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def config(env):
    return (
        env.get("M3_HOST", "yuhuan@192.168.50.55"),
        env.get("M3_KEY", str(Path.home() / ".ssh/yunshu_m3")),
        env.get("M3_REPO", "/Users/yuhuan/Documents/YuhuanStudio/Yunshu"),
    )


VERIFY_TREE_RE = re.compile(re.escape(BUILD) + r"/verify/trees/[^/\s:'\";]+")


def verify_trees(cmd, env):
    """Verify trees (BUILD/verify/trees/<sha>, each a git worktree) a job's cmd or env names."""
    trees = []
    for value in [*cmd, *env.values()]:
        for match in VERIFY_TREE_RE.finditer(str(value)):
            if match[0] not in trees:
                trees.append(match[0])
    return trees


def verify_tree_conflict(cmd, env):
    """Refusal reason when a job names more than one verify tree (M3 syncs one source tree)."""
    trees = verify_trees(cmd, env)
    if len(trees) > 1:
        return (
            "verify trees "
            + ", ".join(trees)
            + " in one job (M3 syncs one source tree)"
        )
    return None


# Absolute paths a job names. Relative paths resolve against its cwd, so they are not named here.
GIT_PATH_RE = re.compile(r"(?<![\w./-])/[^\s:'\";=,()]+")
# The laptop lane replaces these, or they name the cwd rather than what the job runs.
UNSYNCED_ENV = {"PATH", "HOME", "TMPDIR", "UV_CACHE_DIR", "GPUQ_DIR", "PWD", "OLDPWD"}


def _checkout(path, cache):
    """(toplevel, common git dir) of the nearest existing directory at or above path, or None
    outside a checkout. Cached per directory: a job names a dozen paths in one checkout."""
    d = path
    while not os.path.isdir(d):
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent
    if d not in cache:
        top = subprocess.run(
            ["git", "-C", d, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        )
        common = subprocess.run(
            ["git", "-C", d, "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
        )
        ok = top.returncode == 0 and common.returncode == 0
        cache[d] = (
            (
                top.stdout.strip(),
                os.path.abspath(os.path.join(d, common.stdout.strip())),
            )
            if ok
            else None
        )
    return cache[d]


def _tracked(top, path):
    """Whether git tracks path (a file or a directory with tracked files) in top's index."""
    return (
        subprocess.run(
            ["git", "-C", top, "ls-files", "--error-unmatch", "--", path],
            capture_output=True,
        ).returncode
        == 0
    )


def source_tree_refusals(cmd, env, cwd):
    """Reasons the laptop cannot run a job from its one synced source tree. The laptop gets the
    job's checkout (or verify tree) at HEAD plus tracked edits, so refused: a named path in
    another checkout or worktree, and a named path that exists but is untracked (2026-11-12:
    several jobs lost a queue round to rc=4 'file or directory not found' on the laptop).
    Paths that are not in a checkout (checkpoints, outputs, devices) and .venv interpreters,
    which the laptop maps to its own venv, are not checked."""
    cache = {}
    trees = verify_trees(cmd, env)
    home = _checkout(trees[0] if trees else cwd, cache)
    src = home[0] if home else None
    # Only checkouts of this repo count: other repos named in a job (Homebrew, a plugin) are not
    # source the laptop copies. Without a checkout at cwd, this repo is the one gpuq lives in.
    main = _checkout(str(Path(__file__).resolve().parent), cache)
    main_top = main[0] if main else None
    own = (home or main)[1]
    texts = [str(v) for v in cmd] + [
        str(v) for k, v in env.items() if k not in UNSYNCED_ENV
    ]
    reasons, seen = [], set()
    for text in texts:
        for match in GIT_PATH_RE.finditer(text):
            path = match[0]
            if "/.venv" in path or path in seen:
                continue
            seen.add(path)
            found = _checkout(path, cache)
            if found is None or found[1] != own:
                continue
            top = found[0]
            if trees and top == main_top:
                # yv runs its harness (scripts/) from the main checkout and the engine from the
                # verify tree; the laptop keeps its own copy of the main checkout at that path
                continue
            if top != src:
                reasons.append(f"{path} is in {top}, not the job's source tree {src}")
            elif os.path.exists(path) and not _tracked(src, path):
                reasons.append(f"{path} is untracked in {src}")
    return reasons


def run_cwd(job, top, wt, pairs):
    """The job's cwd on the laptop. A verify-tree job's cwd is the main checkout, which the
    laptop must not run: it runs the synced worktree instead."""
    cwd = job["cwd"]
    if verify_trees(job["cmd"], job["env"]) and not (
        cwd == top or cwd.startswith(top + "/")
    ):
        return wt
    return mapped(cwd, pairs)


def paths(job, repo):
    trees = verify_trees(job["cmd"], job["env"])
    if len(trees) > 1:
        raise ValueError("M3: " + verify_tree_conflict(job["cmd"], job["env"]))
    # A verify tree is itself a git worktree: its toplevel is the source the laptop runs.
    top = subprocess.check_output(
        [
            "git",
            "-C",
            trees[0] if trees else job["cwd"],
            "rev-parse",
            "--show-toplevel",
        ],
        text=True,
    ).strip()
    wt = repo + "/.m3-wt/gpuq-" + job["id"]
    pairs = [(top, wt), (BUILD, repo + "/.m3-home/out")]
    # Both main-checkout and worktree venv interpreters map to the shared M3 venv.
    for value in [*job["cmd"], *job["env"].values()]:
        for match in re.finditer(r"/[^\s:'\";]+/\.venv/bin/[^\s:'\";]+", str(value)):
            source = match.group().split("/.venv/bin/")[0] + "/.venv/bin"
            pairs.append((source, repo + "/.venv/bin"))
    sources, synced = model_sources(job["cmd"], job["env"])
    models = {}
    for source in sources:
        dest = repo + "/.m3-home/models/" + Path(source).name
        if dest in models.values():
            raise ValueError("model basename collision; use distinct checkpoint names")
        models[source] = dest
    pairs.extend(models.items())
    pairs.extend(input_files(job["cmd"], job["env"], repo).items())
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return top, wt, pairs, {k: v for k, v in models.items() if k in synced}


def model_sources(cmd, env):
    """(checkpoint dirs a job names under MODEL_ROOTS, the subset that is copied to the laptop).
    Pure: no git, no filesystem beyond config.json probes. Shared by paths() and by
    gpuq.submit, which refuses an M3 job here instead of after the lane was chosen."""
    sources, synced = [], set()
    # local.env exports a catalogue of a dozen checkpoints (27B, 30B Omni, ...) into every
    # job's env. Syncing all of them stalled every M3 job (2026-10-06): only checkpoints
    # named in the command, in a non-catalogue env var, or in GPUQ_M3_SYNC (comma-separated
    # env keys) are synced; catalogue paths are still mapped.
    catalogue = _catalogue_keys() - set(
        filter(None, env.get("GPUQ_M3_SYNC", "").split(","))
    )
    values = [(None, v) for v in cmd] + list(env.items())
    for key, value in values:
        for root in MODEL_ROOTS:
            for match in re.finditer(re.escape(root) + r"/([^\s:'\";]+)", str(value)):
                source = root + "/" + match[1].split("/")[0]
                candidate = Path(match[0])
                while str(candidate).startswith(root + "/"):
                    if (candidate / "config.json").is_file():
                        source = str(candidate)
                        break
                    candidate = candidate.parent
                if source not in sources:
                    sources.append(source)
                if key is None or key not in catalogue:
                    synced.add(source)
    return sources, synced


# DYLD_* values name interposer dylibs (tfbench passes them as `--env K=V` argv tokens). BUILD maps to
# an output-only directory on the laptop, so a dylib under BUILD is copied in as a job input instead.
# TFB_WORK names tfbench's prompt directory (under BUILD too): a directory input, copied whole.
INPUT_RE = re.compile(r"(?:DYLD_[A-Z_]+|TFB_WORK)=([^\s'\";]+)")


def input_files(cmd, env, repo):
    """{local file or directory under BUILD: laptop path in <repo>/.m3-home/inputs} named by a
    DYLD_* (dylib) or TFB_WORK (prompt directory) variable."""
    values = [str(v) for v in cmd] + [
        f"{k}={v}" for k, v in env.items() if k.startswith(("DYLD_", "TFB_WORK"))
    ]
    out = {}
    for text in values:
        for match in INPUT_RE.finditer(text):
            for item in match[1].split(":"):
                if item.startswith(BUILD + "/") and (
                    os.path.isfile(item) or os.path.isdir(item)
                ):
                    out[item] = repo + "/.m3-home/inputs/" + Path(item).name
    return out


def m3_input_refusals(cmd, env):
    """Reasons a copied dylib cannot load on the laptop (arm64 is its only architecture)."""
    reasons = []
    for source in input_files(cmd, env, ""):
        if os.path.isdir(source):
            continue
        try:
            archs = subprocess.check_output(
                ["lipo", "-archs", source], text=True, stderr=subprocess.DEVNULL
            )
        except (OSError, subprocess.CalledProcessError):
            reasons.append(f"{Path(source).name}: not a Mach-O file")
            continue
        if "arm64" not in archs.split():
            reasons.append(
                f"{Path(source).name}: no arm64 slice ({archs.strip() or 'none'})"
            )
    return reasons


def refused_m3_models(cmd, env):
    """Basenames of the checkpoints a job would copy to the laptop that M3_MODELS does not allow."""
    _, synced = model_sources(cmd, env)
    return sorted(Path(s).name for s in synced if Path(s).name not in M3_MODELS)


def _catalogue_keys():
    env = Path(__file__).resolve().parents[1] / "research" / "local.env"
    try:
        lines = env.read_text().splitlines()
    except OSError:
        return set()
    keys = set()
    for line in lines:
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=", line)
        if m:
            keys.add(m[1])
    return keys


def tfb_outputs(job, pairs):
    """(laptop path, remote path) of tfbench's TFB_OUT directory. The verify stage reads its engine
    logs (out/server-*.log) from there, but no --out names it, so it must be copied back too."""
    values = [
        str(v)[len("TFB_OUT=") :] for v in job["cmd"] if str(v).startswith("TFB_OUT=")
    ]
    if job["env"].get("TFB_OUT"):
        values.append(job["env"]["TFB_OUT"])
    return [(v, mapped(v, pairs)) for v in dict.fromkeys(values)]


def mapped(value, pairs):
    # Replace once: a remote checkout may itself be beneath the source checkout.
    pattern = re.compile(
        "|".join(re.escape(a) + r"(?=/|$|[\s:'\";])" for a, _ in pairs)
    )
    replacements = dict(pairs)
    return pattern.sub(lambda m: replacements[m[0]], str(value))


class InterruptedError(Exception):
    pass


class Remote:
    def __init__(self, job, path, api, log):
        self.job, self.path, self.api, self.log = job, path, api, log
        self.host, self.key, self.repo = config(job["env"])
        self.ssh = [
            "ssh",
            "-i",
            self.key,
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
            self.host,
        ]
        self.deadline = time.monotonic() + job["timeout_s"]
        self.reason = None
        self.last_growth = time.monotonic()
        self.size = -1
        self.pidfile = self.repo + "/.m3-home/gpuq/" + job["id"] + ".pid"

    def ssh_cmd(self, script):
        return self.ssh + ["/bin/bash -c " + shlex.quote(script)]

    def call(self, cmd, env=None):
        proc = subprocess.Popen(
            cmd,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self.job["pid"] = proc.pid
        if self.api._read(self.path).get("cancel"):
            self.job["cancel"] = True
        self.api._write(self.path, self.job)
        while proc.poll() is None:
            now = time.monotonic()
            size = os.fstat(self.log.fileno()).st_size
            if size != self.size:
                self.size, self.last_growth = size, now
            if self.api._read(self.path).get("cancel"):
                self.reason = "cancelled"
            elif now > self.deadline:
                self.reason = "timeout"
            elif now - self.last_growth > self.job.get("stall_s", 600):
                self.reason = "stalled"
            if self.reason:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise InterruptedError(self.reason)
            time.sleep(0.1)
        if proc.returncode:
            raise RuntimeError(
                f"transport/command exited rc={proc.returncode}: {cmd[0]}"
            )

    def sync(self, source, target, back=False, delete=True):
        cmd = [
            "rsync",
            "-a",
            "-v",  # one line per transferred file: a multi-GB copy is not a stall
            "-e",
            shlex.join(self.ssh[:-1]),
            source,
            target,
        ]
        if not back and delete:
            cmd.insert(2, "--delete")
        if back:
            # Cleanup is bounded separately and runs even after cancellation/timeout.
            result = subprocess.run(
                cmd, stdout=self.log, stderr=subprocess.STDOUT, timeout=120
            )
            if result.returncode:
                raise RuntimeError(f"rsync-back failed rc={result.returncode}")
        else:
            self.call(cmd)

    def stop(self):
        script = f'touch {shlex.quote(self.pidfile + ".cancel")}; if [ -f {shlex.quote(self.pidfile)} ]; then p=$(cat {shlex.quote(self.pidfile)}); kill -TERM -- -"$p" 2>/dev/null || true; sleep 2; kill -KILL -- -"$p" 2>/dev/null || true; fi'
        result = subprocess.run(
            self.ssh_cmd(script), stdout=self.log, stderr=subprocess.STDOUT, timeout=20
        )
        if result.returncode:
            raise RuntimeError("cannot confirm remote cleanup; M3 lane quarantined")

    def cleanup(self):
        if (
            self.api._read(self.api.ROOT / "m3-quarantine.json").get("job")
            == self.job["id"]
        ):
            return
        q = shlex.quote
        wt = self.repo + "/.m3-wt/gpuq-" + self.job["id"]
        script = (
            kill_tagged_script(self.job["id"])
            + f"; cd {q(self.repo)} && git worktree remove --force {q(wt)} 2>/dev/null; git -C {q(self.repo)} update-ref -d {q('refs/m5/gpuq-' + self.job['id'])}"
        )
        subprocess.run(
            self.ssh_cmd(script), stdout=self.log, stderr=subprocess.STDOUT, timeout=20
        )

    def collect(self, outputs=None):
        if outputs is None:
            from gpuq_digest import output_paths

            _, _, pairs, _ = paths(self.job, self.repo)
            outputs = [(str(p), mapped(p, pairs)) for p in output_paths(self.job)]
            outputs += tfb_outputs(self.job, pairs)
        errors = []
        for local, remote in outputs:
            try:
                kind = subprocess.check_output(
                    self.ssh_cmd(
                        f"if [ -d {shlex.quote(remote)} ]; then printf directory; fi"
                    ),
                    text=True,
                    timeout=20,
                ).strip()
                Path(local).parent.mkdir(parents=True, exist_ok=True)
                if kind == "directory":
                    Path(local).mkdir(parents=True, exist_ok=True)
                    self.sync(self.host + ":" + remote + "/", local + "/", back=True)
                else:
                    self.sync(self.host + ":" + remote, local, back=True)
                import sys

                sys.path.insert(
                    0, str(Path(__file__).resolve().parents[1] / "research")
                )
                from device_evidence import stamp_output

                stamp_output(local, "m3", self.host)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))

    def check_space(self, models):
        """Refuse a job whose new checkpoints would leave the laptop under M3_MIN_FREE_GIB free
        (2026-10-07: its data volume was at 97%, 27 GiB free, with 45 GB of ours on it)."""
        q = shlex.quote
        probe = "df -k " + q(self.repo) + " | tail -1 | awk '{print $4}'"
        for dest in models.values():
            probe += "; test -d " + q(dest) + " && echo have || echo new"
        lines = subprocess.check_output(
            self.ssh_cmd(probe), text=True, timeout=30
        ).split()
        need = space_needed(models, lines[1:])
        if not space_ok(int(lines[0]) * 1024, need):
            raise ValueError(
                f"M3: {need / 2**30:.1f} GiB of new checkpoints would leave the laptop under "
                f"{M3_MIN_FREE_GIB} GiB free ({int(lines[0]) / 2**20:.1f} GiB free now)"
            )

    def run(self):
        top, wt, pairs, models = paths(self.job, self.repo)
        refused = sorted(Path(m).name for m in models if Path(m).name not in M3_MODELS)
        if refused:
            raise ValueError(
                f"M3: checkpoints {refused} are not allowed on the laptop; allowed: {sorted(M3_MODELS)}"
            )
        from gpuq_digest import output_paths

        outputs = [(str(p), mapped(p, pairs)) for p in output_paths(self.job)]
        outputs += tfb_outputs(self.job, pairs)
        if any(
            a == b
            or not b.startswith(wt + "/")
            and not b.startswith(self.repo + "/.m3-home/out/")
            for a, b in outputs
        ):
            raise ValueError("M3 outputs must be inside the worktree or yunshu-build")
        q = shlex.quote
        rcpath = self.repo + "/.m3-home/gpuq/" + self.job["id"] + ".rc"
        self.call(
            self.ssh_cmd(
                f"mkdir -p {q(self.repo + '/.m3-home/gpuq')} {q(self.repo + '/.m3-home/models')} {q(self.repo + '/.m3-home/inputs')} {q(self.repo + '/.uv-cache')} {q(self.repo + '/.m3-wt')}"
            )
        )
        # Temporary index creates HEAD + tracked edits without touching the stash stack.
        index = self.api.ROOT / (self.job["id"] + ".index")
        env = {**os.environ, "GIT_INDEX_FILE": str(index)}
        try:
            base = subprocess.check_output(
                ["git", "-C", top, "write-tree"], text=True
            ).strip()
            self.call(["git", "-C", top, "read-tree", base], env)
            self.call(["git", "-C", top, "add", "-u"], env)
            tree = subprocess.check_output(
                ["git", "-C", top, "write-tree"], env=env, text=True
            ).strip()
            sha = subprocess.check_output(
                [
                    "git",
                    "-C",
                    top,
                    "-c",
                    "user.name=yuhuanowo",
                    "-c",
                    "user.email=huhu11256@gmail.com",
                    "commit-tree",
                    tree,
                    "-p",
                    "HEAD",
                    "-m",
                    "M3 queue snapshot",
                ],
                text=True,
            ).strip()
            self.call(
                [
                    "git",
                    "-C",
                    top,
                    "push",
                    "-q",
                    "ssh://" + self.host + self.repo,
                    sha + ":refs/m5/gpuq-" + self.job["id"],
                ],
                {**os.environ, "GIT_SSH_COMMAND": shlex.join(self.ssh[:-1])},
            )
        finally:
            index.unlink(missing_ok=True)
        self.check_space(models)
        for source, dest in models.items():
            self.sync(source + "/", self.host + ":" + dest + "/")
        for source, dest in input_files(
            self.job["cmd"], self.job["env"], self.repo
        ).items():
            if os.path.isdir(source):
                self.sync(source + "/", self.host + ":" + dest + "/", delete=False)
            else:
                self.sync(source, self.host + ":" + dest, delete=False)
        self.call(
            self.ssh_cmd(
                f"cd {q(self.repo)} && git worktree add -q --detach {q(wt)} {q(sha)}"
            )
        )
        env = {
            k: mapped(v, pairs)
            for k, v in self.job["env"].items()
            if k
            not in {"HOME", "PATH", "TMPDIR", "PYTHONPATH", "UV_CACHE_DIR", "GPUQ_DIR"}
        }
        env.update(
            HOME=self.repo + "/.m3-home",
            UV_CACHE_DIR=self.repo + "/.uv-cache",
            TMPDIR=self.repo + "/.m3-home/tmp",
            PATH=self.repo + "/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            PYTHONPATH=wt + "/python",
            GPUQ_DEVICE="m3",
            GPUQ_JOB_ID=self.job[
                "id"
            ],  # tags every process of the job, see _KILL_TAGGED
            GPUQ_REMOTE_HOST=self.host,
            M3_MODELS=self.repo + "/.m3-home/models",
            XDG_CACHE_HOME=self.repo + "/.m3-home/cache",
            HF_HOME=self.repo + "/.m3-home/cache/huggingface",
            HUGGINGFACE_HUB_CACHE=self.repo + "/.m3-home/cache/huggingface/hub",
            TRANSFORMERS_CACHE=self.repo + "/.m3-home/cache/huggingface/hub",
            TORCH_HOME=self.repo + "/.m3-home/cache/torch",
            MPLCONFIGDIR=self.repo + "/.m3-home/cache/matplotlib",
            NUMBA_CACHE_DIR=self.repo + "/.m3-home/cache/numba",
            UV_PROJECT_ENVIRONMENT=self.repo + "/.venv",
        )
        cmd = [mapped(v, pairs) for v in self.job["cmd"]]
        # The remote supervisor owns a new process group, shared lock and hard timeout.
        code = Path(__file__).with_name("gpuq_remote_worker.py").read_text()
        payload = dict(
            cmd=cmd,
            cwd=run_cwd(self.job, top, wt, pairs),
            env=env,
            pid=self.pidfile,
            rc=rcpath,
            lock=self.repo + "/.m3-home/m3run.lock",
            timeout=max(1, self.deadline - time.monotonic()),
        )
        dirs = {str(Path(b).parent) for _, b in outputs}
        dirs.add(env["TMPDIR"])
        script = (
            "mkdir -p "
            + " ".join(q(p) for p in dirs)
            + "; exec "
            + q(self.repo + "/.venv/bin/python")
            + " -u -c "
            + q(code)
            + " "
            + q(json.dumps(payload))
        )
        launched = False
        error = None
        try:
            launched = True
            self.call(self.ssh_cmd(script))
            self.job["rc"] = read_remote_rc(self.ssh_cmd("cat " + q(rcpath)))
        except Exception as exc:
            error = exc
        finally:
            if launched:
                try:
                    self.stop()
                except Exception as exc:
                    self.api._write(
                        self.api.ROOT / "m3-quarantine.json",
                        {"error": str(exc), "job": self.job["id"]},
                    )
                    error = exc
                try:
                    self.collect(outputs)
                except Exception as exc:
                    error = exc
        if error:
            raise error
        if self.job["rc"] == 124:
            self.reason = "timeout"
