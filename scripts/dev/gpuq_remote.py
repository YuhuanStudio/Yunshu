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


def paths(job, repo):
    top = subprocess.check_output(
        ["git", "-C", job["cwd"], "rev-parse", "--show-toplevel"], text=True
    ).strip()
    wt = repo + "/.m3-wt/gpuq-" + job["id"]
    pairs = [(top, wt), (BUILD, repo + "/.m3-home/out")]
    # Both main-checkout and worktree venv interpreters map to the shared M3 venv.
    for value in [*job["cmd"], *job["env"].values()]:
        for match in re.finditer(r"/[^\s:'\";]+/\.venv/bin/[^\s:'\";]+", str(value)):
            source = match.group().split("/.venv/bin/")[0] + "/.venv/bin"
            pairs.append((source, repo + "/.venv/bin"))
    models, synced = {}, set()
    # local.env exports a catalogue of a dozen checkpoints (27B, 30B Omni, ...) into every
    # job's env. Syncing all of them stalled every M3 job (2026-10-06): only checkpoints
    # named in the command, in a non-catalogue env var, or in GPUQ_M3_SYNC (comma-separated
    # env keys) are synced; catalogue paths are still mapped.
    catalogue = _catalogue_keys() - set(
        filter(None, job["env"].get("GPUQ_M3_SYNC", "").split(","))
    )
    values = [(None, v) for v in job["cmd"]] + list(job["env"].items())
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
                dest = repo + "/.m3-home/models/" + Path(source).name
                if dest in models.values() and models.get(source) != dest:
                    raise ValueError(
                        "model basename collision; use distinct checkpoint names"
                    )
                models[source] = dest
                if key is None or key not in catalogue:
                    synced.add(source)
    pairs.extend(models.items())
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return top, wt, pairs, {k: v for k, v in models.items() if k in synced}


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

    def sync(self, source, target, back=False):
        cmd = [
            "rsync",
            "-a",
            "-v",  # one line per transferred file: a multi-GB copy is not a stall
            "-e",
            shlex.join(self.ssh[:-1]),
            source,
            target,
        ]
        if not back:
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

    def run(self):
        top, wt, pairs, models = paths(self.job, self.repo)
        refused = sorted(Path(m).name for m in models if Path(m).name not in M3_MODELS)
        if refused:
            raise ValueError(
                f"M3: checkpoints {refused} are not allowed on the laptop; allowed: {sorted(M3_MODELS)}"
            )
        from gpuq_digest import output_paths

        outputs = [(str(p), mapped(p, pairs)) for p in output_paths(self.job)]
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
                f"mkdir -p {q(self.repo + '/.m3-home/gpuq')} {q(self.repo + '/.m3-home/models')} {q(self.repo + '/.uv-cache')} {q(self.repo + '/.m3-wt')}"
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
        for source, dest in models.items():
            self.sync(source + "/", self.host + ":" + dest + "/")
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
            cwd=mapped(self.job["cwd"], pairs),
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
            rc = subprocess.check_output(
                self.ssh_cmd("cat " + q(rcpath)), text=True, timeout=20
            )
            self.job["rc"] = int(rc.strip())
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
