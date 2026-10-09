"""Shared plumbing: paths, run directory with resumable JSONL evidence, trees, gpuq client.

Python 3.9 compatible (the wrapper runs on the system interpreter, like gpuq).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # the checkout this tool lives in
VERIFY_ROOT = Path(os.environ.get("YV_ROOT", "/Volumes/P5Plus/yunshu-build/verify"))
TREES = VERIFY_ROOT / "trees"
RUNS = VERIFY_ROOT / "runs"
MAIN_CHECKOUT = Path(
    os.environ.get("YUNSHU_MAIN", "/Users/yuhuan/Documents/YuhuanStudio/Yunshu")
)
VENV_PY = os.environ.get("YV_PY", str(MAIN_CHECKOUT / ".venv/bin/python"))
GPUQ_BIN = os.environ.get("YV_GPUQ", str(REPO / "scripts/dev/gpuq"))
PAIRED_PY = os.environ.get(
    "PAIRED_PY", "/Volumes/P5Plus/yunshu-test-envs/paired-eval/bin/python"
)
TERMINAL = {"done", "failed", "timeout", "stalled", "cancelled", "lost"}


class InfraError(RuntimeError):
    """Something in the tool or the queue broke (exit 2), not a verdict about the candidate."""


def now() -> float:
    return time.time()


def sha(*parts: object, n: int = 12) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:n]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # torn last line from a killed writer
            continue
    return rows


def write_json_atomic(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def git(*args: str, cwd: Path | str | None = None) -> str:
    r = subprocess.run(
        ["git", *args], cwd=str(cwd or REPO), capture_output=True, text=True
    )
    if r.returncode != 0:
        raise InfraError(f"git {' '.join(args)}: {r.stderr.strip()[:300]}")
    return r.stdout.strip()


# ── trees ────────────────────────────────────────────────────────────────
class Arm:
    def __init__(self, name: str, spec: str, commit: str, path: Path, dirty: str):
        self.name, self.spec, self.commit, self.path, self.dirty = (
            name,
            spec,
            commit,
            path,
            dirty,
        )
        self.runtime_fingerprint = self._runtime_fingerprint()

    def _runtime_fingerprint(self) -> str:
        """Hash marked wheel receipts without importing or initializing MLX."""
        venv = self.path / ".venv"
        if (
            not (self.path / ".yv-own-venv").exists()
            or not (venv / "bin/python").is_file()
        ):
            return ""
        digest = hashlib.sha256()
        digest.update(str((venv / "bin/python").resolve()).encode())
        cfg = venv / "pyvenv.cfg"
        if cfg.exists():
            digest.update(cfg.read_bytes())
        # RECORD includes native binary checksums; direct_url records source
        # installs. A rebuilt MLX wheel must not reuse another build's verdict.
        for pattern in ("RECORD", "direct_url.json"):
            for receipt in sorted(
                venv.glob(f"lib/python*/site-packages/*.dist-info/{pattern}")
            ):
                digest.update(str(receipt.relative_to(venv)).encode())
                digest.update(receipt.read_bytes())
        return digest.hexdigest()[:12]

    def python(self, fallback: str) -> str:
        """Honor an explicitly marked dependency-upgrade environment."""
        own = self.path / ".venv/bin/python"
        if (self.path / ".yv-own-venv").exists() and own.is_file():
            return str(own)
        return fallback

    @property
    def key(self) -> str:
        """Identity of the code under test: commit, plus a hash of uncommitted changes."""
        return (
            self.commit[:12]
            + (f"-d{self.dirty}" if self.dirty else "")
            + (f"-v{self.runtime_fingerprint}" if self.runtime_fingerprint else "")
        )

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "spec": self.spec,
            "commit": self.commit,
            "key": self.key,
            "path": str(self.path),
            "dirty": bool(self.dirty),
            "runtime_fingerprint": self.runtime_fingerprint,
        }


def resolve_arm(name: str, spec: str, trees: Path | None = None) -> Arm:
    """`spec` is a git ref (a pinned detached worktree is built, reused by commit hash)
    or an existing directory (a checkout / worktree used as is; uncommitted changes are hashed
    into the arm key so a resume never mixes two states of the same tree)."""
    trees = trees or TREES
    p = Path(spec)
    if p.is_dir() and (p / ".git").exists():
        commit = git("rev-parse", "HEAD", cwd=p)
        diff = git("diff", "HEAD", cwd=p)
        dirty = sha(diff, n=8) if diff else ""
        return Arm(name, spec, commit, p.resolve(), dirty)
    commit = git("rev-parse", "--verify", spec + "^{commit}")
    path = trees / commit[:12]
    if not (path / ".git").exists():
        trees.mkdir(parents=True, exist_ok=True)
        git("worktree", "add", "--detach", str(path), commit)
    if git("rev-parse", "HEAD", cwd=path) != commit:
        raise InfraError(f"{path} is not at {commit[:12]}")
    return Arm(name, spec, commit, path, "")


def changed_files(base: Arm, cand: Arm) -> list[str]:
    if cand.dirty or base.dirty:
        out = git("diff", "--name-only", base.commit, cwd=cand.path)
        extra = git("ls-files", "--others", "--exclude-standard", cwd=cand.path)
        return sorted({x for x in (out + "\n" + extra).splitlines() if x})
    if base.commit == cand.commit:
        return []
    return sorted(
        x
        for x in git(
            "diff", "--name-only", f"{base.commit}..{cand.commit}", cwd=cand.path
        ).splitlines()
        if x
    )


def related_tests(files: list[str], tree: Path) -> list[str]:
    """Map changed files to the unit tests that touch them: changed test files, tests named
    after the module stem, and tests that import / mention the dotted module path or its stem."""
    tests_dir = tree / "tests"
    all_tests = sorted(tests_dir.rglob("test_*.py")) if tests_dir.exists() else []
    texts: dict[Path, str] = {}

    def text(p: Path) -> str:
        if p not in texts:
            texts[p] = p.read_text(errors="replace")
        return texts[p]

    found: set[str] = set()
    for f in files:
        if not f.endswith(".py"):
            continue
        fp = Path(f)
        if fp.name.startswith("test_") and (tree / f).exists():
            found.add(f)
            continue
        if fp.parts[0] not in ("python", "scripts"):
            continue
        stem = fp.stem
        if stem in ("__init__", "__main__"):
            stem = fp.parent.name
        parts = list(fp.with_suffix("").parts)
        if parts[0] == "python":
            parts = parts[1:]
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        dotted = ".".join(parts)
        pat = re.compile(r"\b" + re.escape(stem) + r"\b")
        for t in all_tests:
            rel = str(t.relative_to(tree))
            if t.stem.startswith("test_" + stem) or (dotted and dotted in text(t)):
                found.add(rel)
            elif pat.search(text(t)):
                # the stem is mentioned: only count real references (import lines
                # or path strings), not prose
                ref = (
                    rf"(from\s+[\w.]*\b{stem}\b|import\s+[\w., ]*\b{stem}\b"
                    rf"|[/\"']{stem}(\.py)?[/\"'])"
                )
                if re.search(ref, text(t)):
                    found.add(rel)
    return sorted(found)


# ── run directory ────────────────────────────────────────────────────────
class RunDir:
    """Evidence of one verification: <stage>.jsonl per stage + state.json + cells/ files."""

    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "cells").mkdir(exist_ok=True)

    def stage_file(self, stage: str) -> Path:
        return self.path / f"{stage}.jsonl"

    def append(self, stage: str, row: dict) -> None:
        row = dict(row, t=round(now(), 1))
        with self.stage_file(stage).open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def rows(self, stage: str) -> list[dict]:
        return read_jsonl(self.stage_file(stage))

    def state(self) -> dict:
        p = self.path / "state.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def save_state(self, st: dict) -> None:
        st["updated"] = now()
        write_json_atomic(self.path / "state.json", st)

    def cell_path(self, stage: str, cell: str, suffix: str = "jsonl") -> Path:
        return self.path / "cells" / f"{stage}.{cell}.{suffix}"


# ── gpuq client ──────────────────────────────────────────────────────────
class Job:
    def __init__(self, d: dict):
        self.d = d
        self.id = d.get("id", "")
        self.state = d.get("state", "")
        self.rc = d.get("rc")
        self.contended = bool(d.get("contended"))

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL


def gpuq_dir() -> Path:
    d = os.environ.get("GPUQ_DIR")
    if d:
        return Path(d)
    env = MAIN_CHECKOUT / "scripts/research/local.env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GPUQ_DIR="):
                return Path(line.split("=", 1)[1].strip())
    raise InfraError("GPUQ_DIR is not set")


class Gpuq:
    def __init__(self, binary: str | None = None, jobs_dir: Path | None = None):
        self.bin = binary or GPUQ_BIN
        self._jobs_dir = jobs_dir

    @property
    def jobs_dir(self) -> Path:
        return self._jobs_dir or (gpuq_dir() / "jobs")

    def submit(
        self,
        label: str,
        argv: list[str],
        *,
        timeout_min: float,
        mem_gb: float,
        stall_min: float = 10,
        priority: int = 0,
        quiet: bool = False,
        out: Path | None = None,
        expect_complete: bool = False,
        cwd: Path | None = None,
        device: str = "",
    ) -> str:
        cmd = [
            self.bin,
            "submit",
            "--label",
            label,
            "--timeout",
            str(timeout_min),
            "--stall",
            str(stall_min),
            "--mem-gb",
            str(mem_gb),
            "--priority",
            str(priority),
        ]
        if timeout_min <= 10:
            # Preserve the declared short lane even when learned_timeout would
            # otherwise inflate this wrapper beyond ten minutes. gpuq --short
            # enforces the lane cap and kills an overrun.
            cmd.append("--short")
        if quiet:
            cmd.append("--quiet")
        if device:
            cmd += ["--device", device]
        if out is not None:
            cmd += ["--out", str(out)]
            if expect_complete:
                cmd.append("--expect-complete")
        cmd += ["--", *argv]
        r = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(cwd) if cwd else None
        )
        text = (r.stdout + r.stderr).strip()
        if r.returncode != 0:
            raise InfraError(f"gpuq submit refused: {text[-600:]}")
        ids = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        if not ids:
            raise InfraError(f"gpuq submit printed no job id: {text[-300:]}")
        return ids[-1].split()[-1]

    def job(self, jid: str) -> Job:
        p = self.jobs_dir / f"{jid}.json"
        for _ in range(5):
            try:
                return Job(json.loads(p.read_text()))
            except (OSError, ValueError):
                time.sleep(0.5)
        raise InfraError(f"job file unreadable: {p}")

    def resolve(self, jid: str) -> Job:
        """Follow a requeue (gpuq stops a backlog job for p>=0 work and queues a copy)."""
        j = self.job(jid)
        for _ in range(20):
            nxt = j.d.get("requeued_as")
            if not nxt:
                return j
            j = self.job(nxt)
        return j

    def wait(self, jid: str, max_seconds: int = 3000) -> Job:
        """Block until the job (or its requeued copy) is terminal. Never concludes from a cut-off wait."""
        while True:
            j = self.resolve(jid)
            if j.terminal:
                return j
            subprocess.run(
                [self.bin, "wait", "--max-seconds", str(max_seconds), j.id],
                capture_output=True,
            )

    def cancel(self, jid: str) -> None:
        subprocess.run([self.bin, "cancel", jid], capture_output=True)

    def log_tail(self, jid: str, n: int = 12) -> str:
        r = subprocess.run([self.bin, "log", jid], capture_output=True, text=True)
        return "\n".join((r.stdout + r.stderr).strip().splitlines()[-n:])
