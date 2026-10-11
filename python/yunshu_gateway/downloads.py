"""Model download manager: a registry of background Hugging Face downloads.

``POST /v1/yunshu/downloads`` and Ollama's ``/api/pull`` both submit jobs here. A job
lists the repo's files first (so total bytes are known before the first byte moves),
refuses when the disk cannot hold the remainder, then downloads on a single worker thread
(one at a time; the rest queue). Progress comes from the hub's own tqdm bars, so byte
counts are real, not estimated. Cancelling raises inside the progress callback; the hub's
``.incomplete`` files stay on disk, so submitting the same repo again resumes.

The hub is behind a small interface (:class:`Hub`) so tests run with a fake and never
touch the network.
"""

from __future__ import annotations

import collections
import fnmatch
import logging
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

MAX_JOBS = 50  # finished jobs kept for the UI
RATE_WINDOW_S = 10.0
SAMPLE_EVERY_S = 0.25
DISK_MARGIN_MIN = 256 * 1024 * 1024
ACTIVE = ("queued", "running")


class DownloadCancelledError(Exception):
    """Raised inside a hub progress callback to abort the transfer."""


class InsufficientDiskError(Exception):
    def __init__(self, needed: int, free: int, path: str) -> None:
        self.needed, self.free, self.path = needed, free, path
        super().__init__(
            f"not enough disk space at {path}: need {needed / 1e9:.1f} GB, "
            f"{free / 1e9:.1f} GB free"
        )


class Hub(Protocol):
    def list_files(
        self, repo: str, revision: str | None, patterns: list[str] | None
    ) -> list[tuple[str, int]]: ...

    def download(
        self,
        repo: str,
        revision: str | None,
        patterns: list[str] | None,
        local_dir: Path | None,
        on_file: Callable[[str, int, int], None],
        on_bytes: Callable[[str, int], None],
    ) -> Path: ...

    def cache_dir(self) -> Path: ...


def matches(name: str, patterns: list[str] | None) -> bool:
    return not patterns or any(fnmatch.fnmatch(name, p) for p in patterns)


class HFHub:
    """The real hub: ``HfApi`` for sizes, ``snapshot_download`` (resumable) for bytes."""

    def list_files(self, repo, revision, patterns):
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        return [
            (s.rfilename, int(s.size or 0))
            for s in info.siblings or []
            if matches(s.rfilename, patterns)
        ]

    def download(self, repo, revision, patterns, local_dir, on_file, on_bytes):
        from huggingface_hub import snapshot_download
        from tqdm.auto import tqdm

        class Bar(tqdm):
            def __init__(self, *a, **kw):
                kw["disable"] = False
                super().__init__(*a, **kw)
                self._name = str(kw.get("desc") or "")
                if self.unit == "B":
                    on_file(self._name, int(self.total or 0), int(self.n or 0))

            def update(self, n=1):
                r = super().update(n)
                if self.unit == "B" and n:
                    on_bytes(self._name, int(n))
                return r

        kw: dict[str, Any] = {"repo_id": repo, "tqdm_class": Bar}
        if revision:
            kw["revision"] = revision
        if patterns:
            kw["allow_patterns"] = patterns
        if local_dir is not None:
            kw["local_dir"] = str(local_dir)
        return Path(snapshot_download(**kw))

    def cache_dir(self) -> Path:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)


def _existing_parent(p: Path) -> Path:
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


class Job:
    def __init__(
        self,
        repo: str,
        revision: str | None,
        patterns: list[str] | None,
        local_dir: Path | None,
        on_complete: Callable[[Job], None] | None = None,
    ) -> None:
        self.id = "dl_" + uuid.uuid4().hex[:12]
        self.repo, self.revision, self.patterns = repo, revision, patterns
        self.local_dir = local_dir
        self.on_complete = on_complete
        self.state = "queued"
        self.error: str | None = None
        self.path: str | None = None
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.total = 0
        self.done = 0
        self.files: dict[str, list[int]] = {}  # name -> [total, done]
        self.registered = False
        self.already_present = False
        self.cancel = threading.Event()
        self.finished_event = threading.Event()
        self._lock = threading.Lock()
        self._samples: collections.deque[tuple[float, int]] = collections.deque()

    # ── progress (called from the download thread) ─────────────────────
    def file_start(self, name: str, total: int, initial: int) -> None:
        with self._lock:
            if name not in self.files:
                self.files[name] = [total, initial]
                self.done += initial
            self._check_cancel()

    def add_bytes(self, name: str, n: int) -> None:
        now = time.monotonic()
        with self._lock:
            rec = self.files.setdefault(name, [0, 0])
            rec[1] += n
            self.done += n
            if not self._samples or now - self._samples[-1][0] >= SAMPLE_EVERY_S:
                self._samples.append((now, self.done))
            while self._samples and now - self._samples[0][0] > RATE_WINDOW_S:
                self._samples.popleft()
            self._check_cancel()

    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise DownloadCancelledError()

    # ── views ──────────────────────────────────────────────────────────
    def rate_bps(self) -> float | None:
        with self._lock:
            if self.state != "running" or len(self._samples) < 2:
                return None
            (t0, b0), (t1, b1) = self._samples[0], self._samples[-1]
        return (b1 - b0) / (t1 - t0) if t1 > t0 else None

    def to_dict(self, detail: bool = False) -> dict[str, Any]:
        rate = self.rate_bps()
        with self._lock:
            total, done = self.total, min(self.done, self.total or self.done)
            files_total = len(self.files)
            files_done = sum(1 for t, d in self.files.values() if t and d >= t)
            active = [n for n, (t, d) in self.files.items() if not (t and d >= t)]
            out: dict[str, Any] = {
                "id": self.id,
                "repo": self.repo,
                "revision": self.revision,
                "allow_patterns": self.patterns,
                "state": self.state,
                "error": self.error,
                "path": self.path,
                "bytes_total": total,
                "bytes_done": done,
                "files_total": files_total,
                "files_done": files_done,
                "active_files": active[:8],
                "rate_bps": round(rate, 1) if rate else None,
                "eta_s": round((total - done) / rate, 1)
                if rate and total > done
                else None,
                "created": self.created,
                "started": self.started,
                "finished": self.finished,
                "registered": self.registered,
                "already_present": self.already_present,
            }
            if detail:
                out["files"] = [
                    {"name": n, "bytes_total": t, "bytes_done": d}
                    for n, (t, d) in list(self.files.items())[:200]
                ]
        return out

    def wait(self, timeout: float | None = None) -> bool:
        return self.finished_event.wait(timeout)


class DownloadRegistry:
    def __init__(self, hub: Hub | None = None) -> None:
        self.hub: Hub = hub or HFHub()
        self._jobs: collections.OrderedDict[str, Job] = collections.OrderedDict()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="yunshu-download")

    # ── API ────────────────────────────────────────────────────────────
    def submit(
        self,
        repo: str,
        *,
        revision: str | None = None,
        allow_patterns: list[str] | None = None,
        local_dir: Path | None = None,
        on_complete: Callable[[Job], None] | None = None,
    ) -> Job:
        """Queue a download; returns the already-active job for the same request."""
        patterns = list(allow_patterns) if allow_patterns else None
        with self._lock:
            for j in self._jobs.values():
                if j.state in ACTIVE and (
                    j.repo,
                    j.revision,
                    j.patterns,
                    j.local_dir,
                ) == (repo, revision, patterns, local_dir):
                    return j
            job = Job(repo, revision, patterns, local_dir, on_complete)
            self._jobs[job.id] = job
            self._trim()
        self._pool.submit(self._run, job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def jobs(self) -> list[Job]:
        with self._lock:
            return list(reversed(self._jobs.values()))

    def cancel(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is None:
            return None
        if job.state in ACTIVE:
            job.cancel.set()
            if job.state == "queued":
                self._finish(job, "cancelled")
        return job

    def _trim(self) -> None:
        done = [k for k, j in self._jobs.items() if j.state not in ACTIVE]
        for k in done[: max(0, len(self._jobs) - MAX_JOBS)]:
            del self._jobs[k]

    # ── worker ─────────────────────────────────────────────────────────
    def _finish(self, job: Job, state: str, error: str | None = None) -> None:
        with job._lock:
            if job.finished is not None:
                return
            job.state, job.error, job.finished = state, error, time.time()
        job.finished_event.set()

    def _run(self, job: Job) -> None:
        if job.cancel.is_set() or job.finished is not None:
            self._finish(job, "cancelled")
            return
        job.state = "running"
        job.started = time.time()
        try:
            files = self.hub.list_files(job.repo, job.revision, job.patterns)
            if not files:
                raise RuntimeError("the repository has no files matching the request")
            with job._lock:
                job.total = sum(s for _, s in files)
            self._check_disk(job.local_dir, files, job.total)
            job.path = str(
                self.hub.download(
                    job.repo,
                    job.revision,
                    job.patterns,
                    job.local_dir,
                    job.file_start,
                    job.add_bytes,
                )
            )
            if job.on_complete is not None:
                job.on_complete(job)
            with job._lock:
                job.done = max(job.done, job.total)
            self._finish(job, "done")
        except DownloadCancelledError:
            self._finish(job, "cancelled")
        except InsufficientDiskError as exc:
            self._finish(job, "failed", str(exc))
        except Exception as exc:  # noqa: BLE001 - every failure is a job state, not a crash
            logger.info("download %s failed: %s", job.repo, exc, exc_info=True)
            self._finish(job, "failed", f"{type(exc).__name__}: {exc}")

    def preflight(
        self,
        repo: str,
        revision: str | None,
        patterns: list[str] | None,
        local_dir: Path | None,
    ) -> int:
        """List the files and check the disk; returns the total bytes. Raises
        :class:`InsufficientDiskError` (or whatever the hub raises for an unknown repo)."""
        files = self.hub.list_files(repo, revision, patterns)
        if not files:
            raise RuntimeError("the repository has no files matching the request")
        total = sum(s for _, s in files)
        self._check_disk(local_dir, files, total)
        return total

    def _check_disk(
        self, local_dir: Path | None, files: list[tuple[str, int]], total: int
    ) -> None:
        target = local_dir if local_dir is not None else self.hub.cache_dir()
        have = 0
        if local_dir is not None:
            for name, size in files:
                f = local_dir / name
                try:
                    if f.is_file() and f.stat().st_size >= size:
                        have += size
                except OSError:
                    pass
        remaining = max(0, total - have)
        needed = remaining + max(DISK_MARGIN_MIN, remaining // 20)
        probe = _existing_parent(target)
        free = shutil.disk_usage(probe).free
        if needed > free:
            raise InsufficientDiskError(needed, free, str(probe))

    def record_present(self, repo: str, path: str) -> Job:
        """A finished job for a model that was already on disk (nothing downloaded)."""
        job = Job(repo, None, None, None)
        job.path, job.already_present = path, True
        job.started = time.time()
        with self._lock:
            self._jobs[job.id] = job
            self._trim()
        self._finish(job, "done")
        return job


_registry: DownloadRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> DownloadRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = DownloadRegistry()
        return _registry


def set_registry(reg: DownloadRegistry | None) -> None:
    global _registry
    with _registry_lock:
        _registry = reg
