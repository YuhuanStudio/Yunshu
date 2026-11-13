"""``yunshu prepare``: the one-time repack of a Flash-Next PLE table into ssd-rows (see ple_ssd).

One job class serves the CLI and the console's admin endpoint. The repack runs in a child process
(``python -m yunshu_engine.ple_prepare <model>``: numpy only, no GPU), so a cancel is a clean kill
and a long copy never holds the server's GIL. Progress is the number of blocks the (sparse) temp
file has allocated; the finished manifest appears only after the data is complete, so a killed or
failed run is rebuilt from scratch and a finished one is skipped.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import ple_ssd

# free space kept after the repack: the larger of 10 GiB and 5% of the table
SAFETY_MIN_BYTES = 10 * 2**30


class InsufficientDiskError(RuntimeError):
    def __init__(self, needed: int, free: int, path: str) -> None:
        self.needed, self.free, self.path = needed, free, path
        super().__init__(
            f"not enough free disk space on {path}: the repack needs "
            f"{needed / 2**30:.1f} GiB (table plus margin), {free / 2**30:.1f} GiB is free"
        )


def needed_bytes(model_path: Path) -> int:
    table = ple_ssd.ple_payload_bytes(model_path)
    return table + max(SAFETY_MIN_BYTES, table // 20)


def _free_bytes(directory: Path) -> int:
    probe = directory
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def check_disk(model_path: Path) -> None:
    directory = ple_ssd.rows_dir_for(model_path)
    needed, free = needed_bytes(model_path), _free_bytes(directory)
    if free < needed:
        raise InsufficientDiskError(needed, free, str(directory))


def status(model_path: Path) -> dict:
    """What the console and CLI show: whether the pack applies, is prepared, and what is in use."""
    model_path = Path(model_path)
    if not ple_ssd.has_ple_table(model_path):
        return {"applicable": False}
    prepared = ple_ssd.prepared_manifest(model_path)
    storage, why = ple_ssd.effective_storage(model_path)
    directory = ple_ssd.rows_dir_for(model_path)
    return {
        "applicable": True,
        "prepared": prepared is not None,
        "storage": storage,
        "storage_reason": why,
        "table_bytes": ple_ssd.ple_payload_bytes(model_path),
        "needed_bytes": needed_bytes(model_path),
        "free_bytes": _free_bytes(directory),
        "prepared_bytes": sum(
            f.stat().st_size for f in directory.glob("*") if f.is_file()
        )
        if prepared is not None
        else 0,
        "rows_dir": str(directory),
    }


def remove(model_path: Path) -> int:
    """Delete the prepared repack (the checkpoint is untouched); returns the bytes freed."""
    directory = ple_ssd.rows_dir_for(Path(model_path))
    if not directory.is_dir():
        return 0
    freed = sum(f.stat().st_size for f in directory.glob("*") if f.is_file())
    shutil.rmtree(directory, ignore_errors=True)
    return freed


class PrepareJob:
    """One repack in a child process. ``poll()`` is cheap and safe from any thread."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self.total = ple_ssd.ple_payload_bytes(self.model_path)
        self.directory = ple_ssd.rows_dir_for(self.model_path)
        self.started = time.time()
        self.finished: float | None = None
        self.error: str | None = None
        self.cancelled = False
        self._proc: subprocess.Popen[bytes] | None = None
        self._log = b""

    def start(self) -> None:
        check_disk(self.model_path)
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "yunshu_engine.ple_prepare", str(self.model_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        self._log = self._proc.stderr.read()[-4000:]

    def bytes_done(self) -> int:
        try:
            return sum(
                f.stat().st_blocks * 512
                for f in self.directory.glob(".ple-q4.rows.*.partial")
            )
        except OSError:
            return 0

    def state(self) -> str:
        if self._proc is None:
            return "queued"
        code = self._proc.poll()
        if code is None:
            return "running"
        if self.finished is None:
            self.finished = time.time()
            if self.cancelled:
                ple_ssd.clear_partials(self.directory)
            elif code != 0:
                self.error = (
                    self._log.decode(errors="replace").strip().splitlines()
                    or [f"exit {code}"]
                )[-1]
        if self.cancelled:
            return "cancelled"
        return "failed" if code != 0 else "done"

    def poll(self) -> dict:
        state = self.state()
        done = self.total if state == "done" else min(self.bytes_done(), self.total)
        elapsed = (self.finished or time.time()) - self.started
        eta = (
            elapsed * (self.total - done) / done
            if state == "running" and done > 0
            else None
        )
        return {
            "state": state,
            "bytes_done": done,
            "bytes_total": self.total,
            "eta_s": eta,
            "elapsed_s": elapsed,
            "error": self.error,
        }

    def cancel(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self.cancelled = True
            self._proc.terminate()
            try:
                self._proc.wait(10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self.state()  # clean the temp file now

    def wait(self) -> int:
        assert self._proc is not None
        return self._proc.wait()


_jobs: dict[str, PrepareJob] = {}
_lock = threading.Lock()


def job_for(model_path: Path) -> PrepareJob | None:
    with _lock:
        return _jobs.get(str(Path(model_path).resolve()))


def start_job(model_path: Path) -> PrepareJob:
    """Start (or return the running) job for ``model_path``; raises InsufficientDiskError."""
    key = str(Path(model_path).resolve())
    with _lock:
        current = _jobs.get(key)
        if current is not None and current.state() == "running":
            return current
        job = PrepareJob(Path(model_path))
        job.start()
        _jobs[key] = job
        return job


def main(argv: list[str]) -> int:
    ple_ssd.rows_manifest_for(Path(argv[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
