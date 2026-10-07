"""Run a stage's cells as gpuq jobs: resumable, fail-fast, fail closed."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .core import VERIFY_ROOT, Gpuq, InfraError, RunDir, sha

Validator = Callable[[Path], "tuple[bool, str]"]


@dataclass
class Cell:
    stage: str
    key: str  # unique within the stage, e.g. "base-1024"
    argv: list  # may contain "{out}" (replaced by the attempt file)
    mem_gb: float = 24
    timeout_min: float = 20
    stall_min: float = 10
    quiet: bool = False
    needs_out: bool = (
        True  # False: the tool writes its own evidence (quality); only rc matters
    )
    validate: Validator | None = None
    retries: int = 1  # extra attempts for a contended timing cell
    priority: int = 0
    meta: dict = field(default_factory=dict)
    # Deterministic evidence (greedy digests of a base arm) that any run with the same inputs may
    # reuse: a content key (commit, env, model, harness hash, cell parameters), never run paths.
    share_key: str = ""
    device: str = ""  # "any": small-model liveness cells may run on the M3 lane
    cwd: Path | None = None  # remote snapshots must come from this pinned arm

    @property
    def sig(self) -> str:
        return sha(self.argv, self.quiet)


@dataclass
class CellResult:
    key: str
    ok: bool
    reason: str = ""
    job: str = ""
    evidence: Path | None = None
    contended: bool = False
    cached: bool = False


def default_validate(path: Path) -> tuple[bool, str]:
    """Fail closed: evidence must exist and end with a record that says complete=true."""
    from .core import read_jsonl

    rows = read_jsonl(path)
    if not rows:
        return False, f"no evidence in {path.name}"
    if not any(r.get("complete") is True for r in rows):
        return False, f"{path.name} has no complete=true record (job ended early)"
    return True, ""


class Executor:
    def __init__(
        self,
        run: RunDir,
        gq: Gpuq,
        label: str,
        log: Callable[[str], None] = print,
        priority: int = 0,
        cache_dir: Path | None = None,
    ):
        self.cache_dir = (
            cache_dir if cache_dir is not None else VERIFY_ROOT / "cellcache"
        )
        self.run, self.gq, self.label, self.log, self.priority = (
            run,
            gq,
            label,
            log,
            priority,
        )
        self.jobs: list = []  # (stage, cell, job id) for the verdict

    # -- helpers
    def _cached(self, cell: Cell) -> CellResult | None:
        done = None
        for r in self.run.rows(cell.stage):
            if r.get("ev") == "cell_done" and r.get("cell") == cell.key:
                done = r
        if done and done.get("ok") and done.get("sig") == cell.sig:
            ev = self.run.cell_path(cell.stage, cell.key)
            if ev.exists() or not cell.needs_out:
                self.jobs.append((cell.stage, cell.key, done.get("job", ""), True))
                return CellResult(
                    cell.key, True, "", done.get("job", ""), ev, False, True
                )
        return None

    def _shared(self, cell: Cell) -> CellResult | None:
        """Evidence another run already produced for the same inputs (see Cell.share_key)."""
        if not cell.share_key or not cell.needs_out:
            return None
        src = self.cache_dir / f"{cell.share_key}.jsonl"
        ok, _ = (
            (cell.validate or default_validate)(src) if src.exists() else (False, "")
        )
        if not ok:
            return None
        final = self.run.cell_path(cell.stage, cell.key)
        shutil.copyfile(src, final)
        self.run.append(
            cell.stage,
            {
                "ev": "cell_done",
                "cell": cell.key,
                "job": f"cache:{cell.share_key}",
                "sig": cell.sig,
                "ok": True,
                "rc": 0,
                "state": "cached",
                "contended": False,
                "reason": "",
            },
        )
        self.jobs.append((cell.stage, cell.key, f"cache:{cell.share_key}", True))
        return CellResult(
            cell.key, True, "", f"cache:{cell.share_key}", final, False, True
        )

    def _attempts(self, cell: Cell) -> int:
        return sum(
            1
            for r in self.run.rows(cell.stage)
            if r.get("ev") == "cell_submitted" and r.get("cell") == cell.key
        )

    def _inflight(self, cell: Cell) -> tuple[str, int] | None:
        """A job submitted by an earlier invocation that has no recorded outcome: re-attach."""
        sub = None
        for r in self.run.rows(cell.stage):
            if r.get("cell") != cell.key:
                continue
            if r.get("ev") == "cell_submitted" and r.get("sig") == cell.sig:
                sub = r
            elif r.get("ev") in ("cell_done", "cell_cancelled"):
                sub = None
        if sub:
            try:
                j = self.gq.resolve(sub["job"])
            except InfraError:
                return None
            if j.state in ("pending", "running") or (
                j.terminal and sub.get("attempt") is not None
            ):
                return j.id, int(sub["attempt"])
        return None

    def _submit(self, cell: Cell, attempt: int) -> str:
        out = self.run.cell_path(cell.stage, cell.key, f"a{attempt}.jsonl")
        if out.exists():
            out.unlink()
        argv = [str(a).replace("{out}", str(out)) for a in cell.argv]
        jid = self.gq.submit(
            f"{self.label}-{cell.stage}-{cell.key}-a{attempt}-{cell.sig[:4]}",
            argv,
            timeout_min=cell.timeout_min,
            stall_min=cell.stall_min,
            mem_gb=cell.mem_gb,
            priority=cell.priority if cell.priority else self.priority,
            quiet=cell.quiet,
            out=out if cell.needs_out else None,
            expect_complete=cell.needs_out,
            device=cell.device,
            cwd=cell.cwd,
        )
        self.run.append(
            cell.stage,
            {
                "ev": "cell_submitted",
                "cell": cell.key,
                "job": jid,
                "attempt": attempt,
                "sig": cell.sig,
                "argv": argv,
            },
        )
        self.log(f"submitted {cell.stage}/{cell.key} -> {jid}")
        return jid

    def _finish(self, cell: Cell, jid: str, attempt: int) -> CellResult:
        j = self.gq.wait(jid)
        out = self.run.cell_path(cell.stage, cell.key, f"a{attempt}.jsonl")
        ok, reason = True, ""
        if j.state != "done" or j.rc not in (0, None):
            ok = False
            reason = f"job {j.id} {j.state} rc={j.rc}: " + self.gq.log_tail(j.id, 6)
        elif cell.needs_out:
            ok, reason = (cell.validate or default_validate)(out)
            if not ok:
                reason = f"job {j.id}: {reason}"
        if ok and cell.quiet and j.contended:
            ok, reason = False, f"job {j.id} was CPU-contended (timing not trusted)"
        final = self.run.cell_path(cell.stage, cell.key)
        if ok and cell.needs_out:
            os.replace(out, final)
            if cell.share_key:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = self.cache_dir / f".{cell.share_key}.tmp"
                shutil.copyfile(final, tmp)
                os.replace(tmp, self.cache_dir / f"{cell.share_key}.jsonl")
        self.run.append(
            cell.stage,
            {
                "ev": "cell_done",
                "cell": cell.key,
                "job": j.id,
                "attempt": attempt,
                "sig": cell.sig,
                "ok": ok,
                "rc": j.rc,
                "state": j.state,
                "contended": j.contended,
                "reason": reason,
            },
        )
        self.jobs.append((cell.stage, cell.key, j.id, False))
        self.log(f"{cell.stage}/{cell.key}: {'ok' if ok else 'FAILED'} {reason[:200]}")
        return CellResult(
            cell.key,
            ok,
            reason,
            j.id,
            final if ok and cell.needs_out else None,
            j.contended,
        )

    # -- API
    def run_cells(self, cells: list) -> dict:
        """Run every cell not already completed; returns key -> CellResult. Stops at the first
        failure (fail-fast): pending cells of the batch are cancelled and left unresults."""
        results: dict = {}
        todo: list = []
        for c in cells:
            hit = self._cached(c) or self._shared(c)
            if hit:
                results[c.key] = hit
                self.log(f"{c.stage}/{c.key}: reused (job {hit.job})")
            else:
                todo.append(c)
        live: dict = {}
        for c in todo:
            att = self._inflight(c)
            if att:
                live[c.key] = att
                self.log(f"{c.stage}/{c.key}: re-attached to {att[0]}")
            else:
                n = self._attempts(c) + 1
                live[c.key] = (self._submit(c, n), n)
        for c in todo:
            jid, attempt = live[c.key]
            r = self._finish(c, jid, attempt)
            tries = 0
            while not r.ok and (r.contended or "CPU-contended" in r.reason):
                tries += 1
                if tries > c.retries:
                    break
                self.log(f"{c.stage}/{c.key}: contended, rerunning")
                n = self._attempts(c) + 1
                r = self._finish(c, self._submit(c, n), n)
            results[c.key] = r
            if not r.ok:
                for k2, (j2, _a) in live.items():
                    if k2 != c.key and k2 not in results:
                        self.gq.cancel(j2)
                        self.run.append(
                            c.stage,
                            {
                                "ev": "cell_cancelled",
                                "cell": k2,
                                "job": j2,
                                "why": f"{c.key} failed",
                            },
                        )
                break
        return results
