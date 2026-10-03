"""Digest of gpuq jobs that finished since the last look (stdlib only, python 3.9 safe).

    gpuq digest                 # jobs finished since the last digest (first time: last 24 h), then mark
    gpuq digest --since 6h      # or 90m / 2d / an ISO time / epoch seconds; does not move the marker
    gpuq digest --label-prefix worker- --since 6h  # only this label prefix; does not move the marker
    gpuq digest --peek          # same window as the marker but leave the marker alone

Finished jobs are grouped by label family (the label without its round / shard suffix: ``paired-gsm8k-ref-r3``
-> ``paired-gsm8k-ref``). Each family shows its states, the log of its last job and every output path found
at submission or in commands (the argument after ``--out`` / ``--output``, also ``--out=PATH``). Problems are
flagged and listed first, because they are the results nobody read:

* a contended perf/quiet job (CPU timing is not trustworthy; rerun after contention);
* a job that ended failed, lost, stalled, timeout or cancelled, or has an absent/nonzero rc;
* an output that is missing/empty, or lacks a line containing ``complete`` when expect_complete is set.
  A directory counts as empty when it has no files; complete checking requires a file.

Exit status 1 when anything is flagged, so a coordinator loop can stop on it. The marker
(``$GPUQ_DIR/.digest_marker``) is the start of the last digest's window, taken before the jobs were read, so a job
that ends while the digest runs shows up in the next one.
"""

# ruff: noqa: UP031, UP045  (python 3.9 stdlib-only script: keep %-format / Optional)
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

BAD_STATES = ("failed", "lost", "stalled", "timeout", "cancelled")
FINAL = BAD_STATES + ("done",)
OUT_FLAGS = ("--out", "--output")
DEFAULT_WINDOW_S = 24 * 3600
_SUFFIX = re.compile(r"-(?:r\d+|p\d+s\d+|s\d+|\d+)$")


def gpuq_dir() -> Path:
    return Path(os.environ.get("GPUQ_DIR", "~/.cache/yunshu/gpuq")).expanduser()


def family(label: str) -> str:
    """Label without trailing round / shard counters (repeatedly: ``x-p1s2`` and ``x-r3``)."""
    prev = None
    while prev != label:
        prev, label = label, _SUFFIX.sub("", label) or label
    return label


def output_paths(job: dict) -> list:
    """Paths named by --out / --output in the job command, resolved against the job's cwd."""
    cmd = [str(x) for x in job.get("cmd") or []]
    found = [str(p) for p in job.get("outputs", [])]
    for i, arg in enumerate(cmd):
        val = None
        if arg in OUT_FLAGS and i + 1 < len(cmd):
            val = cmd[i + 1]
        else:
            for flag in OUT_FLAGS:
                if arg.startswith(flag + "="):
                    val = arg[len(flag) + 1 :]
        if val and not val.startswith("-"):
            found.append(val)
    paths = []
    for val in found:
        p = Path(val).expanduser()
        if not p.is_absolute():
            p = Path(job.get("cwd") or ".") / p
        if p not in paths:
            paths.append(p)
    return paths


def output_problem(path: Path, expect_complete: bool = False) -> Optional[str]:
    """None when the output exists and has content, else a short reason."""
    try:
        if path.is_dir():
            if expect_complete:
                return "complete requires a file"
            return None if any(q.is_file() for q in path.rglob("*")) else "empty dir"
        if not path.exists():
            return "missing"
        if path.stat().st_size == 0:
            return "empty"
        if expect_complete:
            with path.open(encoding="utf-8", errors="replace") as f:
                if not any("complete" in line for line in f):
                    return "no complete line"
        return None
    except OSError as e:
        return "unreadable (%s)" % e.__class__.__name__


def parse_since(text: str, now: float) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", text.strip())
    if m:
        return (
            now
            - float(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
        )
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError as e:
        raise SystemExit(
            "bad --since %r (use 90m, 6h, 2d, epoch seconds or ISO time)" % text
        ) from e


def read_marker(root: Path) -> Optional[float]:
    try:
        return float((root / ".digest_marker").read_text().strip())
    except (OSError, ValueError):
        return None


def write_marker(root: Path, t: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=root, prefix=".digest_marker.", delete=False
    ) as f:
        tmp = Path(f.name)
        f.write("%.17g\n" % t)
    try:
        tmp.replace(root / ".digest_marker")
    finally:
        tmp.unlink(missing_ok=True)


def load_jobs(root: Path) -> list:
    jobs = []
    for f in sorted((root / "jobs").glob("*.json")):
        try:
            jobs.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue
    return jobs


def output_issues(job: dict) -> list:
    """Shared wait/digest output contract, including legacy command output flags."""
    issues = []
    paths = output_paths(job)
    for path in paths:
        why = output_problem(path, bool(job.get("expect_complete")))
        if why:
            issues.append("output %s: %s" % (why, path))
    if job.get("expect_complete") and not paths:
        issues.append("output missing: --expect-complete requires an output path")
    return issues


def collect(root: Path, since: float, until: float, label_prefix: str = "") -> dict:
    """Finished jobs with ended in (since, until], plus problems, grouped by family."""
    fams: dict = {}
    problems = []
    for j in load_jobs(root):
        if not j.get("label", "").startswith(label_prefix):
            continue
        if j.get("state") not in FINAL:
            continue
        ended = j.get("ended") or 0
        if not (since < ended <= until):
            continue
        outs = output_paths(j)
        issues = []
        if j["state"] in BAD_STATES or j.get("rc") != 0:
            issues.append("state %s rc=%s" % (j["state"], j.get("rc")))
        if j.get("contended") and (j.get("priority", 0) >= 0 or j.get("quiet")):
            issues.append(
                "contended: CPU timing is not trustworthy; rerun after contention"
            )
        issues.extend(output_issues(j))
        entry = {
            "id": j["id"],
            "device": j.get("device", "m5"),
            "remote_host": j.get("remote_host"),
            "evidence": "portability evidence (not M5)"
            if j.get("device") == "m3"
            else "M5 evidence",
            "label": j.get("label", ""),
            "state": "contended"
            if j["state"] == "done" and j.get("contended")
            else j["state"],
            "contended": bool(j.get("contended")),
            "foreign_cpu_max_pct": j.get("foreign_cpu_max_pct"),
            "foreign_cpu_mean_pct": j.get("foreign_cpu_mean_pct"),
            "rc": j.get("rc"),
            "dur": max(0.0, ended - (j.get("started") or ended)),
            "ended": ended,
            "log": str(root / "logs" / (j["id"] + ".log")),
            "outputs": [str(p) for p in outs],
            "issues": issues,
        }
        fams.setdefault(family(entry["label"]), []).append(entry)
        if issues:
            problems.append(entry)
    return {"families": fams, "problems": problems}


def _fmt_t(t: float) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(t))


def _fmt_dur(s: float) -> str:
    return (
        "%dh%02dm" % (s // 3600, s % 3600 // 60)
        if s >= 3600
        else "%dm%02ds" % (s // 60, s % 60)
    )


def render(res: dict, since: float, until: float) -> str:
    fams, problems = res["families"], res["problems"]
    total = sum(len(v) for v in fams.values())
    out = [
        "gpuq digest %s -> %s: %d finished jobs, %d flagged"
        % (_fmt_t(since), _fmt_t(until), total, len(problems))
    ]
    if problems:
        out.append("")
        out.append("FLAGGED (read these first)")
        for e in sorted(problems, key=lambda e: e["ended"]):
            for issue in e["issues"]:
                out.append("  %s  %s  %s" % (e["id"], _fmt_dur(e["dur"]), issue))
            out.append("      log %s" % e["log"])
    if fams:
        out.append("")
        out.append("BY FAMILY")
    for name in sorted(fams):
        es = sorted(fams[name], key=lambda e: e["ended"])
        states: dict = {}
        for e in es:
            states[e["state"]] = states.get(e["state"], 0) + 1
        summary = " ".join("%s=%d" % kv for kv in sorted(states.items()))
        out.append(
            "  %s  [%s]  %s total, last %s"
            % (
                name,
                summary,
                _fmt_dur(sum(e["dur"] for e in es)),
                _fmt_t(es[-1]["ended"]),
            )
        )
        for e in es:
            out.append(
                "      %s device=%s %s rc=%s"
                % (
                    e["id"],
                    e.get("device", "m5"),
                    e.get("evidence", "M5 evidence"),
                    e["rc"],
                )
            )
        outs = sorted({o for e in es for o in e["outputs"]})
        for o in outs:
            out.append("      out %s" % o)
        out.append("      log %s" % es[-1]["log"])
    if not fams:
        out.append("(nothing finished in the window)")
    return "\n".join(out)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="gpuq digest", description=__doc__.split("\n")[0])
    ap.add_argument(
        "--since",
        help="window start: 90m, 6h, 2d, epoch seconds or ISO time (default: last digest)",
    )
    ap.add_argument(
        "--label-prefix", default="", help="only labels starting with this prefix"
    )
    ap.add_argument("--peek", action="store_true", help="do not move the marker")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args(argv)
    root = gpuq_dir()
    now = time.time()  # taken before the jobs are read
    marker = read_marker(root)
    if a.since:
        since = parse_since(a.since, now)
    else:
        since = marker if marker is not None else now - DEFAULT_WINDOW_S
    res = collect(root, since, now, a.label_prefix)
    if a.json:
        json.dump(res, sys.stdout, indent=1)
        sys.stdout.write("\n")
    else:
        print(render(res, since, now))
    if not a.since and not a.peek and not a.label_prefix:
        write_marker(root, now)
    return 1 if res["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
