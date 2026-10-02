"""Digest of gpuq jobs that finished since the last look (stdlib only, python 3.9 safe).

    gpuq digest                 # jobs finished since the last digest (first time: last 24 h), then mark
    gpuq digest --since 6h      # or 90m / 2d / an ISO time / epoch seconds; does not move the marker
    gpuq digest --peek          # same window as the marker but leave the marker alone

Finished jobs are grouped by label family (the label without its round / shard suffix: ``paired-gsm8k-ref-r3``
-> ``paired-gsm8k-ref``). Each family shows its states, the log of its last job and every output path found
in the commands (the argument after ``--out`` / ``--output``, also ``--out=PATH``). Two kinds of problem are
flagged and listed first, because they are the results nobody read:

* a job that ended failed, lost, stalled or timeout;
* a ``done`` job whose output file is missing or empty (a directory counts as empty when it has no files).

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
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

BAD_STATES = ("failed", "lost", "stalled", "timeout")
FINAL = BAD_STATES + ("done", "cancelled")
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
    found = []
    for i, arg in enumerate(cmd):
        val = None
        if arg in OUT_FLAGS and i + 1 < len(cmd):
            val = cmd[i + 1]
        else:
            for flag in OUT_FLAGS:
                if arg.startswith(flag + "="):
                    val = arg[len(flag) + 1 :]
        if val and not val.startswith("-"):
            p = Path(val).expanduser()
            if not p.is_absolute():
                p = Path(job.get("cwd") or ".") / p
            found.append(p)
    return found


def output_problem(path: Path) -> Optional[str]:
    """None when the output exists and has content, else a short reason."""
    try:
        if path.is_dir():
            return None if any(q.is_file() for q in path.rglob("*")) else "empty dir"
        if not path.exists():
            return "missing"
        return None if path.stat().st_size > 0 else "empty"
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
    tmp = root / ".digest_marker.tmp"
    tmp.write_text("%.3f\n" % t)
    tmp.replace(root / ".digest_marker")


def load_jobs(root: Path) -> list:
    jobs = []
    for f in sorted((root / "jobs").glob("*.json")):
        try:
            jobs.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue
    return jobs


def collect(root: Path, since: float, until: float) -> dict:
    """Finished jobs with ended in (since, until], plus problems, grouped by family."""
    fams: dict = {}
    problems = []
    for j in load_jobs(root):
        if j.get("state") not in FINAL:
            continue
        ended = j.get("ended") or 0
        if not (since < ended <= until):
            continue
        outs = output_paths(j)
        issues = []
        if j["state"] in BAD_STATES:
            issues.append("state %s rc=%s" % (j["state"], j.get("rc")))
        if j["state"] == "done":
            for p in outs:
                why = output_problem(p)
                if why:
                    issues.append("output %s: %s" % (why, p))
        entry = {
            "id": j["id"],
            "label": j.get("label", ""),
            "state": j["state"],
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
    res = collect(root, since, now)
    if a.json:
        json.dump(res, sys.stdout, indent=1)
        sys.stdout.write("\n")
    else:
        print(render(res, since, now))
    if not a.since and not a.peek:
        write_marker(root, now)
    return 1 if res["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
