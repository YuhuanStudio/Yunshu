"""Report what changed upstream for everything Yunshu copies or builds on.

Reads ``vendor.json`` and, using the local clones under ``reference/``:

1. For each vendored file: did upstream change it since the commit we copied
   (diffstat + commit subjects), and how does our copy differ from the version
   we copied (so intended local changes are visible before a sync).
2. For each watched repo: commits touching the watched paths since our pinned
   point (the vendored commit, or the last fetch), and new files matching the
   globs that we have not vendored.
3. Installed vs latest PyPI version for the packages we depend on.

    python scripts/vendor/check_upstream.py            # fetch + report
    python scripts/vendor/check_upstream.py --no-fetch  # offline, current clones

Exit code 1 when anything is behind, so it can gate a periodic job.
"""

import argparse
import fnmatch
import importlib.metadata
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git(clone: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(clone), *args], capture_output=True, text=True, check=False
    )
    return out.stdout.strip() if out.returncode == 0 else ""


def upstream_ref(clone: Path) -> str:
    for ref in ("origin/main", "origin/master"):
        if git(clone, "rev-parse", "--verify", "-q", ref):
            return ref
    return "HEAD"


def check_vendored(entries, fetched: set) -> int:
    behind = 0
    for e in entries:
        clone = ROOT / e["clone"]
        ref = upstream_ref(clone)
        src = e["upstream_path"]
        log = git(clone, "log", "--oneline", f"{e['commit']}..{ref}", "--", src)
        ours = (ROOT / e["path"]).read_text()
        base = git(clone, "show", f"{e['commit']}:{src}")
        local_lines = sum(
            1
            for a, b in zip(ours.splitlines(), base.splitlines(), strict=False)
            if a != b
        )
        local_lines += abs(len(ours.splitlines()) - len(base.splitlines()))
        status = "UPSTREAM CHANGED" if log else "up to date"
        behind += bool(log)
        print(
            f"- {e['path']}  [{status}]  (copied at {e['commit']}, {local_lines} local line diffs)"
        )
        if log:
            stat = git(clone, "diff", "--shortstat", e["commit"], ref, "--", src)
            print(f"    upstream {ref}: {stat}")
            for line in log.splitlines()[:10]:
                print(f"      {line}")
            print(
                f"    intended local changes to keep: {', '.join(e.get('local_changes', []))}"
            )
    return behind


def check_watch(watch, vendored_paths: set, pins: dict) -> int:
    news = 0
    for w in watch:
        clone = ROOT / w["clone"]
        if not clone.exists():
            print(f"- {w['repo']}: no local clone at {w['clone']}")
            continue
        ref = upstream_ref(clone)
        since = pins.get(w["clone"])
        if w.get("pin_package"):
            # Compare against the release we actually run (tag v<installed>).
            try:
                tag = "v" + importlib.metadata.version(w["pin_package"])
                since = git(clone, "rev-parse", "--verify", "-q", tag) and tag or since
            except importlib.metadata.PackageNotFoundError:
                pass
        since = since or git(clone, "rev-parse", "HEAD")
        files = git(clone, "ls-tree", "-r", "--name-only", ref).splitlines()
        matched = [f for f in files if any(fnmatch.fnmatch(f, g) for g in w["globs"])]
        commits = (
            git(clone, "log", "--oneline", f"{since}..{ref}", "--", *matched)
            if matched
            else ""
        )
        new_files = [
            f
            for f in matched
            if f not in vendored_paths and w["clone"] == "reference/omlx"
        ]
        print(
            f"- {w['repo']} ({w['why']}): {len(commits.splitlines()) if commits else 0} commits on watched paths since {since[:10]}"
        )
        for line in commits.splitlines()[:8]:
            print(f"      {line}")
        if new_files:
            print(
                f"    not vendored ({len(new_files)}): {', '.join(sorted(new_files)[:12])}"
            )
        news += bool(commits)
    return news


def check_packages(packages) -> int:
    behind = 0
    for name in packages:
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = "-"
        try:
            with urllib.request.urlopen(
                f"https://pypi.org/pypi/{name}/json", timeout=10
            ) as r:
                latest = json.load(r)["info"]["version"]
        except Exception:  # noqa: BLE001
            latest = "?"
        flag = "" if latest in ("?", installed) else "  <- newer on PyPI"
        behind += bool(flag)
        print(f"- {name}: installed {installed}, PyPI {latest}{flag}")
    return behind


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    manifest = json.loads((ROOT / "vendor.json").read_text())
    clones = {e["clone"] for e in manifest["vendored"]} | {
        w["clone"] for w in manifest["watch"]
    }
    fetched = set()
    if not a.no_fetch:
        for c in sorted(clones):
            if (ROOT / c).exists():
                subprocess.run(
                    ["git", "-C", str(ROOT / c), "fetch", "-q", "origin"], check=False
                )
                fetched.add(c)
    print("## Vendored files")
    behind = check_vendored(manifest["vendored"], fetched)
    print("\n## Watched upstream paths")
    pins = {e["clone"]: e["commit"] for e in manifest["vendored"]}
    vendored_paths = {e["upstream_path"] for e in manifest["vendored"]}
    check_watch(manifest["watch"], vendored_paths, pins)
    print("\n## Packages")
    pkgs = check_packages(manifest["packages"])
    sys.exit(1 if (behind or pkgs) else 0)


if __name__ == "__main__":
    main()
