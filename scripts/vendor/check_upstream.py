"""Report what changed upstream for everything Yunshu copies or builds on.

Reads ``vendor.json`` (kinds: vendored, derived, inspired, patches; see
docs/guides/UPSTREAM_TRACKING.md) and, using the local clones under ``reference/``:

0. For each ``patches`` entry (an upstream symbol we monkeypatch): hash the
   symbol's source in the installed package and compare with the recorded
   ``source_sha256``. A change is reported loudly and exits 2: our patch was
   written against the recorded code and may now be wrong.
0b. For ``derived`` / ``inspired`` entries: upstream commits touching the
   upstream paths since the recorded base commit.

1. For each vendored file: did upstream change it since the commit we copied
   (diffstat + commit subjects), and how does our copy differ from the version
   we copied (so intended local changes are visible before a sync).
2. For each watched repo: commits touching the watched paths since our pinned
   point (the vendored commit, or the last fetch), and new files matching the
   globs that we have not vendored.
3. Installed vs latest PyPI version for the packages we depend on.

    python scripts/vendor/check_upstream.py            # fetch + report
    python scripts/vendor/check_upstream.py --no-fetch  # offline, current clones
    python scripts/vendor/check_upstream.py --update-hashes  # re-record patch hashes after review

Exit code 1 when anything is behind, so it can gate a periodic job.
"""

import argparse
import ast
import fnmatch
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def clone_dir(rel: str) -> Path:
    """``reference/<x>`` in this checkout, else in the main checkout (worktrees)."""
    here = ROOT / rel
    if here.exists():
        return here
    common = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if common:
        alt = (ROOT / common).resolve().parent / rel
        if alt.exists():
            return alt
    return here


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
        clone = clone_dir(e["clone"])
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


def check_history(entries) -> int:
    """derived / inspired: upstream commits on the upstream paths since our base."""
    behind = 0
    for e in entries:
        clone = clone_dir(e["clone"])
        if not clone.exists():
            print(f"- {e['path']}  [no clone at {e['clone']}]")
            continue
        ref = upstream_ref(clone)
        paths = e["upstream_paths"]
        log = git(clone, "log", "--oneline", f"{e['commit']}..{ref}", "--", *paths)
        behind += bool(log)
        n = len(log.splitlines()) if log else 0
        flag = f"{n} upstream commits" if n else "up to date"
        print(
            f"- {e['path']}  <- {e['repo'].split('github.com/')[-1]} {', '.join(paths)}"
            f"  [{flag}]  (base {e['commit']}, {e['license']})"
        )
        for line in log.splitlines()[:6]:
            print(f"      {line}")
    return behind


def _find_symbol(source: str, symbol: str):
    """The ast node for ``name`` or ``Class.name`` (functions, methods, classes)."""
    body = ast.parse(source).body
    node = None
    for part in symbol.split("."):
        node = next(
            (
                n
                for n in body
                if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                and n.name == part
            ),
            None,
        )
        if node is None:
            return None
        body = getattr(node, "body", [])
    return node


def symbol_fingerprint(module: str, symbol: str):
    """(sha256 of the normalized source, signature) of an installed symbol, or None."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    node = _find_symbol(Path(spec.origin).read_text(), symbol)
    if node is None:
        return None
    sig = ast.unparse(node.args) if hasattr(node, "args") else ""
    digest = hashlib.sha256(ast.unparse(node).encode()).hexdigest()
    return digest, "(" + sig + ")"


def norm_sig(sig: str) -> str:
    return "".join(sig.split())


def check_patches(entries, update: bool) -> int:
    """Patched upstream symbols: did the code we patch change in the installed package?"""
    broken = 0
    for e in entries:
        got = symbol_fingerprint(e["module"], e["symbol"])
        label = f"{e['module']}:{e['symbol']}"
        try:
            version = importlib.metadata.version(e["package"])
        except importlib.metadata.PackageNotFoundError:
            version = "-"
        if got is None:
            print(
                f"- {label}  [MISSING in installed {e['package']} {version}]  <- {e['path']}"
            )
            broken += 1
            continue
        digest, sig = got
        if update:
            e["source_sha256"] = digest
            e["installed_version"] = version
            print(f"- {label}  recorded {digest[:12]} ({e['package']} {version})")
            continue
        if not e.get("source_sha256"):
            print(f"- {label}  [NO HASH RECORDED] run --update-hashes")
            broken += 1
        elif digest != e["source_sha256"]:
            broken += 1
            print(
                f"!! {label}  [UPSTREAM SOURCE CHANGED]  installed {e['package']} {version}"
            )
            print(f"   patched by {e['path']}: {e['why']}")
            print(f"   recorded {e['source_sha256'][:12]}, now {digest[:12]}")
            if norm_sig(sig) != norm_sig(e["signature"]):
                print(f"   SIGNATURE CHANGED: recorded {e['signature']}  now {sig}")
            print("   review our patch against the new source, then --update-hashes")
        else:
            print(f"- {label}  [same source]  ({e['package']} {version}, {e['path']})")
    return broken


def check_watch(watch, vendored_paths: set, pins: dict) -> int:
    news = 0
    for w in watch:
        clone = clone_dir(w["clone"])
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
    ap.add_argument(
        "--update-hashes",
        action="store_true",
        help="record the installed sources of every patched symbol (after review)",
    )
    a = ap.parse_args()
    manifest = json.loads((ROOT / "vendor.json").read_text())
    tracked = [
        e
        for k in ("vendored", "derived", "inspired", "patches")
        for e in manifest.get(k, [])
    ]
    clones = {e["clone"] for e in tracked} | {w["clone"] for w in manifest["watch"]}
    if a.update_hashes:
        check_patches(manifest["patches"], update=True)
        (ROOT / "vendor.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return
    fetched = set()
    if not a.no_fetch:
        for c in sorted(clones):
            if clone_dir(c).exists():
                subprocess.run(
                    ["git", "-C", str(clone_dir(c)), "fetch", "-q", "origin"],
                    check=False,
                )
                fetched.add(c)
    print("## Patched upstream symbols (a change here can break us silently)")
    broken = check_patches(manifest["patches"], update=False)
    if broken:
        print(f"\n!!!!! {broken} PATCHED UPSTREAM SYMBOL(S) CHANGED OR MISSING !!!!!\n")
    print("\n## Vendored files")
    behind = check_vendored(manifest["vendored"], fetched)
    print("\n## Derived from upstream (rewritten; sync by hand)")
    behind += check_history(manifest["derived"])
    print("\n## Inspired by upstream (same idea, own code)")
    behind += check_history(manifest["inspired"])
    print("\n## Watched upstream paths")
    pins = {e["clone"]: e["commit"] for e in manifest["vendored"]}
    vendored_paths = {e["upstream_path"] for e in manifest["vendored"]}
    check_watch(manifest["watch"], vendored_paths, pins)
    print("\n## Packages")
    pkgs = check_packages(manifest["packages"])
    sys.exit(2 if broken else 1 if (behind or pkgs) else 0)


if __name__ == "__main__":
    main()
