"""Diff two census runs: new paths, headers, beta values, body fields, tool / block types.

    python census_diff.py <pinned_run_dir> <latest_run_dir> [--json out.json]

Exit 1 when the latest run contains anything the pinned one did not. The caller then has to
give every new item a row in docs/guides/AGENT_COMPAT.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HDR_VALUES = ("anthropic-beta", "openai-beta", "anthropic-version", "originator")
SKIP_HDR = ("host", "content-length", "user-agent", "x-api-key", "authorization")


def _walk(obj, prefix, out, depth=0):
    """Collect dotted body-field names (lists collapse to `[]`) and every `type` value."""
    if depth > 4:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out["fields"].add(p)
            if k == "type" and isinstance(v, str):
                out["types"].add(f"{prefix or '.'}={v}")
            _walk(v, p, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:50]:
            _walk(v, prefix + "[]", out, depth + 1)


def _is_volatile(name: str) -> bool:
    """Per-session ids and per-tool schema internals are not wire-contract fields."""
    return (
        name.startswith("metadata.")
        or ".input_schema." in name
        or ".parameters." in name
        or ".properties." in name
    )


def signature(root: Path) -> dict[str, set[str]]:
    keys = ("endpoints", "headers", "header_values", "fields", "types")
    sig: dict[str, set[str]] = {k: set() for k in keys}
    for f in sorted(root.glob("*/requests.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ep = r["path"].split("?")[0]
            sig["endpoints"].add(f"{r['method']} {ep}")
            q = r["path"].partition("?")[2]
            for part in q.split("&") if q else []:
                sig["endpoints"].add(f"{r['method']} {ep}?{part.split('=')[0]}")
            for h, v in (r.get("headers") or {}).items():
                hl = h.lower()
                if hl in SKIP_HDR:
                    continue
                sig["headers"].add(hl)
                if hl in HDR_VALUES:
                    for item in str(v).split(","):
                        sig["header_values"].add(f"{hl}={item.strip()}")
            body = r.get("body")
            if isinstance(body, dict):
                out: dict[str, set[str]] = {"fields": set(), "types": set()}
                _walk(body, "", out)
                sig["fields"].update(
                    f"{ep} {x}" for x in out["fields"] if not _is_volatile(x)
                )
                sig["types"].update(f"{ep} {x}" for x in out["types"])
    return sig


def diff(old: dict, new: dict) -> dict[str, list[str]]:
    return {
        k: sorted(new[k] - old.get(k, set())) for k in new if new[k] - old.get(k, set())
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    so, sn = signature(Path(a.old)), signature(Path(a.new))
    d, gone = diff(so, sn), diff(sn, so)
    for k, v in d.items():
        print(f"== new in latest: {k} ({len(v)})")
        for x in v:
            print("  +", x)
    for k, v in gone.items():
        print(f"== only in pinned: {k} ({len(v)})")
        for x in v[:20]:
            print("  -", x)
    if a.json:
        Path(a.json).write_text(json.dumps({"new": d, "gone": gone}, indent=1))
    return 1 if d else 0


if __name__ == "__main__":
    sys.exit(main())
