#!/usr/bin/env python3
"""Discover published MLX checkpoints without importing MLX or downloading weights."""

from __future__ import annotations

import argparse
import json
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def family_name(value: str) -> str:
    return value.strip().rstrip("/").rsplit("/", 1)[-1]


def is_mlx(row: dict) -> bool:
    name = str(row.get("id", "")).lower()
    return (
        name.startswith("mlx-community/")
        or any(x in name for x in ("mlx", "mxfp"))
        or "mlx" in row.get("tags", [])
    )


def discover(value: str, fetch=None) -> list[dict]:
    """Follow HF pagination; do not treat an API failure as an empty search."""
    fetch = fetch or fetch_page
    url = "https://huggingface.co/api/models?" + urlencode(
        {"search": family_name(value), "full": "true", "limit": 100}
    )
    found = {}
    seen = set()
    while url:
        if url in seen:
            raise ValueError("HF pagination cycle")
        seen.add(url)
        rows, url = fetch(url)
        for row in rows:
            if is_mlx(row):
                found[row["id"]] = row
    return sorted(
        found.values(), key=lambda row: (-int(row.get("downloads", 0)), row["id"])
    )


def fetch_page(url: str):
    with urlopen(
        Request(url, headers={"User-Agent": "yunshu-prior-art-check/1"}), timeout=30
    ) as response:
        rows = json.load(response)
        link = response.headers.get("Link", "")
    next_url = None
    for part in link.split(","):
        if 'rel="next"' in part:
            next_url = part.split("<", 1)[1].split(">", 1)[0]
    return rows, next_url


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", help="HF model id or family name; repeat for aliases separately"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        rows = discover(args.model)
    except (OSError, ValueError) as exc:
        print(f"Prior-art search failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            license_id = (row.get("cardData") or {}).get("license") or next(
                (t[8:] for t in row.get("tags", []) if t.startswith("license:")),
                "unknown",
            )
            print(
                f"https://huggingface.co/{row['id']}\tlicense={license_id}\tdownloads={row.get('downloads', 'unreported')}\tupdated={row.get('lastModified', 'unknown')}"
            )
        print(
            f"{len(rows)} candidates; inspect runtime, head layout, license and reference parity. No results do not prove originality.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
