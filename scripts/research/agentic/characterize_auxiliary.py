"""Inventory captured request fingerprints, including unknown no-tools traffic.

CPU-only; scans saved *req.json bodies. Counts are physical captures, not unique
sessions (runs may replay / copy a body). No client-version guesses from prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def inventory(root: Path) -> list[dict]:
    groups: dict[str, dict] = {}
    counts: dict[str, int] = defaultdict(int)
    for path in sorted(root.rglob("*req.json")):
        if "error-req" in path.name:
            continue
        body = json.loads(path.read_text())
        client = next(
            (
                c
                for c in ("opencode", "claude", "codex")
                if c in str(path.relative_to(root))
            ),
            "unknown",
        )
        messages = body.get("messages") or []
        response_input = body.get("input")
        inputs = response_input if isinstance(response_input, list) else []
        parts = []
        for field in ("system", "instructions"):
            if body.get(field) is not None:
                parts.append((field, body[field]))
        for field, items in (("messages", messages), ("input", inputs)):
            for index, message in enumerate(items):
                if isinstance(message, dict) and message.get("role") in (
                    "system",
                    "developer",
                ):
                    parts.append(
                        (
                            f"{field}[{index}].{message['role']}",
                            message.get("content", ""),
                        )
                    )
        encoded_parts = [
            (
                source,
                value if isinstance(value, str) else json.dumps(value, sort_keys=True),
            )
            for source, value in parts
        ]
        encoded = "\n".join(value for _, value in encoded_parts)
        record = dict(
            client=client,
            model=body.get("model"),
            tools=len(body.get("tools") or []),
            max_tokens=body.get("max_tokens"),
            max_output_tokens=body.get("max_output_tokens"),
            system_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
            system_chars=len(encoded),
            system_parts=[
                dict(
                    source=source,
                    sha256=hashlib.sha256(value.encode()).hexdigest(),
                    chars=len(value),
                )
                for source, value in encoded_parts
            ],
            roles=[
                m.get("role")
                for m in [*messages, *inputs]
                if isinstance(m, dict) and m.get("role")
            ],
        )
        # Role growth is part of an agent conversation, not a new system fingerprint.
        key = json.dumps(
            {k: v for k, v in record.items() if k != "roles"}, sort_keys=True
        )
        counts[key] += 1
        if key not in groups:
            record["example"] = str(path.relative_to(root))
            groups[key] = record
    return [dict(**groups[k], captures=n) for k, n in counts.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = inventory(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    for record in records:
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
