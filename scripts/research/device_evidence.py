"""Fail closed on cross-device comparisons and M3 performance verdicts.

Unlabelled historical receipts predate M3 offload and are M5. New queue outputs
are stamped at collection, so this fallback never promotes new M3 evidence.
"""

from __future__ import annotations

import json
from pathlib import Path


def devices(rows):
    found = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "device" in row:
            value = str(row["device"]).lower()
            if value not in {"m3", "m5"}:
                raise ValueError(f"unknown evidence device {value!r}")
            found.add(value)
        for value in row.values():
            if isinstance(value, dict):
                found.update(devices([value]))
            elif isinstance(value, list):
                found.update(devices(value))
    return found


def require_same_device(*arms, performance=False):
    per_arm = [devices(rows) or {"m5"} for rows in arms]
    all_devices = set().union(*per_arm)
    if len(all_devices) != 1:
        raise ValueError(
            "mixed-device arms refused: both arms must run on the same device"
        )
    device = next(iter(all_devices))
    if performance and device != "m5":
        raise ValueError(
            "M3 portability evidence (not M5): performance verdict refused"
        )
    return device


def stamp_output(path, device, host=None):
    """Stamp every JSON/JSONL row, including failure artifacts, before publication."""
    p = Path(path)
    if p.is_dir():
        for child in p.rglob("*"):
            if child.is_file() and child.suffix in {".json", ".jsonl"}:
                stamp_output(child, device, host)
        return
    if not p.exists() or p.suffix not in {".json", ".jsonl"}:
        return

    def stamp(value):
        if isinstance(value, dict):
            # Preserve source evidence when a local analysis consumes remote rows.
            # Collection records execution separately; it must never relabel an M3
            # measurement as M5 merely because its report ran on the M5.
            value.setdefault("device", device)
            value["execution_device"] = device
            value.setdefault("remote_host", host)
            for child in list(value.values()):
                if isinstance(child, (dict, list)):
                    stamp(child)
        elif isinstance(value, list):
            for child in value:
                stamp(child)

    text = p.read_text()
    if p.suffix == ".jsonl":
        lines = []
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                lines.append(line)
                continue
            stamp(row)
            lines.append(json.dumps(row))
        p.write_text("\n".join(lines) + "\n")
    else:
        row = json.loads(text)
        stamp(row)
        p.write_text(json.dumps(row, indent=2) + "\n")
