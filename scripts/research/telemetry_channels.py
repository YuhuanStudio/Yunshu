"""No-model native channel diagnostic; run through gpuq, never imports MLX."""

from __future__ import annotations

import argparse
import json
import plistlib
import struct
import subprocess
import time
from pathlib import Path


def pmgr_tables(nodes):
    result = []
    for node in nodes if isinstance(nodes, list) else [nodes]:
        blob = node.get("voltage-states9")
        if isinstance(blob, bytes) and len(blob) % 8 == 0:
            result.append(
                {
                    "name": node.get("IORegistryEntryName"),
                    "pairs": [
                        struct.unpack("<II", blob[i : i + 8])
                        for i in range(0, len(blob), 8)
                    ],
                }
            )
        result.extend(pmgr_tables(node.get("IORegistryEntryChildren", [])))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        a.out.write_text(json.dumps({"complete": "dry-run"}) + "\n")
        return
    from yunshu_engine.telemetry import apple

    energy = apple.EnergySampler()
    try:
        time.sleep(2)
        reading = energy.read()
        raw = subprocess.run(
            ["ioreg", "-a", "-r", "-d2", "-c", "AppleARMIODevice"],
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout
        record = {
            "complete": True,
            "watts": reading.watts,
            "reasons": reading.reasons,
            "gpu_states": reading.gpu_states,
            "current_table_mhz": apple.gpu_frequency_table_mhz(),
            "pmgr_tables": pmgr_tables(plistlib.loads(raw)),
        }
        a.out.write_text(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    finally:
        energy.close()


if __name__ == "__main__":
    main()
