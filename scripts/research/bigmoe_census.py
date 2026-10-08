"""CPU-only safetensors census; never imports MLX or reads weight payloads."""

import argparse
import json
import struct
from pathlib import Path

GIB = 2**30


def census(root: Path, system_gib: float = 10, workspace_gib: float = 8) -> dict:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    expected = index["weight_map"]
    found = {}
    files = {}
    for name in sorted(set(expected.values())):
        path = root / name
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f"truncated header: {name}")
            size = struct.unpack("<Q", raw)[0]
            if size > min(path.stat().st_size - 8, 64 * 1024 * 1024):
                raise ValueError(f"invalid header size: {name}")
            header = json.loads(stream.read(size))
        payload = path.stat().st_size - 8 - size
        spans = []
        for key, tensor in header.items():
            if key == "__metadata__":
                continue
            lo, hi = tensor["data_offsets"]
            if not 0 <= lo <= hi <= payload:
                raise ValueError(f"invalid payload span: {key}")
            spans.append((lo, hi))
            if key in found:
                raise ValueError(f"duplicate tensor: {key}")
            found[key] = (name, hi - lo)
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:], strict=False)):
            raise ValueError(f"overlapping payloads: {name}")
        files[name] = path.stat().st_size
    if set(found) != set(expected) or any(
        found[k][0] != v for k, v in expected.items()
    ):
        raise ValueError("index/header mismatch")
    mtp = sum(n for k, (_, n) in found.items() if k.startswith("mtp."))
    base = sum(n for _, n in found.values()) - mtp
    cfg = json.loads((root / "config.json").read_text())
    total_gib, reserve_gib = 128, 30
    headroom = total_gib - reserve_gib - system_gib - workspace_gib - base / GIB
    return dict(
        schema=1,
        model_type=cfg["model_type"],
        tensor_count=len(found),
        file_bytes=sum(files.values()),
        base_bytes=base,
        mtp_bytes=mtp,
        budget=dict(
            machine_gib=total_gib,
            reserve_gib=reserve_gib,
            system_other_gib=system_gib,
            workspace_allocator_gib=workspace_gib,
            base_gib=base / GIB,
            cache_headroom_gib=headroom,
        ),
        admission="estimate_only" if headroom >= 0 else "reject",
        warning="Header census is not a measured resident or peak-memory verdict.",
        complete=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--system-gib", type=float, default=10)
    parser.add_argument("--workspace-gib", type=float, default=8)
    args = parser.parse_args()
    if args.system_gib < 0 or args.workspace_gib < 0:
        parser.error("memory allowances must be nonnegative")
    print(json.dumps(census(args.model, args.system_gib, args.workspace_gib), indent=2))


if __name__ == "__main__":
    main()
