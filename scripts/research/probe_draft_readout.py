"""Diagnostic only: capture real reduced draft readouts, replay identical inputs across MLX versions.

All GPU commands must run through gpuq. Capture wraps tfbench's server launcher and changes no
serving output. Instrumented runs make no timing claim. Replay loads only captured arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_result(path, result):
    """Keep complete=true on one JSON line, including after gpuq device stamping."""
    Path(path).write_text(json.dumps(result) + "\n")


def read_manifest(directory):
    root = Path(directory)
    data = json.loads((root / "manifest.json").read_text())
    if data.get("complete") is not True or not data.get("frames"):
        raise ValueError("incomplete capture")
    for row in data["frames"]:
        name = row["file"]
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("invalid frame path")
        if digest(root / name) != row["sha256"]:
            raise ValueError("frame checksum mismatch")
    return data


def server_command(command, directory):
    # Pinned tfbench servers use this exact module launcher; preserve their interpreter.
    if len(command) >= 4 and list(command[1:4]) == ["-m", "yunshu_cli", "serve"]:
        return [
            command[0],
            str(Path(__file__).resolve()),
            "serve",
            str(directory),
            *command[3:],
        ]
    return command


def install(directory):
    import mlx.core as mx

    from yunshu_engine.draft_vocab import DraftVocab

    root = Path(directory)
    original = DraftVocab._matmul
    seen = set()
    (root / "runtime.json").write_text(
        json.dumps(
            {"mlx": importlib.metadata.version("mlx"), "device": mx.metal.device_info()}
        )
    )

    def matmul(self, x, rows):
        result = original(self, x, rows)
        w, s, b = rows
        # The small extension is the changed split-K branch. Bound perturbation/storage.
        if 1 <= w.shape[0] <= 32 and len(seen) < 16:
            mx.eval(x, w, s, b, result)
            # Every call receives a unique sequential frame: do not deduplicate by printed tensors.
            idx = len(seen)
            seen.add(idx)
            file = root / f"frame-{idx:02}.safetensors"
            mx.save_safetensors(
                str(file), {"x": x, "w": w, "s": s, "b": b, "y": result}
            )
            meta = {
                "file": file.name,
                "sha256": digest(file),
                "shape": list(x.shape),
                "width": w.shape[0],
                "bits": self.bits,
                "group_size": self.group_size,
            }
            (root / f"frame-{idx:02}.json").write_text(json.dumps(meta))
        return result

    DraftVocab._matmul = matmul


def capture(args):
    import runpy
    import subprocess

    root = Path(args.directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    original = subprocess.Popen

    def launch(command, *a, **kw):
        return original(server_command(command, root), *a, **kw)

    subprocess.Popen = launch
    old = sys.argv
    sys.argv = ["tfbench.py", *args.tfbench]
    try:
        runpy.run_path(str(Path(__file__).with_name("tfbench.py")), run_name="__main__")
    finally:
        subprocess.Popen = original
        sys.argv = old
    frames = [json.loads(p.read_text()) for p in sorted(root.glob("frame-*.json"))]
    if not frames:
        raise RuntimeError("no reduced extension readout captured")
    data = {
        "complete": True,
        "frames": frames,
        "diagnostic_only": True,
        "source": str(Path(__file__).resolve()),
        "probe_sha256": digest(__file__),
        "runtime": json.loads((root / "runtime.json").read_text()),
    }
    write_result(root / "manifest.json", data)
    print(
        json.dumps({"complete": True, "frames": len(frames), "diagnostic_only": True})
    )


def replay(args):
    data = read_manifest(args.directory)
    import mlx.core as mx

    frames = []
    for row in data["frames"]:
        a = mx.load(str(Path(args.directory) / row["file"]))
        y = mx.quantized_matmul(
            a["x"],
            a["w"],
            a["s"],
            a["b"],
            transpose=True,
            group_size=row["group_size"],
            bits=row["bits"],
        )
        mx.eval(y)
        frames.append(
            {
                "file": row["file"],
                "input_sha256": row["sha256"],
                "shape": list(y.shape),
                "logits": y.astype(mx.float32).tolist(),
                "argmax": mx.argmax(y, axis=-1).tolist(),
                "max_abs_capture_delta": float(
                    mx.max(
                        mx.abs(y.astype(mx.float32) - a["y"].astype(mx.float32))
                    ).item()
                ),
            }
        )
    result = {
        "complete": True,
        "diagnostic_only": True,
        "manifest_sha256": digest(Path(args.directory) / "manifest.json"),
        "mlx": importlib.metadata.version("mlx"),
        "device": mx.metal.device_info(),
        "frames": frames,
    }
    write_result(args.out, result)
    print(json.dumps({"complete": True, "mlx": result["mlx"], "frames": len(frames)}))


def compare_replays(base, cand):
    import math

    if not all(x.get("complete") is True and x.get("frames") for x in (base, cand)):
        raise ValueError("incomplete replay")
    if (
        base["manifest_sha256"] != cand["manifest_sha256"]
        or base["device"] != cand["device"]
    ):
        raise ValueError("inputs or device differ")
    if len(base["frames"]) != len(cand["frames"]):
        raise ValueError("frame coverage differs")
    changes = []
    for a, b in zip(base["frames"], cand["frames"], strict=True):
        if any(a[k] != b[k] for k in ("file", "input_sha256", "shape")):
            raise ValueError("frame inputs differ")
        if len(a["logits"]) != len(b["logits"]) or any(
            len(u) != len(v) for u, v in zip(a["logits"], b["logits"], strict=True)
        ):
            raise ValueError("logit coverage differs")
        deltas = [
            abs(u - v)
            for ar, br in zip(a["logits"], b["logits"], strict=True)
            for u, v in zip(ar, br, strict=True)
        ]
        if not deltas or not all(math.isfinite(d) for d in deltas):
            raise ValueError("missing or nonfinite logits")
        changes.append(
            {
                "file": a["file"],
                "max_abs_delta": max(deltas),
                "changed_values": sum(d != 0 for d in deltas),
                "argmax_changed": a["argmax"] != b["argmax"],
            }
        )
    return {
        "complete": True,
        "diagnostic_only": True,
        "base_mlx": base["mlx"],
        "cand_mlx": cand["mlx"],
        "frames": changes,
    }


def pilot(args):
    """Tiny bounded exercise of the same hook, serializer and replay; no timing claim."""
    from types import SimpleNamespace

    import mlx.core as mx

    from yunshu_engine.draft_vocab import DraftVocab

    root = Path(args.directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    install(root)
    mx.random.seed(7)
    w, s, b = mx.quantize(
        mx.random.normal((8, 128)).astype(mx.bfloat16), group_size=64, bits=4
    )
    x = mx.random.normal((15, 128)).astype(mx.bfloat16)
    DraftVocab._matmul(SimpleNamespace(bits=4, group_size=64), x, (w, s, b))
    frames = [json.loads(p.read_text()) for p in sorted(root.glob("frame-*.json"))]
    if len(frames) != 1:
        raise RuntimeError("pilot failed to capture one frame")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "frames": frames,
                "pilot": True,
                "runtime": json.loads((root / "runtime.json").read_text()),
            }
        )
    )
    replay(args)


def main():
    if sys.argv[1:2] == ["serve"]:
        install(sys.argv[2])
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        from yunshu_cli import main as cli

        cli()
        return
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--directory", "--cache-dir", dest="directory", required=True)
    c.add_argument("tfbench", nargs=argparse.REMAINDER)
    r = sub.add_parser("replay")
    r.add_argument("--directory", required=True)
    r.add_argument("--out", required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--base", required=True)
    cmp.add_argument("--cand", required=True)
    cmp.add_argument("--out", required=True)
    pilot_parser = sub.add_parser("pilot")
    pilot_parser.add_argument("--cache-dir", dest="directory", required=True)
    pilot_parser.add_argument("--out", required=True)
    a = p.parse_args()
    if a.mode == "capture":
        if a.tfbench[:1] == ["--"]:
            a.tfbench.pop(0)
        capture(a)
    elif a.mode == "pilot":
        pilot(a)
    elif a.mode == "replay":
        replay(a)
    else:
        result = compare_replays(
            json.loads(Path(a.base).read_text()), json.loads(Path(a.cand).read_text())
        )
        Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
