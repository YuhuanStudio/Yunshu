"""Run the ordinary cold/partial/full HTTP identity gate on one research arm."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("async", "barrier", "paired32", "spans", "cow"),
        required=True,
    )
    parser.add_argument("--model")
    parser.add_argument("--out", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    import apc_restore_identity as identity
    import tfbench

    args.out.parent.mkdir(parents=True, exist_ok=True)
    launcher = args.out.with_suffix(".launcher.py")
    source = Path(__file__).resolve().parents[2] / "python"
    module = {
        "async": "async_restore",
        "cow": "cow_restore",
        "spans": "span_forward",
        "barrier": "lane_final_barrier",
        "paired32": "lane_paired32",
    }[args.variant]
    launcher.write_text(
        f"#!{sys.executable}\nimport sys\n"
        f"sys.path[:0] = {[str(source), str(Path(__file__).resolve().parent)]!r}\n"
        f"from {module} import install\n_installation = install()\n"
        "from yunshu_cli import main\nmain()\n"
    )
    launcher.chmod(0o700)
    tfbench.YUNSHU_BIN = str(launcher)
    tfbench.YUNSHU_SRC = str(source)
    if args.model:
        tfbench.M = args.model
    children = []
    original_popen = tfbench.subprocess.Popen

    def tracked_popen(*positional, **kwargs):
        child = original_popen(*positional, **kwargs)
        children.append(child)
        return child

    tfbench.subprocess.Popen = tracked_popen
    sys.argv = ["identity", "--out", str(args.out), *remaining]
    try:
        identity.main()
    finally:
        tfbench.subprocess.Popen = original_popen
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=20)
        launcher.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
