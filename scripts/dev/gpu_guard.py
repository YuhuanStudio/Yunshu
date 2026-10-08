#!/usr/bin/env python3
"""Claude Code PreToolUse hook: GPU work on this Mac must go through gpuq.

gpuq exists so a timing job never shares the GPU with anything else. A direct
`python -c "import mlx..."`, a server start or a benchmark run from a shell
bypasses it and silently contaminates whatever gpuq is measuring. This hook
refuses such Bash commands unless they go through gpuq. Unit tests (pytest),
compile / lint / --help invocations and the M3 helper (m3run) are allowed.

Reads the hook JSON on stdin; exit 2 with a reason on stderr blocks the call.
"""

from __future__ import annotations

import json
import re
import sys

GPU = re.compile(
    r"""
    \bimport\s+mlx | \bfrom\s+mlx[\w.]*\s+import | \bmlx\.core\b
    | \byunshu_cli\s+serve\b | \byunshu\s+serve\b | \brapid_mlx\b
    | \bmlx_(lm|vlm)\.(generate|server)\b | \bmlx_(lm|vlm)\s+(generate|server)\b
    | scripts/(research|realmodel|bench)\S*\.py
    | \b(memory_ab|soak_\w+|probe_\w+|bench_\w+)\.py\b
    """,
    re.X,
)
# Only something that executes Python or a server can touch the GPU; a grep for
# "import mlx" is not GPU work.
EXEC = re.compile(
    r"(^|[\s;&|(/])python[\d.]*(\s|$)|\buv\s+run\b|\.venv/bin/|\byunshu\s|\brapid_mlx\b|\bmlx_(lm|vlm)\b"
)
ALLOWED = re.compile(
    r"\bgpuq\b|\bpytest\b|\bpy_compile\b|\bruff\b|\bmypy\b|\bm3run\b|--help\b"
)


SPLIT = re.compile(r"&&|\|\||[;|\n]")


def _segment_blocked(segment: str) -> bool:
    return bool(
        GPU.search(segment) and EXEC.search(segment) and not ALLOWED.search(segment)
    )


# Pattern kills hit processes this session did not start (the user's apps, other workers'
# servers, gpuq jobs); agents kept using them despite the written rule (2026-10-03, twice
# on 2026-10-08). Kill by PID only.
PATTERN_KILL = re.compile(
    r"^\s*(?:sudo\s+|nice(?:\s+-n\s*-?\d+)?\s+|exec\s+)*(pkill|killall)\b"
)


def verdict(command: str) -> str | None:
    """Reason to block ``command``, or None when it may run. Each shell segment is
    judged on its own (a `git diff x.py; python3 tool.py` is not GPU work); with a
    heredoc the whole command is one segment, since its body is the program."""
    if any(PATTERN_KILL.search(seg) for seg in SPLIT.split(command)):
        return (
            "pkill / killall are not allowed: they can hit processes this session did "
            "not start. Find your own process's PID and use `kill <pid>`."
        )
    segments = [command] if "<<" in command else SPLIT.split(command)
    if not any(_segment_blocked(s) for s in segments):
        return None
    return (
        "GPU work must go through gpuq (scripts/dev/gpuq submit/run ...): a direct "
        "MLX / server / benchmark run shares the GPU with whatever gpuq is timing. "
        "Wrap the command in `gpuq run --label ... -- <cmd>`."
    )


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:  # noqa: BLE001
        return 0
    if data.get("tool_name") != "Bash":
        return 0
    reason = verdict((data.get("tool_input") or {}).get("command") or "")
    if reason:
        print(reason, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
