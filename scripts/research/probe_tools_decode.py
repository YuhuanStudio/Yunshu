"""Decode speed of a long request with and without declared tools (27B, DFlash default).

Agent clients (Claude Code, Codex, opencode) declare tools on every request. A declared
tool set compiles a structural-tag tool guide, and a guided job turns off the dense lane,
the batch-invariant kernels, the DFlash fast tree and copy rounds. This measures what that
costs on the same 32K exact snapshot prompt: one server with the tool grammar on (default)
and one with YUNSHU_TOOL_GRAMMAR=0, each serving the prompt without and with tools.

Run only through gpuq:
    TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_EXACT_PROMPTS=1 \
    python scripts/research/probe_tools_decode.py --src <tree>/python --out <jsonl>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tfbench  # noqa: E402


def tool(name, desc, props, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


S = {"type": "string"}
TOOLS = [
    tool(
        "Bash",
        "Run a shell command.",
        {"command": S, "timeout": {"type": "integer"}},
        ["command"],
    ),
    tool(
        "Read",
        "Read a file.",
        {"file_path": S, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
        ["file_path"],
    ),
    tool(
        "Edit",
        "Replace text in a file.",
        {"file_path": S, "old_string": S, "new_string": S},
        ["file_path", "old_string", "new_string"],
    ),
    tool(
        "Write",
        "Write a file.",
        {"file_path": S, "content": S},
        ["file_path", "content"],
    ),
    tool("Grep", "Search file contents.", {"pattern": S, "path": S}, ["pattern"]),
    tool("Glob", "Find files by pattern.", {"pattern": S}, ["pattern"]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="python/ of the tree to serve")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default="code")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--tokens", type=int, default=512)
    a = ap.parse_args()
    out = Path(a.out)
    tfbench.YUNSHU_SRC = a.src
    tfbench.OUT = out.parent / (out.stem + "-servers")
    tfbench.OUT.mkdir(parents=True, exist_ok=True)
    text = tfbench.decode_prompt(a.kind, a.ctx, True)
    rows = []
    for grammar in ("1", "0"):
        env = {
            "YUNSHU_VLM_DRAFT": tfbench.D,
            "YUNSHU_AUTH_DISABLED": "1",
            "YUNSHU_TOOL_GRAMMAR": grammar,
        }
        srv = tfbench.Srv("yunshu", env, f"grammar{grammar}")
        try:
            for arm, extra in (
                ("plain", None),
                ("tools", {"tools": TOOLS, "tool_choice": "auto"}),
            ):
                r = tfbench.send(
                    srv.url, tfbench.req("m", text, a.tokens, extra=extra), timeout=900
                )
                row = {
                    "grammar": grammar,
                    "arm": arm,
                    "pt": r["pt"],
                    "ct": r["ct"],
                    "ttft_s": r["ttft_s"],
                    "dec_tps": r["dec_tps"],
                    "finish": r["finish"],
                    "speculative": (r["xy"] or {}).get("speculative"),
                    "head": r["head"],
                }
                print(json.dumps(row), flush=True)
                rows.append(row)
        finally:
            srv.kill()
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        ok = len(rows) == 4 and all(r["ct"] and r["ct"] > 32 for r in rows)
        f.write(json.dumps({"complete": ok}) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
