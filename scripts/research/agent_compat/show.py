"""Print a compact digest of a census session: every request's method/path/status, request body
shape (top-level keys, tool list, thinking / reasoning fields), and selected headers.

    python show.py <session> [--full N]    # N: dump request N's full JSON
"""

import glob
import json
import sys

root = sorted(
    glob.glob(
        "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/docs/research/runs/*-agent-census"
    )
)[-1]
name = sys.argv[1]
full = int(sys.argv[sys.argv.index("--full") + 1]) if "--full" in sys.argv else None
with open(f"{root}/{name}/requests.jsonl") as f:
    rows = [json.loads(x) for x in f]
for i, r in enumerate(rows):
    b = r["body"] if isinstance(r["body"], dict) else {}
    print(f"[{i}] {r['method']} {r['path']} -> {r['status']}")
    h = {k.lower(): v for k, v in r["headers"].items()}
    for k in (
        "anthropic-beta",
        "openai-beta",
        "originator",
        "version",
        "session_id",
        "x-client-request-id",
        "x-codex-turn-state",
        "x-codex-window-id",
        "user-agent",
    ):
        if k in h:
            print(f"     {k}: {h[k][:200]}")
    if b:
        print(
            "     keys:",
            {
                k: (
                    v
                    if not isinstance(v, (list, dict, str))
                    else type(v).__name__ + str(len(v))
                )
                for k, v in b.items()
            },
        )
        for k in (
            "thinking",
            "reasoning",
            "output_config",
            "context_management",
            "include",
            "text",
            "tool_choice",
            "service_tier",
            "prompt_cache_key",
            "max_tokens",
            "max_output_tokens",
            "metadata",
            "mcp_servers",
            "reasoning_effort",
            "temperature",
        ):
            if k in b:
                print(f"     {k}: {json.dumps(b[k])[:200]}")
        tools = b.get("tools") or []
        if tools:
            print(
                "     tools:",
                [
                    (t.get("type") or "?")
                    + ":"
                    + str(t.get("name") or (t.get("function") or {}).get("name"))
                    for t in tools
                ],
            )
    if full is not None and full == i:
        print(json.dumps(r, indent=1)[:12000])
