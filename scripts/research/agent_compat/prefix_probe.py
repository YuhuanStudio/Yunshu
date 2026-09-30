"""Which request fields change between consecutive requests of one census session (prefix-cache killers)?

python prefix_probe.py cc_bash_edit
"""

import glob
import json
import sys

root = sorted(
    glob.glob(
        "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/docs/research/runs/*-agent-census"
    )
)[-1]
with open(f"{root}/{sys.argv[1]}/requests.jsonl") as f:
    rows = [json.loads(x) for x in f]
prev = None
for i, r in enumerate(rows):
    b = r["body"] or {}
    sysb = b.get("system")
    first = sysb[0]["text"] if isinstance(sysb, list) and sysb else str(sysb)[:100]
    print(i, "system[0]:", first[:90].replace("\n", " "))
    if isinstance(sysb, list):
        print(
            "   n_system_blocks",
            len(sysb),
            "cache_control on",
            [j for j, x in enumerate(sysb) if x.get("cache_control")],
        )
    msgs = b.get("messages") or []
    print("   roles:", [m.get("role") for m in msgs])
    if prev is not None:
        pm = prev.get("messages") or []
        same = 0
        for a, c in zip(pm, msgs, strict=False):
            if a == c:
                same += 1
            else:
                break
        print(f"   messages identical with previous request: first {same} of {len(pm)}")
        print(
            "   tools identical:",
            prev.get("tools") == b.get("tools"),
            " system identical:",
            prev.get("system") == b.get("system"),
        )
    prev = b
