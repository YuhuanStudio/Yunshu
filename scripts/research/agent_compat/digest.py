"""Digest the message structure of one census request: roles and content block types per message.

python digest.py <session> <request index>
"""

import glob
import json
import sys

root = sorted(
    glob.glob(
        "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/docs/research/runs/*-agent-census"
    )
)[-1]
name, idx = sys.argv[1], int(sys.argv[2])
with open(f"{root}/{name}/requests.jsonl") as f:
    r = [json.loads(x) for x in f][idx]
b = r["body"]
print("system:", json.dumps(b.get("system"))[:300] if b.get("system") else None)
for i, m in enumerate(b.get("messages") or b.get("input") or []):
    c = m.get("content")
    if isinstance(c, list):
        d = []
        for x in c:
            t = x.get("type")
            extra = ""
            if t == "tool_result":
                inner = x.get("content")
                extra = "->" + (
                    json.dumps([y.get("type") for y in inner])
                    if isinstance(inner, list)
                    else str(inner)[:80]
                )
            if t == "tool_use":
                extra = ":" + x.get("name", "")
            if t in ("text", "input_text", "output_text"):
                extra = ":" + x.get("text", "")[:70].replace("\n", " ")
            d.append(str(t) + extra)
        print(i, m.get("role"), d)
    else:
        print(
            i,
            m.get("role") or m.get("type"),
            (str(c) or json.dumps(m))[:160].replace("\n", " "),
        )
