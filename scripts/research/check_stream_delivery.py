"""Generated == delivered: compare a soak's delivered text with the engine's capture.

    YUNSHU_DEBUG_STREAM_CAPTURE=capture.jsonl yunshu serve ...
    python scripts/research/soak_realistic.py ... --output soak.jsonl
    python scripts/research/check_stream_delivery.py soak.jsonl capture.jsonl

The server capture holds, per generation, the token ids and the text pieces the engine
handed to the gateway. Every non-aborted soak request is matched to a capture row by
its prompt and completion token counts, and the text the client received
(reasoning + content) must equal the capture's joined text (tool-call requests are
matched on counts only: their markup is parsed out of the text). Exits 1 on a
mismatch or an unmatched request.
"""

import json
import sys
from pathlib import Path


def norm(text):
    return (text or "").strip()


def main():
    soak_path, cap_path = sys.argv[1:3]
    reqs = [
        r
        for r in (
            json.loads(x) for x in Path(soak_path).read_text().splitlines() if x.strip()
        )
        if r.get("kind") == "req"
    ]
    caps = [json.loads(x) for x in Path(cap_path).read_text().splitlines() if x.strip()]
    used = [False] * len(caps)
    checked = mismatched = unmatched = skipped = 0
    bad = []
    for r in reqs:
        if r.get("completion_tokens") is None or r.get("prompt_tokens") is None:
            skipped += 1  # aborted by the client or errored: no usage was delivered
            continue
        found = None
        for i, c in enumerate(caps):
            if (
                not used[i]
                and c["prompt_tokens"] == r["prompt_tokens"]
                and 0
                <= r["completion_tokens"] - len(c["token_ids"])
                <= 4  # think tags are not events
            ):
                found = i
                break
        if found is None:
            unmatched += 1
            bad.append((r["n"], r["type"], "no capture row", None))
            continue
        used[found] = True
        c = caps[found]
        checked += 1
        if r.get("tool_calls"):
            continue
        delivered = norm((r.get("reasoning") or "") + (r.get("content") or ""))
        generated = norm(c["text"])
        if delivered != generated:
            mismatched += 1
            bad.append((r["n"], r["type"], delivered[:60], generated[:60]))
    summary = {
        "requests": len(reqs),
        "checked": checked,
        "mismatched": mismatched,
        "unmatched": unmatched,
        "skipped_no_usage": skipped,
        "first_problems": bad[:10],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    sys.exit(1 if mismatched or unmatched else 0)


if __name__ == "__main__":
    main()
