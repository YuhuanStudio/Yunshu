"""Tool-guided requests: spec on == spec off, and their decode speed (27B, 32K).

Run only through gpuq. One tree, three servers (DFlash, MTP, drafting off), each serving:
  A  32K prompt + declared tools, a long free-text reply (streamed: decode tok/s)
  B  32K prompt + declared tools, asked to call tools (exercises the masked call rows)
  C  32K prompt, no tools (control)
Every case's content and tool calls must be identical across the three servers.
Writes JSON lines and a final ``complete`` record; exits nonzero on any mismatch.

    TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_EXACT_PROMPTS=1 \
    python scripts/research/probe_tools_identity.py --src <tree>/python --out <jsonl>
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tfbench  # noqa: E402
from probe_tools_decode import TOOLS  # noqa: E402

CALL_ASK = (
    "\n\nDo not review yet. First call the Read tool on README.md and the Grep tool for "
    "the pattern 'def main' in the repository root. Only call the tools now."
)


def post(url, body, timeout=900):
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(dict(body, stream=False)).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer k"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    msg = d["choices"][0]["message"]
    calls = [
        (c["function"]["name"], c["function"]["arguments"])
        for c in msg.get("tool_calls") or []
    ]
    return {
        "content_sha": hashlib.sha256((msg.get("content") or "").encode()).hexdigest()[
            :16
        ],
        "calls": calls,
        "finish": d["choices"][0]["finish_reason"],
        "ct": d["usage"]["completion_tokens"],
        "total_s": round(time.perf_counter() - t0, 2),
        "spec": (d.get("x_yunshu") or {}).get("speculative"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    a = ap.parse_args()
    out = Path(a.out)
    tfbench.YUNSHU_SRC = a.src
    tfbench.OUT = out.parent / (out.stem + "-servers")
    tfbench.OUT.mkdir(parents=True, exist_ok=True)
    text = tfbench.decode_prompt("code", 32768, True)
    tools = {"tools": TOOLS, "tool_choice": "auto"}
    cases = {
        "A": tfbench.req("m", text, a.tokens, extra=tools),
        "B": tfbench.req("m", text + CALL_ASK, a.tokens, extra=tools),
        "C": tfbench.req("m", text, a.tokens),
    }
    rows, results = [], {}
    for mode, draft in (("dflash", tfbench.D), ("mtp", "mtp"), ("off", "off")):
        env = {
            "YUNSHU_VLM_DRAFT": draft,
            "YUNSHU_AUTH_DISABLED": "1",
            "YUNSHU_TOOL_GRAMMAR": "1",
        }
        srv = tfbench.Srv("yunshu", env, f"ident-{mode}")
        try:
            for case, body in cases.items():
                r = post(srv.url, body)
                if case in ("A", "C"):
                    s = tfbench.send(srv.url, body, timeout=900)
                    r["dec_tps"] = s["dec_tps"]
                    r["stream_sha"] = s["sha"]
                row = {"mode": mode, "case": case, **r}
                print(json.dumps(row), flush=True)
                rows.append(row)
                results[(mode, case)] = (r["content_sha"], r["calls"])
        finally:
            srv.kill()
    ok = True
    for case in cases:
        ref = results.get(("off", case))
        for mode in ("dflash", "mtp"):
            same = results.get((mode, case)) == ref and ref is not None
            ok &= same
            rows.append({"compare": f"{mode}_vs_off", "case": case, "identical": same})
            print(json.dumps(rows[-1]), flush=True)
    called = any(r.get("case") == "B" and r.get("calls") for r in rows)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(
            json.dumps({"complete": True, "identical": ok, "tool_called": called})
            + "\n"
        )
    sys.exit(0 if ok and called else 1)


if __name__ == "__main__":
    main()
