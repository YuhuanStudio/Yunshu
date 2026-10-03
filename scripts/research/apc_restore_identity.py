"""APC restore identity: a prefix restored from APC continues exactly like a cold prefill, same build.

  apc_restore_identity.py --ctx 8192 [--env K=V ...] --out F.jsonl

Server A: Q = P + tail, cold.                 -> tokens + logprobs (reference)
Server B: P first (stores its checkpoint), then Q (prefix restored from P's checkpoint, the tail prefilled as a
          shorter chunk) and Q again (full restore).
Identical = same tokens and bit-equal per-token logprobs. Any request without [DONE] / finish / usage is an error.
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tfbench as t  # noqa: E402


def assert_identity(out):
    for mode in ("partial", "full"):
        result = out[mode]
        if not result["cached"] or not (
            result["tokens_equal"] and result["logprobs_equal"]
        ):
            raise RuntimeError(f"APC {mode} restore is not bit-equal: {result}")


def ask(url, model, text, n=64):
    body = dict(
        model=model,
        messages=[{"role": "user", "content": text}] if isinstance(text, str) else text,
        max_tokens=n,
        temperature=0,
        logprobs=True,
        stream=True,
        stream_options={"include_usage": True},
        chat_template_kwargs={"enable_thinking": False},
    )
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer k"},
    )
    toks, lps, done, usage, finish = [], [], False, None, None
    t0 = time.perf_counter()
    first = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            p = line[5:].strip()
            if p == b"[DONE]":
                done = True
                break
            d = json.loads(p)
            if d.get("error"):
                raise RuntimeError(d["error"])
            usage = d.get("usage") or usage
            for ch in d.get("choices") or []:
                for c in (ch.get("logprobs") or {}).get("content") or []:
                    toks.append(c["token"])
                    lps.append(c["logprob"])
                    first = first or time.perf_counter() - t0
                finish = ch.get("finish_reason") or finish
    if not (done and finish and usage and toks):
        raise RuntimeError("incomplete stream")
    return dict(
        toks=toks,
        lps=lps,
        ttft=round(first, 3),
        cached=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        pt=usage["prompt_tokens"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--kind", default="prose")
    ap.add_argument("--turn2", action="store_true")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env = dict(kv.split("=", 1) for kv in a.env)
    P = t.load_prompt(f"{a.kind}-{a.ctx}")
    Q = (
        P
        + "\n\nAdditionally, answer in exactly one short paragraph and begin with the word Overall."
    )
    if a.turn2:
        Q = [
            {"role": "user", "content": P},
            {
                "role": "assistant",
                "content": (
                    "The reference describes events, names, places, and their explanations. "
                    * 24
                ),
            },
            {
                "role": "user",
                "content": "Continue with the next part, at the same length.",
            },
        ]
    res = {}
    s = t.Srv("yunshu", env, f"apcid-A-{a.ctx}")
    try:
        t.send(s.url, t.req(s.model, "Say hi.", 8))
        res["cold"] = ask(s.url, s.model, Q)
    finally:
        s.kill()
    s = t.Srv("yunshu", env, f"apcid-B-{a.ctx}")
    try:
        t.send(s.url, t.req(s.model, "Say hi.", 8))
        res["prime"] = ask(s.url, s.model, P, 8)
        res["partial"] = ask(s.url, s.model, Q)
        res["full"] = ask(s.url, s.model, Q)
    finally:
        s.kill()
    ref = res["cold"]
    out = dict(ctx=a.ctx, kind=a.kind, turn2=a.turn2, env=env)
    for k in ("partial", "full"):
        r = res[k]
        out[k] = dict(
            cached=r["cached"],
            pt=r["pt"],
            ttft=r["ttft"],
            tokens_equal=r["toks"] == ref["toks"],
            logprobs_equal=r["lps"] == ref["lps"],
            max_abs_dlp=max(
                abs(x - y) for x, y in zip(r["lps"], ref["lps"], strict=True)
            )
            if len(r["lps"]) == len(ref["lps"])
            else None,
        )
    out["cold_ttft"] = ref["ttft"]
    print(json.dumps(out))
    with open(a.out, "a") as f:
        f.write(json.dumps(out) + "\n")
        assert_identity(out)
        f.write(json.dumps(dict(phase="complete", success=True)) + "\n")


if __name__ == "__main__":
    main()
