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
    for mode in ("partial", "full", *(["spec_off"] if "spec_off" in out else [])):
        result = out[mode]
        if (mode != "spec_off" and not result["cached"]) or not (
            result["tokens_equal"] and result["logprobs_equal"]
        ):
            raise RuntimeError(f"APC {mode} restore is not bit-equal: {result}")


def ask(url, model, text, n=64, sampling=None, capture=None):
    capture_start = (
        len(capture.read_text().splitlines())
        if capture is not None and capture.exists()
        else 0
    )
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
    body.update(sampling or {})
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
    if capture is not None:
        deadline = time.monotonic() + 5
        while True:
            rows = (
                [
                    json.loads(line)
                    for line in capture.read_text().splitlines()[capture_start:]
                ]
                if capture.exists()
                else []
            )
            row = next(
                (
                    r
                    for r in reversed(rows)
                    if r["prompt_tokens"] == usage["prompt_tokens"]
                    and len(r["token_ids"]) == len(toks)
                ),
                None,
            )
            if row is not None:
                toks = row["token_ids"]
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("missing raw generated token capture")
            time.sleep(0.01)
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
    ap.add_argument("--turn2-repeat", type=int, default=24)
    ap.add_argument("--compare-spec-off", action="store_true")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--top-p", type=float, default=1)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--raw-ids", action="store_true")
    ap.add_argument("--trace-tail", action="store_true")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sampling = dict(
        temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, seed=a.seed
    )
    if a.trace_tail:
        import subprocess

        from turn2prefill_profile import server_command

        popen = subprocess.Popen

        def launch(command, *args, **kwargs):
            return popen(server_command(command, "--serve-tail"), *args, **kwargs)

        subprocess.Popen = launch
    env = dict(kv.split("=", 1) for kv in a.env)
    capture = Path(a.out).with_suffix(".raw-ids.jsonl") if a.raw_ids else None
    if capture is not None:
        capture.parent.mkdir(parents=True, exist_ok=True)
        env["YUNSHU_DEBUG_STREAM_CAPTURE"] = str(capture.resolve())
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
                    * a.turn2_repeat
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
        res["cold"] = ask(s.url, s.model, Q, a.tokens, sampling, capture)
    finally:
        s.kill()
    s = t.Srv("yunshu", env, f"apcid-B-{a.ctx}")
    try:
        t.send(s.url, t.req(s.model, "Say hi.", 8))
        res["prime"] = ask(s.url, s.model, P, 8, sampling, capture)
        res["partial"] = ask(s.url, s.model, Q, a.tokens, sampling, capture)
        res["full"] = ask(s.url, s.model, Q, a.tokens, sampling, capture)
    finally:
        s.kill()
    if a.compare_spec_off:
        s = t.Srv("yunshu", {**env, "YUNSHU_VLM_DRAFT": "off"}, f"apcid-off-{a.ctx}")
        try:
            t.send(s.url, t.req(s.model, "Say hi.", 8))
            res["spec_off"] = ask(s.url, s.model, Q, a.tokens, sampling, capture)
        finally:
            s.kill()
    ref = res["cold"]
    out = dict(
        ctx=a.ctx,
        kind=a.kind,
        turn2=a.turn2,
        requested_tokens=a.tokens,
        generated_tokens=len(ref["toks"]),
        env=env,
        sampling=sampling,
    )
    for k in ("partial", "full", *(["spec_off"] if a.compare_spec_off else [])):
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
