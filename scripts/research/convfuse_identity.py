"""Identity check for the fused GDN prefill conv: one server, cold then warm (APC hit) request per
context; writes tokens + logprobs + engagement per request. Run via gpuq.

  convfuse_identity.py --src <tree>/python --tag base --out F.json --ctx 32768 --ctx 131072
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

TFB_DIR = "/Volumes/P5Plus/yunshu-build/wt-kernelstudy/scripts/research"


def summarize(body):
    ch = body["choices"][0]
    lp = (ch.get("logprobs") or {}).get("content") or []
    toks = [e["token"] for e in lp]
    vals = [e["logprob"] for e in lp]
    return {
        "text_sha": hashlib.sha256(
            (ch["message"].get("content") or "").encode()
        ).hexdigest()[:16],
        "n": len(toks),
        "tokens": toks,
        "logprobs": vals,
        "finish": ch["finish_reason"],
        "prompt_tokens": body["usage"]["prompt_tokens"],
        "cached": (body["usage"].get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, action="append")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        print("dry-run ok", summarize_selftest())
        return
    os.environ.setdefault("TFB_WORK", "/Volumes/P5Plus/yunshu-build/snapshot014/corpus")
    os.environ["TFB_EXACT_PROMPTS"] = "1"
    os.environ["TFB_YUNSHU_SRC"] = a.src
    os.environ.setdefault("TFB_PORT_LAST", "18996")
    os.environ.setdefault("TFB_OUT", os.path.dirname(os.path.abspath(a.out)) + "/tfb")
    sys.path.insert(0, TFB_DIR)
    import tfbench

    env = {
        "YUNSHU_VLM_APC_DISK": "0",
        "YUNSHU_VLM_DRAFT": "/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2",
    }
    srv = tfbench.start_server("yunshu", env, f"convfuse-id-{a.tag}")
    result = {"tag": a.tag, "runs": []}
    try:
        for ctx in a.ctx or [32768]:
            text = tfbench.decode_prompt("prose", ctx, True)
            for phase in ("cold", "warm"):
                body = {
                    "model": srv.model,
                    "messages": [{"role": "user", "content": text}],
                    "max_tokens": a.tokens,
                    "temperature": 0,
                    "logprobs": True,
                    "top_logprobs": 1,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                req = urllib.request.Request(
                    srv.url + "/v1/chat/completions",
                    json.dumps(body).encode(),
                    {"Content-Type": "application/json", "Authorization": "Bearer k"},
                )
                with urllib.request.urlopen(req, timeout=1500) as r:
                    rec = summarize(json.load(r))
                if rec["n"] != a.tokens or rec["finish"] != "length":
                    raise RuntimeError(
                        f"bad reply ctx={ctx} {phase}: {rec['n']} {rec['finish']}"
                    )
                rec.update(ctx=ctx, phase=phase)
                result["runs"].append(rec)
                print(ctx, phase, rec["text_sha"], rec["cached"], flush=True)
        log = srv.log.read_text(errors="replace")
        result["fused_engaged"] = "fused conv+silu engaged" in log
    finally:
        srv.kill()
    with open(a.out, "w") as f:
        json.dump(result, f)
    print("fused_engaged", result["fused_engaged"])


def summarize_selftest():
    body = {
        "choices": [
            {
                "message": {"content": "hi"},
                "finish_reason": "length",
                "logprobs": {"content": [{"token": "hi", "logprob": -0.5}]},
            }
        ],
        "usage": {"prompt_tokens": 3},
    }
    r = summarize(body)
    assert r["n"] == 1 and r["cached"] is None
    return r["text_sha"]


if __name__ == "__main__":
    main()
