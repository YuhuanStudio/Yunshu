"""Same-version rehearsal/plain identity and follow-up timing, always run via gpuq."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def metrics(t, server):
    req = urllib.request.Request(
        server.url + "/metrics", headers={"Authorization": "Bearer k"}
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return t.completed_rehearsals(response.read().decode())


def check_reply_len(row, tokens, what):
    """Natural stop is fine once the reply reached half the budget (long budgets)."""
    if row.get("finish") == "length" or (
        row.get("finish") == "stop" and row.get("ct", 0) >= tokens // 2
    ):
        return
    raise RuntimeError(f"{what}: finish={row.get('finish')} ct={row.get('ct')}")


def same_output(left, right):
    """Require complete chosen-token logprobs as well as raw token identity."""
    return bool(
        left.get("toks")
        and left["toks"] == right.get("toks")
        and left.get("lps") is not None
        and len(left["lps"]) == len(left["toks"])
        and left["lps"] == right.get("lps")
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--kind", choices=["code", "prose"], default="code")
    p.add_argument("--reply-tokens", type=int, nargs="+", default=[256, 512, 2048])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top-p", type=float, default=1)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--apc-dir", default="/Volumes/Micron/yunshu-apc/splitinv-rehearsal")
    # A logprobs request takes the stock-serial arithmetic and its own APC namespace
    # (vlm_batch_runner._serial_rows), so it never reads the invariant checkpoints a
    # rehearsal writes: rehearsal == plain is a raw-token-ID gate. --logprobs asks for
    # them anyway and shows such a request missing the rehearsal (cached 0).
    p.add_argument(
        "--logprobs",
        action="store_true",
        help="Also request and compare chosen-token logprobs (served outside the rehearsal namespace)",
    )
    a = p.parse_args()
    if a.dry_run:
        print(json.dumps(vars(a), default=str))
        return
    import tfbench as t

    sampling = dict(
        temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, seed=a.seed
    )

    prompt = t.decode_prompt(a.kind, a.ctx, True)
    env = {
        "YUNSHU_VLM_APC_DISK": "0",
        "YUNSHU_VLM_DRAFT": t.D,
        "YUNSHU_NEXT_TURN_DELAY_S": "0",
        "YUNSHU_VLM_APC_DISK_DIR": a.apc_dir,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    capture = a.out.with_suffix(".raw-ids.jsonl")
    env["YUNSHU_DEBUG_STREAM_CAPTURE"] = str(capture)
    fixtures = []

    def ask_plain(server, messages):
        body = t.req(server.model, "", 32)
        body.update(sampling)
        body["messages"] = messages
        body["logprobs"] = a.logprobs
        capture_start = len(capture.read_text().splitlines()) if capture.exists() else 0
        result = t.send(server.url, body)
        deadline = time.monotonic() + 3
        while True:
            rows = (
                [
                    json.loads(line)
                    for line in capture.read_text().splitlines()[capture_start:]
                ]
                if capture.exists()
                else []
            )
            match = next(
                (
                    row
                    for row in reversed(rows)
                    if row["prompt_tokens"] == result["pt"]
                    and len(row["token_ids"]) == result["ct"]
                ),
                None,
            )
            if match is not None:
                result["toks"] = match["token_ids"]
                result.pop("_text", None)
                return result
            if time.monotonic() >= deadline:
                raise RuntimeError("missing raw generated token capture")
            time.sleep(0.01)

    def emit(row):
        with a.out.open("a") as out:
            out.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    for arrival in ["completed", "immediate"]:
        server = t.start_server(
            "yunshu",
            {**env, "YUNSHU_NEXT_TURN_PREFILL": "stride"},
            f"splitinv-rehearsal-{a.ctx}-{a.kind}-{arrival}",
        )
        try:
            for tokens in a.reply_tokens:
                # Vary the final user text so full-hit turn-two caches from earlier
                # cells cannot masquerade as newly rehearsed checkpoints.
                body = t.req(
                    server.model,
                    prompt
                    + f"\nResponse budget: {tokens} tokens. Arrival case: {arrival}.",
                    tokens,
                )
                body.update(sampling)
                before = metrics(t, server)
                first = t.send(server.url, body)
                check_reply_len(first, tokens, f"splitinv reply {tokens}")
                if not first["_text"].strip():
                    raise RuntimeError("empty assistant reply")
                if arrival == "completed":
                    deadline = time.monotonic() + 180
                    while metrics(t, server) <= before:
                        if time.monotonic() >= deadline:
                            raise RuntimeError("rehearsal never completed")
                        time.sleep(0.1)
                messages = body["messages"] + [
                    {"role": "assistant", "content": first["_text"].strip()},
                    {
                        "role": "user",
                        "content": "Continue with the next part, at the same length.",
                    },
                ]
                result = ask_plain(server, messages)
                if not result["cached"] or result["cached"] % 64:
                    raise RuntimeError(f"missing aligned APC hit: {result}")
                emit(
                    dict(
                        ctx=a.ctx,
                        kind=a.kind,
                        reply_tokens=tokens,
                        arrival=arrival,
                        arm="rehearsal",
                        **{k: v for k, v in result.items() if k not in ("toks", "lps")},
                    )
                )
                fixtures.append((tokens, arrival, messages, result))
            server.verify_spec_mode()
        finally:
            server.kill()
    # A separate process with APC and rehearsal disabled computes the exact same
    # follow-up bodies from scratch. Cross-version outputs are not compared here.
    server = t.start_server(
        "yunshu",
        {**env, "YUNSHU_NEXT_TURN_PREFILL": "off", "YUNSHU_VLM_APC_MEMORY_GB": "0"},
        f"splitinv-plain-{a.ctx}-{a.kind}",
    )
    try:
        for tokens, arrival, messages, reference in fixtures:
            plain = ask_plain(server, messages)
            equal = (
                plain["toks"] == reference["toks"]
                if not a.logprobs
                else same_output(plain, reference)
            )
            emit(
                dict(
                    ctx=a.ctx,
                    kind=a.kind,
                    reply_tokens=tokens,
                    arrival=arrival,
                    arm="plain",
                    tokens_equal=plain["toks"] == reference["toks"],
                    raw_ids_equal=plain["toks"] == reference["toks"],
                    logprobs_equal=(
                        plain["lps"] == reference["lps"] if a.logprobs else None
                    ),
                    **{k: v for k, v in plain.items() if k not in ("toks", "lps")},
                )
            )
            if plain["cached"] or not equal:
                raise RuntimeError("rehearsal != plain cold follow-up")
        server.verify_spec_mode()
    finally:
        server.kill()
    emit(dict(complete=True, cases=len(fixtures), logprobs_checked=a.logprobs))


if __name__ == "__main__":
    main()
