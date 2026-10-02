"""Official SDK concurrent agent replay; only launch through gpuq.

A/B changes admission coalescing only. Each burst has a new, uncached prefix.
Uniform agents share one generator; mixed logprob settings use separate groups.
"""

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
from pathlib import Path

import openai
import tfbench as t


def ask(server, prefix, row, mixed):
    client = openai.OpenAI(base_url=server.url + "/v1", api_key="k", timeout=900)
    start = time.perf_counter()
    first = None
    content = []
    tokens = []
    lps = []
    usage = None
    finish = None
    stream = client.chat.completions.create(
        model=server.model,
        messages=[
            {"role": "system", "content": prefix},
            {"role": "user", "content": f"Agent {row}: reply with one short sentence."},
        ],
        max_tokens=8,
        temperature=0,
        logprobs=True,
        top_logprobs=(row % 2 if mixed else 0),
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for chunk in stream:
        usage = chunk.usage or usage
        for choice in chunk.choices:
            for lp in (
                choice.logprobs.content
                if choice.logprobs and choice.logprobs.content
                else []
            ):
                first = first or time.perf_counter() - start
                tokens.append(lp.token)
                lps.append(lp.logprob)
            if choice.delta.content:
                content.append(choice.delta.content)
            finish = choice.finish_reason or finish
    if not (first and finish and usage and tokens):
        raise RuntimeError("incomplete SDK stream")
    cached = usage.prompt_tokens_details.cached_tokens
    return {
        "row": row,
        "ttft": first,
        "elapsed": time.perf_counter() - start,
        "cached": cached,
        "computed": usage.prompt_tokens - cached,
        "usage": usage.model_dump(),
        "tokens": tokens,
        "lps": lps,
        "digest": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768])
    p.add_argument("--parallel", type=int, nargs="+", default=[2, 4, 8])
    a = p.parse_args()
    from cache_probe_source import freeze

    source = freeze(a.out, t)
    root = Path(a.out).parent
    root.mkdir(parents=True, exist_ok=True)
    # Test harness switches a Python class attribute; no experimental serving
    # flag or losing production path is introduced by the benchmark.
    original = t.YUNSHU_BIN
    wrappers = {}
    for mode in ("off", "on"):
        path = root / f"cachesf-serve-{mode}"
        path.write_text(
            "#!" + str(Path(original).parent / "python") + "\n"
            "from yunshu_engine.vlm_batch_runner import VLMBatchRunner\n"
            f"VLMBatchRunner.singleflight = {mode == 'on'!r}\n"
            "from yunshu_cli import main\nmain()\n"
        )
        path.chmod(0o755)
        wrappers[mode] = str(path)
    records = []
    failures = []
    with open(a.out, "w") as out:

        def record(r):
            records.append(r)
            out.write(json.dumps(r) + "\n")
            out.flush()
            print(
                json.dumps({k: v for k, v in r.items() if k != "requests"}), flush=True
            )

        for ctx in a.contexts:
            text = t.load_prompt(f"prose-{ctx}")
            for rep in range(a.runs):
                paired = {}
                for mode in ("off", "on") if rep % 2 == 0 else ("on", "off"):
                    t.YUNSHU_BIN = wrappers[mode]
                    server = t.Srv(
                        "yunshu",
                        {"YUNSHU_VLM_APC_DISK": "0", "YUNSHU_VLM_DRAFT": t.D},
                        f"cachesf-{ctx}-{rep}-{mode}-{time.time_ns()}",
                    )
                    try:
                        ask(server, "Warmup.", 0, False)
                        for mixed in (False, True):
                            for parallel in a.parallel:
                                prefix = (
                                    f"Cold burst ctx={ctx} rep={rep} groups={mixed} n={parallel}.\n"
                                    + text
                                )
                                with concurrent.futures.ThreadPoolExecutor(
                                    max_workers=parallel
                                ) as pool:
                                    requests = list(
                                        pool.map(
                                            lambda row: ask(server, prefix, row, mixed),
                                            range(parallel),
                                        )
                                    )
                                ts = sorted(r["ttft"] for r in requests)
                                paired[(mode, mixed, parallel)] = requests
                                record(
                                    {
                                        "ctx": ctx,
                                        "rep": rep,
                                        "mode": mode,
                                        "mixed": mixed,
                                        "parallel": parallel,
                                        "p50": statistics.median(ts),
                                        "p90": ts[min(len(ts) - 1, int(0.9 * len(ts)))],
                                        "computed": sum(
                                            r["computed"] for r in requests
                                        ),
                                        "cached": sum(r["cached"] for r in requests),
                                        "requests": requests,
                                        "server_log": str(server.log),
                                        "rc": 0,
                                    }
                                )
                    except Exception as exc:
                        failures.append(str(exc))
                        record(
                            {
                                "ctx": ctx,
                                "rep": rep,
                                "mode": mode,
                                "rc": 1,
                                "error": repr(exc),
                            }
                        )
                    finally:
                        server.kill()
                    log = server.log.read_text(errors="replace")
                    record(
                        {
                            "ctx": ctx,
                            "rep": rep,
                            "mode": mode,
                            "engaged": {
                                "singleflight_waits": log.count(
                                    "APC single-flight wait:"
                                ),
                                "dflash_installed": "Speculative decoding: dflash"
                                in log,
                                "prefix_invariant": "APC prefix-invariant dispatch engaged:"
                                in log,
                                "shared_gdn": "APC shared GDN singleton arithmetic engaged:"
                                in log,
                                "shared_attention_tile": "APC shared attention tile arithmetic engaged"
                                in log,
                                "source": source,
                            },
                            "server_log": str(server.log),
                        }
                    )
                for mixed in (False, True):
                    for parallel in a.parallel:
                        before = paired.get(("off", mixed, parallel))
                        after = paired.get(("on", mixed, parallel))
                        if before is None or after is None:
                            continue
                        equal = all(
                            x["tokens"] == y["tokens"] and x["lps"] == y["lps"]
                            for x, y in zip(before, after, strict=True)
                        )
                        record(
                            {
                                "ctx": ctx,
                                "rep": rep,
                                "mixed": mixed,
                                "parallel": parallel,
                                "identity": equal,
                            }
                        )
                        if not equal:
                            failures.append(f"identity {ctx} {rep} {mixed} {parallel}")
        record({"complete": True, "failures": failures, "ok": not failures})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
