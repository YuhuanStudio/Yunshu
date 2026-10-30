"""Per-step cost of concurrent Flash-Next decode: main, row-exact QSA, and batched sparse QSA.

  bqsa_cost.py --ctx-tokens 8192 --rows 1 2 4 --arms base row cand --out F.jsonl

Per arm one server (draft off, so only the batch layout differs). B requests with distinct salts start together;
the decode window is the span in which all B rows are streaming (last first token to first finish). Each
row's tok/s over that window and the step time 1000 / tok/s are reported. A window with fewer than
--min-window tokens per row fails the cell (the cell is not a decode measurement).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20)


def window_rates(rows: list[dict], min_tokens: int) -> dict:
    """Per-row tok/s while every row is decoding."""
    lo = max(r["times"][0] for r in rows)
    hi = min(r["times"][-1] for r in rows)
    if hi <= lo:
        raise RuntimeError("rows never decoded together")
    rates = []
    for r in rows:
        inside = [t for t in r["times"] if lo <= t <= hi]
        if len(inside) < min_tokens:
            raise RuntimeError(f"decode window too short: {len(inside)} tokens")
        rates.append((len(inside) - 1) / (inside[-1] - inside[0]))
    return dict(
        window_s=round(hi - lo, 2),
        row_tps=[round(x, 2) for x in rates],
        mean_tps=round(statistics.mean(rates), 2),
        step_ms=round(1000 / statistics.mean(rates), 2),
        aggregate_tps=round(sum(rates), 2),
    )


def require_warm_cache(cell: dict) -> None:
    """Reject missing accounting and short prefix hits in every measured row."""
    cached = cell["cached"]
    prompts = cell["prompt_tokens"]
    if len(cached) != len(prompts) or not cached:
        raise RuntimeError("missing cache accounting")
    if any(
        isinstance(c, bool)
        or not isinstance(c, int)
        or not isinstance(p, int)
        or c < max(1, p - 128)
        or c > p
        for c, p in zip(cached, prompts, strict=True)
    ):
        raise RuntimeError(
            f"missing full-prefix APC hits: {cached}, prompts: {prompts}"
        )


def calibrated_chars(token_count, target: int) -> int:
    """Find a prefix close to the target; fail rather than mislabel a context."""
    lo, hi = 0, max(1024, target * 16)
    while lo < hi:
        mid = (lo + hi) // 2
        if token_count(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    if abs(token_count(lo) - target) > 128:
        raise RuntimeError("could not calibrate prompt token count")
    return lo


def require_context(cell: dict, target: int) -> None:
    if not cell["prompt_tokens"] or any(
        isinstance(p, bool) or not isinstance(p, int) or abs(p - target) > 256
        for p in cell["prompt_tokens"]
    ):
        raise RuntimeError(f"unexpected prompt token counts: {cell['prompt_tokens']}")


def warm(url, model, rows, ctx_chars, salt):
    """Prefill every row's prompt once, alone, so the measured cells start decoding together (APC hit)."""
    import conc_probe
    import sampledspec_conc as sc

    for i in range(rows):
        prompt = (
            conc_probe.filler(ctx_chars[i], salt + i)
            + f"\n\n[{salt + i}] Review the material above carefully. Reason step by step about "
            "its design, then write a detailed technical critique with concrete improvement proposals."
        )
        b = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            stream=True,
            stream_options={"include_usage": True},
            seed=17 + i,
            **SAMPLING,
        )
        sc.stream(url, b, time.perf_counter())


def run_cell(url, model, rows, ctx_chars, tokens, salt, min_tokens):
    import conc_probe
    import sampledspec_conc as sc

    out, errs = {}, []
    t_zero = time.perf_counter()

    def go(i, tokens=tokens):
        try:
            prompt = (
                conc_probe.filler(ctx_chars[i], salt + i)
                + f"\n\n[{salt + i}] Review the material above carefully. Reason step by step about "
                "its design, then write a detailed technical critique with concrete improvement proposals."
            )
            b = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=tokens,
                stream=True,
                stream_options={"include_usage": True},
                seed=17 + i,
                **SAMPLING,
            )
            out[i] = sc.stream(url, b, t_zero)
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))

    threads = [threading.Thread(target=go, args=(i,)) for i in range(rows)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errs or len(out) != rows:
        raise RuntimeError(f"cell failed: {errs}")
    got = [out[i] for i in range(rows)]
    if any(r["finish"] != "length" for r in got):
        raise RuntimeError("a row did not fill its budget")
    cell = (
        window_rates(got, min_tokens)
        if rows > 1
        else window_rates(
            [dict(times=got[0]["times"][len(got[0]["times"]) // 10 :])], min_tokens
        )
    )
    cell.update(
        rows=rows,
        prompt_tokens=[r["usage"].get("prompt_tokens") for r in got],
        cached=[
            (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
            for r in got
        ],
        ttft_s=[round(r["ttft"], 2) for r in got],
        digests=[r["digest"][:12] for r in got],
    )
    return cell


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-src", type=Path, required=True)
    p.add_argument("--cand-src", type=Path, required=True)
    p.add_argument("--row-src", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--ctx-tokens", type=int, required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument(
        "--arms", nargs="+", default=["base", "cand"], choices=["base", "row", "cand"]
    )
    p.add_argument("--tokens", type=int, default=900)
    p.add_argument("--min-window", type=int, default=150)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    import conc_probe
    import tfbench as t
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    lengths = [
        calibrated_chars(
            lambda n, salt=5000 + i: len(tokenizer.encode(conc_probe.filler(n, salt))),
            a.ctx_tokens - 128,
        )
        for i in range(max(a.rows))
    ]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for arm_index, arm in enumerate(a.arms):
        t.YUNSHU_SRC = str(
            {"base": a.base_src, "row": a.row_src, "cand": a.cand_src}[arm].resolve()
        )
        label = f"bqsa-cost-{a.out.stem}-{arm}-{arm_index}"
        server = t.Srv(
            "yunshu",
            {
                "YUNSHU_VLM_DRAFT": "off",
                "YUNSHU_VLM_APC_DISK": "0",
                "YUNSHU_NEXT_TURN_PREFILL": "off",
                "YUNSHU_CONSOLE": "0",
            },
            label,
            model=a.model,
        )
        try:
            t.send(server.url, t.req(server.model, "Say hi.", 16))
            for rows in a.rows:
                warm(server.url, server.model, rows, lengths, 5000)
                cell = run_cell(
                    server.url,
                    server.model,
                    rows,
                    lengths,
                    a.tokens,
                    5000,
                    a.min_window,
                )
                require_warm_cache(cell)
                require_context(cell, a.ctx_tokens)
                cell.update(
                    arm=arm,
                    arm_index=arm_index,
                    ctx_tokens=a.ctx_tokens,
                    ctx_chars=lengths[:rows],
                )
                with a.out.open("a") as fh:
                    fh.write(json.dumps(cell) + "\n")
                print(json.dumps(cell), flush=True)
        finally:
            server.kill()
    with a.out.open("a") as fh:
        fh.write(json.dumps(dict(complete=True)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
