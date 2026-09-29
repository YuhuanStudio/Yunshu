"""Greedy divergence: 50 fixed prompts x 256 tokens, reference vs Yunshu.

Every run decodes the same token-id prompts greedily and stores the ids. A
pair of runs is scored by exact match (the whole 256-token continuation is
identical) and by the position of the first differing token; one early flip
diverges everything after it, so the histogram shows how far the two paths
stay together, not how many tokens differ.

Sub-commands
    build                     write the fixed prompt set (token ids, private)
    run --config ref          stock mlx-vlm greedy loop, no Yunshu kernel imported
    run --config engine       VLMEngine's batch runner, in-process. Each request
                              runs twice, spec off then spec on (allow_draft),
                              so one process yields both paths. With
                              YUNSHU_ROUND_DRIVER=1 it is the round driver.
    compare A B [C ...]       first run is the base; every other run is scored against it
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402, N812

N_PROMPTS, MAX_TOKENS = 50, 256
EOS = (248046, 248044)  # <|im_end|>, <|endoftext|> (checked in build)
HIST = [(0, 8), (8, 32), (32, 64), (64, 128), (128, 256)]


def prompts_path() -> Path:
    return C.DATA / f"prompts-{C.CORPUS_VERSION}.json"


def build(model_dir: str) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    ev = C.EVAL_DATA

    def rows(name, n, skip=0):
        return C._jsonl(ev / name, n + skip)[skip:]

    items: list[tuple[str, str]] = []
    for r in rows("arc_challenge.jsonl", 12):
        q = r.get("question") or r.get("prompt") or json.dumps(r)
        items.append(("en", f"{q}\nExplain your answer briefly."))
    for r in rows("humaneval.jsonl", 10, 20):
        items.append(("code", f"Complete this Python function.\n\n{r['prompt']}"))
    for r in rows("gsm8k_test.jsonl", 10, 50):
        items.append(("math", r["question"]))
    for r in C._jsonl(C.CN_JSONL, 600)[590:600]:
        items.append(("zh", r["conversations"][0]["value"]))
    books = C._en_books()
    for i in range(8):  # ~2-3K-token passages to summarize
        text = books[i % 3]
        a = 60000 + 30000 * i
        items.append(
            (
                "long",
                f"{text[a : a + 9000]}\n\nSummarize the passage above in three sentences.",
            )
        )
    assert len(items) == N_PROMPTS, len(items)
    out = []
    for cat, text in items:
        msgs = [{"role": "user", "content": text}]
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False
        )
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        out.append({"cat": cat, "ids": [int(i) for i in ids]})
    eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
    C.DATA.mkdir(parents=True, exist_ok=True)
    prompts_path().write_text(json.dumps({"eos": sorted(eos), "prompts": out}))
    lens = [len(p["ids"]) for p in out]
    print(f"{len(out)} prompts, {min(lens)}-{max(lens)} tokens, eos {sorted(eos)}")


def load_prompts() -> dict:
    return json.loads(prompts_path().read_text())


def ref_generate(lm, ids, eos, step) -> list[int]:
    import mlx.core as mx
    from mlx_vlm.models import cache as vcache

    cache = vcache.make_prompt_cache(lm)
    x = np.asarray(ids)[None]
    lg = None
    for c in range(0, x.shape[1], step):
        lg = lm(mx.array(x[:, c : c + step]), cache=cache).logits[:, -1]
        mx.eval(lg)
    out: list[int] = []
    for _ in range(MAX_TOKENS):
        tok = int(mx.argmax(lg, axis=-1).item())
        if tok in eos:
            break
        out.append(tok)
        lg = lm(mx.array([[tok]]), cache=cache).logits[:, -1]
        mx.eval(lg)
    return out


def run(a) -> None:
    P = load_prompts()
    eos = set(P["eos"])
    res: dict = {"config": a.config, "model": Path(a.model).name, "runs": {}}
    t0 = time.perf_counter()
    if a.config == "ref":
        model, _ = C.load_model(a.model)
        assert not any(m.startswith("yunshu_engine.kernels") for m in sys.modules)
        toks = []
        for i, p in enumerate(P["prompts"]):
            toks.append(ref_generate(model.language_model, p["ids"], eos, a.step))
            print(f"ref {i} {len(toks[-1])} tok", flush=True)
        res["runs"]["ref"] = toks
    else:
        import asyncio

        from yunshu_engine.vlm_engine import VLMEngine

        engine = VLMEngine(a.model)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(engine.start())
        runner = engine._batch_runner
        res["engine"] = {
            "driver": runner.driver is not None,
            "draft": bool(getattr(runner, "drafter", None)),
            "ragged": runner.ragged_kv,
            "env": {k: v for k, v in os.environ.items() if k.startswith("YUNSHU_")},
        }
        print("engine:", res["engine"], flush=True)
        from yunshu_engine.vlm_batch_runner import RunStats

        for name, draft in (("spec_off", False), ("spec_on", True)):
            toks, drafted = [], 0
            for i, p in enumerate(P["prompts"]):
                stats = RunStats()
                out = list(
                    runner.iter_tokens(
                        p["ids"],
                        max_tokens=MAX_TOKENS,
                        temperature=0.0,
                        allow_draft=draft,
                        stats=stats,
                    )
                )
                out = [t for t in out if t not in eos]
                drafted += int(stats.used_draft)
                toks.append(out)
                print(f"{name} {i} {len(out)} tok", flush=True)
            res["runs"][name] = toks
            res["engine"][f"{name}_used_draft_requests"] = drafted
        loop.run_until_complete(engine.stop())
    res["wall_s"] = round(time.perf_counter() - t0)
    C.DATA.mkdir(parents=True, exist_ok=True)
    (C.DATA / f"greedy-{a.tag}.json").write_text(json.dumps(res))
    print("wrote", C.DATA / f"greedy-{a.tag}.json", res["wall_s"], "s")


def first_div(x: list[int], y: list[int]) -> int | None:
    n = min(len(x), len(y))
    for i in range(n):
        if x[i] != y[i]:
            return i
    return None if len(x) == len(y) else n


def score(base: list, other: list) -> dict:
    div = [first_div(x, y) for x, y in zip(base, other, strict=True)]
    hist = {
        f"{lo}-{hi - 1}": sum(d is not None and lo <= d < hi for d in div)
        for lo, hi in HIST
    }
    hist["identical"] = sum(d is None for d in div)
    got = [d for d in div if d is not None]
    return {
        "exact_match": hist["identical"],
        "n": len(div),
        "exact_pct": 100 * hist["identical"] / len(div),
        "hist": hist,
        "median_first_div": float(np.median(got)) if got else None,
    }


def compare(a) -> None:
    runs: dict[str, list] = {}
    for f in a.files:
        j = json.loads((C.DATA / f"greedy-{f}.json").read_text())
        for k, v in j["runs"].items():
            runs[f"{f}:{k}"] = v
    names = list(runs)
    base = a.base or names[0]
    print(f"base: {base}\n")
    print(
        "| run | exact match | identical | 0-7 | 8-31 | 32-63 | 64-127 | 128-255 | median first div |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    pairs = [(base, n) for n in names if n != base]
    pairs += [tuple(p.split("~")) for p in a.pair]
    for x, y in pairs:
        s = score(runs[x], runs[y])
        h = s["hist"]
        print(
            f"| {y} vs {x} | {s['exact_pct']:.0f}% | {s['exact_match']}/{s['n']} | "
            f"{h['0-7']} | {h['8-31']} | {h['32-63']} | {h['64-127']} | {h['128-255']} | "
            f"{s['median_first_div']} |"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--config", choices=("ref", "engine"), required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--step", type=int, default=C.PREFILL_STEP, help="ref prefill chunk")
    c = sub.add_parser("compare")
    c.add_argument("files", nargs="+", help="run tags (greedy-<tag>.json)")
    c.add_argument("--base")
    c.add_argument("--pair", nargs="*", default=[], help="extra X~Y pairs (file:run)")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.model)
    elif a.cmd == "run":
        run(a)
    else:
        compare(a)


if __name__ == "__main__":
    main()
