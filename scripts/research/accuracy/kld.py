"""Teacher-forced logit alignment (KLD / top-1 / perplexity) against a reference.

The method llama.cpp's ``perplexity --kl-divergence`` and vLLM's logprob
closeness tests use: feed the same fixed token ids to two implementations and
compare their next-token distributions at every position. No sampling, no
generation drift, so a 100K-token corpus resolves numeric differences that a
300-question benchmark (+-13 questions of 95% noise) cannot.

Sub-commands
    corpus                build the versioned corpus (token ids -> docs/research/accuracy)
    run                   one condition; without --ref it *writes* a reference
                          (top-64 logprobs for every position, full-vocab
                          log-softmax for a fixed random subset), with --ref it
                          *compares* against it
    report                merge the JSON summaries of several runs into tables

Configurations (--config)
    stock   mlx-vlm exactly as installed; no Yunshu kernel module is imported
    serve   what VLMEngine._build_batch_runner installs (oMLX verify kernels,
            batch-invariant + NAX-packed projections, ragged KV)

Parts (--part): which serving path is exercised
    prefill   the whole corpus through 2048-token prefill chunks (all positions)
    dec1      decode windows, one token per step. serve: the speculative lane
              (invariant kernels on, dense-lane ragged attention)
    dec6      decode windows in 6-token verify blocks (serve only; the MTP
              verify forward, with the same activation switches as the lane)
    decb      decode windows in pairs as a two-row shared batch on a ragged
              KV cache (serve only; invariant kernels off, as in the runner)
    decb8     as decb with YUNSHU_KV_PRECISION=int8 codes
    stockb2   stock config, decode windows in pairs as a two-row batch
              (the reference's own batch-variance floor)

Metrics are KLD(ref || cand) per position. Positions outside the stored full
subset use the reference's top-64 plus one lumped tail bucket, a lower bound of
the true KLD; the full-distribution subset gives the exact value beside it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402, N812

N_FULL_PREFILL, N_FULL_DEC = 256, 128
SLICE = 256  # rows per log-softmax slice (float32 [256, V] = 256 MB)


# ── layout: record ids for every (sequence, position) ─────────────────────
def layout(corpus: dict) -> dict:
    """Prefill records: position p of sequence s predicts token p + 1, so a
    length-n sequence has n - 1 records. Decode records: window w, step j feeds
    position start + j and predicts start + j + 1."""
    names = [s["name"] for s in corpus["seqs"]]
    lens = [len(s["ids"]) for s in corpus["seqs"]]
    p_off = np.concatenate([[0], np.cumsum([n - 1 for n in lens])])
    p_seq = np.concatenate([np.full(n - 1, i) for i, n in enumerate(lens)])
    p_pos = np.concatenate([np.arange(n - 1) for n in lens])
    wins = C.decode_windows()
    d_off = np.concatenate([[0], np.cumsum([w[2] for w in wins])])
    d_seq = np.concatenate([np.full(w[2], names.index(w[0])) for w in wins])
    d_pos = np.concatenate([w[1] + np.arange(w[2]) for w in wins])
    d_win = np.concatenate([np.full(w[2], i) for i, w in enumerate(wins)])
    return {
        "names": names,
        "cats": [s["cat"] for s in corpus["seqs"]],
        "wins": wins,
        "prefill": {"off": p_off, "seq": p_seq, "pos": p_pos},
        "dec": {"off": d_off, "seq": d_seq, "pos": d_pos, "win": d_win},
    }


def targets_of(corpus: dict, lay: dict, section: str) -> np.ndarray:
    s = lay[section]
    ids = [x["ids"] for x in corpus["seqs"]]
    return np.array(
        [ids[q][p + 1] for q, p in zip(s["seq"], s["pos"], strict=True)], dtype=np.int64
    )


# ── device-side processing ────────────────────────────────────────────────
def _logprobs(lg):
    import mlx.core as mx

    x = lg.astype(mx.float32)
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


class Sink:
    """Receives logits [n, V] for a run of consecutive records of a section."""

    def __init__(self, section: str, n: int, targets: np.ndarray, full: np.ndarray):
        self.section, self.n, self.targets = section, n, targets
        self.full = full  # record ids (sorted) that keep the whole distribution
        self.full_at = {int(r): i for i, r in enumerate(full)}
        self.done = np.zeros(n, dtype=bool)


class RefSink(Sink):
    def __init__(self, *a, vocab: int | None = None):
        super().__init__(*a)
        self.idx = np.zeros((self.n, C.TOPK), dtype=np.int32)
        self.lp = np.zeros((self.n, C.TOPK), dtype=np.float32)
        self.tgt = np.zeros(self.n, dtype=np.float32)
        self.full_lp: np.ndarray | None = None

    def feed(self, rec0: int, logits) -> None:
        import mlx.core as mx

        n = logits.shape[0]
        for a in range(0, n, SLICE):
            b = min(a + SLICE, n)
            lp = _logprobs(logits[a:b])
            part = mx.argpartition(-lp, C.TOPK - 1, axis=-1)[:, : C.TOPK]
            val = mx.take_along_axis(lp, part, axis=-1)
            order = mx.argsort(-val, axis=-1)
            idx = mx.take_along_axis(part, order, axis=-1)
            val = mx.take_along_axis(val, order, axis=-1)
            tg = mx.array(self.targets[rec0 + a : rec0 + b])
            tlp = mx.take_along_axis(lp, tg[:, None], axis=-1)[:, 0]
            fulls = [
                (r - rec0 - a, r)
                for r in range(rec0 + a, rec0 + b)
                if r in self.full_at
            ]
            rows = (
                mx.stack([lp[i] for i, _ in fulls]).astype(mx.float16)
                if fulls
                else None
            )
            mx.eval(idx, val, tlp, *([rows] if rows is not None else []))
            self.idx[rec0 + a : rec0 + b] = np.array(idx)
            self.lp[rec0 + a : rec0 + b] = np.array(val)
            self.tgt[rec0 + a : rec0 + b] = np.array(tlp)
            if rows is not None:
                if self.full_lp is None:
                    self.full_lp = np.zeros(
                        (len(self.full), rows.shape[1]), dtype=np.float16
                    )
                arr = np.array(rows)
                for (_, r), row in zip(fulls, arr, strict=True):
                    self.full_lp[self.full_at[r]] = row
            self.done[rec0 + a : rec0 + b] = True


class CandSink(Sink):
    def __init__(self, *a, ref: dict):
        super().__init__(*a)
        self.ref = ref
        n = self.n
        self.kld = np.full(n, np.nan, dtype=np.float64)
        self.agree = np.zeros(n, dtype=bool)
        self.dp = np.zeros(n, dtype=np.float64)
        self.tgt = np.zeros(n, dtype=np.float32)
        self.kld_full = np.full(len(self.full), np.nan)

    def feed(self, rec0: int, logits) -> None:
        import mlx.core as mx

        n = logits.shape[0]
        for a in range(0, n, SLICE):
            b = min(a + SLICE, n)
            r0, r1 = rec0 + a, rec0 + b
            lp = _logprobs(logits[a:b])
            ridx = mx.array(self.ref["idx"][r0:r1])
            rlp = mx.array(self.ref["lp"][r0:r1])
            clp = mx.take_along_axis(lp, ridx, axis=-1)
            pr, pc = mx.exp(rlp), mx.exp(clp)
            tail_r = mx.maximum(1.0 - pr.sum(-1), 1e-7)
            tail_c = mx.maximum(1.0 - pc.sum(-1), 1e-7)
            kld = (pr * (rlp - clp)).sum(-1) + tail_r * mx.log(tail_r / tail_c)
            agree = mx.argmax(lp, axis=-1) == ridx[:, 0]
            dp = mx.abs(pr[:, 0] - pc[:, 0])
            tg = mx.array(self.targets[r0:r1])
            tlp = mx.take_along_axis(lp, tg[:, None], axis=-1)[:, 0]
            fulls = [(r - r0, r) for r in range(r0, r1) if r in self.full_at]
            fk = []
            for i, r in fulls:
                lr = mx.array(self.ref["full_lp"][self.full_at[r]]).astype(mx.float32)
                lr = lr - mx.logsumexp(lr)
                fk.append((mx.exp(lr) * (lr - lp[i])).sum())
            mx.eval(kld, agree, dp, tlp, *fk)
            self.kld[r0:r1] = np.array(kld)
            self.agree[r0:r1] = np.array(agree)
            self.dp[r0:r1] = np.array(dp)
            self.tgt[r0:r1] = np.array(tlp)
            for (_, r), v in zip(fulls, fk, strict=True):
                self.kld_full[self.full_at[r]] = float(v.item())
            self.done[r0:r1] = True


# ── model drivers ─────────────────────────────────────────────────────────
def _cache(lm):
    from mlx_vlm.models import cache as vcache

    return vcache.make_prompt_cache(lm)


def _prefill_rows(lm, ids: np.ndarray, step: int, cache=None):
    """Feed ``ids`` ([B, n]) in chunks; yield (first position, logits[B, T, V])."""
    import mlx.core as mx

    n = ids.shape[1]
    for c in range(0, n, step):
        out = lm(mx.array(ids[:, c : c + step]), cache=cache)
        yield c, out.logits


def run_prefill(lm, corpus, lay, sink, step: int, log) -> None:
    import mlx.core as mx

    for i, s in enumerate(corpus["seqs"]):
        t0 = time.perf_counter()
        ids = np.asarray(s["ids"])[None, :-1]
        cache = _cache(lm)
        base = int(lay["prefill"]["off"][i])
        for c, lg in _prefill_rows(lm, ids, step, cache):
            sink.feed(base + c, lg[0])
        mx.clear_cache()
        log(f"prefill {s['name']} {ids.shape[1]} tok {time.perf_counter() - t0:.1f}s")


def _serve_switches(spec: bool):
    from yunshu_engine.kernels import batch_invariant, ragged_kv

    batch_invariant.set_active(spec)
    ragged_kv.set_dense_lane(spec)


def run_decode(lm, corpus, lay, sink, mode: str, serve: bool, log) -> None:
    """mode: step1 | block6 | ragged | ragged8 ; pairs when the mode needs two rows."""
    import mlx.core as mx

    seqs = {s["name"]: np.asarray(s["ids"]) for s in corpus["seqs"]}
    wins = lay["wins"]
    pair = mode in ("ragged", "ragged8", "step1b2")
    groups = (
        [[i, i + 1] for i in range(0, len(wins), 2)]
        if pair
        else [[i] for i in range(len(wins))]
    )
    for grp in groups:
        t0 = time.perf_counter()
        name, start, steps = wins[grp[0]]
        assert all(wins[g][1:] == (start, steps) for g in grp)
        ids = np.stack([seqs[wins[g][0]] for g in grp])
        cache = _cache(lm)
        if serve:
            _serve_switches(False)
        for _c, _lg in _prefill_rows(lm, ids[:, :start], C.PREFILL_STEP, cache):
            mx.eval(_lg[:, -1:])
        if mode in ("ragged", "ragged8"):
            from yunshu_engine.kernels.ragged_kv import RaggedKVCache, _dense_kv_classes

            fmt = "int8" if mode == "ragged8" else "bf16"
            kv = _dense_kv_classes()
            cache = [
                RaggedKVCache.from_cache(c, fmt) if type(c) in kv else c for c in cache
            ]
            assert any(isinstance(c, RaggedKVCache) for c in cache)
        if serve and mode in ("step1", "block6"):
            _serve_switches(True)
        block = 6 if mode == "block6" else 1
        base = [int(lay["dec"]["off"][g]) for g in grp]
        for j in range(0, steps, block):
            t = min(block, steps - j)
            x = mx.array(ids[:, start + j : start + j + t])
            if mode == "block6":
                out = lm(x, cache=cache, speculative_verify=True)
                lg = out.logits
                out.gdn_states.commit(t)
            else:
                lg = lm(x, cache=cache).logits
            for b in range(len(grp)):
                sink.feed(base[b] + j, lg[b])
        if serve:
            _serve_switches(False)
        mx.clear_cache()
        log(
            f"decode {[wins[g][0] for g in grp]} {mode} {time.perf_counter() - t0:.1f}s"
        )


# ── reporting ─────────────────────────────────────────────────────────────
def _stats(kld, agree, dp, tgt_c, tgt_r) -> dict:
    kld = kld[~np.isnan(kld)]
    return {
        "n": int(len(agree)),
        "kld_mean": float(kld.mean()),
        "kld_median": float(np.median(kld)),
        "kld_p99": float(np.percentile(kld, 99)),
        "kld_max": float(kld.max()),
        "top1_pct": float(agree.mean() * 100),
        "dp_top1": float(dp.mean()),
        "ppl_ref": float(np.exp(-tgt_r.astype(np.float64).mean())),
        "ppl_cand": float(np.exp(-tgt_c.astype(np.float64).mean())),
    }


def summarize(sink: CandSink, ref: dict, lay: dict, section: str) -> dict:
    s = lay[section]
    cats = np.array([lay["cats"][q] for q in s["seq"]])
    pos = s["pos"]
    out: dict = {"overall": _stats(sink.kld, sink.agree, sink.dp, sink.tgt, ref["tgt"])}
    for c in sorted(set(cats)):
        m = cats == c
        out.setdefault("by_type", {})[c] = _stats(
            sink.kld[m], sink.agree[m], sink.dp[m], sink.tgt[m], ref["tgt"][m]
        )
    for (lo, hi), nm in zip(C.BUCKETS, C.BUCKET_NAMES, strict=True):
        m = (pos >= lo) & (pos < hi)
        if m.any():
            out.setdefault("by_position", {})[nm] = _stats(
                sink.kld[m], sink.agree[m], sink.dp[m], sink.tgt[m], ref["tgt"][m]
            )
    kf = sink.kld_full[~np.isnan(sink.kld_full)]
    if len(kf):
        rows = np.array(sorted(sink.full_at.values()))
        trunc = sink.kld[np.array(sorted(sink.full_at))]
        out["full_vocab_subset"] = {
            "n": int(len(kf)),
            "kld_mean_exact": float(kf.mean()),
            "kld_mean_top64_estimate": float(trunc.mean()),
            "kld_max_exact": float(kf.max()),
        }
        del rows
    return out


def fmt_row(label: str, s: dict) -> str:
    return (
        f"| {label} | {s['n']} | {s['kld_mean']:.2e} | {s['kld_median']:.2e} | "
        f"{s['kld_p99']:.2e} | {s['kld_max']:.2e} | {s['top1_pct']:.3f} | "
        f"{s['dp_top1']:.2e} | {s['ppl_ref']:.4f} | {s['ppl_cand']:.4f} |"
    )


HEAD = (
    "| slice | n | KLD mean | median | p99 | max | top-1 % | mean abs dp | ppl ref | ppl cand |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)


def cmd_report(a) -> None:
    for f in a.files:
        j = json.loads(Path(f).read_text())
        print(f"\n### {j['label']}  ({j['part']}, {j['config']}, model {j['model']})\n")
        print(HEAD)
        print(fmt_row("overall", j["summary"]["overall"]))
        for k in ("by_type", "by_position"):
            for name, s in j["summary"].get(k, {}).items():
                print(fmt_row(f"{k[3:]} {name}", s))
        if "full_vocab_subset" in j["summary"]:
            print("\nfull vocab subset:", j["summary"]["full_vocab_subset"])


# ── commands ──────────────────────────────────────────────────────────────
PARTS = {
    "prefill": ("prefill", None),
    "dec1": ("dec", "step1"),
    "dec6": ("dec", "block6"),
    "decb": ("dec", "ragged"),
    "decb8": ("dec", "ragged8"),
    "stockb2": ("dec", "step1b2"),
}


def cmd_corpus(a) -> None:
    from tokenizers import Tokenizer

    class Tok:  # tokenizer only: building the corpus needs no model or GPU
        def __init__(self, path):
            self.t = Tokenizer.from_file(path)

        def encode(self, text, add_special_tokens=False):
            return self.t.encode(text, add_special_tokens=add_special_tokens).ids

    corpus = C.build_corpus(Tok(str(Path(a.model) / "tokenizer.json")))
    C.save_corpus(corpus)
    tot = sum(len(s["ids"]) for s in corpus["seqs"])
    print(
        f"corpus {corpus['version']} digest {C.corpus_digest(corpus)}: "
        f"{len(corpus['seqs'])} sequences, {tot} tokens -> {C.corpus_path()}"
    )


def cmd_run(a) -> None:
    import mlx.core as mx

    corpus = C.load_corpus()
    lay = layout(corpus)
    section, mode = PARTS[a.part]
    serve = a.config == "serve"
    if a.part in ("dec6", "decb", "decb8") and not serve:
        sys.exit(f"--part {a.part} needs --config serve")
    model, _tok = C.load_model(a.model)
    lm = model.language_model
    info = C.setup_serve(model, driver=a.driver) if serve else {}
    if serve:
        print("serve setup:", json.dumps(info, default=str), flush=True)
    else:
        assert not any(m.startswith("yunshu_engine.kernels") for m in sys.modules), (
            "a Yunshu kernel module is imported in a stock run"
        )
    targets = targets_of(corpus, lay, section)
    n = len(targets)
    outdir = C.DATA
    outdir.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    t0 = time.perf_counter()
    if a.ref is None:
        rng = np.random.default_rng(7 if section == "prefill" else 11)
        k = N_FULL_PREFILL if section == "prefill" else N_FULL_DEC
        full = np.sort(rng.choice(n, size=k, replace=False))
        sink: Sink = RefSink(section, n, targets, full)
    else:
        z = np.load(outdir / f"{a.ref}.{section}.npz", allow_pickle=False)
        ref = {k: z[k] for k in z.files}
        if str(ref["corpus_digest"]) != C.corpus_digest(corpus):
            sys.exit("reference was built from a different corpus")
        sink = CandSink(section, n, targets, ref["full_rows"], ref=ref)
    if section == "prefill":
        run_prefill(lm, corpus, lay, sink, a.step, log)
    else:
        run_decode(lm, corpus, lay, sink, mode, serve, log)
    assert sink.done.all(), "records missing"
    wall = time.perf_counter() - t0
    if a.ref is None:
        np.savez_compressed(
            outdir / f"{a.tag}.{section}.npz",
            idx=sink.idx,
            lp=sink.lp,
            tgt=sink.tgt,
            full_rows=sink.full,
            full_lp=sink.full_lp,
            corpus_digest=np.array(C.corpus_digest(corpus)),
        )
        print(f"reference {a.tag}.{section} written ({wall:.0f}s)")
        return
    summary = summarize(sink, ref, lay, section)
    res = {
        "label": a.label or a.tag,
        "tag": a.tag,
        "ref": a.ref,
        "part": a.part,
        "config": a.config,
        "model": Path(a.model).name,
        "step": a.step,
        "setup": info,
        "wall_s": round(wall),
        "summary": summary,
    }
    (outdir / f"{a.tag}.{a.part}.json").write_text(
        json.dumps(res, indent=1, default=str)
    )
    np.savez_compressed(
        outdir / f"{a.tag}.{a.part}.perpos.npz",
        kld=sink.kld.astype(np.float32),
        agree=sink.agree,
        dp=sink.dp.astype(np.float32),
    )
    mx.clear_cache()
    print(json.dumps(res, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("--model", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--config", choices=("stock", "serve"), required=True)
    r.add_argument("--part", choices=sorted(PARTS), required=True)
    r.add_argument("--tag", required=True, help="name of the reference or the run")
    r.add_argument("--ref", help="reference tag to compare against (omit to write one)")
    r.add_argument("--label")
    r.add_argument("--step", type=int, default=C.PREFILL_STEP, help="prefill chunk")
    r.add_argument(
        "--driver", action="store_true", help="also apply lane_linear.convert"
    )
    p = sub.add_parser("report")
    p.add_argument("files", nargs="+")
    a = ap.parse_args()
    {"corpus": cmd_corpus, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
