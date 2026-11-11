"""Teacher-forced capture for the yv ``kld`` stage: one cold prefill per corpus window, top-K logprobs at every
position, from the engine of whatever source tree PYTHONPATH names.

  capture.py run --arm base|cand|ref --model DIR --cache-dir D --shard k/n [--env K=V ...] --out F.jsonl
  capture.py serve ARM <yunshu serve args>        (internal: the server process with the capture hooks)

Hooks (installed before the CLI starts, so any tree is measured with its own code): the final hidden state of
every prefill forward goes through the model's own ``lm_head`` in blocks, and the top-K ids / logprobs, the
logprob of the token that follows, and the input ids are kept per request. When the request's prompt phase
ends (``PromptProcessingBatch.generate``) the window is written to ``D/<window id>.npz``. The ``ref`` arm
replaces the QSA attention with dense fp32 attention over the same selected keys (slow, exact), for
Flash-Next (qwen4_exp); other architectures have no reference mode and fail closed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

K = 32
BLOCK = 128
H, KV, D, TOPK, QBLOCK = 24, 2, 256, 512, 4


def reference_attend(
    queries, keys, values, block_indices, query_ends, scale, rows_per_chunk=64
):
    """Dense fp32 attention over each row's selected blocks and tail block: ``[1, 24, T, 256]``."""
    import mlx.core as mx

    rows = int(queries.shape[2])
    bi = mx.sort(block_indices.astype(mx.int32), axis=-1).reshape(rows, TOPK)
    qe = query_ends.astype(mx.int32).reshape(rows)
    kf, vf, qf = (
        keys[0].astype(mx.float32),
        values[0].astype(mx.float32),
        queries[0].astype(mx.float32),
    )
    offs = mx.arange(QBLOCK, dtype=mx.int32)
    tail_offs = mx.arange(QBLOCK - 1, dtype=mx.int32)
    outs = []
    for r0 in range(0, rows, rows_per_chunk):
        r1 = min(rows, r0 + rows_per_chunk)
        b, e = bi[r0:r1], qe[r0:r1]
        ids_sel = (b[:, :, None] * QBLOCK + offs).reshape(r1 - r0, -1)
        valid_sel = mx.repeat(b >= 0, QBLOCK, axis=1)
        ids_tail = mx.minimum(
            ((e // QBLOCK) * QBLOCK)[:, None] + tail_offs, kf.shape[1] - 1
        )  # invalid tail slots stay inside the keys
        tail_valid = ((e // QBLOCK) * QBLOCK)[:, None] + tail_offs < e[:, None]
        ids = mx.concatenate([mx.maximum(ids_sel, 0), ids_tail], axis=1)
        valid = mx.concatenate([valid_sel, tail_valid], axis=1)
        heads = []
        for g in range(KV):
            k_c, v_c = kf[g][ids], vf[g][ids]
            q_c = qf[12 * g : 12 * g + 12, r0:r1].transpose(1, 0, 2)
            s = mx.where(
                valid[:, None, :],
                mx.matmul(q_c, k_c.transpose(0, 2, 1)) * scale,
                -mx.inf,
            )
            heads.append(mx.matmul(mx.softmax(s, axis=-1), v_c))
        o = mx.concatenate(heads, axis=1)
        mx.eval(o)
        outs.append(o)
    return mx.concatenate(outs, axis=0).transpose(1, 0, 2)[None].astype(queries.dtype)


def _eligible(queries, keys, values, block_indices, kw) -> bool:
    import mlx.core as mx

    return bool(
        kw.get("causal")
        and not kw.get("allow_sparse_decode")
        and isinstance(keys, mx.array)
        and queries.ndim == 4
        and queries.shape[0] == 1
        and queries.shape[1] == H
        and queries.shape[2] > 1
        and queries.shape[3] == D
        and keys.shape[:2] == (1, KV)
        and keys.shape[3] == D
        and block_indices.shape[-1] == TOPK
        and kw.get("block_size") == QBLOCK
    )


def detect_arch(model_dir: str) -> str:
    """The architecture family the capture hooks know; anything else fails closed."""
    config = json.loads((Path(model_dir) / "config.json").read_text())
    kind = config.get("model_type") or config.get("text_config", {}).get("model_type")
    from yunshu_engine.qwen4_family import is_qwen4_type

    if is_qwen4_type(kind):
        return "qwen4"
    if kind in ("qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"):
        return "qwen3_5"
    raise RuntimeError(f"kld capture has no hooks for model_type {kind!r}")


def upcast_attention(ref_calls: list[int]) -> None:
    """Reference mode for every architecture: standard attention accumulates in fp32."""
    import mlx.core as mx

    original = mx.fast.scaled_dot_product_attention

    def sdpa(queries, keys, values, *args, **kwargs):
        mask = kwargs.get("mask")
        if isinstance(mask, mx.array) and mask.dtype != mx.bool_:
            kwargs["mask"] = mask.astype(mx.float32)
        out = original(
            queries.astype(mx.float32),
            keys.astype(mx.float32),
            values.astype(mx.float32),
            *args,
            **kwargs,
        )
        ref_calls[0] += 1
        return out.astype(queries.dtype)

    mx.fast.scaled_dot_product_attention = sdpa


def install(arm: str, arch: str = "qwen4") -> None:
    import mlx.core as mx
    from mlx_vlm.generate.ar import PromptProcessingBatch

    if arch == "qwen4":
        from yunshu_engine.qwen4_family import qwen4_first_module

        l4 = qwen4_first_module("language")
        if l4 is None:
            raise RuntimeError("no mlx-vlm language module for the Qwen4 family")
    elif arch == "qwen3_5":
        from mlx_vlm.models.qwen3_5 import (
            language as l4,
        )  # same hook names: LanguageModel and the *Model body
    else:
        raise RuntimeError(f"no capture hooks for {arch}")
    body_class = l4.Qwen4ExpModel if arch == "qwen4" else l4.Qwen3_5Model

    outdir = Path(os.environ["KLD_CACHE_DIR"])
    expect_file = outdir / "_current.json"
    state: dict = {
        "lm": None,
        "tokens": [],
        "ids": [],
        "lp": [],
        "next": [],
        "carry": None,
        "active": False,
        "bad": False,
    }

    ref_calls = [
        0
    ]  # forwards the reference attention replaced (saved with each window)

    if arm == "ref":
        upcast_attention(ref_calls)
    if arm == "ref" and arch == "qwen4":
        original = l4.dispatch_qsa_attention

        def dispatch(queries, keys, values, block_indices, query_ends, **kw):
            if _eligible(queries, keys, values, block_indices, kw):
                ref_calls[0] += 1
                return reference_attend(
                    queries, keys, values, block_indices, query_ends, float(kw["scale"])
                )
            return original(queries, keys, values, block_indices, query_ends, **kw)

        l4.dispatch_qsa_attention = dispatch
    elif arm == "cand":
        pass  # the candidate's own flags arrive through the environment

    if os.environ.get("KLD_DECODE"):
        install_decode(arm, expect_file, outdir, ref_calls)
        return  # decode capture replaces the prefill hooks: the prompt phase is not recorded

    lm_call = l4.LanguageModel.__call__

    def language_model_call(self, *args, **kwargs):
        state["lm"] = self
        return lm_call(self, *args, **kwargs)

    l4.LanguageModel.__call__ = language_model_call
    model_call = body_class.__call__

    def record(inputs, hidden, head=None):
        if head is None:
            lm = state["lm"]
            head = getattr(lm, "lm_head", None) or lm.model.embed_tokens.as_linear
        tokens = inputs[0]
        rows = int(tokens.shape[0])
        tok_np = [int(x) for x in tokens.tolist()]
        for start in range(0, rows, BLOCK):
            stop = min(rows, start + BLOCK)
            logits = head(hidden[:, start:stop])[0].astype(mx.float32)  # [b, V]
            lse = mx.logsumexp(logits, axis=-1, keepdims=True)
            vocab = logits.shape[-1]
            idx = mx.argpartition(logits, kth=vocab - K, axis=-1)[:, vocab - K :]
            vals = mx.take_along_axis(logits, idx, axis=-1)
            order = mx.argsort(-vals, axis=-1)
            idx, vals = (
                mx.take_along_axis(idx, order, axis=-1),
                mx.take_along_axis(vals, order, axis=-1),
            )
            lp = vals - lse
            # logprob of the token that follows each position inside this block (the last one waits)
            nxt = mx.array(tok_np[start + 1 : stop] + [0], dtype=mx.int32)
            chosen = mx.take_along_axis(logits, nxt[:, None], axis=-1)[:, 0] - lse[:, 0]
            mx.eval(idx, lp, chosen, logits[-1:], lse[-1:])
            if (
                state["carry"] is not None
            ):  # the previous block's last position: its next token is our first
                first = int(tok_np[start])
                state["next"][-1] = float(
                    (state["carry"][0][first] - state["carry"][1]).item()
                )
                state["carry"] = None
            state["ids"].append(idx.astype(mx.int32))
            state["lp"].append(lp)
            state["next"].extend(float(x) for x in chosen.tolist()[:-1])
            state["next"].append(float("nan"))
            state["carry"] = (logits[-1], lse[-1, 0])
        state["tokens"].extend(tok_np)

    def qwen4_model_call(self, inputs, *args, **kwargs):
        out = model_call(self, inputs, *args, **kwargs)
        if (
            state["lm"] is not None
            and not state.get(
                "in_driver"
            )  # the driver hook records this forward itself
            and inputs.shape[0] == 1
            and expect_file.exists()
        ):
            state["active"] = True
            record(inputs, out)
        return out

    body_class.__call__ = qwen4_model_call

    def write_window() -> None:
        import numpy as np

        wid = json.loads(expect_file.read_text())["wid"]
        ids = np.concatenate([np.array(a) for a in state["ids"]], axis=0)
        lp = np.concatenate([np.array(a) for a in state["lp"]], axis=0).astype(
            np.float32
        )
        tmp = outdir / f"{wid}.tmp.npz"
        np.savez(
            tmp,
            ids=ids.astype(np.int32),
            lp=lp,
            next_lp=np.array(state["next"], dtype=np.float32),
            tokens=np.array(state["tokens"], dtype=np.int32),
            ref_calls=np.int64(ref_calls[0]),
        )
        ref_calls[0] = 0
        tmp.rename(outdir / f"{wid}.npz")
        expect_file.unlink()

    def reset() -> None:
        state.update(
            tokens=[], ids=[], lp=[], next=[], carry=None, active=False, bad=False
        )

    generate = PromptProcessingBatch.generate

    def finish(self, *args, **kwargs):
        result = generate(self, *args, **kwargs)
        if state["active"]:
            write_window()
            reset()
        return result

    PromptProcessingBatch.generate = finish
    install_driver(state, record, write_window, reset, expect_file)


def topk_record(logits, forced_next: int | None = None):
    """Top-K ids / logprobs of one logits row ``[1, V]`` and the logprob of ``forced_next`` (NaN: none)."""
    import mlx.core as mx

    lg = logits.reshape(1, -1).astype(mx.float32)
    lse = mx.logsumexp(lg, axis=-1, keepdims=True)
    vocab = lg.shape[-1]
    idx = mx.argpartition(lg, kth=vocab - K, axis=-1)[:, vocab - K :]
    vals = mx.take_along_axis(lg, idx, axis=-1)
    order = mx.argsort(-vals, axis=-1)
    idx = mx.take_along_axis(idx, order, axis=-1)
    vals = mx.take_along_axis(vals, order, axis=-1)
    lp = vals - lse
    chosen = (
        lg[0, forced_next] - lse[0, 0]
        if forced_next is not None
        else lse[0, 0] * mx.nan
    )
    mx.eval(idx, lp, chosen)
    return idx[0].astype(mx.int32), lp[0], float(chosen.item())


def reference_decode_attend(
    queries, keys, values, lengths, scale, slots, ref_calls=None
):
    """Dense fp32 attention for the decode window: token t of a row sees its first ``length - T + t + 1`` keys."""
    import mlx.core as mx

    T = int(queries.shape[2])
    span = int(lengths.max().item())
    ks = mx.take(keys, slots, axis=0)[..., :span, :].astype(mx.float32)
    vs = mx.take(values, slots, axis=0)[..., :span, :].astype(mx.float32)
    visible = lengths[:, None] - T + mx.arange(T)[None, :] + 1  # [B, T]
    mask = mx.arange(span)[None, None, :] < visible[:, :, None]
    out = mx.fast.scaled_dot_product_attention(
        queries.astype(mx.float32), ks, vs, scale=scale, mask=mask[:, None, :, :]
    )
    if ref_calls is not None:
        ref_calls[0] += 1
    return out.astype(queries.dtype)


class DecodeTap:
    """Teacher-forced decode capture: the window's forced tokens replace every draw, and the logits of each
    single-token decode step are recorded. A step that is not one row at one position (a speculative
    window, a second request) marks the window bad: nothing is written and the run fails closed."""

    def __init__(self, expect_file: Path, outdir: Path, ref_calls: list[int]):
        self.expect_file, self.outdir, self.ref_calls = expect_file, outdir, ref_calls
        self.reset()

    def reset(self) -> None:
        self.wid: str | None = None
        self.forced: list[int] = []
        self.drawn = 0
        self.ids: list = []
        self.lp: list = []
        self.next_lp: list[float] = []
        self.tokens: list[int] = []
        self.bad = ""

    def _arm(self) -> bool:
        if not self.expect_file.exists():
            return False
        if self.wid is None:
            doc = json.loads(self.expect_file.read_text())
            self.wid, self.forced = doc["wid"], [int(t) for t in doc["forced"]]
        return True

    def on_draw(self, kinds: list[str], logits, tokens):
        """One ``_draw`` call: returns the token array to use in place of the sampled one."""
        import mlx.core as mx

        if not self._arm() or self.bad:
            return tokens
        if kinds != ["p"] and kinds != ["d"]:
            self.bad = f"draw of {kinds} is not one row at one position"
            return tokens
        if logits is None or logits.size != logits.shape[-1] or int(tokens.size) != 1:
            self.bad = "decode window wider than one token (speculative decode on?)"
            return tokens
        i = self.drawn
        if i >= len(self.forced):
            return tokens
        if kinds == ["d"]:
            if i == 0:
                self.bad = "decode step before the prompt's first draw"
                return tokens
            ids, lp, nxt = topk_record(logits, self.forced[i])
            self.ids.append(ids)
            self.lp.append(lp)
            self.next_lp.append(nxt)
            self.tokens.append(self.forced[i - 1])
        self.drawn += 1
        out = mx.array([self.forced[i]], dtype=mx.int32)
        if len(self.tokens) == len(self.forced) - 1:
            self.write()  # resets the tap
        return out

    def write(self) -> None:
        import numpy as np

        tmp = self.outdir / f"{self.wid}.tmp.npz"
        np.savez(
            tmp,
            ids=np.stack([np.array(a) for a in self.ids]).astype(np.int32),
            lp=np.stack([np.array(a) for a in self.lp]).astype(np.float32),
            next_lp=np.array(self.next_lp, dtype=np.float32),
            tokens=np.array(self.tokens, dtype=np.int32),
            ref_calls=np.int64(self.ref_calls[0]),
        )
        self.ref_calls[0] = 0
        tmp.rename(self.outdir / f"{self.wid}.npz")
        self.expect_file.unlink()
        self.reset()


def install_decode(arm: str, expect_file: Path, outdir: Path, ref_calls) -> None:
    """Hook the round driver's ``_logits`` / ``_draw`` (the one place every decode path draws) and, for
    the reference arm, the ragged decode kernel (dense fp32 attention over the same cache)."""
    from yunshu_engine.kernels import ragged_attention as ra
    from yunshu_engine.round_driver import driver as rd

    tap = DecodeTap(expect_file, outdir, ref_calls)
    if arm == "ref":
        decode = ra.ragged_decode_attention

        def dense(queries, keys, values, lengths, scale, *a, **kw):
            if kw.get("k_scales") is not None or kw.get("slots") is None:
                return decode(queries, keys, values, lengths, scale, *a, **kw)
            return reference_decode_attend(
                queries, keys, values, lengths, scale, kw["slots"], ref_calls
            )

        ra.ragged_decode_attention = dense
    logits_fn, draw_fn = rd.RoundDriver._logits, rd.RoundDriver._draw
    seen: dict = {}

    def logits_hook(self, hidden):
        out = logits_fn(self, hidden)
        seen["logits"] = out
        return out

    def draw_hook(self, items, hidden, target_logits=None):
        seen["logits"] = target_logits
        draws = draw_fn(self, items, hidden, target_logits)
        if len(draws) == 1:
            it, tok, extra = draws[0]
            return [(it, tap.on_draw([it.kind], seen.get("logits"), tok), extra)]
        if draws:
            tap.on_draw(["many"] * len(draws), None, draws[0][1])
        return draws

    rd.RoundDriver._logits = logits_hook
    rd.RoundDriver._draw = draw_hook


def _cache_offset(cache) -> int | None:
    """Tokens already in a row's attention cache (None: no cache exposes an offset)."""
    for layer in cache:
        off = getattr(layer, "offset", None)
        if isinstance(off, int):
            return off
    return None


def install_driver(state, record, write_window, reset, expect_file) -> None:
    """The round driver prefills through ``lane_forward`` (one segment per step) instead of the
    upstream prefill: each segment's hidden states go through the driver's own head, in prompt order,
    and the window is written when ``_prefill_step`` emits the row's first token (a one-token request).
    A segment that does not continue the recorded prompt (an APC-restored prefix, a second request)
    marks the window bad: nothing is written, so the run fails closed."""
    try:
        from yunshu_engine.round_driver import driver as rd
        from yunshu_engine.round_driver import forward as fwd
    except ImportError:
        return  # a tree without the round driver: the upstream hooks above are the whole path

    def hooked(forward_fn, head_of):
        """Wrap a segment forward ``(lm, seg, **kw) -> hidden`` so each prompt segment is recorded."""

        def run(lm, seg, **kwargs):
            offset = _cache_offset(seg.cache) if expect_file.exists() else None
            state["in_driver"] = (
                True  # Flash-Next's prefill calls the (hooked) model body inside
            )
            try:
                out = forward_fn(lm, seg, **kwargs)
            finally:
                state["in_driver"] = False
            if expect_file.exists():
                if offset != len(
                    state["tokens"]
                ):  # a restored prefix, or an interleaved row
                    state["bad"] = True
                elif not state["bad"]:
                    state["active"] = (
                        True  # state["lm"] stays unset: the body hook must not record too
                    )
                    record(seg.tokens[None], out[None], head=head_of(lm))
            return out

        return run

    rd.lane_forward = hooked(
        rd.lane_forward, lambda lm: lambda h: fwd.logits(lm, h[0])[None]
    )
    # Flash-Next (Qwen4 family) prefills through flashnext.prefill, not lane_forward: without this hook
    # no window is ever recorded ("the capture hook wrote no window").
    from yunshu_engine.round_driver import flashnext as fn

    fn.prefill = hooked(fn.prefill, lambda lm: lambda h: fn.logits(lm, h[0])[None])
    prefill_step = rd.RoundDriver._prefill_step

    def hooked_prefill_step(self, *args, **kwargs):
        events = prefill_step(self, *args, **kwargs)
        if state["active"] or state["bad"]:
            if events:  # the first token left: the prompt phase is over
                if state["active"] and not state["bad"]:
                    write_window()
                reset()
        return events

    rd.RoundDriver._prefill_step = hooked_prefill_step


def decode_split(tokenizer, text: str, steps: int) -> tuple[str, list[int]]:
    """The window as (prefix text to prefill, the ``steps + 1`` reference tokens to force one at a time)."""
    ids = [int(t) for t in tokenizer(text)["input_ids"]]
    if len(ids) < steps + 2:
        raise RuntimeError(f"window of {len(ids)} tokens is shorter than {steps} steps")
    cut = len(ids) - steps - 1
    return tokenizer.decode(ids[:cut]), ids[cut:]


def ask(url: str, model: str, text: str, max_tokens: int = 1) -> tuple[int, float]:
    """One completion of ``max_tokens`` tokens; (prompt tokens, wall seconds)."""
    import urllib.error
    import urllib.request

    body = dict(model=model, prompt=text, max_tokens=max_tokens, temperature=0)
    req = urllib.request.Request(
        url + "/v1/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer k"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            reply = json.load(r)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"/v1/completions {exc.code}: {exc.read()[:300]!r}") from exc
    return int((reply.get("usage") or {})["prompt_tokens"]), time.perf_counter() - t0


def serve_command(command, arm: str, arch: str):
    """The harness's server command with this script's hooked server in its place. The harness launches
    ``<python> -m <cli module> serve ...`` (or the legacy console script): both are rewritten, otherwise
    no hook is installed and no window is ever written."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research"))
    from bench_engines import YUNSHU_MODULE, is_yunshu_command, yunshu_args

    if command and is_yunshu_command(command):
        return [
            command[0]
            if list(command[1:3]) == ["-m", YUNSHU_MODULE]
            else sys.executable,
            str(Path(__file__).resolve()),
            "serve",
            arm,
            arch,
            *yunshu_args(command),
        ]
    return command


def run(args) -> None:
    import subprocess

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research"))
    import tfbench as t

    from . import corpus

    arch = detect_arch(args.model)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["KLD_CACHE_DIR"] = str(cache)
    k, n = (int(x) for x in args.shard.split("/"))
    windows = corpus.load_texts()[k::n]
    todo = [w for w in windows if not (cache / f"{w['id']}.npz").exists()]
    popen = subprocess.Popen
    steps = int(getattr(args, "decode_steps", 0) or 0)
    tokenizer = None
    if steps and todo:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)

    def launch(command, *a, **kw):
        return popen(serve_command(command, args.arm, arch), *a, **kw)

    out = Path(args.out)
    rows = []
    if todo:
        subprocess.Popen = launch
        env = {
            "YUNSHU_VLM_DRAFT": "off",
            "YUNSHU_VLM_APC_DISK": "0",
            "YUNSHU_VLM_APC_MEMORY_GB": "0",
            "YUNSHU_NEXT_TURN_PREFILL": "off",
            "YUNSHU_CONSOLE": "0",
            "KLD_CACHE_DIR": str(cache),
            **({"KLD_DECODE": "1"} if steps else {}),
            # the reference must not use Yunshu's tile attention kernel: stock attention is what gets upcast
            # the engine's tensor-unit sparse prefill wraps dispatch_qsa_attention at model load, outside
            # the reference hook, and would answer every eligible call itself (ref == base, KLD 0)
            **(
                {
                    "YUNSHU_PREFILL_ATTENTION": "stock",
                    "YUNSHU_QSA_SPARSE_PREFILL": "0",
                }
                if args.arm == "ref"
                else {}
            ),
            **dict(kv.split("=", 1) for kv in args.env),
        }
        srv = t.Srv("yunshu", env, f"kld-{args.arm}-{k}of{n}", model=args.model)
        try:
            t.send(srv.url, t.req(srv.model, "Say hi.", 8))
            for w in todo:
                text, forced = w["text"], []
                if steps:
                    text, forced = decode_split(tokenizer, w["text"], steps)
                (cache / "_current.json").write_text(
                    json.dumps(dict(wid=w["id"], forced=forced))
                )
                tokens, wall = ask(
                    srv.url, srv.model, text, len(forced) if steps else 1
                )
                if not (cache / f"{w['id']}.npz").exists():
                    raise RuntimeError(
                        f"{w['id']}: the capture hook wrote no window (prompt_tokens={tokens})"
                    )
                rows.append(
                    dict(id=w["id"], prompt_tokens=tokens, wall_s=round(wall, 2))
                )
                print(json.dumps(rows[-1]), flush=True)
        finally:
            srv.kill()
            (cache / "_current.json").unlink(missing_ok=True)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        fh.write(
            json.dumps(dict(complete=True, windows=len(windows), captured=len(rows)))
            + "\n"
        )


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["serve"]:
        argv.pop(0)
        arm = argv.pop(0)
        arch = argv.pop(0)
        if os.environ.get(
            "KLD_PREFILL_STEP"
        ):  # noise measurement: the same arithmetic with another chunking
            from yunshu_engine import memory_admission, vlm_batch_runner

            vlm_batch_runner.PREFILL_STEP = memory_admission.PREFILL_STEP = int(
                os.environ["KLD_PREFILL_STEP"]
            )
        install(arm, arch)
        from yunshu_cli import main as cli

        sys.argv = [sys.argv[0], *argv]
        cli()
        return 0
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=["base", "cand", "ref"], required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--cache-dir", required=True)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--env", action="append", default=[])
    r.add_argument(
        "--decode-steps",
        type=int,
        default=0,
        help="teacher-forced decode capture of this many T=1 steps after the window's prefix",
    )
    r.add_argument("--out", required=True)
    a = p.parse_args(argv)
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
