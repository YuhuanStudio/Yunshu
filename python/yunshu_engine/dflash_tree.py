# Upstream (inspired): ashhart/TensorFold (MIT) src/tensorfold/drafters/dflash_tree.py @ 34bae79a
# Patches upstream mlx-vlm symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""DFlash2 draft trees for the single-request speculative lane.

DFlash2 proposes, for a block of positions after the pending token, the top-K
candidate tokens per position (``candidate_selector``) with a unary score and a
pairwise (predecessor -> successor) score. Upstream's greedy round follows one
argmax path (a chain). This module keeps the whole lattice: the ``r``-th best
candidate under ``(unary + 0.6 * pairwise) / 1.5`` given the parent's token fills
each node of a fixed-shape draft *tree* (``spec_topology``), on the GPU, and
``tree_verify`` checks the tree in one forward. The walk keeps the longest root
path whose tokens equal the target's greedy tokens, so the output is the plain
greedy decode's (``tree_verify`` gives every row single-step arithmetic).

The lattice scoring follows TensorFold's draft-tree search
(``drafters/dflash_tree.py``, MIT): per-position scores ``unary + edge *
pairwise`` softened by ``tau``. TensorFold grows the tree best-first on the CPU;
here the shape is fixed (a rank-path topology) so no host round trip sits
between the drafter and the verify.
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import mlx.core as mx

from . import tree_verify as tv
from .spec_schedule import NodeBudget

logger = logging.getLogger(__name__)

CHILDREN = 4  # candidates expanded under each node
BEAM = 15  # paths kept per level (the tree has at most NODES nodes)
EDGE = 0.6  # weight of the pairwise score
TAU = 1.5  # softmax temperature of the node scores


def walk(window: list[int], parents: list[int], target: list[int]) -> list[int]:
    """Rows of the accepted path (root first): follow the child whose token is
    the target's greedy token after the current row."""
    children: dict[int, list[int]] = {}
    for row in range(1, len(window)):
        children.setdefault(parents[row], []).append(row)
    path = [0]
    while True:
        want = target[path[-1]]
        nxt = next((c for c in children.get(path[-1], ()) if window[c] == want), None)
        if nxt is None:
            return path
        path.append(nxt)


class GpuLattice:
    """A drafter forward's candidates and scores, still on the GPU."""

    __slots__ = ("cands", "unary", "hproj", "succ", "pred", "anchor")

    def __init__(self, cands, unary, hproj, succ, pred, anchor):
        self.cands, self.unary, self.hproj = cands, unary, hproj
        self.succ, self.pred, self.anchor = succ, pred, anchor


TOP_K_CHUNK = 1024


def top_k_ids(logits: mx.array, k: int, chunk: int = TOP_K_CHUNK) -> mx.array:
    """Indices of each row's ``k`` largest values (``[D, n]`` -> ``[D, k]``, any order).

    Two stages: the ``k`` best of every ``chunk``-wide slice, then the ``k`` best of those.
    One argpartition over 65536 ids cost 1.09 ms for 15 rows; this 0.27 ms, same set (M5 Max).
    A draft candidate only changes what is verified, never the output."""
    d, n = logits.shape
    if n <= chunk * 2:
        return mx.argpartition(logits, -k, axis=-1)[..., -k:]
    pad = -n % chunk
    if pad:
        logits = mx.concatenate(
            [logits, mx.full((d, pad), -float("inf"), dtype=logits.dtype)], axis=-1
        )
    c = logits.shape[-1] // chunk
    x = logits.reshape(d, c, chunk)
    local = mx.argpartition(x, -k, axis=-1)[..., -k:]  # [D, C, k] within each slice
    values = mx.take_along_axis(x, local, axis=-1).reshape(d, c * k)
    ids = (local + (mx.arange(c, dtype=local.dtype) * chunk)[None, :, None]).reshape(
        d, c * k
    )
    best = mx.argpartition(values, -k, axis=-1)[..., -k:]
    return mx.take_along_axis(ids, best, axis=-1)


def compute_lattice_gpu(
    drafter,
    anchor: int,
    hidden: mx.array,
    cache,
    positions: int,
    *,
    keyed=None,
    committed=0,
) -> GpuLattice:
    """``compute_lattice`` without the host round trip (nothing is evaluated)."""
    inputs = mx.array(
        [[int(anchor)] + [int(drafter.config.mask_token_id)] * positions],
        dtype=mx.int32,
    )
    from . import tree_verify as tv

    prof = tv._profile_on()
    t0 = 0.0
    if prof:
        import time

        mx.eval(inputs, hidden)
        from . import gpu_window

        gpu_window.stage_begin()
        t0 = time.perf_counter()
    if prof:
        # the vendor ``_hidden`` loop, with a counted stage after each piece
        h = drafter._embed_input_tokens(inputs)
        t0 = tv._tick("drafter_embed_ms", h, t0)
        h_ctx = drafter.hidden_norm(drafter.fc(hidden))
        t0 = tv._tick("drafter_fc_ms", h_ctx, t0)
        projected = drafter._project_context_kv(h_ctx)
        t0 = tv._tick(
            "drafter_ctxkv_ms", [h_ctx] if projected is None else projected, t0
        )
        for index, (layer, c) in enumerate(zip(drafter.layers, cache, strict=True)):
            layer_ctx = None if projected is None else projected[index]
            h = layer(h, h_ctx, drafter.rope, c, layer_ctx)
            t0 = tv._tick(f"drafter_layer{index}_ms", h, t0)
        dh = drafter.norm(h)[:, 1:]
    else:
        dh = drafter._hidden(inputs, hidden, cache)[:, 1:]
    if prof:
        t0 = tv._tick("drafter_layers_ms", dh, t0)
    return lattice_from_hidden(
        drafter,
        dh,
        mx.array([int(anchor)]),
        prof=prof,
        t0=t0,
        keyed=keyed,
        committed=committed,
    )


def lattice_from_hidden(
    drafter, dh, anchor, *, prof=False, t0=0.0, keyed=None, committed=0
) -> GpuLattice:
    """The lattice from the proposal rows' hidden states ``dh`` [1, D, H]; ``anchor`` is the
    round's bonus token as a [1] array (it may still be on the GPU)."""
    from . import tree_verify as tv

    sel = drafter.candidate_selector
    vocab = getattr(drafter, "_draft_vocab", None)
    subset = (
        vocab.logits_subset(dh)
        if keyed is None
        and vocab is not None
        and drafter.config.final_logit_softcapping is None
        else None
    )
    if subset is not None:
        logits, ids = subset  # [D, n] over the searched ids
        if prof:
            t0 = tv._tick("drafter_head_ms", logits, t0)
        part = top_k_ids(logits, sel.top_k)
        cands = mx.take(ids, part)  # [D, K]
        unary = mx.take_along_axis(logits, part, axis=-1).astype(mx.float32)
    else:
        logits = drafter._logits(dh)
        if keyed is not None:
            from .constrained_spec import normalize_rows
            from .keyed_sampling import coupled_scores

            # Perturb the complete vocabulary before selecting candidates. A token
            # omitted by a greedy shortlist can be the keyed winner.
            positions = mx.arange(int(dh.shape[1])) + committed
            logits = (TAU * coupled_scores(normalize_rows(logits), keyed, positions))[
                None
            ]
        if prof:
            t0 = tv._tick("drafter_head_ms", logits, t0)
        cands = top_k_ids(logits[0], sel.top_k)
        unary = mx.take_along_axis(logits[0], cands, axis=-1).astype(mx.float32)
    if prof:
        t0 = tv._tick("drafter_topk_ms", cands, t0)
    hproj = sel.hidden_projection(dh)[0].astype(mx.float32)
    succ = sel.successor_codebook(cands).astype(mx.float32)  # [D, K, R]
    pred = sel.predecessor_codebook(cands).astype(mx.float32)
    anchor_row = sel.predecessor_codebook(anchor.reshape(1))[0].astype(mx.float32)
    if keyed is not None:
        # Candidate ranking must remain the coupled distribution's ranking. Pairwise
        # calibration was trained for greedy proposals, not Gumbel-perturbed scores.
        hproj = mx.zeros_like(hproj)
    if prof:
        tv._tick("drafter_selector_ms", (unary, hproj, succ, pred, anchor_row), t0)
    return GpuLattice(cands, unary, hproj, succ, pred, anchor_row)


_SEARCH_COMPILED: dict = {}


def search_tree(
    lat: GpuLattice,
    nodes: int,
    children: int = CHILDREN,
    beam: int = BEAM,
    edge: float = EDGE,
    tau: float = TAU,
):
    """``_search_tree`` compiled once (about 150 small kernels per round fused where MLX can:
    1.67 -> 1.14 ms for 15 depths on M5 Max, same tokens and parents).

    It always searches ``NODES`` and keeps the first ``nodes``: a node's ancestors rank above
    it, so the best ``nodes`` and their parent rows are the same as a ``nodes`` search's. One
    compiled shape: a per-``nodes`` compile cost the requests that first met each width."""
    full = max(int(nodes), NODES)
    key = (full, int(children), int(beam), float(edge), float(tau))
    fn = _SEARCH_COMPILED.get(key)
    if fn is None:

        def run(cands, unary, hproj, succ, pred, anchor):
            return _search_tree(
                GpuLattice(cands, unary, hproj, succ, pred, anchor), *key
            )

        fn = _SEARCH_COMPILED[key] = mx.compile(run)
    tokens, parents = fn(
        lat.cands, lat.unary, lat.hproj, lat.succ, lat.pred, lat.anchor
    )
    if full == nodes:
        return tokens, parents
    return tokens[:nodes], parents[: nodes + 1]


def _search_tree(
    lat: GpuLattice,
    nodes: int,
    children: int = CHILDREN,
    beam: int = BEAM,
    edge: float = EDGE,
    tau: float = TAU,
):
    """Best-first draft tree on the GPU: the ``nodes`` most probable prefixes of
    the lattice by summed log-probability, ``(unary + edge * pairwise) / tau``
    normalized per position (each node expands to its ``children`` best
    candidates). A level keeps its ``beam`` best paths (enough to contain every
    member of the global top ``nodes``, whose ancestors all rank above them).

    Returns ``(tokens [n], parents [n + 1])``: tokens in decreasing path
    probability, parents in window rows (row 0 = the pending token, parent -1),
    every parent before its children so any prefix is a tree. Nothing is read
    back, and the shape depends on the data: use it with ``DynamicShape``."""
    depth_count = int(lat.cands.shape[0])
    neg = -1e9
    floor = -1e-4  # strictly decreasing along a path: parents sort before children

    def log_probs(scores):
        lp = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
        return mx.minimum(lp, floor)

    root = (lat.unary[0] + edge * ((lat.anchor * lat.hproj[0]) @ lat.succ[0].T)) / tau
    lp = log_probs(root)
    order = mx.argsort(-lp)[:children]
    cum = mx.concatenate([mx.take(lp, order), mx.full((beam - children,), neg)])
    cidx = mx.concatenate(
        [order.astype(mx.int32), mx.zeros((beam - children,), mx.int32)]
    )
    tok = [mx.take(lat.cands[0], cidx)]
    par = [mx.full((beam,), -1, dtype=mx.int32)]
    tot = [cum]
    for d in range(1, depth_count):
        pr = mx.take(lat.pred[d - 1], cidx, axis=0)  # [B, R]
        scores = (
            lat.unary[d][None] + edge * ((pr * lat.hproj[d][None]) @ lat.succ[d].T)
        ) / tau
        lp = log_probs(scores)
        order = mx.argsort(-lp, axis=-1)[:, :children]  # [B, C]
        flat = (cum[:, None] + mx.take_along_axis(lp, order, axis=1)).reshape(-1)
        top = mx.argsort(-flat)[:beam]
        cum = mx.take(flat, top)
        cidx = mx.take(order.reshape(-1), top).astype(mx.int32)
        tok.append(mx.take(lat.cands[d], cidx))
        par.append((d - 1) * beam + (top // children).astype(mx.int32))
        tot.append(cum)
    pool_tok = mx.concatenate(tok).astype(mx.int32)
    pool_par = mx.concatenate(par)
    pool_cum = mx.concatenate(tot)
    sel = mx.argsort(-pool_cum)[:nodes].astype(mx.int32)
    where = mx.full((int(pool_cum.shape[0]),), -1, dtype=mx.int32)
    where[sel] = mx.arange(nodes, dtype=mx.int32)
    parent = mx.take(pool_par, sel)
    rows = mx.where(parent < 0, 0, mx.take(where, mx.maximum(parent, 0)) + 1)
    return mx.take(pool_tok, sel), mx.concatenate(
        [mx.array([-1], dtype=mx.int32), rows.astype(mx.int32)]
    )


def quantize_drafter(drafter: Any, bits: int = 8, group_size: int = 64) -> int:
    """Quantize the drafter's projections in place (before ``bind``): drafts are
    verified, so this changes only how often they land, and the drafter reads
    ``bits / 16`` of the bytes per cycle. Returns the layers converted."""
    import mlx.nn as nn

    before = sum(
        1 for _, m in drafter.named_modules() if isinstance(m, nn.QuantizedLinear)
    )
    nn.quantize(
        drafter,
        group_size=group_size,
        bits=bits,
        class_predicate=lambda _p, m: (
            isinstance(m, nn.Linear) and m.weight.shape[-1] % group_size == 0
        ),
    )
    mx.eval(drafter.parameters())
    return (
        sum(1 for _, m in drafter.named_modules() if isinstance(m, nn.QuantizedLinear))
        - before
    )


def _stack_linears(first: Any, second: Any) -> Any:
    """One projection computing ``first(x) | second(x)`` (outputs concatenated) for linear
    layers of equal bits and group size; each output column is the separate layer's."""
    import mlx.nn as nn

    out: Any
    if isinstance(first, nn.QuantizedLinear) and isinstance(second, nn.QuantizedLinear):
        if (first.bits, first.group_size, first.mode) != (
            second.bits,
            second.group_size,
            second.mode,
        ):
            return None
        out = nn.QuantizedLinear(
            first.weight.shape[1] * 32 // first.bits, 1, bias=False
        )
        out.group_size, out.bits, out.mode = first.group_size, first.bits, first.mode
        out.weight = mx.concatenate(
            [mx.array(first.weight), mx.array(second.weight)], axis=0
        )
        out.scales = mx.concatenate([first.scales, second.scales], axis=0)
        if "biases" in first and "biases" in second:
            out.biases = mx.concatenate(
                [first.biases, second.biases],  # type: ignore[list-item]
                axis=0,
            )
        return out
    if isinstance(first, nn.Linear) and isinstance(second, nn.Linear):
        out = nn.Linear(first.weight.shape[1], 1, bias=False)
        out.weight = mx.concatenate([first.weight, second.weight], axis=0)
        return out
    return None


def _stacked_mlp(mlp: Any, gate_up: Any) -> Any:
    """The drafter's gated MLP with gate and up stacked (one matmul over the input)."""
    import mlx.nn as nn
    from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import swiglu

    class StackedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up = gate_up
            self.down_proj = mlp.down_proj

        def __call__(self, x):
            both = self.gate_up(x)
            half = both.shape[-1] // 2
            return self.down_proj(swiglu(both[..., :half], both[..., half:]))

    return StackedMLP()


def compiled_layer_class():
    """DFlash2 decoder layer whose fixed-shape pre/post halves are ``mx.compile`` d, the way
    TensorFold compiles its drafter (``dflash_proposer``). The growing attention and the MLP stay
    eager; the compiled halves are the same ops grouped differently, so proposals must stay
    bit-identical (checked by tests and by the corpus A/B's round counts)."""
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DecoderLayer

    class CompiledDecoderLayer(DFlash2DecoderLayer):
        def _compiled(self):
            fns = self.__dict__.get("_yunshu_compiled")
            if fns is None:

                def pre(x):
                    return self.attention_conv.prepare(self.input_layernorm(x))

                def mid(residual, attended, kernel):
                    x = residual + self.attention_conv.finish(attended, kernel)
                    prepared, kernel2 = self.mlp_conv.prepare(
                        self.post_attention_layernorm(x)
                    )
                    return x, prepared, kernel2

                def post(x, mlp_out, kernel2):
                    return x + self.mlp_conv.finish(mlp_out, kernel2)

                fns = (mx.compile(pre), mx.compile(mid), mx.compile(post))
                self.__dict__["_yunshu_compiled"] = fns
            return fns

        def __call__(self, x, x_ctx, rope, cache, projected_context=None):
            pre, mid, post = self._compiled()
            prepared, kernel = pre(x)
            attended = self.self_attn(prepared, x_ctx, rope, cache, projected_context)
            x1, prepared2, kernel2 = mid(x, attended, kernel)
            return post(x1, self.mlp(prepared2), kernel2)

    return CompiledDecoderLayer


def compile_layers(view) -> int:
    """Switch the private view's DFlash2 decoder layers to the compiled-halves class."""
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DecoderLayer

    cls = compiled_layer_class()
    count = 0
    for layer in getattr(view, "layers", []):
        if type(layer) is DFlash2DecoderLayer:
            layer.__class__ = cls
            count += 1
    return count


def share_masks() -> bool:
    """Build each attention mask once per round instead of once per drafter layer.

    The five layers of a round share (rows, context length, window): the mask is a pure
    function of those, so later layers reuse the first one's array (TensorFold keeps the
    same per-round mask table in its ``_dflash_attend``). Idempotent."""
    from mlx_vlm.speculative.drafters.qwen3_dflash import dflash as vendor

    if getattr(vendor.create_causal_mask, "_yunshu_shared", False):
        return True
    original = vendor.create_causal_mask
    table: dict = {}

    def shared(n, offset=0, window_size=None, *args, **kwargs):
        if args or kwargs:
            return original(n, offset, window_size, *args, **kwargs)
        key = (int(n), int(offset), window_size)
        mask = table.get(key)
        if mask is None:
            if len(table) >= 32:
                table.clear()
            mask = table[key] = original(n, offset=offset, window_size=window_size)
        return mask

    shared._yunshu_shared = True  # type: ignore[attr-defined]
    vendor.create_causal_mask = shared
    return True


def stack_projections(drafter: Any) -> int:
    """Stack each layer's K|V and gate|up projections into single matmuls (ported from
    TensorFold ``lane_fuse``: attn_kv, mlp_gate_up). Drafts only propose, so this changes
    speed and never a token. Returns the groups stacked."""
    count = 0
    for layer in drafter.layers:
        attn = layer.self_attn
        kv = _stack_linears(attn.k_proj, attn.v_proj)
        if kv is not None:

            def project(x, _kv=kv):
                both = _kv(x)
                half = both.shape[-1] // 2
                return both[..., :half], both[..., half:]

            object.__setattr__(attn, "_project_kv", project)
            object.__setattr__(attn, "_stacked_kv", kv)
            attn.k_proj = attn.v_proj = None  # the stacked copy owns the bytes
            mx.eval(kv.parameters())
            count += 1
        gu = _stack_linears(layer.mlp.gate_proj, layer.mlp.up_proj)
        if gu is not None:
            layer.mlp = _stacked_mlp(layer.mlp, gu)
            count += 1
    mx.eval(drafter.parameters())
    return count


def supported(model: Any, draft_model: Any) -> bool:
    lm = model.language_model if hasattr(model, "language_model") else model
    return hasattr(draft_model, "candidate_selector") and tv.supported(lm)


NODES = 15  # draft nodes at most per round (window = pending token + nodes <= 16 rows)
# landing probability of the i-th best node before the round history says otherwise
PRIOR = [
    0.75,
    0.5,
    0.35,
    0.25,
    0.2,
    0.15,
    0.1,
    0.1,
    0.08,
    0.08,
    0.06,
    0.06,
    0.05,
    0.05,
    0.05,
]
POSITIONS = 7  # masked positions the drafter fills (its trained block minus one)


def dflash_tree_rounds(
    model,
    draft_model,
    prompt_cache,
    hidden,
    *,
    first_bonus,
    max_tokens,
    sampler,
    draft_block_size=None,
    token_dtype=mx.int32,
    use_model_initial_block_size=True,
    greedy_sampling=True,
    _original=None,
):
    """Drop-in for upstream ``_dflash_rounds`` (single row): tree drafts."""
    from . import mtp_lane
    from .constrained_spec import current_request
    from .keyed_sampling import KeyedSampler, tree_supported, tree_target_rows
    from .spec_metrics import record as _record_speculative_round

    keyed = sampler if isinstance(sampler, KeyedSampler) else None
    request = current_request()

    lm = model.language_model if hasattr(model, "language_model") else model
    if (
        not tree_supported(sampler, greedy_sampling)
        or mtp_lane._STATE["guide"] is not None
        or callable(getattr(draft_model, "prepare_target_hidden", None))
        or not supported(model, draft_model)
        or not tv.lane_ready(lm, prompt_cache)
    ):
        yield from _original(
            model,
            draft_model,
            prompt_cache,
            hidden,
            first_bonus=first_bonus,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=draft_block_size,
            token_dtype=token_dtype,
            use_model_initial_block_size=use_model_initial_block_size,
            greedy_sampling=greedy_sampling,
        )
        return
    target_ids = list(draft_model.config.target_layer_ids)
    draft_cache = draft_model.reset(model)
    budget = NodeBudget(NODES, prior=PRIOR)
    root_shape = tv.TreeShape([-1])
    b = int(first_bonus)
    emitted = 1
    while emitted < max_tokens:
        started = time.perf_counter()
        n = budget.choose(max_tokens - emitted)
        if n:
            lat = compute_lattice_gpu(
                draft_model,
                b,
                hidden,
                draft_cache,
                POSITIONS,
                keyed=keyed,
                committed=emitted,
            )
            toks, parents = search_tree(lat, n)
            window = mx.concatenate(
                [mx.array([b], dtype=mx.int32), toks.astype(mx.int32)]
            )[None]
            shape = tv.DynamicShape(parents, POSITIONS)
            hidden = None  # the drafter has read the committed positions
        else:
            window, parents, shape = mx.array([[b]], dtype=mx.int32), None, root_shape
        res = tv.tree_forward(lm, window, shape, prompt_cache, target_ids)
        if keyed is not None or (request is not None and request.logprobs):
            target, probs = tree_target_rows(
                lm.speculative_logits_from_hidden(res.hidden),
                shape,
                emitted,
                keyed,
                request,
            )
        else:
            target = lm.speculative_argmax_from_hidden(res.hidden)
        try:
            if parents is None:
                mx.async_eval(target, window)
                wparents = [-1]
            else:
                mx.async_eval(target, window, parents)
                wparents = [int(t) for t in parents.tolist()]
            row_tokens = [int(t) for t in target.reshape(-1).tolist()]
            tokens = [int(t) for t in window.reshape(-1).tolist()]
        except BaseException:
            tv.tree_abort(prompt_cache, res)
            raise
        path = walk(tokens, wparents, row_tokens)
        new_tokens = [tokens[r] for r in path[1:]] + [row_tokens[path[-1]]]
        if request is not None and request.logprobs:
            request.record(
                probs[mx.array(path, dtype=mx.int32)],
                new_tokens[: max_tokens - emitted],
            )
        if n:
            _record_speculative_round(draft_model, len(path) - 1, n, parents=wparents)
        tv.tree_commit(lm, prompt_cache, res, path)
        kept = mx.concatenate(res.captured, axis=-1)[:, mx.array(path, dtype=mx.int32)]
        # a round without drafts leaves the committed positions for the next draft
        hidden = kept if hidden is None else mx.concatenate([hidden, kept], axis=1)
        b = new_tokens[-1]
        budget.observe(
            n,
            [r - 1 for r in path[1:]],
            (time.perf_counter() - started) * 1e3,
            first=emitted == 1,
        )
        for tok in new_tokens:
            yield tok, None
            emitted += 1
            if emitted >= max_tokens:
                return


def install() -> bool:
    """Route upstream's single-row DFlash round loop through the tree rounds
    (idempotent). Rounds that cannot run a tree fall back to upstream's."""
    from mlx_vlm.speculative import utils as spec_utils

    current = spec_utils._dflash_rounds
    if getattr(current, "_yunshu_tree", False):
        return True
    original = current

    def rounds(*args, **kwargs):
        from . import settings

        if settings.get("YUNSHU_SPEC_TREE") != "tree":
            yield from original(*args, **kwargs)
            return
        yield from dflash_tree_rounds(*args, _original=original, **kwargs)

    rounds._yunshu_tree = True
    cast(Any, rounds)._yunshu_copy_underneath = bool(
        getattr(original, "_yunshu_copy", False)
        or getattr(original, "_yunshu_copy_underneath", False)
    )
    spec_utils._dflash_rounds = rounds
    return True


__all__ = [
    "dflash_tree_rounds",
    "install",
    "walk",
]
