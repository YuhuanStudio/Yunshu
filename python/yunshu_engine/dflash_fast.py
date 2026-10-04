"""Transaction-local GDN helpers for the certified DFlash fast tree.

The gate calculations and replay retain tree_verify's exact arithmetic. Gate
arrays live in the current TreeResult, rather than a global id-keyed cache.
"""

import hashlib
from typing import Any

import mlx.core as mx

from . import tree_verify as tv
from .dflash_plan import live_bound

_KERNELS: dict[str, Any] = {}


def _name(base, source):
    return (
        base + "_" + hashlib.sha256((tv.gv._HELPERS + source).encode()).hexdigest()[:16]
    )


def _gate_values(a, b, alog, dt):
    source = """
    uint gi=thread_position_in_grid.x;
    if(gi>=N)return;
    uint hv=gi%Hv;
    float neg_a=-metal::precise::exp(static_cast<float>(A_log[hv]));
    InT dtb=dt_bias[hv];
    G[gi]=gdn_decay(a[gi],dtb,neg_a);
    BETA[gi]=gdn_beta(b[gi]);
    """
    if "gates" not in _KERNELS:
        _KERNELS["gates"] = mx.fast.metal_kernel(
            name=_name("yunshu_tree_gates", source),
            input_names=["a", "b", "A_log", "dt_bias"],
            output_names=["G", "BETA"],
            source=source,
            header=tv.gv._HELPERS,
        )
    return _KERNELS["gates"](
        inputs=[a, b, alog, dt],
        template=[("InT", a.dtype), ("Hv", a.shape[-1]), ("N", a.size)],
        grid=(a.size, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[a.shape] * 2,
        output_dtypes=[mx.float32] * 2,
    )


def _forward_kernel():
    if "forward" not in _KERNELS:
        source = tv._TREE_GDN
        edits = {
            "gdn_decay(a[gi], dtb, neg_a)": "G[gi]",
            "gdn_beta(b[gi])": "BETA[gi]",
            "float states[T][NK];": "float states[LIVE][NK];",
            "states[par][i]": "states[read_slots[t]][i]",
            "states[t][i] = st[i];": "if (write_slots[t] >= 0) states[write_slots[t]][i] = st[i];",
        }
        for old, new in edits.items():
            if source.count(old) != 1:
                raise RuntimeError("tree recurrence source changed: " + old)
            source = source.replace(old, new)
        _KERNELS["forward"] = mx.fast.metal_kernel(
            name=_name("yunshu_tree_last_use", source),
            input_names=[
                "state_in",
                "A_log",
                "dt_bias",
                "q",
                "k",
                "v",
                "a",
                "b",
                "parents",
                "G",
                "BETA",
                "read_slots",
                "write_slots",
            ],
            output_names=["y"],
            source=source,
            header=tv.gv._HELPERS,
        )
    return _KERNELS["forward"]


def _replay_kernel():
    if "replay" not in _KERNELS:
        source = tv._REPLAY_PATH.replace(
            "gdn_decay(pa[gi], dtb, neg_a)", "G[gi]"
        ).replace("gdn_beta(pb[gi])", "BETA[gi]")
        _KERNELS["replay"] = mx.fast.metal_kernel(
            name=_name("yunshu_tree_gate_replay", source),
            input_names=[
                "state_in",
                "A_log",
                "dt_bias",
                "pk",
                "pv",
                "G",
                "BETA",
                "path",
                "count",
            ],
            output_names=["state_out"],
            source=source,
            header=tv.gv._HELPERS,
        )
    return _KERNELS["replay"]


class GateRows(tuple):
    def __new__(cls, k, v, a, b, g, beta):
        result = super().__new__(cls, (k, v, a, b))
        result.gates = (g, beta)
        return result

    def replay(self, layer, state, path, count):
        k, v, _, _ = self
        g, beta = self.gates
        geo = tv.gv._geometry(layer, 1)
        return _replay_kernel()(
            inputs=[state, layer.A_log, layer.dt_bias, k, v, g, beta, path, count],
            template=[
                ("InT", k.dtype),
                ("Hk", geo["Hk"]),
                ("Hv", geo["Hv"]),
                ("Dk", geo["Dk"]),
                ("Dv", geo["Dv"]),
                ("P", k.shape[1]),
            ],
            grid=geo["grid"],
            threadgroup=(32, 4, 1),
            output_shapes=[state.shape],
            output_dtypes=[mx.float32],
        )[0]


def gdn_forward(shape, layer, state, q, k, v, a, b):
    g, beta = _gate_values(a, b, layer.A_log, layer.dt_bias)
    hk, hv = layer.num_k_heads, layer.num_v_heads
    dk, dv = layer.head_k_dim, layer.head_v_dim
    y = _forward_kernel()(
        inputs=[
            state,
            layer.A_log,
            layer.dt_bias,
            q,
            k,
            v,
            a,
            b,
            shape.parents_array(),
            g,
            beta,
            *shape.state_slots(),
        ],
        template=[
            ("InT", q.dtype),
            ("Hk", hk),
            ("Hv", hv),
            ("Dk", dk),
            ("Dv", dv),
            ("T", shape.width),
            ("LIVE", 1 if shape.is_chain else live_bound(shape.width)),
        ],
        grid=(32, dv, hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(1, shape.width, hv, dv)],
        output_dtypes=[q.dtype],
    )[0]
    return y, GateRows(k, v, a, b, g, beta)


class Verifier:
    """Use virtual column groups without mutating the shared verifier."""

    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def _linears(self, members, x):
        from .kernels.lane_virtual import grouped_linears

        members = tuple(members)
        result = grouped_linears(members, x)
        return self.original._linears(members, x) if result is None else result

    def _feed_forward(self, feed_forward, x):
        method = self.original._feed_forward
        if hasattr(method, "__func__"):
            return method.__func__(self, feed_forward, x)
        return method(feed_forward, x)


class PrivateProjections:
    """Keep original drafter projections for fallback requests; parents are weak."""

    def __init__(self, draft):
        import weakref

        import mlx.nn as nn

        from .kernels import lane_linear

        self.bindings = []
        pending = []
        for _, parent in list(draft.named_modules()):
            for key, child in parent.children().items():
                if isinstance(child, nn.QuantizedLinear) and lane_linear.eligible(
                    child
                ):
                    converted = lane_linear.LaneLinear.from_quantized(child)
                    self.bindings.append((weakref.ref(parent), key, child, converted))
                    pending.extend([converted.weight, converted.sbt])
        if pending:
            mx.eval(pending)

    def bind(self, enabled):
        for parent, key, original, converted in self.bindings:
            module = parent()
            if module is not None:
                module.update_modules({key: converted if enabled else original})


def prepare(draft):
    private = getattr(draft, "_yunshu_fast_projections", None)
    if private is None:
        private = PrivateProjections(draft)
        object.__setattr__(draft, "_yunshu_fast_projections", private)
    return private


def rounds(
    model,
    draft,
    cache,
    hidden,
    *,
    first_bonus,
    max_tokens,
    sampler,
    token_dtype=mx.int32,
    **_,
):
    """Measured deep15 proposal recipe, keeping the target's cache transaction."""
    import time

    from mlx_vlm.models.qwen3_5 import language as q35
    from mlx_vlm.speculative.common import _record_speculative_round

    from . import dflash_tree as dt
    from . import mtp_lane
    from .copy_drafter import CopyDrafter
    from .dflash_context import context_window
    from .dflash_plan import FastShape, remap_landed, reorder, search_tree
    from .kernels import lane_linear
    from .spec_schedule import NodeBudget

    lm = getattr(model, "language_model", model)
    private = prepare(draft)
    target_ids = list(draft.config.target_layer_ids)
    draft_cache = draft.reset(model)
    budget = NodeBudget(15, prior=dt.PRIOR)
    context = mtp_lane._STATE["context"]
    copy_rows = mtp_lane.copy_rows_for_model(lm)
    copy = CopyDrafter(max_draft=copy_rows - 1) if copy_rows >= 3 else None
    if copy is not None:
        copy.extend(context)
        copy.extend([first_bonus])
    keep = context_window(draft)
    skipped = 0
    verifier = Verifier(q35._EXACT_SPECULATIVE_VERIFIER)
    bonus = int(first_bonus)
    emitted = 1
    try:
        while emitted < max_tokens:
            previous_sums = lane_linear.sum_reuse_enabled()
            lane_linear.set_sum_reuse(True)
            try:
                started = time.perf_counter()
                room = max_tokens - emitted
                n = budget.choose(max(0, room - 1))
                copied = (
                    copy.draft(min(copy.max_draft, room - 1))
                    if copy is not None
                    else []
                )
                if copied:
                    n = len(copied)
                    window = mx.array([[bonus, *copied]], dtype=token_dtype)
                    parents = mx.arange(-1, n, dtype=mx.int32)
                    shape = FastShape(parents, n, is_chain=True)
                elif n:
                    if skipped:
                        for entry in draft_cache:
                            entry.offset += skipped
                        skipped = 0
                    private.bind(True)
                    try:
                        lattice = dt.compute_lattice_gpu(
                            draft, bonus, hidden, draft_cache, 15
                        )
                    finally:
                        private.bind(False)
                    tokens, parents = search_tree(lattice, n)
                    tokens, parents, ranks = reorder(tokens, parents)
                    window = mx.concatenate(
                        [mx.array([bonus], dtype=token_dtype), tokens]
                    )[None]
                    shape = FastShape(parents, 15, original_ranks=ranks)
                    hidden = None
                else:
                    parents = mx.array([-1], dtype=mx.int32)
                    shape = FastShape(parents, 0, is_chain=True)
                    window = mx.array([[bonus]], dtype=token_dtype)
                result = tv.tree_forward(
                    lm, window, shape, cache, target_ids, verifier=verifier
                )
                try:
                    target = lm.speculative_argmax_from_hidden(result.hidden)
                    if shape.original_ranks is None:
                        mx.async_eval(target, window, parents)
                        ranks = None
                    else:
                        mx.async_eval(target, window, parents, shape.original_ranks)
                        ranks = shape.original_ranks.tolist()
                    parent_ids = parents.tolist()
                    target_ids_row = target.reshape(-1).tolist()
                    window_ids = window.reshape(-1).tolist()
                except BaseException:
                    tv.tree_abort(cache, result)
                    raise
                path = dt.walk(window_ids, parent_ids, target_ids_row)
                new_tokens = [window_ids[row] for row in path[1:]] + [
                    target_ids_row[path[-1]]
                ]
                if n:
                    _record_speculative_round(draft, len(path) - 1, n)
                tv.tree_commit(lm, cache, result, path)
                captured = mx.concatenate(result.captured, axis=-1)[
                    :, mx.array(path, dtype=mx.int32)
                ]
                hidden = (
                    captured
                    if hidden is None
                    else mx.concatenate([hidden, captured], axis=1)
                )
                if copied:
                    copy.observe_copy(n, len(path) - 1)
                    if keep is not None and hidden.shape[1] > keep:
                        dropped = int(hidden.shape[1]) - keep
                        hidden = hidden[:, dropped:]
                        skipped += dropped
                elif copy is not None:
                    copy.observe_model(len(new_tokens))
                landed = [row - 1 for row in path[1:]]
                if ranks is not None:
                    landed = remap_landed(landed, ranks)
                budget.observe(
                    n,
                    landed,
                    (time.perf_counter() - started) * 1000,
                    first=emitted == 1,
                )
                bonus = new_tokens[-1]
            finally:
                lane_linear.set_sum_reuse(previous_sums)
            counted = False
            for token in new_tokens:
                if copy is not None:
                    copy.extend([token])
                if copied:
                    if not counted:
                        draft.copy_total_rounds = (
                            getattr(draft, "copy_total_rounds", 0) + 1
                        )
                        counted = True
                    draft.copy_total_tokens = getattr(draft, "copy_total_tokens", 0) + 1
                yield token, None
                emitted += 1
                if emitted >= max_tokens:
                    return
    finally:
        private.bind(False)
        hidden = None
        draft_cache = None


def supported(lm, draft):
    """The same wide, dense 27B Q4 lane certified by the serving A/B."""
    from .dflash_context import context_window
    from .kernels.lane_linear import LaneLinear
    from .kernels.tensorfold import lane_qmm

    if (
        not hasattr(draft, "candidate_selector")
        or getattr(draft.config, "block_size", None) != 8
    ):
        return False
    if (
        callable(getattr(draft, "prepare_target_hidden", None))
        or context_window(draft) is None
    ):
        return False
    if (
        lane_qmm._resolve_variant() != "m5"
        or not tv.supported(lm)
        or not tv.lane_projections(lm)
    ):
        return False
    layers = lm.model.layers
    if len(layers) != 64 or layers[0].input_layernorm.weight.size != 5120:
        return False
    projections = [
        module for _, module in lm.named_modules() if isinstance(module, LaneLinear)
    ]
    return (
        bool(projections)
        and sum(module.bits == 4 for module in projections) > len(projections) // 2
        and all(
            module.bits in (4, 5) and module.group_size == 64 for module in projections
        )
    )


def eligible(lm, draft, cache, kwargs):
    """Conservative 1K-class policy; long requests keep trained chain+copy."""
    from . import mtp_lane
    from .keyed_sampling import KeyedSampler

    context = mtp_lane._STATE["context"]
    maximum = int(kwargs.get("max_tokens", 0))
    return bool(
        kwargs.get("greedy_sampling", True)
        and not isinstance(kwargs.get("sampler"), KeyedSampler)
        and mtp_lane._STATE["guide"] is None
        and context is not None
        and len(context) >= 512
        and 64 <= maximum <= 256
        and len(context) + maximum <= 1536
        and kwargs.get("draft_block_size") in (None, 8)
        and mtp_lane.copy_rows_for_model(lm) == 16
        and supported(lm, draft)
        and tv.lane_ready(lm, cache)
    )
