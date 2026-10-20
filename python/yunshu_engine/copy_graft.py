"""A tree suffix graft shares the MTP prefix and the target verify round.

Only proposals change. The ordinary verifier, sampling, and cache commit own
acceptance. Width never exceeds the backend's certified row bound.
"""

from collections import deque


class GraftBudget:
    """Request-local extra acceptance versus marginal round cost.

    Start with a four-token probe; grow only after accepting its entire tail.
    Price subsequent widths from observed whole-round costs at this depth,
    excluding intervals contaminated by publication suspension.
    """

    def __init__(self) -> None:
        self.samples: deque[tuple[int, int, float]] = deque(maxlen=32)
        self.base_ms: float | None = None
        self.base_tokens = 3.0
        self.width = 4
        self.calls = 0
        self.graft_rate: float | None = None  # tokens per ms of graft rounds (EMA)
        self.base_rate: float | None = None  # tokens per ms of plain rounds (EMA)

    def choose(self, available: int) -> int:
        self.calls += 1
        cap = min(available, self.width)
        if cap <= 0:
            return 0
        if not self.samples or self.base_ms is None:
            return cap
        if (
            self.graft_rate is not None
            and self.base_rate is not None
            and self.graft_rate < self.base_rate
        ):
            # measured whole-round tokens/ms of graft rounds are worse than plain
            # rounds: never regress, re-probe a short tail now and then
            return min(cap, 4) if self.calls % 16 == 0 else 0
        if (
            cap > max(s[0] for s in self.samples)
            and self.samples[-1][1] == self.samples[-1][0]
        ):
            return cap  # one bounded wider probe earned by a fully accepted tail
        best, best_rate = 0, self.base_tokens / self.base_ms
        for n in range(1, cap + 1):
            eligible = [s for s in self.samples if s[0] >= n]
            if not eligible:
                continue
            extra = sum(min(n, a) for _, a, _ in eligible) / len(eligible)
            # A smaller width is conservatively charged the observed full cost.
            cost = sum(ms for _, _, ms in eligible) / len(eligible)
            rate = (self.base_tokens + extra) / max(cost, self.base_ms)
            if rate > best_rate * 1.01:
                best, best_rate = n, rate
        return min(cap, 4) if best == 0 and self.calls % 16 == 0 else best

    def observe(
        self,
        prefix: int,
        extra: int,
        accepted: int,
        ms: float | None,
        tokens: int | None = None,
    ) -> None:
        if tokens is not None and ms is not None and ms > 0:
            rate = tokens / ms
            if extra:
                self.graft_rate = (
                    rate
                    if self.graft_rate is None
                    else 0.5 * self.graft_rate + 0.5 * rate
                )
            else:
                self.base_rate = (
                    rate
                    if self.base_rate is None
                    else 0.8 * self.base_rate + 0.2 * rate
                )
        if extra:
            landed = max(0, accepted - prefix)
            if landed == extra:
                self.width = min(31, max(self.width, extra * 2))
            if ms is not None and ms > 0:
                self.samples.append((extra, landed, ms))
        elif ms is not None and ms > 0:
            self.base_ms = ms if self.base_ms is None else (self.base_ms + ms) / 2
            self.base_tokens = (self.base_tokens + accepted + 1) / 2


def graft_suffix(model_drafts: int, copied: list[int], cap: int) -> list[int]:
    """Append only the copy rows after the shared MTP positions.

    If the model prefix differs, the verifier rejects at that position, so no
    copied descendant can be committed. No GPU-to-host draft read is needed.
    """
    return copied[model_drafts : model_drafts + max(0, cap)]


def verify_graft(lm, cache, bonus, model_drafts, copied, token_dtype):
    """Verify both branches, redirecting the copy's agreed prefix to MTP nodes.

    Duplicate prefix slots remain allocated to keep GPU shapes static; the walk
    and all descendants share the MTP prefix. No model draft host read precedes
    the forward. The selected path is absorbed by the MTP head afterwards.
    """
    import mlx.core as mx

    from . import tree_verify as tv

    m, n = int(model_drafts.size), len(copied)
    shared = min(m, n)
    matches = mx.cumprod(
        (
            model_drafts.reshape(-1)[:shared]
            == mx.array(copied[:shared], dtype=token_dtype)
        ).astype(mx.int32)
    )
    previous_shared = mx.concatenate([mx.array([1], dtype=mx.int32), matches])
    copy_parents = []
    for j in range(n):
        if j <= shared:
            parent = mx.where(previous_shared[j], j, m + j)
        else:
            parent = mx.array(m + j, dtype=mx.int32)
        copy_parents.append(parent.reshape(1))
    parents = mx.concatenate([mx.arange(-1, m, dtype=mx.int32), *copy_parents])
    shape = tv.DynamicShape(parents, max(m, n))
    window = mx.concatenate(
        [
            mx.array([[bonus]], dtype=token_dtype),
            model_drafts.reshape(1, -1),
            mx.array([copied], dtype=token_dtype),
        ],
        axis=1,
    )
    result = tv.tree_forward(lm, window, shape, cache)
    try:
        target = lm.speculative_argmax_from_hidden(result.hidden).reshape(-1)
        packed = mx.concatenate([window.reshape(-1), target, parents])
        values = packed.tolist()
        width = int(window.size)
        tokens, targets, host_parents = (
            values[:width],
            values[width : 2 * width],
            values[2 * width :],
        )
        path = [0]
        while True:
            child = next(
                (
                    r
                    for r in range(1, width)
                    if host_parents[r] == path[-1] and tokens[r] == targets[path[-1]]
                ),
                None,
            )
            if child is None:
                break
            path.append(child)
        rows = mx.array(path, dtype=mx.int32)
        hidden = result.hidden[:, rows, :]
        target_tokens = target[rows]
        drafted = mx.array([[tokens[r] for r in path[1:]]], dtype=token_dtype)
    except BaseException:
        tv.tree_abort(cache, result)
        raise

    class SelectedPath:
        def __init__(self):
            self.hidden = hidden
            self.target_tokens = target_tokens
            self.graft_nodes = m + n
            self.copy_accepted = 0
            for row, proposed in zip(path[1:], copied, strict=False):
                if tokens[row] != proposed:
                    break
                self.copy_accepted += 1

        def commit(self, model, prompt_cache, accepted, bs):
            tv.tree_commit(model, prompt_cache, result, path[: accepted + 1])

        def abort(self):
            tv.tree_abort(cache, result)

    return SelectedPath(), drafted, len(path)


def longest_verified_path(tokens, parents, targets):
    """Duplicate shared-prefix leaves must not hide a longer verified branch."""
    paths = [[0]]
    for row in range(1, len(tokens)):
        parent = parents[row]
        if tokens[row] == targets[parent]:
            prefix = next((p for p in paths if p[-1] == parent), None)
            if prefix is not None:
                paths.append([*prefix, row])
    return max(paths, key=len)


def plan_tree_graft(proposal, model_nodes, budget, room=None):
    """Return ``(source, rows)``: the copied chain and how many new rows to append.

    The copied chain is walked down the model tree on the device, so up to
    ``model_nodes`` leading tokens may coincide with model rows and cost nothing;
    ``rows`` extra rows continue from there. ``rows`` is bounded by the budget,
    the proposal length and ``room`` (tokens the request may still emit).
    """
    if not proposal:
        return [], 0
    rows = budget.choose(len(proposal))
    if room is not None:
        rows = min(rows, max(0, room - model_nodes))
    if rows <= 0:
        return [], 0
    return list(proposal[: model_nodes + rows]), rows
