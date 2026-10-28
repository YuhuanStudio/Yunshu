# Patches upstream (MIT): mlx-vlm qwen4_exp LanguageModel.speculative_logits_from_hidden gains a logit stash
# (tracked in vendor.json; Blaizzy/mlx-vlm).
"""Position-keyed sampled speculative decode for the Qwen4 native MTP loop.

Upstream's native loop already draws verify targets through ``sampler.sample_target(...,
positions=...)``, so a ``KeyedSampler`` keeps the stream identical with speculation on and off. Two
things are missing for a useful, reportable sampled request:

* drafts are drawn by ``sampler(logits)`` with the sampler's serial cursor, which no longer matches
  the round's positions after a rejection, so the draft noise differs from the target noise and
  acceptance collapses. ``NativeKeyedSampler`` keeps the draft cursor on the position each draft
  token will occupy, so a draft equals the target draw whenever the drafter's distribution agrees;
* logprobs: the verifier hands over bf16 logprobs of a T-row window. The target rows are rebuilt
  from the fp32 logits of the same window with serial one-row reductions (``normalize_rows``), the
  arithmetic of the non-speculative path, and the logprob of every committed token is reported
  from that row.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from .keyed_sampling import KeyedSampler


def install_logit_stash(lm: Any) -> bool:
    """Make the language model remember the logits of the last verify projection.

    The wrapper widens them to float32 for a request that asks for logprobs (what the serial path's
    ``Fp32LogitsProcessor`` does) and keeps the array for ``NativeKeyedSampler.sample_target``."""
    if getattr(lm, "_yunshu_logit_stash_installed", False):
        return True
    project = getattr(lm, "speculative_logits_from_hidden", None)
    if not callable(project):
        return False

    def stashing(hidden: mx.array) -> mx.array:
        from .constrained_spec import current_request

        logits: mx.array = project(hidden)
        request = current_request()
        if request is not None and getattr(request, "logprobs", False):
            logits = logits.astype(mx.float32)
        lm._yunshu_logit_stash = logits
        return logits

    lm.speculative_logits_from_hidden = stashing
    lm._yunshu_logit_stash = None
    lm._yunshu_logit_stash_installed = True
    return True


class _Draw:
    """One verify window's normalized target rows, read back lazily per committed token."""

    def __init__(self, rows: mx.array, tokens: mx.array):
        self.rows = rows
        self.tokens = tokens
        self._chosen: list[float] | None = None

    def chosen(self) -> list[float]:
        if self._chosen is None:
            ids = self.tokens.reshape(-1, 1).astype(mx.int32)
            picked = mx.take_along_axis(self.rows, ids, axis=-1).reshape(-1)
            values: Any = picked.tolist()
            self._chosen = [float(x) for x in values]
        return self._chosen


class NativeKeyedSampler(KeyedSampler):
    """``KeyedSampler`` for the Qwen4 native loop: coupled drafts and exact logprob reports."""

    def __init__(self, params: Any, seed: int, lm: Any = None, request: Any = None):
        super().__init__(params, seed)
        self.lm = lm
        self.request = request
        self._draft_pos = (
            1  # the first draft token follows the prefill token at index 0
        )
        self._drafts: list[mx.array] = []
        self._draws: dict[int, tuple[_Draw, int]] = {}
        self.rounds = 0
        self.accepted = 0

    @property
    def wants_logprobs(self) -> bool:
        return bool(getattr(self.request, "logprobs", False))

    def sample_positions(self, logprobs: mx.array, positions: Any) -> mx.array:
        """Greedy reporting (``params is None``): the serial path's argmax over the float32 logprobs, which
        is what a greedy request that asks for logprobs uses when speculation is off."""
        if self.params is None:
            return mx.argmax(logprobs, axis=-1)
        return super().sample_positions(logprobs, positions)

    def __call__(self, logprobs: mx.array) -> mx.array:
        """Draft draw: the noise of the position this draft token will occupy."""
        shape = logprobs.shape
        flat = logprobs.reshape(-1, shape[-1])
        n = int(flat.shape[0])
        pos = list(range(self._draft_pos, self._draft_pos + n))
        self._draft_pos += n
        token = self.sample_positions(flat, pos).reshape(shape[:-1])
        self._drafts.append(token)
        return token

    def _window_rows(self, logprobs: mx.array, count: int) -> mx.array:
        """The window's target rows from the stashed logits, one-row reductions as in serial decode."""
        stash = getattr(self.lm, "_yunshu_logit_stash", None)
        if (
            stash is None
            or stash.ndim != 3
            or stash.shape[0] != 1
            or stash.shape[1] != count
        ):
            return logprobs
        self.lm._yunshu_logit_stash = None
        from .constrained_spec import normalize_rows

        rows: mx.array = normalize_rows(stash)
        return rows

    def sample_target(self, logprobs, row_ids=None, positions=None):
        if positions is None:
            return self(logprobs)
        positions = [int(p) for p in positions]
        first = positions[0]
        rows = (
            self._window_rows(logprobs, len(positions))
            if first > 0 and logprobs.shape[0] == len(positions)
            else logprobs
        )
        tokens = super().sample_target(rows, row_ids, positions)
        draws = self._drafts
        self._drafts = []
        accepted = 0
        if draws:
            drafted = [int(d.reshape(-1)[0].item()) for d in draws]
            target = tokens.reshape(-1).tolist()
            while (
                accepted < len(drafted)
                and accepted < len(target)
                and drafted[accepted] == target[accepted]
            ):
                accepted += 1
            self.rounds += 1
            self.accepted += accepted
        # the next draft token follows the round's last committed token
        self._draft_pos = first + accepted + 1
        if self.wants_logprobs:
            draw = _Draw(rows.astype(mx.float32), tokens)
            for i, p in enumerate(positions):
                self._draws[p] = (draw, i)
        return tokens

    def report(self, index: int, token: int) -> tuple[float, list]:
        """(logprob, top logprobs) of the committed token at generation ``index``."""
        for stale in [k for k in self._draws if k < index]:
            del self._draws[stale]
        draw, row = self._draws.pop(index)
        if self.request is not None and getattr(self.request, "top_logprobs", None):
            _, lp, top = self.request.reports(draw.rows[row : row + 1], [token])[0]
            return lp, top
        return draw.chosen()[row], []
