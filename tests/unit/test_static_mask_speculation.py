"""A static suppress list (checkpoint ``suppress_tokens`` / ``ignore_eos``) keeps speculative
decoding: the same ban applies to the verify rows and to copy drafts, and speculation on equals
speculation off token for token. (CPU simulation of the lane's accept loop; the Metal kernels
are covered by the Mac A/B.)"""

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine.constrained_spec import ProcessorWindow, SpecRequest, static_ban
from yunshu_engine.copy_drafter import CopyDrafter
from yunshu_engine.vlm_batch_runner import TokenMaskProcessor, static_masks_only

VOCAB = 40


def _row(prev):
    """The next-token logits after ``prev`` (a fixed function of the last token)."""
    rng = np.random.default_rng(1000 + prev)
    return mx.array(rng.standard_normal((1, VOCAB)).astype(np.float32))


def test_static_masks_are_recognised():
    assert static_masks_only([]) and static_masks_only(None)
    assert static_masks_only([TokenMaskProcessor(suppress=[1], eos_ids=[2])])
    assert static_masks_only([TokenMaskProcessor(eos_ids=[2], ignore_eos=True)])
    assert not static_masks_only([TokenMaskProcessor(eos_ids=[2], min_tokens=3)])
    assert not static_masks_only([TokenMaskProcessor(top_n_sigma=2.0)])
    assert not static_masks_only([lambda t, lg: lg])
    assert static_ban([TokenMaskProcessor(suppress=[1, 7], eos_ids=[2])]) == {1, 7}
    assert static_ban(
        [TokenMaskProcessor(suppress=[1], eos_ids=[2], ignore_eos=True)]
    ) == {1, 2}
    assert static_ban(
        [TokenMaskProcessor(suppress=[1], eos_ids=[2], min_tokens=3)]
    ) == {1}


def _serial(mask, first, n):
    out, prev = [first], first
    for _ in range(n):
        row = mask.process_last_token(prev, _row(prev))
        prev = int(mx.argmax(row, axis=-1).item())
        out.append(prev)
    return out


def _speculative(mask, first, n, draft_fn, width=4):
    """The lane's loop: verify [bonus, *drafts] under the masked rows, accept the matching prefix."""
    request = SpecRequest()
    request.processors = [mask]
    window = ProcessorWindow.for_request(request, first)
    out, bonus, emitted = [first], first, 1
    while len(out) < n + 1:
        drafted = draft_fn(bonus, width - 1)
        inputs = [bonus, *drafted]
        logits = mx.concatenate([_row(t)[:, None, :] for t in inputs], axis=1)
        logits = window.apply(logits, drafted, emitted)
        tgt = mx.argmax(logits, axis=-1).reshape(-1).tolist()
        accepted = 0
        while accepted < len(drafted) and drafted[accepted] == tgt[accepted]:
            accepted += 1
        tokens = (drafted[:accepted] + [tgt[accepted]])[: n + 1 - len(out)]
        window.commit(tokens)
        out += tokens
        emitted += len(tokens)
        bonus = tokens[-1]
    return out


def _unmasked_argmax_draft(bonus, k):
    """A draft that proposes what the UNMASKED model would say -- banned ids included."""
    out, prev = [], bonus
    for _ in range(k):
        prev = int(mx.argmax(_row(prev), axis=-1).item())
        out.append(prev)
    return out


@pytest.mark.parametrize("seed", range(6))
def test_speculation_on_equals_off_with_a_static_ban(seed):
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, VOCAB))
    # ban the ids the unmasked model likes best, so drafts really hit the ban
    liked = {int(mx.argmax(_row(t), axis=-1).item()) for t in range(VOCAB)}
    suppress = sorted(int(t) for t in rng.choice(sorted(liked), size=3, replace=False))
    off = _serial(TokenMaskProcessor(suppress=suppress), first, 40)
    on = _speculative(
        TokenMaskProcessor(suppress=suppress), first, 40, _unmasked_argmax_draft
    )
    assert on == off
    assert not set(on[1:]) & set(suppress)  # nothing banned was ever emitted
    # also with a draft that is simply wrong half of the time
    wrong = lambda b, k: [(b + 3 * i + 1) % VOCAB for i in range(k)]  # noqa: E731
    assert _speculative(TokenMaskProcessor(suppress=suppress), first, 40, wrong) == off


def test_copy_drafts_stop_before_a_banned_token():
    import random

    rng = random.Random(1)
    src = [rng.randrange(10, 200) for _ in range(40)]
    noise = [rng.randrange(10, 200) for _ in range(30)]

    def drafter(banned=()):
        d = CopyDrafter(max_draft=7)
        d.extend(noise + src + noise)
        d.extend(src[:12])
        d.banned = frozenset(banned)
        return d

    plain = drafter().draft()
    assert len(plain) >= 3 and plain == src[12 : 12 + len(plain)]
    assert drafter({plain[2]}).draft() == plain[:2]  # cut before the banned id
    assert drafter({plain[0]}).draft() == []  # under two tokens left: the model drafts
    assert drafter({999}).draft() == plain  # an id that is not proposed changes nothing
