"""Reduced-vocabulary draft readout: base ids plus the request's prompt / committed ids."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from yunshu_engine.draft_vocab import DraftVocab  # noqa: E402

K, V, KEEP = 128, 1000, 300


def _head(seed=0):
    mx.random.seed(seed)
    head = nn.QuantizedLinear(K, V, bias=False, group_size=64, bits=4)
    q, s, b = mx.quantize(mx.random.normal((V, K)), group_size=64, bits=4)
    head.weight, head.scales, head.biases = q, s, b
    mx.eval(head.parameters())
    return head


def _full_logits(head, x):
    return mx.quantized_matmul(
        x,
        head.weight,
        head.scales,
        head.biases,
        transpose=True,
        group_size=64,
        bits=4,
    )


def _restricted_argmax(head, x, allowed):
    logits = _full_logits(head, x)
    mask = mx.zeros((V,), dtype=mx.bool_)
    mask = mask.at[mx.array(sorted(allowed))].add(True)
    return mx.argmax(mx.where(mask, logits, -mx.inf), axis=-1)


def test_base_only_matches_restricted_argmax():
    head = _head()
    vocab = DraftVocab(head, KEEP)
    x = mx.random.normal((6, K))
    got = vocab.argmax(x)
    assert got.tolist() == _restricted_argmax(head, x, range(KEEP)).tolist()
    assert max(got.tolist()) < KEEP


def test_prompt_and_learned_ids_widen_the_search():
    head = _head(1)
    vocab = DraftVocab(head, KEEP)
    x = mx.random.normal((32, K))
    full = mx.argmax(_full_logits(head, x), axis=-1).tolist()
    wanted = sorted({t for t in full if t >= KEEP})
    assert wanted, "test data should draft above the base set"

    # nothing outside the base set is reachable before the request's ids arrive
    assert max(vocab.argmax(x).tolist()) < KEEP

    half = wanted[: len(wanted) // 2]
    assert vocab.set_context([5, 7] + half) == len(half)
    allowed = set(range(KEEP)) | set(half)
    assert vocab.argmax(x).tolist() == _restricted_argmax(head, x, allowed).tolist()

    rest = wanted[len(wanted) // 2 :]
    assert vocab.learn(rest + half + [3]) == len(rest)  # only unseen ids >= keep
    allowed |= set(rest)
    assert vocab.argmax(x).tolist() == full
    assert vocab.argmax(x).tolist() == _restricted_argmax(head, x, allowed).tolist()


def test_set_context_resets_the_extension():
    head = _head(2)
    vocab = DraftVocab(head, KEEP)
    vocab.set_context([KEEP + 1, KEEP + 2])
    vocab.learn([KEEP + 3])
    assert vocab.set_context([]) == 0
    assert vocab.extra_ids is None
    assert vocab.learn([KEEP + 1]) == 1


def test_ids_beyond_the_vocabulary_are_ignored():
    vocab = DraftVocab(_head(3), KEEP)
    assert vocab.learn([V, V + 5, -1]) == 0
    x = mx.random.normal((2, K))
    assert vocab.argmax(x).shape == (2,)


def test_a_non_latin_prompt_or_output_falls_back_to_the_full_readout():
    head = _head(4)
    x = mx.random.normal((3, K))
    full = mx.argmax(_full_logits(head, x), axis=-1).tolist()
    calls = []

    def fallback(hidden):
        calls.append(1)
        return mx.argmax(_full_logits(head, hidden.reshape(-1, K)), axis=-1)

    vocab = DraftVocab(head, KEEP)
    vocab.fallback = fallback
    high = [KEEP + i for i in range(20)]

    # prompt: half or more of 16+ ids above the base set
    vocab.set_context(high + list(range(10)))
    assert vocab.full and vocab.argmax(x).tolist() == full and calls

    # a Latin prompt keeps the reduced readout ...
    vocab.set_context(list(range(40)))
    assert not vocab.full
    # ... until the committed tokens turn out to be in another script
    vocab.learn(list(range(20)))
    assert not vocab.full
    vocab.learn(high[:8])  # 8 of 28: still Latin-like
    assert not vocab.full
    vocab.learn(high[8:14])  # 14 of 34
    assert not vocab.full
    vocab.learn(high + high)  # 54 of 74
    assert vocab.full and vocab.extra_ids is None
    assert vocab.learn([KEEP + 50]) == 0

    # the next request starts reduced again
    vocab.set_context([1, 2, 3])
    assert not vocab.full


def test_logits_subset_matches_the_full_head_on_the_searched_ids():
    head = _head()
    vocab = DraftVocab(head, KEEP)
    vocab.set_context([700, 701, 5])  # two ids above the base set join the extension
    x = mx.random.normal((3, K)).astype(mx.bfloat16)
    logits, ids = vocab.logits_subset(x)
    full = _full_logits(head, x)
    assert logits.shape == (3, KEEP + 2) and ids.shape == (KEEP + 2,)
    assert mx.allclose(logits, mx.take(full, ids, axis=-1), atol=1e-2).item()
    assert sorted(ids.tolist())[-2:] == [700, 701]


def test_logits_subset_is_none_once_the_request_reads_the_full_vocabulary():
    vocab = DraftVocab(_head(), KEEP)
    vocab.full = True
    assert vocab.logits_subset(mx.zeros((1, K), dtype=mx.bfloat16)) is None


class _UnboundMTP:
    """Upstream's MTP drafter before bind(): its own greedy readout has no head (the
    round driver never calls bind, so this is the served state)."""

    _lm_head_fn = None

    def _greedy_token(self, hidden):
        return mx.argmax(self._lm_head_fn(hidden), axis=-1)


def test_installed_full_readout_needs_no_upstream_bind():
    """2026-11-09: an image prompt (mostly image-pad ids) on 27B MTP crashed with
    'NoneType' object is not callable in the full-vocabulary fallback."""
    from types import SimpleNamespace

    from yunshu_engine.draft_vocab import install

    head = _head(5)
    lm = SimpleNamespace(lm_head=head)
    drafter = _UnboundMTP()
    vocab = install(drafter, lm, KEEP)
    assert vocab is not None
    vocab.set_context([KEEP + i for i in range(20)])
    assert vocab.full
    x = mx.random.normal((2, K))
    full = mx.argmax(_full_logits(head, x), axis=-1).tolist()
    assert drafter._greedy_token(x).tolist() == full


def test_media_placeholder_ids_do_not_switch_to_the_full_vocabulary():
    """An image prompt is mostly image-pad ids above the base set; they are not a script."""
    head = _head(6)
    vocab = DraftVocab(head, KEEP)
    vocab.fallback = lambda h: mx.argmax(_full_logits(head, h), axis=-1)
    pad = KEEP + 7
    prompt = [pad] * 200 + list(range(20))
    vocab.set_context(prompt)
    assert vocab.full  # without the skip set the pads look like a non-Latin script
    vocab.skip_ids = frozenset({pad})
    vocab.set_context(prompt)
    assert not vocab.full and vocab.extra_ids is None  # and the pad is not searched
    import numpy as np

    vocab.set_context(np.array(prompt, dtype=np.int32))
    assert not vocab.full


# --- Chinese script set (2026-11-09): a mostly-Chinese request searches base + Han ids
# instead of the whole head (before: the full vocabulary, i.e. every draft read all 248k rows).

SCRIPT = list(range(KEEP + 100, KEEP + 400))


def _script_vocab(seed=7):
    head = _head(seed)
    vocab = DraftVocab(head, KEEP)
    vocab.fallback = lambda h: mx.argmax(_full_logits(head, h), axis=-1)
    vocab.set_script_ids(SCRIPT)
    return head, vocab


def test_a_chinese_prompt_searches_base_plus_script_not_the_full_head():
    head, vocab = _script_vocab()
    prompt = SCRIPT[:30] + list(range(10))
    vocab.set_context(prompt)
    assert vocab.script and not vocab.full
    x = mx.random.normal((4, K))
    allowed = set(range(KEEP)) | set(SCRIPT)
    assert vocab.argmax(x).tolist() == _restricted_argmax(head, x, allowed).tolist()
    logits, ids = vocab.logits_subset(x)
    got = ids.tolist()
    assert len(got) == len(set(got)) == len(allowed)  # no id twice


def test_other_scripts_still_read_the_full_head():
    _, vocab = _script_vocab()
    vocab.set_context(
        [KEEP + 600 + i for i in range(30)]
    )  # high, not in the script set
    assert vocab.full and not vocab.script


def test_chinese_output_after_a_latin_prompt_switches_to_the_script_set():
    head, vocab = _script_vocab()
    vocab.set_context(list(range(40)) + [KEEP + 650])  # extension holds one outside id
    assert not vocab.script
    vocab.learn(SCRIPT[:60])
    assert vocab.script and not vocab.full
    x = mx.random.normal((2, K))
    allowed = set(range(KEEP)) | set(SCRIPT) | {KEEP + 650}
    assert vocab.argmax(x).tolist() == _restricted_argmax(head, x, allowed).tolist()
    _, ids = vocab.logits_subset(x)
    assert len(ids.tolist()) == len(set(ids.tolist()))
    vocab.set_context([1, 2, 3])  # next request starts on the base set again
    assert not vocab.script and not vocab.full


def test_cjk_ids_covers_simplified_and_traditional():
    from yunshu_engine.draft_vocab import cjk_ids

    table = ["a", "的", "們", "们", "，", "hello", "Ａ", " 中", "x"]

    class Tok:
        def __len__(self):
            return len(table)

        def decode(self, ids):
            return table[ids[0]]

    assert cjk_ids(Tok(), 100).tolist() == [1, 2, 3, 4, 6, 7]
