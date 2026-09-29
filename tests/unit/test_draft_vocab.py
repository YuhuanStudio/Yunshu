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
