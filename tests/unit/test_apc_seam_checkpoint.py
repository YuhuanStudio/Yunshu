"""YUNSHU_APC_SEAM_CHECKPOINT: one aligned cut inside a long final user message (TensorFold #1262)."""

import random

import pytest

mx = pytest.importorskip("mlx.core")

from tests.unit.test_apc_manager import _coordinator, _mgr  # noqa: E402

IM_START, USER = 90, 91


@pytest.fixture(autouse=True)
def cpu_arrays():
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(device)


def _prompt(system=300, message=10_000, question=100, rnd=None):
    rnd = rnd or random.Random(4)
    body = [rnd.randrange(100, 800) for _ in range(system)]
    doc = [rnd.randrange(100, 800) for _ in range(message - question)]
    q = [rnd.randrange(100, 800) for _ in range(question)]
    return body, doc, q


def _ids(body, doc, q):
    return [7, *body, IM_START, USER, *doc, *q]


def _manager(**kw):
    kw.setdefault("prefill_alignment", 64)
    kw.setdefault("seam_checkpoint", True)
    m = _mgr(**kw)
    m.head_marker = (IM_START, USER)
    return m


def test_off_by_default_adds_nothing():
    body, doc, q = _prompt()
    m = _manager(seam_checkpoint=False)
    assert m.seam_boundary(_ids(body, doc, q), len(_ids(body, doc, q)) - 1) == 0


def test_not_aligned_mode_adds_nothing():
    body, doc, q = _prompt()
    ids = _ids(body, doc, q)
    m = _manager(prefill_alignment=0)
    assert m.seam_boundary(ids, len(ids) - 1) == 0


def test_short_final_message_adds_nothing():
    body, doc, q = _prompt(message=8000)  # < 8192
    ids = _ids(body, doc, q)
    m = _manager()
    assert m.seam_boundary(ids, len(ids) - 1) == 0


def test_cut_is_aligned_inside_the_message_and_2048_before_its_end():
    body, doc, q = _prompt(message=12_000, question=700)
    ids = _ids(body, doc, q)
    final = len(ids) - 1
    m = _manager()
    start = m.last_user_boundary(ids)
    assert ids[start : start + 2] == [IM_START, USER]
    seam = m.seam_boundary(ids, final)
    assert seam % 64 == 0 and start < seam <= final - 2048 and final - 2048 - seam < 64
    # a different question after the same document shares the whole prefix up to the cut
    other = _ids(body, doc, [5] * 900)
    assert other[:seam] == ids[:seam]


def test_plan_gets_the_seam_unless_an_interval_cut_already_covers_it():
    # interval cut (2048 grid) lands inside the question: the seam is added
    body, doc, q = _prompt(system=100, message=12_000, question=2300)
    ids = _ids(body, doc, q)
    final = len(ids) - 1
    m = _manager()
    plan = _coordinator(m).checkpoint_lengths(ids, set(), begin=False)
    start = m.last_user_boundary(ids)
    seam = m.seam_boundary(ids, final)
    assert seam in plan and all(n % 64 == 0 for n in plan)
    assert start < seam <= final - 2048
    # an interval cut already >= 2048 before the end and inside the message: nothing to add
    for question in range(0, 4096, 61):
        body, doc, q = _prompt(system=100, message=12_000 + question, question=50)
        ids = _ids(body, doc, q)
        base = _coordinator(_manager(seam_checkpoint=False)).checkpoint_lengths(
            ids, set(), begin=False
        )
        with_seam = _coordinator(_manager()).checkpoint_lengths(ids, set(), begin=False)
        final, start = len(ids) - 1, _manager().last_user_boundary(ids)
        covered = any(start < n <= final - 2048 for n in base)
        assert (with_seam == base) == covered
        assert set(with_seam) - set(base) <= {_manager().seam_boundary(ids, final)}


def test_a_media_span_is_never_cut():
    body, doc, q = _prompt(message=12_000, question=700)
    ids = _ids(body, doc, q)
    final = len(ids) - 1
    m = _manager()
    seam = m.seam_boundary(ids, final)
    image = 777
    for i in range(seam - 100, seam + 100):  # an image straddling the natural cut
        ids[i] = image
    plan = _coordinator(m).checkpoint_lengths(ids, {image}, begin=False)
    assert all(not (seam - 100 < n <= seam + 100) for n in plan if n != final)
    assert all(n % 64 == 0 for n in plan)


def test_auxiliary_policy_plans_no_seam():
    from yunshu_engine.serving.auxiliary_prefill import AuxiliaryPrefillPolicy

    body, doc, q = _prompt()
    ids = _ids(body, doc, q)
    assert AuxiliaryPrefillPolicy(_manager()).seam_boundary(ids, len(ids) - 1) == 0
