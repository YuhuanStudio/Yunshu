"""Real small-array DFlash transactions: per-row identity and fair progress."""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")


def _draft(lm):
    from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel

    from yunshu_engine.dflash_tree import quantize_drafter

    cfg = DFlash2Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        vocab_size=512,
        mask_token_id=7,
        max_position_embeddings=4096,
        block_size=8,
        num_target_layers=4,
        target_layer_ids=[0, 3],
        conv_kernel_size=2,
        conv_group_size=16,
        selector_rank=8,
        selector_top_k=4,
        layer_types=["sliding_attention"] * 2,
        sliding_window=32,
    )
    d = DFlash2DraftModel(cfg)
    d.set_dtype(mx.bfloat16)
    quantize_drafter(d, group_size=64, bits=8)
    mx.eval(d.parameters())
    return d


@pytest.fixture(scope="module", autouse=True)
def restore_probe_lane_mode():
    # The probe enables a process-global kernel route for the tiny model.
    # A later module must see the mode it had before this module ran.
    from yunshu_engine.kernels import ragged_kv

    previous = ragged_kv._STATE["dense_lane"]
    yield
    ragged_kv.set_dense_lane(previous)


@pytest.fixture(scope="module")
def target():
    from scripts.research.phaseb_verify import tiny

    from yunshu_engine.kernels.ragged_attention import tile_ready

    if not mx.metal.is_available() or not tile_ready():
        pytest.skip("needs lane-capable Metal")
    return tiny()[0]


def _run(
    lm,
    prompts,
    *,
    draft,
    sampled=False,
    stagger=False,
    logprobs=True,
    backend=None,
    apc=None,
    tokens=12,
    guide_factory=None,
):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    driver = RoundDriver(
        lm,
        dflash_drafter=_draft(lm) if (draft if backend is None else backend) else None,
        chunk=16,
        apc=apc,
    )
    output = [[] for _ in prompts]
    todo = list(enumerate(prompts))
    steps = 0
    while driver.busy() or todo:
        if todo and (not stagger or steps % 3 == 0):
            i, prompt = todo.pop(0)
            params = (
                SimpleNamespace(
                    temperature=1.0,
                    top_p=0.95,
                    top_k=20,
                    min_p=0.0,
                    xtc_probability=0.0,
                    seed=17 + prompt[0],
                )
                if sampled
                else None
            )
            driver.add(
                Request(
                    prompt,
                    tokens,
                    sampling=params,
                    handle=i,
                    logprobs=logprobs,
                    top_logprobs=2,
                    draft=draft,
                    guide=guide_factory() if guide_factory else None,
                )
            )
        for event in driver.step():
            output[event.handle].append((event.token, event.logprob))
        steps += 1
        assert steps < 100
    return output, driver


@pytest.mark.parametrize("logprobs", [False, True])
@pytest.mark.parametrize("rows", [1, 2, 4])
@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("stagger", [False, True])
def test_dflash_rows_match_solo_target_tokens_and_logprobs(
    target, sampled, stagger, rows, logprobs
):
    prompts = [list(range(4 + b, 25 + 7 * b)) for b in range(rows)]
    ref = [
        _run(
            target,
            [p],
            draft=False,
            sampled=sampled,
            logprobs=logprobs,
            backend=not logprobs,
        )[0][0]
        for p in prompts
    ]
    got, driver = _run(
        target, prompts, draft=True, sampled=sampled, stagger=stagger, logprobs=logprobs
    )
    assert got == ref
    assert driver.drafted > 0
    assert (
        not driver.batch.rows and not driver.dflash.rings and not driver.dflash.hidden
    )


def test_cancel_drops_prefill_capture_and_live_ring(target):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    driver = RoundDriver(target, dflash_drafter=_draft(target), chunk=16)
    a, b = object(), object()
    driver.add(Request(list(range(4, 60)), 20, handle=a))
    driver.add(Request(list(range(8, 70)), 20, handle=b))
    driver.idle_budget = 16
    driver.step()
    row = next(r for r in driver.rows if r.req.handle is a)
    assert id(row) in driver.dflash.hidden
    driver.remove(a)
    assert id(row) not in driver.dflash.hidden
    while driver.rows[0].pending is None:
        driver.step()
    live = driver.rows[0]
    # the first token is out; the decode-batch join (which builds the ring) runs at the next step
    assert id(live) not in driver.dflash.rings and id(live) in driver.dflash.hidden
    driver.step()
    assert id(live) in driver.dflash.rings
    driver.remove(b)
    assert not driver.busy()
    assert not driver.dflash.hidden and not driver.dflash.rings


def test_cancel_between_the_first_token_and_the_deferred_join_leaks_nothing(
    target, monkeypatch
):
    """A row cancelled after its first token but before its (deferred) join keeps no capture
    window and never gets a ring; with ``dflash.leave`` broken the same sequence DOES leak, so
    the assertions can fail on a real cancel leak."""
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    def run(break_leave):
        driver = RoundDriver(target, dflash_drafter=_draft(target), chunk=16)
        a = object()
        driver.add(Request(list(range(4, 60)), 20, handle=a))
        while driver.rows[0].pending is None:
            driver.step()
        assert len(driver._pending_join) == 1
        if break_leave:
            monkeypatch.setattr(driver.dflash, "leave", lambda gone: None)
        driver.remove(a)
        driver.step()  # the deferred join must skip the removed row
        return driver

    clean = run(False)
    assert not clean.busy() and not clean.batch.rows and not clean._pending_join
    assert not clean.dflash.hidden and not clean.dflash.rings
    leaked = run(True)
    assert leaked.dflash.hidden  # the negative control: a broken leave() is caught


def test_more_than_node_budget_still_advances_every_row(monkeypatch):
    from yunshu_engine.kernels import lane_linear
    from yunshu_engine.round_driver.dflash import DFlashBatch

    batch = object.__new__(DFlashBatch)
    batch.rows = list(range(35))
    packs = []
    monkeypatch.setattr(
        batch, "_step_rows", lambda driver, rows: packs.append(rows) or rows
    )
    before = lane_linear.sum_reuse_enabled()
    assert batch.step(None) == list(range(35))
    assert list(map(len, packs)) == [32, 3]
    assert lane_linear.sum_reuse_enabled() == before


def test_dflash_apc_hit_matches_cold_and_uses_target_layout_salt(target):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    from .test_round_driver_apc import _manager

    apc = _manager()
    prompt = [4 + (7 * i) % 500 for i in range(700)]
    draft = _draft(target)

    def generate():
        driver = RoundDriver(target, dflash_drafter=draft, chunk=128, apc=apc)
        hit = driver.add(Request(prompt, 12, logprobs=True, handle=0))
        assert driver.apc_salt(0, True) == driver.apc_salt(0, False)
        out = []
        while driver.busy():
            out.extend((e.token, e.logprob) for e in driver.step())
        return hit, out

    hit0, cold = generate()
    hit1, warm = generate()
    assert hit0 == 0 and hit1 > 0
    assert warm == cold


def test_apc_hit_restores_the_drafters_context_window(target):
    """A prefix hit seeds the drafter with the window stored beside the checkpoint (the lane
    does this; without it a follow-up turn drafts from the uncached suffix only)."""
    from yunshu_engine import dflash_apc_window as win
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    from .test_round_driver_apc import _manager

    win.clear()
    win.set_budget(1 << 30)
    apc = _manager()
    prompt = [4 + (7 * i) % 500 for i in range(700)]
    draft = _draft(target)
    seen = []
    for _ in range(2):
        driver = RoundDriver(target, dflash_drafter=draft, chunk=128, apc=apc)
        hit = driver.add(Request(prompt, 4, handle=0))
        row = driver.rows[0]
        seen.append(
            (
                hit,
                int(driver.dflash.hidden[id(row)].shape[1])
                if id(row) in driver.dflash.hidden
                else 0,
            )
        )
        while driver.busy():
            driver.step()
    assert seen[0][0] == 0 and seen[1][0] > 0
    # the hit's context is the stored window (>= the checkpoint's positions up to the draft keep)
    assert seen[1][1] == min(seen[1][0], driver.dflash.keep)
    assert win.stats()["stored"] > 0
    win.clear()


def test_committed_tokens_are_learned_by_the_draft_vocabulary(target):
    """The lane learns every committed token into the reduced draft vocabulary; the
    driver must too (otherwise its drafts lose tokens the output just introduced)."""
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    draft = _draft(target)
    learned = []
    draft.__dict__["_draft_vocab"] = SimpleNamespace(
        learn=learned.extend, logits_subset=lambda hidden: None
    )
    driver = RoundDriver(target, dflash_drafter=draft, chunk=16)
    emitted = []
    driver.add(Request([5, 6, 7, 8, 9, 10], 12, handle=0))
    while driver.busy():
        emitted += [e.token for e in driver.step()]
    assert len(emitted) == 12
    assert learned == emitted[1 : 1 + len(learned)]
    assert len(learned) >= len(emitted) - 2


def test_oracle_copy_windows_commit_more_than_one_token(target, monkeypatch):
    from yunshu_engine.copy_drafter import CopyDrafter

    prompt = list(range(4, 25))
    ref, _ = _run(target, [prompt], draft=False, tokens=48)
    tokens = [t for t, _ in ref[0]]

    def oracle(copy, cap):
        generated = len(copy.ctx) - len(prompt)
        copy.last_match = 100
        return tokens[generated : generated + cap]

    monkeypatch.setattr(CopyDrafter, "draft", oracle)
    got, driver = _run(target, [prompt], draft=True, tokens=48)
    assert got == ref
    assert driver.accepted > 30
    assert driver.steps < 10


def test_guided_rows_stop_tree_at_entry_and_mask_each_body_token(target):
    import numpy as np

    class Guide:
        def __init__(self):
            self.position = 0

        @property
        def constrained(self):
            return 2 <= self.position < 5

        def arms(self, token):
            return self.position == 1

        def feed(self, token):
            self.position += 1

        def mask(self):
            if not self.constrained:
                return None
            words = np.zeros(16, dtype=np.uint32)
            token = 70 + self.position
            words[token // 32] = 1 << (token % 32)
            return words.view(np.int32)

    prompts = [list(range(4, 25)), list(range(10, 47))]
    ref, _ = _run(target, prompts, draft=False, logprobs=False, guide_factory=Guide)
    got, _ = _run(target, prompts, draft=True, logprobs=False, guide_factory=Guide)
    assert got == ref
    assert all([t for t, _ in row][2:5] == [72, 73, 74] for row in got)


def test_text_driver_clears_another_groups_mrope_before_prefill(target):
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    target._rope_deltas = mx.array([[17]], dtype=mx.float32)
    driver = RoundDriver(target, dflash_drafter=_draft(target), chunk=16)
    driver.add(Request(list(range(4, 25)), 4))
    driver.step()
    assert target._rope_deltas is None
    driver.remove(None)


def test_prefill_service_credit_reuses_shared_policy_and_bounds_dispatch(monkeypatch):
    from yunshu_engine.round_driver import driver as rd
    from yunshu_engine.serving.work_scheduler import DECODE_FIRST_BURST_S

    driver = object.__new__(rd.RoundDriver)
    driver.dflash = object()
    driver._prefilled_last = False
    driver.batch = SimpleNamespace(rows=[SimpleNamespace(generated=100)])
    driver.rows = [SimpleNamespace(pending=1), SimpleNamespace(pending=None)]
    calls = []
    monkeypatch.setattr(
        rd,
        "decode_quantum",
        lambda elapsed, early: calls.append((elapsed, early)) or 2.0,
    )
    driver._credit_prefill(4.0, True)
    assert calls == [(4.0, False)]
    assert driver._will_decode()
    driver._decode_credit = 0
    assert not driver._will_decode()  # waiting prompt gets the next atom
    driver.batch.rows[0].generated = 1
    driver._credit_prefill(4.0, False)
    assert calls[-1] == (4.0, True)
    assert driver._decode_credit == DECODE_FIRST_BURST_S
    driver.batch.rows.clear()
    driver._credit_prefill(4.0, True)
    assert driver._decode_credit == 0
    assert not driver._will_decode()


@pytest.mark.parametrize("sampled", [False, True])
def test_plain_verify_driver_without_a_drafter_equals_drafting_rows(target, sampled):
    """Speculation off: a draft-less DFlash batch decodes through the same verify forward and
    emits exactly the tokens of the drafting rows (it used to crash on res.captured and, with
    DecodeBatch, to use a different forward)."""
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    prompts = [list(range(4 + b, 25 + 7 * b)) for b in range(2)]
    want, _ = _run(target, prompts, draft=True, sampled=sampled, logprobs=False)
    driver = RoundDriver(target, plain_verify=True, chunk=16)
    assert driver.dflash is not None and not driver.can_draft
    got = [[] for _ in prompts]
    for i, prompt in enumerate(prompts):
        params = (
            SimpleNamespace(
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                xtc_probability=0.0,
                seed=17 + prompt[0],
            )
            if sampled
            else None
        )
        driver.add(Request(prompt, 12, sampling=params, handle=i, draft=False))
    steps = 0
    while driver.busy():
        for event in driver.step():
            got[event.handle].append((event.token, event.logprob))
        steps += 1
        assert steps < 200
    assert got == want
    assert driver.drafted == 0


def test_last_finished_row_offers_its_kv_buffers_to_the_restore_pool(target):
    """With the DFlash batch the row decodes in its own caches: when the last row finishes its K/V
    buffers go to the restore pool (a following cache hit restores into them)."""
    from yunshu_engine.kernels import buffer_cache
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    buffer_cache.clear_restore_buffers()
    prev = dict(buffer_cache._STATE)
    buffer_cache._STATE["limit"] = 1 << 30
    try:
        driver = RoundDriver(target, dflash_drafter=_draft(target), chunk=16)
        driver.add(Request(list(range(4, 60)), 6, handle=object()))
        while driver.busy():
            driver.step()
        assert buffer_cache.restore_buffer_bytes() > 0
    finally:
        buffer_cache.clear_restore_buffers()
        buffer_cache._STATE.update(prev)


def test_media_row_dflash_matches_plain_decode_beside_text_rows(target):
    """DFlash tree verify takes the row's mRoPE offset per window: a media row (positions with a
    jump, delta 7) decodes the same plainly, with the DFlash drafter, and beside a text row."""
    import mlx.core as mx

    from yunshu_engine.round_driver.driver import Request, RoundDriver

    prompt = list(range(4, 60))
    other = list(range(8, 70))
    n = len(prompt)
    pos = mx.tile(mx.arange(n, dtype=mx.int32)[None, :], (3, 1))
    pos = pos + (mx.arange(n) >= 10).astype(mx.int32) * 7
    embeds = target.model.embed_tokens(mx.array(prompt, dtype=mx.int32))

    def run(draft, others):
        driver = RoundDriver(
            target, dflash_drafter=_draft(target) if draft else None, chunk=16
        )
        driver.add(
            Request(
                prompt,
                20,
                handle="m",
                embeds=embeds,
                position_ids=pos,
                rope_delta=7,
            )
        )
        for i, p in enumerate(others):
            driver.add(Request(p, 20, handle=f"o{i}"))
        toks: dict = {}
        while driver.busy():
            for e in driver.step():
                toks.setdefault(e.handle, []).append(e.token)
        return toks["m"]

    plain = run(False, [])
    assert run(True, []) == plain
    assert run(True, [other]) == plain


def test_row_finishing_at_its_first_token_offers_its_kv_buffers(target):
    """A request that ends in its prefill step (max_tokens 1, the next-turn rehearsal) pools its
    K/V buffers too: the following cache hit restores into them."""
    from yunshu_engine.kernels import buffer_cache
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    buffer_cache.clear_restore_buffers()
    prev = dict(buffer_cache._STATE)
    buffer_cache._STATE["limit"] = 1 << 30
    try:
        driver = RoundDriver(target, dflash_drafter=_draft(target), chunk=16)
        driver.add(Request(list(range(4, 60)), 1, handle=object()))
        while driver.busy():
            driver.step()
        assert buffer_cache.restore_buffer_bytes() > 0
    finally:
        buffer_cache.clear_restore_buffers()
        buffer_cache._STATE.update(prev)


def test_static_token_bans_keep_the_dflash_row_drafting_and_equal_plain_decode(target):  # noqa: F811
    """ignore_eos / suppress_tokens are static bans: the row still drafts (the 27B follow-up turn of the
    bench sends ignore_eos and had lost speculation), applies the ban to every verify row, and produces
    the tokens a plain row with the same processor does."""
    from yunshu_engine.round_driver.driver import Request, RoundDriver
    from yunshu_engine.vlm_batch_runner import TokenMaskProcessor

    prompt = [(7 * i + 11) % 500 for i in range(120)]

    def run(drafter, ban):
        mask = TokenMaskProcessor(suppress=ban, eos_ids=[0], ignore_eos=True)
        d = RoundDriver(
            target,
            dflash_drafter=_draft(target) if drafter else None,
            plain_verify=not drafter,
            stop_tokens=set(),
            chunk=64,
        )
        d.add(Request(prompt, 24, handle=0, processors=[mask]))
        toks = []
        while d.busy():
            toks += [e.token for e in d.step()]
        return toks, d

    free, _ = run(False, [])
    ban = sorted(set(free))[:4]
    assert ban
    plain, _ = run(False, ban)
    drafted, d = run(True, ban)
    assert not set(ban) & set(drafted)
    assert drafted == plain
    assert d.drafted > 0  # the row speculated under the ban


@pytest.mark.parametrize("sampled", [False, True])
def test_presence_penalty_keeps_the_dflash_row_drafting_and_equal_plain_decode(
    target, sampled
):  # noqa: F811
    """A non-thinking request carries the card's presence_penalty 1.5: the row still drafts (the 8K sampled
    gapboard cell had decoded plain, 23 vs 53 tok/s), every verify row is penalized over the tokens fed up to
    it, and the tokens equal the same row decoded without drafting."""
    from yunshu_engine.round_driver.driver import Request, RoundDriver
    from yunshu_engine.text_logit_processors import GeneratedTokenPenalty
    from yunshu_engine.vlm_batch_runner import RowParams

    prompt = [(7 * i + 11) % 500 for i in range(120)]
    params = RowParams(0.7, 0.95, 20, 0.0, seed=1234) if sampled else None

    def run(draft, penalty=True, drafter=True):
        d = RoundDriver(
            target,
            dflash_drafter=_draft(target) if drafter else None,
            plain_verify=not drafter,
            stop_tokens=set(),
            chunk=64,
        )
        procs = [GeneratedTokenPenalty(0.5, 1.5)] if penalty else []
        d.add(
            Request(
                prompt, 32, handle=0, sampling=params, processors=procs, draft=draft
            )
        )
        toks = []
        while d.busy():
            toks += [e.token for e in d.step()]
        return toks, d

    free, _ = run(False, penalty=False)
    plain, _ = run(False)
    assert plain != free  # the penalty changes this decode
    drafted, d = run(True)
    assert drafted == plain
    assert d.drafted > 0
    # a spec-off server (no drafter) still keys a sampled row: the penalty must apply there too
    off, _ = run(True, drafter=False)
    assert off == plain


def test_absorb_keeps_only_the_window_not_every_chunks_captures():
    """The drafter window is evaluated as it is absorbed: left lazy, it chained every prefill chunk's
    captures until the first token (27B 128K prompt: +100 MiB per 2048-row chunk, +5 GiB at the end)."""
    from types import SimpleNamespace

    from yunshu_engine.round_driver.dflash import DFlashBatch

    d = DFlashBatch.__new__(DFlashBatch)
    d.can_draft, d.keep, d.hidden = True, 64, {}
    row = SimpleNamespace(finished=False)
    mx.synchronize()
    base = None
    for step in range(24):
        caps = [mx.random.normal((2048, 256)) for _ in range(2)]  # 4 MiB per chunk
        mx.eval(caps)  # the forward evaluated them with the step's caches
        d.absorb([SimpleNamespace(row=row, at=0, count=2048)], caps)
        del caps
        mx.synchronize()
        if step == 3:
            base = mx.get_active_memory()
    grown = mx.get_active_memory() - base
    assert grown < 8 * 2**20, (
        f"window retained {grown / 2**20:.0f} MiB of earlier chunks"
    )
    assert tuple(d.hidden[id(row)].shape) == (1, 64, 512)


def _oracle_caps(monkeypatch, tokens, prompt, caps, limit=None):
    from yunshu_engine import mtp_lane
    from yunshu_engine.copy_drafter import CopyDrafter

    monkeypatch.setattr(mtp_lane, "copy_rows_for_model", lambda lm, rows=None: 64)

    def oracle(copy, cap):
        generated = len(copy.ctx) - len(prompt)
        copy.last_match = 100
        caps.append(cap)
        return tokens[generated : generated + min(cap, limit or cap)]

    monkeypatch.setattr(CopyDrafter, "draft", oracle)


def test_lone_row_copy_round_commits_past_32_tokens(target, monkeypatch):
    prompt = list(range(4, 25))
    ref, _ = _run(target, [prompt], draft=False, tokens=90)
    tokens = [t for t, _ in ref[0]]
    caps = []
    _oracle_caps(monkeypatch, tokens, prompt, caps)
    got, driver = _run(target, [prompt], draft=True, tokens=90)
    assert got == ref
    assert max(caps) == 63
    assert driver.accepted > 50


def test_lone_row_64_row_round_near_max_tokens_stays_in_cache_buffers(
    target, monkeypatch
):
    """reserve_decode only sets the growth step: a 64-row round starting with less than
    64 tokens of room neither over-commits nor writes past the cache buffers."""
    from yunshu_engine import tree_verify

    prompt = list(range(4, 25))
    for max_tokens in (40, 63, 64, 65):
        ref, _ = _run(target, [prompt], draft=False, tokens=max_tokens)
        tokens = [t for t, _ in ref[0]]
        caps = []
        _oracle_caps(monkeypatch, tokens, prompt, caps)
        real, writes = tree_verify.compact_kv, []

        def spy(caches, src, n0, m, real=real, writes=writes):
            for c in caches:
                writes.append((n0 + m, c.keys.shape[2]))
            return real(caches, src, n0, m)

        monkeypatch.setattr(tree_verify, "compact_kv", spy)
        got, _ = _run(target, [prompt], draft=True, tokens=max_tokens)
        assert got == ref
        assert len(got[0]) <= max_tokens
        assert max(caps) <= 63
        assert all(end <= cap for end, cap in writes)


def test_packed_rows_copy_round_keeps_32_row_budget(target, monkeypatch):
    from yunshu_engine.round_driver import dflash as rd
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    prompts = [list(range(4, 25)), list(range(10, 47))]
    ref, _ = _run(target, prompts, draft=False, tokens=64)
    _oracle_caps(monkeypatch, [t for t, _ in ref[0]], prompts[0], [])
    real, seen = rd.forward, []

    def spy(lm, windows, *a, **k):
        seen.append((len(windows), sum(w.shape.width for w in windows)))
        return real(lm, windows, *a, **k)

    monkeypatch.setattr(rd, "forward", spy)
    driver = RoundDriver(target, dflash_drafter=_draft(target), chunk=16)
    for i, p in enumerate(prompts):
        driver.add(Request(p, 64, handle=i, draft=True))
    got = [[] for _ in prompts]
    for _ in range(100):
        if not driver.busy():
            break
        for e in driver.step():
            got[e.handle].append(e.token)
    packed = [w for n, w in seen if n == 2]
    assert packed, sorted(set(seen))
    assert max(packed) <= 32


@pytest.mark.parametrize("draft", [False, True])
def test_forced_runs_on_packed_rows_match_serial(target, monkeypatch, draft):
    from tests.unit.test_round_driver_generic import _ForcedGuide
    from yunshu_engine import settings

    original = settings.get
    results = []
    for enabled in (False, True):
        monkeypatch.setattr(
            settings,
            "get",
            lambda key, *a, **kw: (
                enabled if key == "YUNSHU_JUMP_FORWARD" else original(key, *a, **kw)
            ),
        )
        out, driver = _run(
            target,
            [list(range(4, 25))],
            draft=draft,
            tokens=15,
            logprobs=True,
            guide_factory=lambda: _ForcedGuide(list(range(7, 22)), vocab=512),
        )
        assert [t for t, _ in out[0]] == list(range(7, 22))
        results.append((out, driver.steps))
    assert results[0][0] == results[1][0]
    assert results[1][1] < results[0][1]


def test_forced_packed_window_hidden_and_next_state_equal_t1_bits(target):
    import copy

    from yunshu_engine import tree_verify as tv
    from yunshu_engine.round_driver.driver import Request, RoundDriver
    from yunshu_engine.round_driver.forward import logits
    from yunshu_engine.round_driver.verify import Window, forward

    driver = RoundDriver(target, plain_verify=True, stop_tokens=set(), chunk=16)
    driver.add(Request(list(range(4, 25)), 30))
    while not driver.batch.rows:
        driver.step()
    cache = driver.batch.rows[0].cache
    serial, window = copy.deepcopy(cache), copy.deepcopy(cache)
    tokens = list(range(7, 15))
    got = forward(
        driver.lm,
        [
            Window(
                mx.array([tokens]),
                tv.TreeShape(list(range(-1, len(tokens) - 1))),
                window,
            )
        ],
        [],
    )[0]
    refs = []
    for token in tokens:
        ref = forward(
            driver.lm, [Window(mx.array([[token]]), tv.TreeShape([-1]), serial)], []
        )[0]
        refs.append(ref.hidden)
        tv.tree_commit(driver.lm, serial, ref, [0])
    expected = mx.concatenate(refs, axis=1)
    assert mx.array_equal(got.hidden, expected).item()
    assert mx.array_equal(
        logits(driver.lm, got.hidden[0]), logits(driver.lm, expected[0])
    ).item()
    tv.tree_commit(driver.lm, window, got, list(range(len(tokens))))
    nxt = [
        forward(driver.lm, [Window(mx.array([[20]]), tv.TreeShape([-1]), c)], [])[0]
        for c in [serial, window]
    ]
    assert mx.array_equal(nxt[0].hidden, nxt[1].hidden).item()
    for c, res in zip([serial, window], nxt, strict=True):
        tv.tree_commit(driver.lm, c, res, [0])


@pytest.mark.parametrize("width", [2, 8, 16])
def test_forced_flash_window_hidden_and_next_state_equal_t1_bits(width):
    import copy

    from scripts.research.phaseb_verify import native_tiny

    from yunshu_engine.round_driver import flashnext

    if not mx.metal.is_available():
        pytest.skip("GPU window identity gate")
    lm = native_tiny()
    cache = lm.make_cache()
    lm(mx.array([[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]]), cache=cache)
    mx.eval([c.state for c in cache])
    serial, window = copy.deepcopy(cache), copy.deepcopy(cache)
    tokens = list(range(7, 7 + width))
    got, transaction = flashnext.chain_forward(lm, [mx.array([tokens])], [window])[0]
    refs = []
    for token in tokens:
        ref, tx = flashnext.chain_forward(lm, [mx.array([[token]])], [serial])[0]
        refs.append(ref)
        tx.commit(1)
    expected = mx.concatenate(refs, axis=1)
    assert mx.array_equal(got, expected).item()
    assert mx.array_equal(
        flashnext.logits(lm, got[0]), flashnext.logits(lm, expected[0])
    ).item()
    transaction.commit(width)
    nxt = [
        flashnext.chain_forward(lm, [mx.array([[40]])], [c])[0]
        for c in [serial, window]
    ]
    assert mx.array_equal(nxt[0][0], nxt[1][0]).item()
    for _, tx in nxt:
        tx.commit(1)
