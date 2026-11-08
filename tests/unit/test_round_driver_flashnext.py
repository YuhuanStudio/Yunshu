"""Flash-Next (Qwen4: HC + QSA + GDN + MoE + PLE) rows on the round driver, on a tiny random 4-bit model
and a random native MTP head, CPU only.

A greedy row's tokens are the same alone, beside other rows, joining mid-flight, with the MTP head drafting or
without (spec on == spec off), and the rows really draft; each row's QSA indexer / PLE history / GDN state
lives in its own caches."""

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm.models.qwen4_exp.language")

from types import SimpleNamespace  # noqa: E402

from yunshu_engine.round_driver import flashnext  # noqa: E402
from yunshu_engine.round_driver.driver import Request, RoundDriver  # noqa: E402

PROMPTS = [
    [1 + (5 * i) % 60 for i in range(18)],
    [3 + (7 * i) % 58 for i in range(11)],
    [2 + (3 * i) % 61 for i in range(23)],
]
CHUNK = 8


@pytest.fixture(scope="module", autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture(autouse=True)
def fresh_replay_store():
    """The tail replay store is process-global: leave nothing of these tiny models behind."""
    from yunshu_engine.round_driver import replay

    replay.clear()
    yield
    replay.clear()


def make_tiny():
    from mlx.utils import tree_flatten
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.config import Qwen4ExpMTPConfig
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp import (
        Qwen4ExpMTPDraftModel,
    )
    from scripts.research.phaseb_verify import native_tiny
    from scripts.research.qwen4_mtp_probe import tiny_config

    from yunshu_engine.qwen4_mtp import load_native_head

    lm = native_tiny()
    mx.random.seed(5)
    ref = Qwen4ExpMTPDraftModel(
        Qwen4ExpMTPConfig.from_dict({"text_config": tiny_config()})
    )
    head = load_native_head(
        {"text_config": tiny_config()}, dict(tree_flatten(ref.parameters()))
    )
    mx.eval(head.parameters())  # float32: CPU gather_mm takes nothing else
    return lm, head


@pytest.fixture(scope="module")
def tiny():
    return make_tiny()


def run(model, ids_list, *, head=None, join_after=None, max_tokens=10):
    """Drive the rows to completion; returns each row's tokens (and the driver)."""
    driver = RoundDriver(
        SimpleNamespace(language_model=model),
        drafter=head,
        plain_verify=head is None,
        stop_tokens=set(),
        chunk=CHUNK,
        max_draft=1,
    )
    out = {i: [] for i in range(len(ids_list))}
    pending = list(enumerate(ids_list))
    steps = 0
    while pending or driver.busy():
        while pending and (join_after is None or steps >= join_after[pending[0][0]]):
            i, ids = pending.pop(0)
            driver.add(Request(ids, max_tokens, handle=i))
        for event in driver.step():
            if event.token is not None:
                out[event.handle].append(int(event.token))
        steps += 1
        assert steps < 400
    return out, driver


def test_flash_next_is_detected_and_takes_flash_batch(tiny):
    lm, head = tiny
    assert flashnext.is_flash_next(lm)
    driver = RoundDriver(SimpleNamespace(language_model=lm), drafter=head, chunk=CHUNK)
    assert driver.flash and isinstance(driver.batch, flashnext.FlashNextBatch)
    assert isinstance(driver.head, flashnext.FlashNextHead)
    assert driver.dflash is None and driver.can_draft
    assert driver.max_draft == 1
    plain = RoundDriver(
        SimpleNamespace(language_model=lm), chunk=CHUNK, plain_verify=True
    )
    assert plain.flash and plain.head is None and not plain.can_draft
    assert (
        plain.dflash is None
    )  # no Qwen3.5 tree-verify batch for a plain Flash-Next row


def test_flash_spans_are_the_grid_plus_checkpoints_only(tiny):
    lm, _ = tiny
    driver = RoundDriver(SimpleNamespace(language_model=lm), chunk=CHUNK)
    row = SimpleNamespace(req=SimpleNamespace(ids=list(range(21))), ckpts=[12])
    # grid 8: spans 0-8, 8-12 (cut at the checkpoint), 12-16, 16-21 (no 64-aligned tail cut)
    ends, at = [], 0
    while at < 21:
        at = driver._run_end(row, at, 10**6)
        ends.append(at)
    assert ends == [8, 12, 16, 21]


def test_solo_spec_on_equals_spec_off(tiny):
    lm, head = tiny
    for ids in PROMPTS:
        plain, _ = run(lm, [ids])
        spec, driver = run(lm, [ids], head=head)
        assert spec[0] == plain[0], ids
        assert len(plain[0]) == 10
        assert driver.drafted > 0, "the row never drafted"


def test_rows_beside_each_other_equal_their_solo_runs(tiny):
    lm, head = tiny
    solo = [run(lm, [ids], head=head)[0][0] for ids in PROMPTS]
    together, driver = run(lm, PROMPTS, head=head)
    assert [together[i] for i in range(len(PROMPTS))] == solo
    assert driver.drafted > 0 and driver.accepted >= 0


def test_rows_joining_mid_flight_and_plain_rows_agree(tiny):
    lm, head = tiny
    solo = [run(lm, [ids])[0][0] for ids in PROMPTS]
    late, _ = run(lm, PROMPTS, head=head, join_after=[0, 3, 7])
    assert [late[i] for i in range(len(PROMPTS))] == solo
    plain_together, _ = run(lm, PROMPTS)
    assert [plain_together[i] for i in range(len(PROMPTS))] == solo


def test_each_row_owns_its_state(tiny):
    lm, head = tiny
    driver = RoundDriver(
        SimpleNamespace(language_model=lm),
        drafter=head,
        stop_tokens=set(),
        chunk=CHUNK,
        max_draft=1,
    )
    for i, ids in enumerate(PROMPTS[:2]):
        driver.add(Request(ids, 6, handle=i))
    for _ in range(40):
        driver.step()
        if len(driver.batch.rows) == 2:
            break
    a, b = driver.batch.rows
    assert a.cache is not b.cache and a.cache is not None
    for ca, cb in zip(a.cache, b.cache, strict=True):
        assert ca is not cb
    # QSA indexer state and the GDN / PLE arrays are per row and of the row's own length
    qsa = [c for c in a.cache if getattr(c, "index_keys", None) is not None]
    assert qsa and all(int(c.index_keys.shape[1]) == int(c.offset) for c in qsa)
    assert a.n != b.n
    heads = driver.head.caches
    assert set(heads) == {id(a), id(b)} and heads[id(a)] is not heads[id(b)]
    # the arrays the driver evaluates between steps cover the indexer keys and the head's cache
    mine = {id(x) for x in flashnext.cache_arrays(a.cache)}
    assert any(id(c.index_keys) in mine for c in qsa)
    assert driver.head.slots.arrays()


def test_head_cache_is_trimmed_to_the_positions_that_landed(tiny):
    lm, head = tiny
    h = flashnext.FlashNextHead(head, lm, lambda x: flashnext.logits(lm, x))
    cache = h.make_cache()
    width = lm.args.hc_count * lm.args.hidden_size
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    state = mx.zeros((1, 4, width), dtype=mx.bfloat16)
    h._step(cache, tokens, state)
    assert int(cache[0].offset) == 4
    h._sync(cache, 2)
    assert int(cache[0].offset) == 2
    assert all(c.index_keys is None or int(c.index_keys.shape[1]) == 2 for c in cache)
    h._step(cache, tokens[:, :1], state[:, :1])
    assert int(cache[0].offset) == 3


def test_packs_split_at_the_fused_kernel_row_limit(tiny):
    lm, _ = tiny
    batch = flashnext.FlashNextBatch(lm)
    assert batch.packs([2] * 10) == [list(range(8)), [8, 9]]
    assert batch.packs([1] * 17) == [list(range(16)), [16]]
    assert batch.packs([2]) == [[0]]


def test_kv_precision_other_than_bf16_is_refused(tiny):
    lm, _ = tiny
    with pytest.raises(ValueError, match="not available for Flash-Next"):
        flashnext.FlashNextBatch(lm, "int8")


def test_accepted_drafts_keep_the_plain_tokens(tiny, monkeypatch):
    """An oracle head (every draft is the token plain decode produces next): windows of two positions
    commit both, the QSA indexer / PLE / GDN state advance by two, and the stream is still the plain one."""
    lm, head = tiny
    plain, _ = run(lm, PROMPTS)
    original = flashnext.FlashNextHead.draft

    def oracle(self, rows, heads, depths, first=None):
        out = original(self, rows, heads, depths, first)
        for b, (row, depth) in enumerate(zip(rows, depths, strict=True)):
            truth = plain[row.req.handle]
            if depth and row.generated < len(truth):
                out[b] = mx.array([truth[row.generated]], dtype=mx.int32)
        return out

    monkeypatch.setattr(flashnext.FlashNextHead, "draft", oracle)
    got, driver = run(lm, PROMPTS, head=head)
    assert got == plain
    assert driver.accepted > 0
    assert driver.accepted >= driver.drafted - len(PROMPTS) * 3


# ── routing: Flash-Next rows take the driver, and draft per request kind as MTP head rows do ──────────
def test_flash_next_takes_the_driver_without_the_lane_self_test(tiny, monkeypatch):
    from yunshu_engine import vlm_engine
    from yunshu_engine.kernels.tensorfold import lane_qmm

    lm, _ = tiny
    monkeypatch.setattr(lane_qmm, "ready", lambda: False)
    engine = object.__new__(vlm_engine.VLMEngine)
    assert engine._round_driver_wanted(lm) is True
    # a Qwen3.5 target still needs the lane arithmetic (the Flash-Next branch is by family)
    assert engine._round_driver_wanted(SimpleNamespace(model=object())) is False


def test_the_not_qwen4_gate_is_gone():
    import inspect

    from yunshu_engine import vlm_engine

    src = inspect.getsource(vlm_engine.VLMEngine._build_batch_runner)
    assert "use_driver = spec_family and self._round_driver_wanted(lm)" in src
    assert "not qwen4 and spec_family" not in src


def _kinds():
    from tests.unit.test_driver_plain_rows import _processors

    for name, procs in _processors().items():
        for logprobs in (False, True):
            for sampled in (False, True):
                yield name, procs, logprobs, sampled


def test_request_kinds_draft_on_flash_rows_as_on_mtp_head_rows(tiny):
    """The runner's ``allow_draft`` rule (``driver_allow_draft`` for an MTP head row) and the driver's own
    ``add`` agree on a Flash-Next row, kind by kind: the rows that decode plain are exactly the listed
    plain kinds of test_driver_plain_rows (sampled + logprobs, XTC, non-invariant guides), nothing new."""
    from yunshu_engine import vlm_batch_runner as vbr

    lm, head = tiny
    model = SimpleNamespace(language_model=lm)
    seen_draft = seen_plain = 0
    for name, procs, logprobs, sampled in _kinds():
        want = vbr.driver_allow_draft(
            True, list(procs), logprobs, None, sampled=sampled
        )
        sampling = vbr.RowParams(0.7, 0.9, 0, 0.0, 5, 0.0) if sampled else None
        driver = RoundDriver(model, drafter=head, chunk=CHUNK, stop_tokens=set())
        assert driver.will_draft(
            sampling=sampling, processors=procs, logprobs=logprobs, draft=want
        ) == (want and driver._processors_draft(procs, logprobs)), name
        driver.add(
            Request(
                PROMPTS[1],
                4,
                sampling=sampling,
                processors=list(procs),
                logprobs=logprobs,
                draft=want,
                handle=0,
            )
        )
        row = driver.rows[0]
        assert row.drafting == bool(want), (name, logprobs, sampled)
        assert (row.mtp_cache is not None) == bool(want)
        seen_draft += bool(want)
        seen_plain += not want
    assert seen_draft > 20 and seen_plain == 0  # XTC alone stays plain (no keyed draw)
    assert not vbr.driver_allow_draft(True, [], False, None, xtc=True)


def test_flash_row_with_processors_and_a_guide_free_request_matches_solo(tiny):
    """A drafting row that applies a static ban is the same stream alone and beside another row."""
    from yunshu_engine import vlm_batch_runner as vbr

    lm, head = tiny
    ban = [vbr.TokenMaskProcessor(suppress=[5, 6, 7])]

    def go(ids_list):
        driver = RoundDriver(
            SimpleNamespace(language_model=lm),
            drafter=head,
            stop_tokens=set(),
            chunk=CHUNK,
            max_draft=1,
        )
        out = {i: [] for i in range(len(ids_list))}
        for i, ids in enumerate(ids_list):
            driver.add(Request(ids, 8, processors=list(ban), handle=i))
        for _ in range(300):
            if not driver.busy():
                break
            for e in driver.step():
                if e.token is not None:
                    out[e.handle].append(int(e.token))
        return out

    alone = go([PROMPTS[0]])[0]
    both = go([PROMPTS[0], PROMPTS[2]])
    assert both[0] == alone and len(alone) == 8
    assert not set(alone) & {5, 6, 7}


def test_flash_next_media_prompts_take_the_driver():
    from yunshu_engine import vlm_batch_runner as vbr

    admitted = []
    r = object.__new__(vbr.VLMBatchRunner)
    r.driver = SimpleNamespace(flash=True)
    r._admit_driver = lambda job, use_apc: admitted.append(job)
    r.apc_manager = None
    r._decode_follows = lambda job: False
    for pkw in (None, {"pixel_values": 1}):
        job = SimpleNamespace(
            ids=[1],
            guide=None,
            priority=0,
            cache_plan=None,
            prompt_kwargs=pkw,
            stats=vbr.RunStats(),
        )
        r._admit(job, True)
        assert (
            admitted[-1] is job
        )  # text and media prompts alike: the driver, no other path


# ── DFlash2 on Flash-Next rows ─────────────────────────────────────────────────────────────────
# ── explicit cache plans ride on the driver ─────────────────────────────────────────────────────
class _Coordinator:
    enabled = True
    is_checkpoint = True
    media_checkpoint = False

    def __init__(self, plan_points):
        self.plan_points = plan_points
        self.policy = None
        self.stored = []
        self.manager = SimpleNamespace(
            exact_cache_min_tokens=1,
            lock=__import__("contextlib").nullcontext(),
            stats=SimpleNamespace(restored_tokens=0),
        )

    def set_request(self, ids, policy):
        self.policy = (tuple(ids), policy)

    def lookup(self, ids, **kw):
        return None

    def prepare_prefill(self, *a, **kw):
        pass

    def checkpoint_lengths(self, ids, media, *, begin=True):
        return list(self.plan_points) if self.policy else [len(ids) - 1]

    def store_checkpoint(self, ids, caches, extra_hash=0):
        self.stored.append(len(ids))


def test_explicit_cache_plan_steers_the_row_checkpoints(tiny):
    lm, _ = tiny
    coord = _Coordinator([12])
    manager = SimpleNamespace(coordinator=lambda model: coord)
    driver = RoundDriver(
        SimpleNamespace(language_model=lm),
        stop_tokens=set(),
        chunk=CHUNK,
        apc=manager,
        plain_verify=True,
    )
    plan = {"points": [(12, 300)], "lookup_points": [12]}
    driver.add(Request(PROMPTS[2], 3, handle=0, cache_plan=plan))
    assert coord.policy == (tuple(PROMPTS[2]), plan)
    for _ in range(100):
        if not driver.busy():
            break
        driver.step()
    driver.flush_checkpoints()
    assert coord.stored == [12]  # stored at the seam, not at an automatic point


def test_cache_plan_requests_take_the_driver_and_salt_their_own_namespace():
    from yunshu_engine import vlm_batch_runner as vbr

    r = object.__new__(vbr.VLMBatchRunner)
    r.driver = SimpleNamespace(flash=True)
    r.apc_semantic_hash = 7
    plain = SimpleNamespace(cache_plan=None, prompt_kwargs=None, salt=None)
    planned = SimpleNamespace(cache_plan={"points": []}, prompt_kwargs=None, salt=None)
    assert r._driver_extra_hash(plain) == 7
    assert r._driver_extra_hash(planned) not in (0, 7)


# ── media prompts (prompt embeddings + mRoPE positions + a per-row decode offset) ───────────────
def _media_request(lm, ids, shift=0, handle=0, max_tokens=10):
    emb = lm.model.embed_tokens(mx.array(ids, dtype=mx.int32))
    pos = mx.broadcast_to(
        (mx.arange(len(ids), dtype=mx.int32) + shift)[None], (3, len(ids))
    )
    return Request(
        ids, max_tokens, handle=handle, embeds=emb, position_ids=pos, rope_delta=shift
    )


def run_requests(lm, requests, *, head=None, dflash=None):
    driver = RoundDriver(
        SimpleNamespace(language_model=lm),
        drafter=head,
        dflash_drafter=dflash,
        plain_verify=head is None and dflash is None,
        stop_tokens=set(),
        chunk=CHUNK,
        max_draft=1,
    )
    out = {r.handle: [] for r in requests}
    for r in requests:
        driver.add(r)
    for _ in range(400):
        if not driver.busy():
            break
        for e in driver.step():
            if e.token is not None:
                out[e.handle].append(int(e.token))
    assert not driver.busy()
    return out, driver


def test_media_rows_equal_text_rows_at_zero_offset_and_solo_equals_concurrent(tiny):
    lm, head = tiny
    ids = PROMPTS[0]
    text = run(lm, [ids])[0][0]
    media = run_requests(lm, [_media_request(lm, ids)])[0][0]
    assert (
        media == text
    )  # embeddings of the tokens with sequential positions are the text prompt
    solo = run_requests(lm, [_media_request(lm, PROMPTS[2], shift=5, handle=7)])[0][7]
    both, _ = run_requests(
        lm,
        [
            _media_request(lm, PROMPTS[2], shift=5, handle=7),
            Request(PROMPTS[1], 10, handle=8),
            _media_request(lm, ids, shift=3, handle=9),
        ],
    )
    assert both[7] == solo
    solo9 = run_requests(lm, [_media_request(lm, ids, shift=3, handle=9)])[0][9]
    assert both[9] == solo9
    assert both[8] == run(lm, [PROMPTS[1]])[0][0]


def test_media_rows_draft_and_keep_their_stream(tiny):
    lm, head = tiny
    request = lambda: _media_request(lm, PROMPTS[2], shift=5, handle=3)  # noqa: E731
    plain = run_requests(lm, [request()])[0][3]
    spec, driver = run_requests(lm, [request()], head=head)
    assert spec[3] == plain and driver.drafted > 0


def test_finished_head_row_stores_target_and_head_caches_together(tiny):
    """The reply cache of a drafting row carries the head's KV after the target's, cut to the same rows."""
    lm, head = tiny
    driver = RoundDriver(
        SimpleNamespace(language_model=lm),
        drafter=head,
        stop_tokens=set(),
        chunk=CHUNK,
        max_draft=1,
    )
    driver.add(Request(PROMPTS[0], 8, handle=0))
    seen = {}

    def hook(row):
        seen["caches"] = flashnext.decode_caches(driver, row)
        seen["n"], seen["hn"] = row.n, row.hn
        return False

    driver.finish_hook = hook
    for _ in range(100):
        if not driver.busy():
            break
        driver.step()
    caches = seen["caches"]
    assert caches is not None and len(caches) == len(lm.layers) + len(head.layers)
    target, hcache = caches[: len(lm.layers)], caches[len(lm.layers) :]
    assert int(target[lm.model.fa_idx].offset) == seen["n"]
    assert int(hcache[0].offset) == seen["n"]
    assert all(
        c.index_keys is None or int(c.index_keys.shape[1]) == seen["n"] for c in hcache
    )
    assert flashnext.decode_caches(driver, SimpleNamespace(hn=0, n=0)) is None


def test_device_walk_equals_the_host_walk():
    from yunshu_engine.round_driver.flashnext_dflash import walk_on_device

    cases = [
        ([5, 1, 2, 3], [1, 2, 9, 4]),  # two drafts land, third misses
        ([5, 1, 2, 3], [7, 2, 9, 4]),  # first misses
        ([5, 1, 2, 3], [1, 2, 3, 8]),  # all land: bonus after the last
        ([5], [8]),  # a window of one
    ]
    for window, target in cases:
        path = [0]
        while path[-1] + 1 < len(window) and window[path[-1] + 1] == target[path[-1]]:
            path.append(path[-1] + 1)
        used, bonus = walk_on_device(
            mx.array(window, dtype=mx.int32), mx.array(target, dtype=mx.int32)
        )
        assert int(used.item()) == len(path)
        assert int(bonus.item()) == target[path[-1]]


# ── tail replay carries the QSA indexer and PLE state ───────────────────────────────────────────
@pytest.mark.parametrize("mode", ["mtp", "plain"])
def test_tail_replay_gives_the_stream_of_a_real_forward(tiny, mode):
    from yunshu_engine.round_driver import replay

    lm, head = tiny
    replay.clear()
    kinds = {"mtp": dict(head=head), "plain": {}}[mode]
    ids = PROMPTS[0]  # 18 tokens on a grid of 8: a 2-row final block (replay-eligible)

    def go():
        driver = RoundDriver(
            SimpleNamespace(language_model=lm),
            drafter=kinds.get("head"),
            dflash_drafter=kinds.get("dflash"),
            plain_verify=not kinds,
            stop_tokens=set(),
            chunk=CHUNK,
            max_draft=1,
        )
        driver.idle_budget = (
            CHUNK  # one span a step, as a 2048-token grid span fills the real budget
        )
        out = []
        driver.add(Request(ids, 10, handle=0))
        for _ in range(200):
            if not driver.busy():
                break
            out += [int(e.token) for e in driver.step() if e.token is not None]
        return out

    first = go()
    assert replay.stats()["stored"] >= 1
    second = go()
    assert replay.stats()["replayed"] >= 1, "the repeat did not replay its tail"
    assert second == first
    replay.clear()


def test_replay_entry_keeps_the_indexer_rows(tiny):
    from yunshu_engine.round_driver import replay

    lm, _ = tiny
    cache = lm.make_cache()
    lm(mx.array([PROMPTS[0][:12]]), cache=cache)
    lm(mx.array([PROMPTS[0][12:18]]), cache=cache)
    mx.eval([c.state for c in cache])
    replay.clear()
    replay.store("k", cache, 12, 18, mx.zeros((6, 4)), [])
    entry = replay._STORE["k"]
    tags = [tag for tag, _ in entry.slots]
    assert "q" in tags and "a" in tags
    fresh = lm.make_cache()
    lm(mx.array([PROMPTS[0][:12]]), cache=fresh)
    mx.eval([c.state for c in fresh])
    assert replay.lookup("k", fresh, 12, 18) is entry
    replay.restore(entry, "k", fresh)
    for a, b in zip(cache, fresh, strict=True):
        if getattr(a, "index_keys", None) is not None:
            assert mx.array_equal(a.index_keys, b.index_keys).item()
            assert mx.array_equal(a.keys[..., :18, :], b.keys[..., :18, :]).item()
        elif hasattr(a, "cache"):
            for x, y in zip(a.cache, b.cache, strict=True):
                assert (x is None) == (y is None)
                if x is not None:
                    assert mx.array_equal(x, y).item()
    replay.clear()


# ── sampled + logprobs rows draft on MTP head rows: the report of a window equals plain steps ───
def _draw_driver():
    driver = object.__new__(RoundDriver)
    driver.dflash = None
    return driver


def _sampled_row(logprobs=True, top=2, seed=11):
    from yunshu_engine import keyed_sampling
    from yunshu_engine import vlm_batch_runner as vbr
    from yunshu_engine.round_driver.driver import _Row

    params = vbr.RowParams(0.8, 0.95, 0, 0.0, seed, 0.0)
    req = Request(
        [1, 2, 3],
        8,
        sampling=params,
        logprobs=logprobs,
        top_logprobs=top,
        handle=0,
    )
    row = _Row(
        req=req,
        cache=None,
        mtp_cache=None,
        keyed=keyed_sampling.KeyedSampler(params, seed),
        base=keyed_sampling.seed_base(seed),
        drafting=True,
    )
    row.generated = 3
    return row


@pytest.mark.parametrize("vocab", [4096, 151936])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_window_report_is_bit_equal_to_plain_steps(vocab, dtype):
    """A two-row verify window (an MTP draft) draws and reports every position with the bits of the plain
    step that would have run there: logprobs, the drawn token and the top-k report."""
    from yunshu_engine.round_driver.driver import _Item

    mx.random.seed(3)
    rows = (mx.random.normal((2, vocab)) * 3).astype(dtype)
    d = _draw_driver()
    window_row = _sampled_row()
    ((it, tok2, extra2),) = d._draw(
        [_Item("d", window_row, at=0, count=2)], None, target_logits=rows
    )
    for j in range(2):
        plain_row = _sampled_row()
        plain_row.generated = 3 + j
        ((_, tok1, extra1),) = d._draw(
            [_Item("d", plain_row, at=0, count=1)], None, target_logits=rows[j : j + 1]
        )
        assert int(tok1[0].item()) == int(tok2[j].item())
        got, want = extra2[j], extra1
        assert mx.array_equal(
            got[0], want[0]
        ).item()  # the chosen token's logprob, bit for bit
        for a, b in zip(got[1:], want[1:], strict=True):
            for x, y in zip(a, b, strict=True):
                assert mx.array_equal(x, y).item()


def test_row_logprobs_reduces_each_row_on_its_own():
    from yunshu_engine.round_driver.driver import row_logprobs

    x = (mx.random.normal((5, 8192)) * 4).astype(mx.bfloat16)
    got = row_logprobs(x)
    for j in range(5):
        want = x[j : j + 1] - mx.logsumexp(x[j : j + 1], axis=-1, keepdims=True)
        assert mx.array_equal(got[j : j + 1], want).item()
    one = row_logprobs(x[:1])
    assert mx.array_equal(one, got[:1]).item()


def test_sampled_logprobs_rows_draft_on_flash_mtp_rows(tiny):
    """End to end on the tiny model: a seeded sampled request with logprobs drafts, and its tokens and
    reports equal the same request decoded plain."""
    from yunshu_engine import vlm_batch_runner as vbr

    lm, head = tiny
    params = vbr.RowParams(0.8, 0.95, 0, 0.0, 21, 0.0)

    def go(drafter):
        driver = RoundDriver(
            SimpleNamespace(language_model=lm),
            drafter=drafter,
            plain_verify=drafter is None,
            stop_tokens=set(),
            chunk=CHUNK,
            max_draft=1,
        )
        driver.add(
            Request(
                PROMPTS[0],
                10,
                sampling=params,
                logprobs=True,
                top_logprobs=2,
                draft=drafter is not None,
                handle=0,
            )
        )
        out = []
        for _ in range(300):
            if not driver.busy():
                break
            for e in driver.step():
                if e.token is not None:
                    out.append((int(e.token), e.logprob))
        return out, driver

    plain, _ = go(None)
    spec, driver = go(head)
    assert driver.rows == [] and driver.drafted > 0
    assert [t for t, _ in spec] == [t for t, _ in plain]
    for (_, a), (_, b) in zip(spec, plain, strict=True):
        assert a is not None and a == b


def test_head_prefill_stays_queued_past_the_first_token_and_changes_nothing(
    tiny, monkeypatch
):
    """The MTP head's prompt prefill is not evaluated before a row's first token (it ran on the
    TTFT path); with the retained-bytes bound at zero it is evaluated per step again, and the
    tokens are the same either way."""
    from yunshu_engine.round_driver import driver as drv

    lm, head = tiny
    ids = PROMPTS[0]
    lazy, d1 = run(lm, [ids], head=head)
    assert d1._lazy_ids and all(d1._lazy_ids.values())
    monkeypatch.setattr(drv, "HEAD_LAZY_MAX", 0)
    eager, d2 = run(lm, [ids], head=head)
    assert d2._lazy_ids and not any(d2._lazy_ids.values())
    assert lazy == eager
