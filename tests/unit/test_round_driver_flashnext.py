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


@pytest.fixture(scope="module")
def tiny():
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
            True, list(procs), logprobs, None, dflash_driver=False, sampled=sampled
        )
        sampling = vbr.RowParams(0.7, 0.9, 0, 0.0, 5, 0.0) if sampled else None
        driver = RoundDriver(model, drafter=head, chunk=CHUNK, stop_tokens=set())
        assert driver.will_draft(
            sampling=sampling, processors=procs, logprobs=logprobs, draft=want
        ) == (
            want
            and driver._processors_draft(procs, logprobs)
            and not (sampled and logprobs)
        ), name
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
    assert seen_draft > 20 and seen_plain > 0


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


def test_flash_next_media_prompts_keep_the_stock_runner():
    from yunshu_engine import vlm_batch_runner as vbr

    def runner(flash):
        r = object.__new__(vbr.VLMBatchRunner)
        r.driver = SimpleNamespace(flash=flash)
        return r

    text = SimpleNamespace(prompt_kwargs=None)
    media = SimpleNamespace(prompt_kwargs={"pixel_values": 1})
    assert runner(True)._driver_takes(text)
    assert not runner(True)._driver_takes(
        media
    )  # the driver has no Flash-Next media input yet
    assert runner(False)._driver_takes(media)  # Qwen3.5 packs media itself
    stock = object.__new__(vbr.VLMBatchRunner)
    stock.driver = None
    assert not stock._driver_takes(text)


# ── DFlash2 on Flash-Next rows ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def dflash(tiny):
    from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel

    from yunshu_engine import qwen4_dflash

    lm, _ = tiny
    from mlx_vlm.models.qwen4_exp.language import LanguageModel, Qwen4ExpDecoderLayer

    saved = {
        (cls, name): cls.__dict__.get(name, None)
        for cls, name in (
            (Qwen4ExpDecoderLayer, "__call__"),
            (LanguageModel, "__call__"),
            (LanguageModel, "_yunshu_dflash_capture"),
            (LanguageModel, "speculative_verify_dflash_hidden"),
        )
    }
    qwen4_dflash.install(lm)
    mx.random.seed(11)
    cfg = DFlash2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=64,
        mask_token_id=7,
        max_position_embeddings=4096,
        block_size=6,
        num_target_layers=2,
        target_layer_ids=[0, 1],
        conv_kernel_size=2,
        conv_group_size=16,
        selector_rank=8,
        selector_top_k=4,
        layer_types=["sliding_attention"] * 2,
        sliding_window=32,
    )
    d = DFlash2DraftModel(cfg)
    mx.eval(d.parameters())
    yield d
    for (
        cls,
        name,
    ), value in saved.items():  # the install patches classes: leave them as found
        if value is None:
            if name in cls.__dict__:
                delattr(cls, name)
        else:
            setattr(cls, name, value)


def run_dflash(model, draft, ids_list, *, join_after=None, max_tokens=10):
    driver = RoundDriver(
        SimpleNamespace(language_model=model),
        dflash_drafter=draft,
        stop_tokens=set(),
        chunk=CHUNK,
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


def test_dflash_flash_next_driver_wiring(tiny, dflash):
    from yunshu_engine.round_driver.flashnext_dflash import FlashNextDFlash

    lm, _ = tiny
    driver = RoundDriver(
        SimpleNamespace(language_model=lm), dflash_drafter=dflash, chunk=CHUNK
    )
    assert isinstance(driver.dflash, FlashNextDFlash) and driver.batch is driver.dflash
    assert driver.head is None and driver.can_draft
    assert driver.dflash.capture_ids == [0, 1]


def test_dflash_solo_equals_plain_and_concurrent(tiny, dflash):
    lm, _ = tiny
    plain = [run(lm, [ids])[0][0] for ids in PROMPTS]
    solo = [run_dflash(lm, dflash, [ids])[0][0] for ids in PROMPTS]
    assert solo == plain
    together, driver = run_dflash(lm, dflash, PROMPTS)
    assert [together[i] for i in range(len(PROMPTS))] == plain
    assert driver.drafted > 0
    late, _ = run_dflash(lm, dflash, PROMPTS, join_after=[0, 4, 9])
    assert [late[i] for i in range(len(PROMPTS))] == plain


@pytest.mark.parametrize("corrupt_at", [None, 2])
def test_dflash_oracle_chains_commit_whole_windows(
    tiny, dflash, monkeypatch, corrupt_at
):
    """Chains that are what plain decode produces next commit all their positions (QSA indexer, PLE history
    and GDN state advance by the whole window); a chain wrong at one depth commits the prefix and rolls the
    rest back. The stream is the plain one either way."""
    from types import SimpleNamespace as NS

    from yunshu_engine import dflash_tree

    lm, _ = tiny
    for ids in PROMPTS:
        truth = run(lm, [ids], max_tokens=10)[0][0]

        def oracle(draft, dh, anchor, *, committed=0, truth=truth, **kw):
            future = (truth[committed : committed + 15] + [0] * 15)[:15]
            if corrupt_at is not None:
                future[corrupt_at] = (future[corrupt_at] + 1) % 60
            return NS(cands=mx.array(future, dtype=mx.int32)[:, None])

        monkeypatch.setattr(dflash_tree, "lattice_from_hidden", oracle)
        got, driver = run_dflash(lm, dflash, [ids])
        assert got[0] == truth, corrupt_at
        if corrupt_at is None:
            assert driver.accepted >= len(truth) - 3  # windows landed whole
