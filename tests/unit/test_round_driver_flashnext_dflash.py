"""DFlash2 on Flash-Next driver rows (tiny random Flash-Next + tiny DFlash2, CPU)."""

# ruff: noqa: F811  (imported fixtures are used as test arguments)

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm.models.qwen4_exp.language")

from tests.unit.test_round_driver_flashnext import (  # noqa: E402,F401
    CHUNK,
    PROMPTS,
    _media_request,
    cpu,
    fresh_replay_store,
    run,
    run_requests,
    tiny,
)
from yunshu_engine.round_driver.driver import Request, RoundDriver  # noqa: E402


def make_dflash(lm):
    """(drafter, restore): the DFlash2 drafter and the undo of ``qwen4_dflash.install`` on the target classes."""
    from mlx_vlm.models.qwen4_exp.language import LanguageModel, Qwen4ExpDecoderLayer
    from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel

    from yunshu_engine import qwen4_dflash

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

    def restore():
        for (cls, name), value in saved.items():
            if value is None:
                if name in cls.__dict__:
                    delattr(cls, name)
            else:
                setattr(cls, name, value)

    return d, restore


@pytest.fixture(scope="module")
def dflash(tiny):
    d, restore = make_dflash(tiny[0])
    yield d
    restore()


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
@pytest.mark.parametrize("lone", [True, False])
def test_dflash_oracle_chains_commit_whole_windows(
    tiny, dflash, monkeypatch, corrupt_at, lone
):
    """Chains that are what plain decode produces next commit all their positions (QSA indexer, PLE history
    and GDN state advance by the whole window); a chain wrong at one depth commits the prefix and rolls the
    rest back. The stream is the plain one either way, with the next round drafted before the verify is read
    back (a lone row) or after (two rows)."""
    from yunshu_engine.round_driver.flashnext_dflash import FlashNextDFlash

    lm, _ = tiny
    for ids in PROMPTS:
        truth = run(lm, [ids], max_tokens=10)[0][0]

        def oracle(self, dh, anchor, row, truth=truth, generated=None):
            n = row.generated if generated is None else generated
            future = (truth[n : n + 15] + [0] * 15)[:15]
            if corrupt_at is not None:
                future[corrupt_at] = (future[corrupt_at] + 1) % 60
            return mx.array(future, dtype=mx.int32)

        monkeypatch.setattr(FlashNextDFlash, "_chain", oracle)
        got, _ = run_dflash(lm, dflash, [ids] if lone else [ids, PROMPTS[1]])
        assert got[0] == truth, (corrupt_at, lone)


def test_media_rows_on_dflash_keep_their_stream_beside_text_rows(tiny, dflash):
    lm, _ = tiny
    plain = run_requests(lm, [_media_request(lm, PROMPTS[2], shift=5, handle=3)])[0][3]
    got, driver = run_requests(
        lm,
        [
            _media_request(lm, PROMPTS[2], shift=5, handle=3),
            Request(PROMPTS[1], 10, handle=4),
        ],
        dflash=dflash,
    )
    assert got[3] == plain
    assert got[4] == run(lm, [PROMPTS[1]])[0][0]


def test_dflash_chain_depth_follows_the_cost_aware_budget(tiny, dflash):
    from yunshu_engine.round_driver.flashnext_dflash import CHAIN_DEPTH

    lm, _ = tiny
    driver = RoundDriver(
        SimpleNamespace(language_model=lm), dflash_drafter=dflash, chunk=CHUNK
    )
    batch = driver.dflash
    row = SimpleNamespace()
    budget = batch._budget(row)
    assert budget.size == min(
        CHAIN_DEPTH, batch.positions
    )  # the old lane's cap of seven drafts
    assert batch._budget(row) is budget
    assert batch.positions == dflash.config.block_size - 1


def test_tail_replay_on_dflash_rows(tiny, dflash):
    from yunshu_engine.round_driver import replay

    lm, _ = tiny
    ids = PROMPTS[0]

    def go():
        driver = RoundDriver(
            SimpleNamespace(language_model=lm),
            dflash_drafter=dflash,
            stop_tokens=set(),
            chunk=CHUNK,
        )
        driver.idle_budget = CHUNK
        out = []
        driver.add(Request(ids, 10, handle=0))
        for _ in range(200):
            if not driver.busy():
                break
            out += [int(e.token) for e in driver.step() if e.token is not None]
        return out

    first = go()
    assert replay.stats()["stored"] >= 1
    assert go() == first
    assert replay.stats()["replayed"] >= 1
