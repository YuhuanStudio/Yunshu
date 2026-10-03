"""YunshuAPCManager through the real mlx-vlm BatchGenerator on a tiny hybrid (GDN) Qwen3.5.

A conversation that grows reuses its own last checkpoint, a new session with the same system
turn reuses the head checkpoint, older checkpoints of a grown conversation are dropped, and
every output equals a cold run's (lossless).
"""

import random
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from yunshu_engine.apc_manager import YunshuAPCManager  # noqa: E402
from yunshu_engine.vlm_batch_runner import RunStats, VLMBatchRunner  # noqa: E402

IM_START, USER = 900, 901


class _Stop:
    def __init__(self):
        self.eos = set()

    def add_eos_token_ids(self, ids):
        self.eos |= set(ids or [])

    def __call__(self, token):
        return int(token) in self.eos


def _processor():
    return SimpleNamespace(tokenizer=SimpleNamespace(stopping_criteria=_Stop()))


@pytest.fixture(scope="module", params=[mx.bfloat16, mx.float32])
def model(request):
    from yunshu_engine.utils.hardware import is_paravirtual_metal

    if request.param == mx.bfloat16 and is_paravirtual_metal():
        pytest.skip(
            "Apple Paravirtual air64 stock BF16 prefill is span-dependent; "
            "serving disables APC there (FP32 cache roundtrips remain tested)"
        )
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    cfg = dict(
        model_type="qwen3_5",
        hidden_size=256,
        intermediate_size=512,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=1024,
        num_key_value_heads=1,
        max_position_embeddings=4096,
        head_dim=256,
        tie_word_embeddings=False,
    )
    mx.random.seed(3)
    lm = LanguageModel(
        TextConfig(**cfg),
        SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2),
            image_token_id=1020,
            video_token_id=1021,
            vision_start_token_id=1022,
        ),
    )
    lm.set_dtype(request.param)
    # Serving calls eval(): training uses a chunked parallel GDN scan whose
    # BF16 arithmetic is not checkpoint-span invariant on every Apple GPU.
    lm.eval()
    mx.eval(lm.parameters())

    class Embeds:
        def __init__(self, e):
            self.e = e

        def to_dict(self):
            return {"inputs_embeds": self.e}

    return SimpleNamespace(
        language_model=lm,
        config=SimpleNamespace(image_token_index=None),
        get_input_embeddings=lambda ids, pv, mask=None, **kw: Embeds(
            lm.model.embed_tokens(ids)
        ),
    )


def _run(runner, ids):
    stats = RunStats()
    out = list(runner.iter_tokens(ids, max_tokens=6, stats=stats, allow_draft=False))
    return out, stats


def test_growth_head_reuse_supersede_and_lossless_output(model):
    rnd = random.Random(1)

    def toks(n):
        return [rnd.randrange(3, 800) for _ in range(n)]

    head = toks(300)
    s1 = [7, *head, IM_START, USER, *toks(200)]
    s1b = s1 + toks(90)  # the same conversation, longer
    s2 = [7, *head, IM_START, USER, *toks(150)]  # a new session, same system turn

    mgr = YunshuAPCManager(
        num_blocks=64,
        block_size=16,
        overrides={"memory_max_gb": 1, "checkpoint_interval_tokens": 128},
        head_marker=(IM_START, USER),
    )
    runner = VLMBatchRunner(
        model, processor=_processor(), apc_manager=mgr, apc_semantic_hash=0
    )
    outs, stats = [], []
    for ids in (s1, s1b, s2):
        o, st = _run(runner, ids)
        outs.append(o)
        stats.append(st)

    assert stats[0].cached_tokens == 0 and stats[0].cache_tier == "none"
    assert stats[1].cached_tokens == len(s1) - 1 and stats[1].cache_tier == "ram"
    assert stats[2].cached_tokens == 301 and stats[2].cache_tier == "ram"  # the head
    # s1's own checkpoints were superseded by s1b's; the head and both sessions' latest remain
    assert sorted(mgr.snapshot()["entry_tokens"]) == [301, 384, 452, 512, 592]
    assert mgr.snapshot()["head_checkpoints"] == 1

    cold = VLMBatchRunner(model, processor=_processor())
    for ids, got in zip((s1, s1b, s2), outs, strict=True):
        assert _run(cold, ids)[0] == got


def test_ssd_reload_gives_the_same_tokens_as_ram_and_cold(model, tmp_path):
    from yunshu_engine.apc_manager import SpillDiskStore

    rnd = random.Random(5)
    a = [rnd.randrange(3, 800) for _ in range(260)]
    b = [rnd.randrange(3, 800) for _ in range(260)]
    a2 = a + [rnd.randrange(3, 800) for _ in range(40)]

    def manager(**kw):
        disk = SpillDiskStore(
            tmp_path, namespace="e2e", num_workers=1, max_bytes=1 << 30
        )
        return YunshuAPCManager(
            num_blocks=64,
            block_size=16,
            disk=disk,
            overrides={"memory_max_gb": 1, "checkpoint_interval_tokens": 0},
            **kw,
        )

    def runner_for(mgr):
        return VLMBatchRunner(
            model, processor=_processor(), apc_manager=mgr, apc_semantic_hash=0
        )

    # one RAM entry: the second conversation pushes the first one out to the SSD
    mgr = manager(max_entries=1)
    runner = runner_for(mgr)
    _run(runner, a)
    _run(runner, b)
    mgr.disk.flush()
    out_ssd, st_ssd = _run(runner, a2)
    assert st_ssd.cache_tier == "ssd" and st_ssd.cached_tokens == len(a) - 1
    assert st_ssd.cache_reload_ms is not None

    runner_ram = runner_for(manager(max_entries=8))
    _run(runner_ram, a)
    out_ram, st_ram = _run(runner_ram, a2)
    assert st_ram.cache_tier == "ram"

    out_cold, _ = _run(VLMBatchRunner(model, processor=_processor()), a2)
    assert out_ssd == out_ram == out_cold
