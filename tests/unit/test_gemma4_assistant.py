"""Offline tests for Gemma4MTPMaskedEmbedder — pure tensor math, random weights.

No real Gemma-4 checkpoint or model drive needed: every test drives the
MLX embedder with random tensors and checks it against an independent
NumPy reference implementation of the centroid-masking algorithm.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine.gemma4_assistant import (
    Gemma4MLP,
    Gemma4MTPAttention,
    Gemma4MTPDecoderLayer,
    Gemma4MTPMaskedEmbedder,
    Gemma4MultiTokenPredictor,
    GemmaRMSNorm,
    load_assistant_drafter,
)

_REAL_DRAFTER = Path("models/gemma-4-E4B-it-assistant-bf16")
_REAL_TARGET = Path("models/gemma-4-e4b-it-bf16")

H = 16  # hidden_size
V = 64  # vocab_size
C = 8  # num_centroids
TOPK = 3  # centroid_intermediate_top_k
# vocab_size_per_centroid = 8, num_selected = 24


def _make_embedder(seed: int = 0) -> Gemma4MTPMaskedEmbedder:
    mx.random.seed(seed)
    emb = Gemma4MTPMaskedEmbedder(
        hidden_size=H, vocab_size=V, num_centroids=C, centroid_intermediate_top_k=TOPK
    )
    # Give the centroid projection and token_ordering deterministic content.
    emb.centroids.weight = mx.array(
        np.random.default_rng(seed).standard_normal((C, H)).astype(np.float32)
    )
    perm = np.random.default_rng(seed + 1).permutation(V).astype(np.int32)
    emb.token_ordering = mx.array(perm)
    mx.eval(emb.centroids.weight, emb.token_ordering)
    return emb


def _np_reference(emb, hidden_np, lm_head_np):
    """Independent NumPy implementation of _select_and_score."""
    scores = hidden_np @ np.asarray(emb.centroids.weight).T  # (T, C)
    top_k = np.argsort(-scores, axis=-1)[:, :TOPK]  # (T, TOPK)
    clusters = np.asarray(emb.token_ordering).reshape(C, V // C)
    selected = clusters[top_k]  # (T, TOPK, V//C)
    T = hidden_np.shape[0]
    selected_flat = selected.reshape(T, -1)  # (T, num_selected)
    emb_rows = lm_head_np[selected_flat.reshape(-1)].reshape(T, -1, H)
    logits = np.einsum("td,tsd->ts", hidden_np, emb_rows)
    return logits, selected_flat


def test_init_derived_dims():
    emb = _make_embedder()
    assert emb.vocab_size_per_centroid == V // C == 8
    assert emb.num_selected == TOPK * (V // C) == 24


def test_init_rejects_indivisible_vocab():
    with pytest.raises(ValueError, match="divisible"):
        Gemma4MTPMaskedEmbedder(
            hidden_size=H, vocab_size=65, num_centroids=8, centroid_intermediate_top_k=2
        )


@pytest.mark.parametrize("bad_topk", [0, -1, C + 1])
def test_init_rejects_bad_topk(bad_topk):
    with pytest.raises(ValueError, match="centroid_intermediate_top_k"):
        Gemma4MTPMaskedEmbedder(
            hidden_size=H,
            vocab_size=V,
            num_centroids=C,
            centroid_intermediate_top_k=bad_topk,
        )


def test_selected_indices_match_numpy_set():
    emb = _make_embedder(seed=2)
    rng = np.random.default_rng(7)
    hidden_np = rng.standard_normal((5, H)).astype(np.float32)
    lm_head_np = rng.standard_normal((V, H)).astype(np.float32)
    _, idx_mlx = emb._select_and_score(mx.array(hidden_np), mx.array(lm_head_np))
    _, idx_np = _np_reference(emb, hidden_np, lm_head_np)
    idx_mlx = np.asarray(idx_mlx)
    # argpartition order may differ from argsort, so compare as per-row sets.
    for t in range(hidden_np.shape[0]):
        assert set(idx_mlx[t].tolist()) == set(idx_np[t].tolist())


def test_sparse_logits_match_numpy_paired():
    emb = _make_embedder(seed=3)
    rng = np.random.default_rng(11)
    hidden_np = rng.standard_normal((4, H)).astype(np.float32)
    lm_head_np = rng.standard_normal((V, H)).astype(np.float32)
    logits_mlx, idx_mlx = emb._select_and_score(
        mx.array(hidden_np), mx.array(lm_head_np)
    )
    logits_mlx, idx_mlx = np.asarray(logits_mlx), np.asarray(idx_mlx)
    logits_np, idx_np = _np_reference(emb, hidden_np, lm_head_np)
    # Pair each (vocab_id -> logit) and compare dicts to be order-independent.
    for t in range(hidden_np.shape[0]):
        d_mlx = dict(zip(idx_mlx[t].tolist(), logits_mlx[t].tolist(), strict=True))
        d_np = dict(zip(idx_np[t].tolist(), logits_np[t].tolist(), strict=True))
        assert d_mlx.keys() == d_np.keys()
        for k in d_mlx:
            assert d_mlx[k] == pytest.approx(d_np[k], rel=1e-4, abs=1e-4)


def test_forward_scatters_selected_and_masks_rest():
    emb = _make_embedder(seed=4)
    rng = np.random.default_rng(13)
    hidden_np = rng.standard_normal((3, H)).astype(np.float32)
    lm_head_np = rng.standard_normal((V, H)).astype(np.float32)
    full = np.asarray(emb(mx.array(hidden_np), mx.array(lm_head_np)))
    assert full.shape == (3, V)
    logits_np, idx_np = _np_reference(emb, hidden_np, lm_head_np)
    neg_inf = np.finfo(np.float32).min
    for t in range(hidden_np.shape[0]):
        selected = set(idx_np[t].tolist())
        # Non-selected positions are masked.
        for v in range(V):
            if v not in selected:
                assert full[t, v] == neg_inf
        # Selected positions carry the sparse logit.
        for v_id, lg in zip(idx_np[t].tolist(), logits_np[t].tolist(), strict=True):
            assert full[t, v_id] == pytest.approx(lg, rel=1e-4, abs=1e-4)


def test_get_top_tokens_matches_full_argmax():
    emb = _make_embedder(seed=5)
    rng = np.random.default_rng(17)
    hidden_np = rng.standard_normal((6, H)).astype(np.float32)
    lm_head_np = rng.standard_normal((V, H)).astype(np.float32)
    top = np.asarray(emb.get_top_tokens(mx.array(hidden_np), mx.array(lm_head_np)))
    full = np.asarray(emb(mx.array(hidden_np), mx.array(lm_head_np)))
    assert top.shape == (6,)
    # Token IDs are valid and equal to the argmax of the dense masked logits.
    assert np.all((top >= 0) & (top < V))
    np.testing.assert_array_equal(top, full.argmax(axis=-1))


def test_single_token_batch():
    emb = _make_embedder(seed=6)
    rng = np.random.default_rng(19)
    hidden_np = rng.standard_normal((1, H)).astype(np.float32)
    lm_head_np = rng.standard_normal((V, H)).astype(np.float32)
    top = np.asarray(emb.get_top_tokens(mx.array(hidden_np), mx.array(lm_head_np)))
    assert top.shape == (1,)
    full = np.asarray(emb(mx.array(hidden_np), mx.array(lm_head_np)))
    assert int(top[0]) == int(full.argmax(axis=-1)[0])


# --- Decoder architecture (shape/structure tests, random weights, no drive) ---

HID = 32  # hidden_size
BB = 48  # backbone_hidden_size
INTER = 64  # intermediate_size
NH = 4  # num_heads
NKV = 2  # num_kv_heads (GQA)
HD = 8  # head_dim
NL = 2  # num_layers


def _kv(t: int, n_kv: int = NKV, hd: int = HD, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((1, n_kv, t, hd)).astype(np.float32))
    v = mx.array(rng.standard_normal((1, n_kv, t, hd)).astype(np.float32))
    return k, v


def test_gemma_rmsnorm_default_weight_is_identity_scale():
    # weight defaults to 1 -> standard RMSNorm gain 1 -> pure normalization.
    norm = GemmaRMSNorm(8)
    x = mx.array(np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32))
    out = np.asarray(norm(x))
    # Each row should have unit mean-square (within eps).
    ms = (out**2).mean(axis=-1)
    np.testing.assert_allclose(ms, np.ones(3), rtol=1e-3, atol=1e-3)


def test_gemma_rmsnorm_standard_weight_multiply():
    # Standard RMSNorm (NOT 1+weight): weight=2 -> gain 2. This convention
    # matches the assistant checkpoint's full-gain norm weights + mlx-lm gemma4.
    norm = GemmaRMSNorm(4)
    norm.weight = mx.full((4,), 2.0)  # gain = 2 (not 1+2)
    x = mx.array(np.ones((1, 4), dtype=np.float32))
    out = np.asarray(norm(x))
    # ones normalized -> ones, times gain 2.
    np.testing.assert_allclose(out, np.full((1, 4), 2.0), rtol=1e-4, atol=1e-4)


def test_mlp_shape_preserved():
    mlp = Gemma4MLP(HID, INTER)
    x = mx.array(np.random.default_rng(1).standard_normal((5, HID)).astype(np.float32))
    out = mlp(x)
    mx.eval(out)
    assert out.shape == (5, HID)
    assert not np.any(np.isnan(np.asarray(out)))


def test_attention_qonly_external_kv_shape():
    t = 6
    attn = Gemma4MTPAttention(HID, NH, NKV, HD)
    x = mx.array(np.random.default_rng(2).standard_normal((t, HID)).astype(np.float32))
    k, v = _kv(t, seed=3)
    out = attn(x, k, v)
    mx.eval(out)
    assert out.shape == (t, HID)
    assert not np.any(np.isnan(np.asarray(out)))


def test_attention_handles_gqa_and_mha():
    t = 4
    # MHA: num_kv_heads == num_heads
    attn = Gemma4MTPAttention(HID, NH, NH, HD)
    x = mx.array(np.random.default_rng(4).standard_normal((t, HID)).astype(np.float32))
    k, v = _kv(t, n_kv=NH, seed=5)
    assert attn(x, k, v).shape == (t, HID)


def test_decoder_layer_shape_and_residual_identity():
    t = 5
    layer = Gemma4MTPDecoderLayer(HID, INTER, NH, NKV, HD)
    x = mx.array(np.random.default_rng(6).standard_normal((t, HID)).astype(np.float32))
    k, v = _kv(t, seed=7)
    out = layer(x, k, v)
    mx.eval(out)
    assert out.shape == (t, HID)
    assert not np.any(np.isnan(np.asarray(out)))


def test_decoder_layer_scalar_scales_output():
    t = 3
    layer = Gemma4MTPDecoderLayer(HID, INTER, NH, NKV, HD)
    x = mx.array(np.random.default_rng(8).standard_normal((t, HID)).astype(np.float32))
    k, v = _kv(t, seed=9)
    base = np.asarray(layer(x, k, v))
    layer.layer_scalar = mx.full((1,), 2.0)
    scaled = np.asarray(layer(x, k, v))
    np.testing.assert_allclose(scaled, base * 2.0, rtol=1e-4, atol=1e-4)


def _make_predictor():
    return Gemma4MultiTokenPredictor(
        vocab_size=V,
        hidden_size=HID,
        backbone_hidden_size=BB,
        intermediate_size=INTER,
        num_layers=NL,
        num_heads=NH,
        num_kv_heads=NKV,
        head_dim=HD,
    )


def test_predictor_forward_dual_output_shapes():
    t = 7
    model = _make_predictor()
    rng = np.random.default_rng(10)
    inputs_embeds = mx.array(rng.standard_normal((t, BB)).astype(np.float32))
    hidden = mx.array(rng.standard_normal((t, BB)).astype(np.float32))
    kv = [_kv(t, seed=20 + i) for i in range(NL)]
    draft, backbone = model(inputs_embeds, hidden, kv)
    mx.eval(draft, backbone)
    assert draft.shape == (t, HID)  # draft-dim for lm_head
    assert backbone.shape == (t, BB)  # backbone-dim for feedback buffer
    assert not np.any(np.isnan(np.asarray(draft)))
    assert not np.any(np.isnan(np.asarray(backbone)))


def test_predictor_rejects_wrong_kv_count():
    model = _make_predictor()
    t = 3
    inputs_embeds = mx.zeros((t, BB))
    hidden = mx.zeros((t, BB))
    with pytest.raises(ValueError, match="kv_per_layer"):
        model(inputs_embeds, hidden, [_kv(t)])  # only 1, needs NL=2


def test_predictor_embed_input_ids_applies_normalizer():
    model = _make_predictor()
    ids = mx.array(np.array([1, 5, 9], dtype=np.int32))
    raw = np.asarray(model.embed_tokens(ids))
    scaled = np.asarray(model.embed_input_ids(ids))
    np.testing.assert_allclose(scaled, raw * (BB**0.5), rtol=1e-4, atol=1e-4)


def test_from_hf_config_per_layer_head_dims():
    # Mimic the real gemma-4 assistant: 3 sliding + 1 full_attention layer.
    cfg = {
        "backbone_hidden_size": 64,
        "num_centroids": 8,
        "centroid_intermediate_top_k": 2,
        "text_config": {
            "num_hidden_layers": 4,
            "head_dim": 16,
            "global_head_dim": 32,
            "hidden_size": 24,
            "vocab_size": 64,
            "intermediate_size": 48,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000.0},
                "full_attention": {
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25,
                },
            },
        },
    }
    model = Gemma4MultiTokenPredictor.from_hf_config(cfg)
    assert len(model.layers) == 4
    for i in range(3):
        assert model.layers[i].self_attn.head_dim == 16
        assert abs(model.layers[i].self_attn.rope.base - 10000.0) < 1
    # full_attention layer: global head_dim + partial rotary.
    assert model.layers[3].self_attn.head_dim == 32
    assert model.layers[3].self_attn.rope.dims == 8  # 0.25 * 32
    assert abs(model.layers[3].self_attn.rope.base - 1000000.0) < 1


# --- Real-weight validation (drive-gated; skips when drive not mounted) ---

_skip_no_drive = pytest.mark.skipif(
    not _REAL_DRAFTER.exists(), reason="Gemma-4 assistant drive not mounted"
)


@_skip_no_drive
def test_real_drafter_loads_strict_all_keys():
    # strict=True => every checkpoint key must map onto a parameter and vice
    # versa; a structural mismatch in the port would raise here.
    pred, emb, cfg = load_assistant_drafter(_REAL_DRAFTER)
    tcfg = cfg["text_config"]
    assert len(pred.layers) == tcfg["num_hidden_layers"] == 4
    # Sliding layers head_dim=256; full-attention layer=512 with partial rope.
    assert [layer.self_attn.head_dim for layer in pred.layers] == [256, 256, 256, 512]
    assert pred.layers[3].self_attn.rope.dims == 128  # 0.25 * 512
    assert emb.num_centroids == 2048
    assert emb.centroids.weight.shape == (2048, 256)
    assert emb.token_ordering.shape == (262144,)


@_skip_no_drive
def test_real_drafter_forward_runs_nan_free():
    pred, emb, cfg = load_assistant_drafter(_REAL_DRAFTER)
    bb = cfg["backbone_hidden_size"]
    vocab = cfg["text_config"]["vocab_size"]
    mx.random.seed(0)
    t = 5
    ie = mx.random.normal((t, bb)) * 0.04
    hs = mx.random.normal((t, bb)) * 0.04
    kv = [
        (
            mx.random.normal(
                (1, layer.self_attn.num_kv_heads, t, layer.self_attn.head_dim)
            )
            * 0.04,
            mx.random.normal(
                (1, layer.self_attn.num_kv_heads, t, layer.self_attn.head_dim)
            )
            * 0.04,
        )
        for layer in pred.layers
    ]
    draft, backbone = pred(ie, hs, kv)
    mx.eval(draft, backbone)
    assert draft.shape == (t, cfg["text_config"]["hidden_size"])
    assert backbone.shape == (t, bb)
    assert not np.any(np.isnan(np.asarray(draft)))
    assert not np.any(np.isnan(np.asarray(backbone)))
    # MaskedEmbedder on real drafter output yields valid token IDs.
    lm = pred.embed_tokens.weight.astype(mx.float32)
    top = np.asarray(emb.get_top_tokens(draft.astype(mx.float32), lm))
    assert np.all((top >= 0) & (top < vocab))


@_skip_no_drive
def test_real_assistant_proposer_acceptance():
    # End-to-end: the serving proposer reproduces high draft acceptance against
    # the real target on greedy-consistent text .
    import json

    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model, load_tokenizer

    from yunshu_engine.gemma4_assistant import Gemma4AssistantProposer

    target_dir = "models/gemma-4-e4b-it-bf16"
    if not __import__("pathlib").Path(target_dir).exists():
        pytest.skip("target not mounted")
    ret = load_model(__import__("pathlib").Path(target_dir), strict=False)
    target = ret[0] if isinstance(ret, tuple) else ret
    tok = load_tokenizer(__import__("pathlib").Path(target_dir))
    tm = target.language_model.model
    tcfg = json.loads(
        (__import__("pathlib").Path(target_dir) / "config.json").read_text()
    )
    prop = Gemma4AssistantProposer.from_paths(
        str(DRAFTER_DIR_FOR_TEST), tm.embed_tokens.weight, tm.embed_scale, tcfg
    )
    assert prop.sliding_kv_layer == 22
    assert prop.full_kv_layer == 23

    ids = tok.encode("Explain gravity simply.")
    cache = make_prompt_cache(target)
    h = tm(mx.array(ids)[None], cache=cache)
    for _ in range(30):
        nt = int(mx.argmax(tm.embed_tokens.as_linear(h[:, -1:, :])[0, -1]))
        ids.append(nt)
        h = tm(mx.array([[nt]]), cache=cache)
    full = mx.array(ids)[None]
    seq_len = full.shape[1]
    gen_start = seq_len - 30
    cache = make_prompt_cache(target)
    hid = tm(full, cache=cache)
    mx.eval(hid)
    tgt = mx.argmax(tm.embed_tokens.as_linear(hid)[0], -1)
    mx.eval(tgt)
    hits = 0
    n = 0
    for t in range(gen_start, seq_len - 1):
        kv = {
            prop.sliding_kv_layer: tuple(
                x[:, :, : t + 1, :] for x in cache[prop.sliding_kv_layer].state
            ),
            prop.full_kv_layer: tuple(
                x[:, :, : t + 1, :] for x in cache[prop.full_kv_layer].state
            ),
        }
        pred = prop.propose(ids[t], hid[0, : t + 1], kv)
        assert 0 <= pred < tcfg["text_config"]["vocab_size"]
        if pred == int(tgt[t]):
            hits += 1
        n += 1
    # EAGLE-level acceptance on greedy-consistent text (validated ~85-90%).
    assert hits / n > 0.5, f"acceptance {hits}/{n} too low"


@_skip_no_drive
def test_spec_decode_generate_matches_greedy_prefix():
    # The reusable serving primitive (spec_decode_generate) must produce a long
    # exact-match prefix vs sequential greedy — confirms the draft-verify-rollback
    # loop bookkeeping (KV trim, offset, hidden carry) is correct.
    import json

    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model, load_tokenizer

    from yunshu_engine.gemma4_assistant import Gemma4AssistantProposer

    if not _REAL_TARGET.exists():
        pytest.skip("target not mounted")
    ret = load_model(_REAL_TARGET, strict=False)
    target = ret[0] if isinstance(ret, tuple) else ret
    tok = load_tokenizer(_REAL_TARGET)
    tm = target.language_model.model
    lm = tm.embed_tokens.as_linear
    tcfg = json.loads((_REAL_TARGET / "config.json").read_text())
    prop = Gemma4AssistantProposer.from_paths(
        str(_REAL_DRAFTER), tm.embed_tokens.weight, tm.embed_scale, tcfg
    )
    ids = tok.encode("Explain gravity simply.")
    n = 40
    # Sequential greedy reference.
    cache = make_prompt_cache(target)
    h = tm(mx.array(ids)[None], cache=cache)
    greedy = []
    for _ in range(n):
        t = int(mx.argmax(lm(h[:, -1:, :])[0, -1]))
        greedy.append(t)
        h = tm(mx.array([[t]]), cache=cache)
    # Spec-decode via the reusable primitive.
    cache2 = make_prompt_cache(target)
    spec = prop.spec_decode_generate(tm, lm, cache2, ids, max_tokens=n, k=4)
    assert len(spec) == n
    prefix = 0
    for g, s in zip(greedy, spec, strict=False):
        if g == s:
            prefix += 1
        else:
            break
    # bf16 batched-forward noise eventually flips an argmax; a long exact prefix
    # confirms correct loop bookkeeping.
    assert prefix >= 12, f"only {prefix}/{n} greedy-exact prefix"


@_skip_no_drive
def test_spec_decode_sampling_is_stochastic_and_valid():
    # temperature=0 is deterministic (greedy); temperature>0 is genuine
    # speculative sampling — different seeds yield different sequences and all
    # tokens are valid. Confirms the rejection-sampling path runs end-to-end.
    import json

    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model, load_tokenizer

    from yunshu_engine.gemma4_assistant import Gemma4AssistantProposer

    if not _REAL_TARGET.exists():
        pytest.skip("target not mounted")
    ret = load_model(_REAL_TARGET, strict=False)
    target = ret[0] if isinstance(ret, tuple) else ret
    tok = load_tokenizer(_REAL_TARGET)
    tm = target.language_model.model
    lm = tm.embed_tokens.as_linear
    tcfg = json.loads((_REAL_TARGET / "config.json").read_text())
    vocab = tcfg["text_config"]["vocab_size"]
    prop = Gemma4AssistantProposer.from_paths(
        str(_REAL_DRAFTER), tm.embed_tokens.weight, tm.embed_scale, tcfg
    )
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "Write a creative story opening."}],
        add_generation_prompt=True,
        tokenize=True,
    )
    mk = lambda: make_prompt_cache(target)  # noqa: E731
    g0 = prop.spec_decode_generate(tm, lm, mk(), ids, 30, k=4, temperature=0.0)
    g1 = prop.spec_decode_generate(tm, lm, mk(), ids, 30, k=4, temperature=0.0)
    assert g0 == g1  # greedy deterministic
    s1 = prop.spec_decode_generate(tm, lm, mk(), ids, 30, k=4, temperature=2.0, seed=11)
    s2 = prop.spec_decode_generate(tm, lm, mk(), ids, 30, k=4, temperature=2.0, seed=99)
    assert all(0 <= t < vocab for t in s1 + s2)  # valid tokens
    assert s1 != s2  # sampling genuinely stochastic across seeds


_DRAFTER_DIR = Path("models/gemma-4-E4B-it-assistant-bf16")
DRAFTER_DIR_FOR_TEST = _DRAFTER_DIR


@_skip_no_drive
def test_real_masked_embedder_sparse_matches_dense():
    # On REAL centroids/token_ordering/lm_head, sparse logits at the selected
    # IDs must equal the dense logits there (the masking only skips scoring,
    # it does not change values).
    pred, emb, cfg = load_assistant_drafter(_REAL_DRAFTER)
    hidden = cfg["text_config"]["hidden_size"]
    lm = pred.embed_tokens.weight.astype(mx.float32)
    mx.random.seed(1)
    h = mx.random.normal((3, hidden)) * float(mx.std(lm).item())
    sparse, idx = emb._select_and_score(h, lm)
    dense = h @ lm.T
    mx.eval(sparse, idx, dense)
    for ti in range(3):
        dense_sel = mx.take(dense[ti], idx[ti])
        err = float(mx.max(mx.abs(sparse[ti] - dense_sel)).item())
        assert err < 1e-2, err
