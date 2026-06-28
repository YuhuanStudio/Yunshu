"""Run the REAL Lance reference decoder layer-0 and diff vs the MLX dump (#176).

Uses scripts/lance_ref_env to import the real `qwen2_navit`, instantiates one
real `Qwen2MoTDecoderLayer`, loads the layer-0 checkpoint weights, and runs its
`forward_inference` in gen mode on the SAME embeddings the MLX harness used
(/tmp/mlx.npz test vector). For layer 0 the text/latent K/V are functions of the
input embeddings only (not of any prior layer), so the latent output must match
the MLX port's layer_0 regardless of the text-causal vs bidirectional choice — a
match confirms the per-layer math is faithful to the *real* reference; a mismatch
localizes the bug to the layer forward.

Run: PYTHONPATH=. uv run python scripts/diff_real_lance_layer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

LANCE = Path("/Volumes/P5Plus/models/Lance-3B-bf16")


def main() -> int:
    if not LANCE.exists():
        print("SKIP: Lance not mounted")
        return 0
    import torch
    from scripts.lance_ref_env import load_reference_qwen2_navit
    nav = load_reference_qwen2_navit("reference/Lance")

    from modeling.qwen2.configuration_qwen2 import Qwen2Config
    cfg_d = json.loads((LANCE / "llm_config.json").read_text())
    cfg = Qwen2Config(
        vocab_size=cfg_d["vocab_size"],
        hidden_size=cfg_d["hidden_size"],
        num_attention_heads=cfg_d["num_attention_heads"],
        num_key_value_heads=cfg_d["num_key_value_heads"],
        num_hidden_layers=cfg_d["num_hidden_layers"],
        intermediate_size=cfg_d["intermediate_size"],
        rms_norm_eps=cfg_d["rms_norm_eps"],
        rope_theta=cfg_d["rope_theta"],
        max_position_embeddings=cfg_d["max_position_embeddings"],
        attention_dropout=0.0,
        qk_norm=True,  # checkpoint has q_norm/k_norm (+ _moe_gen) weights
        layer_module="Qwen2MoTDecoderLayer",
        apply_qwen_2_5_vl_pos_emb=False,
        rope_scaling=None,  # standard 1D rope (mrope unused on this path)
        _attn_implementation="eager",
    )

    layer = nav.Qwen2MoTDecoderLayer(cfg, layer_idx=0)
    layer.eval()
    layer = layer.to(torch.bfloat16)  # reference attention hard-casts to bf16

    # load layer-0 weights from the checkpoint
    w = mx.load(str(LANCE / "model.safetensors"))
    sd = {}
    for k, v in w.items():
        if k.startswith("layers.0."):
            sd[k[len("layers.0."):]] = torch.from_numpy(np.asarray(v.astype(mx.float32)))
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    print(f"layer load: {len(sd)} keys, missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print("  missing:", missing[:6], flush=True)

    # test vector + embeddings (same as MLX dump)
    dump = np.load("/tmp/mlx.npz")
    ids = torch.from_numpy(dump["prompt_ids"].astype(np.int64))[0]
    pos = torch.from_numpy(dump["latent_pos"].astype(np.int64))
    x_t = torch.from_numpy(np.asarray(mx.array(dump["x_t"]).astype(mx.float32)))[0]
    ts = torch.from_numpy(np.asarray(mx.array(dump["t"]).astype(mx.float32)))
    nt, nlat = ids.shape[0], x_t.shape[0]

    def tt(name):
        return torch.from_numpy(np.asarray(w[name].astype(mx.float32)))

    def lin(x, name, bias=True):
        y = x @ tt(f"{name}.weight").T
        if bias and f"{name}.bias" in w:
            y = y + tt(f"{name}.bias")
        return y

    text_emb = tt("embed_tokens.weight")[ids]
    half = 128
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half).float() / half)
    args = ts[:, None] * freqs[None]
    temb = lin(torch.cat([torch.cos(args), torch.sin(args)], -1), "time_embedder.proj_in")
    temb = lin(torch.nn.functional.silu(temb), "time_embedder.proj_out")
    lat_emb = lin(x_t, "vae_in_proj.vae2llm") + temb + tt("latent_pos_embed.pos_embed")[pos]

    seq = torch.cat([text_emb, lat_emb], 0).to(torch.bfloat16)  # (nt+nlat, dim)
    position_ids = torch.cat([torch.arange(nt), torch.arange(nlat) + 1000])

    # standard 1D Qwen2 rope cos/sin (theta=1e6); matches torch reimpl exactly
    hd = cfg.hidden_size // cfg.num_attention_heads
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2).float() / hd))
    fr = torch.outer(position_ids.float(), inv)
    emb = torch.cat([fr, fr], -1)
    pemb = (emb.cos(), emb.sin())

    query_lens = torch.tensor([seq.shape[0]], dtype=torch.int32)
    qidx = torch.arange(seq.shape[0], dtype=torch.int64)
    txt_idx = torch.arange(nt, dtype=torch.int64)
    vae_idx = torch.arange(nt, nt + nlat, dtype=torch.int64)

    with torch.no_grad():
        out, _ = layer(
            packed_query_sequence=seq,
            query_lens=query_lens,
            packed_query_position_embeddings=pemb,
            packed_query_indexes=qidx,
            past_key_values=None,
            key_values_lens=None,
            packed_key_value_indexes=None,
            update_past_key_values=False,
            is_causal=False,
            mode="gen",
            packed_vae_token_indexes=vae_idx,
            packed_text_indexes=txt_idx,
            apply_qwen_2_5_vl_pos_emb=False,
        )
    ref_lat = out[vae_idx].float().detach().numpy()[None]  # (1, nlat, dim)
    mine = dump["layer_0"]
    mad = float(np.abs(ref_lat - mine).max())
    rel = mad / (float(np.abs(ref_lat).max()) + 1e-8)
    print(f"REAL reference layer-0 latent vs MLX layer_0: max|Δ|={mad:.5f} rel={rel:.4f}", flush=True)
    print("LAYER MATCHES real reference" if rel < 0.02
          else "LAYER DIVERGES from real reference <-- bug in layer forward", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
