"""Faithful torch reference of the Lance gen-path velocity forward (CPU).

Reimplements `qwen2_navit.py`'s gen-mode decoder forward directly from the
REFERENCE source (not from the MLX port), using torch SDPA in place of
flash_attn (numerically equivalent for non-causal / causal masks). Runs on this
Mac (no CUDA), loads the SAME test vector the MLX harness dumped, and diffs
per-layer latent hidden states against /tmp/mlx.npz to localize the gen-path bug.

Reference forward per layer (qwen2_navit.py L646 + PackedAttentionMoT L374), for
the dual-expert t2i step:
  text tokens  -> base proj (q/k/v/o_proj, q/k_norm, input/post_attention_layernorm, mlp)
  latent tokens-> *_moe_gen proj + *_moe_gen norms/layernorms/mlp
  text attends causally to text;  latents attend non-causally to [text ++ latents]
  RoPE: text at positions 0..nt-1; latents MaPE-shifted to start at 1000 (theta=1e6)
  final: norm_moe_gen(latents) -> llm2vae -> velocity

Run: PYTHONPATH=python uv run python scripts/torch_lance_ref_velocity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

LANCE = Path("/Volumes/P5Plus/models/Lance-3B-bf16")


def t(x: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x.astype(mx.float32)))


def rms_norm(x, w, eps=1e-6):
    v = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * w


def rope(q, positions, theta=1e6):
    # q: (H, L, D) ; standard Qwen2 rotate_half NeoX style
    d = q.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))
    f = torch.outer(positions.float(), inv)  # (L, d/2)
    emb = torch.cat([f, f], -1)  # (L, d)
    cos, sin = emb.cos()[None], emb.sin()[None]  # (1, L, d)

    def rh(x):
        x1, x2 = x[..., : d // 2], x[..., d // 2:]
        return torch.cat([-x2, x1], -1)
    return q * cos + rh(q) * sin


def main() -> int:
    if not LANCE.exists():
        print("SKIP: Lance not mounted")
        return 0
    mlx_dump = np.load("/tmp/mlx.npz")
    cfg = json.loads((LANCE / "llm_config.json").read_text())
    H, KV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg["hidden_size"] // H
    eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]
    nl_layers = cfg["num_hidden_layers"]


    w = mx.load(str(LANCE / "model.safetensors"))
    W = {k: t(v) for k, v in w.items()}

    def lin(x, name, bias=True):
        y = x @ W[f"{name}.weight"].T
        if bias and f"{name}.bias" in W:
            y = y + W[f"{name}.bias"]
        return y

    # ---- inputs (byte-identical to the MLX dump) ----
    x_t = t(mx.array(mlx_dump["x_t"]))[0]          # (256, 48)
    ts = t(mx.array(mlx_dump["t"]))                # (1,)
    ids = torch.from_numpy(mlx_dump["prompt_ids"].astype(np.int64))[0]  # (nt,)
    pos = torch.from_numpy(mlx_dump["latent_pos"].astype(np.int64))     # (256,)
    nt = ids.shape[0]
    nlat = x_t.shape[0]

    # ---- embeddings ----
    text_h = W["embed_tokens.weight"][ids]  # (nt, dim)
    # latent embedding: vae2llm + time_embedder + latent_pos_embed
    half = 128
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half).float() / half)
    args = ts[:, None] * freqs[None]
    temb_sin = torch.cat([torch.cos(args), torch.sin(args)], -1)  # (1, 256)
    temb = lin(F.silu(lin(temb_sin, "time_embedder.proj_in")), "time_embedder.proj_out")  # (1, dim)
    lat_h = lin(x_t, "vae_in_proj.vae2llm") + temb + W["latent_pos_embed.pos_embed"][pos]

    tpos = torch.arange(nt)
    lpos = torch.arange(nlat) + 1000  # MaPE shift

    diffs = []
    for i in range(nl_layers):
        p = f"layers.{i}"
        # input layernorm (text=base, latent=moe_gen)
        tln = rms_norm(text_h, W[f"{p}.input_layernorm.weight"], eps)
        lln = rms_norm(lat_h, W[f"{p}.input_layernorm_moe_gen.weight"], eps)

        def qkv(x, sfx, ap=p):
            q = lin(x, f"{ap}.self_attn.q_proj{sfx}").reshape(-1, H, hd)
            k = lin(x, f"{ap}.self_attn.k_proj{sfx}").reshape(-1, KV, hd)
            v = lin(x, f"{ap}.self_attn.v_proj{sfx}").reshape(-1, KV, hd)
            q = rms_norm(q, W[f"{ap}.self_attn.q_norm{sfx}.weight"], eps).transpose(0, 1)
            k = rms_norm(k, W[f"{ap}.self_attn.k_norm{sfx}.weight"], eps).transpose(0, 1)
            return q, k, v.transpose(0, 1)  # (H,L,D)/(KV,L,D)

        qt, kt, vt = qkv(tln, "")
        ql, kl, vl = qkv(lln, "_moe_gen")
        qt, kt = rope(qt, tpos, theta), rope(kt, tpos, theta)
        ql, kl = rope(ql, lpos, theta), rope(kl, lpos, theta)

        def gqa(q, k, v):  # repeat kv heads
            rep = H // KV
            return q[None], k.repeat_interleave(rep, 0)[None], v.repeat_interleave(rep, 0)[None]

        # text: causal self-attn
        qtb, ktb, vtb = gqa(qt, kt, vt)
        ot = F.scaled_dot_product_attention(qtb, ktb, vtb, is_causal=True)[0].transpose(0, 1).reshape(nt, -1)
        ot = lin(ot, f"{p}.self_attn.o_proj", bias=False)
        # latents: bidirectional over [text ++ latents]
        k_all = torch.cat([kt, kl], 1)
        v_all = torch.cat([vt, vl], 1)
        qlb, kab, vab = gqa(ql, k_all, v_all)
        ol = F.scaled_dot_product_attention(qlb, kab, vab, is_causal=False)[0].transpose(0, 1).reshape(nlat, -1)
        ol = lin(ol, f"{p}.self_attn.o_proj_moe_gen", bias=False)

        text_h = text_h + ot
        lat_h = lat_h + ol

        def mlp(x, sfx, ap=p):
            g = lin(x, f"{ap}.mlp{sfx}.gate_proj", bias=False)
            u = lin(x, f"{ap}.mlp{sfx}.up_proj", bias=False)
            return lin(F.silu(g) * u, f"{ap}.mlp{sfx}.down_proj", bias=False)

        text_h = text_h + mlp(rms_norm(text_h, W[f"{p}.post_attention_layernorm.weight"], eps), "")
        lat_h = lat_h + mlp(rms_norm(lat_h, W[f"{p}.post_attention_layernorm_moe_gen.weight"], eps), "_moe_gen")

        ref = lat_h.detach().numpy()[None]
        mine = mlx_dump[f"layer_{i}"]
        mad = float(np.abs(ref - mine).max())
        rel = mad / (float(np.abs(ref).max()) + 1e-8)
        diffs.append((i, mad, rel))

    velo = lin(rms_norm(lat_h, W["norm_moe_gen.weight"], eps), "llm2vae")
    vref = velo.detach().numpy()[None]
    vmad = float(np.abs(vref - mlx_dump["velocity"]).max())
    vrel = vmad / (float(np.abs(vref).max()) + 1e-8)

    print(f"{'layer':>6} | {'max|Δ|':>10} | {'rel':>8}")
    first = None
    for i, mad, rel in diffs:
        flag = "  <-- FIRST DIVERGENCE" if rel > 0.02 and first is None else ""
        if rel > 0.02 and first is None:
            first = i
        if i < 3 or rel > 0.02 or i == nl_layers - 1:
            print(f"{i:>6} | {mad:10.5f} | {rel:8.4f}{flag}")
    print(f"{'velo':>6} | {vmad:10.5f} | {vrel:8.4f}")
    print(f"\nfirst diverging layer: {first if first is not None else 'none — MLX matches torch reference'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
