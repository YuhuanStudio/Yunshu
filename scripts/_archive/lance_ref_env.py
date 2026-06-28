"""Make the Lance reference PyTorch modules importable on this Mac (no CUDA).

The reference (`reference/Lance/modeling/lance/qwen2_navit.py`) hard-imports
`flash_attn` and pulls transformers' flash-attention integration, plus optional
media deps (imageio, …). None are installable/needed for *running the model
forward on CPU*. This module stubs them — flash_attn's varlen/func are replaced
by `torch.nn.functional.scaled_dot_product_attention` (numerically equivalent for
causal / non-causal masks) — so the real reference layers can be instantiated and
run here for a layer-by-layer diff against the MLX port (VALIDATION_REPORT §138).

Usage:
    from scripts.lance_ref_env import load_reference_qwen2_navit
    nav = load_reference_qwen2_navit("reference/Lance")
    layer_cls = nav.Qwen2MoTDecoderLayer   # etc.

Confirmed via the config: `apply_qwen_2_5_vl_pos_emb` defaults False, so the
reference uses standard 1D `Qwen2RotaryEmbedding` (theta=1e6) — the MLX port's
1D RoPE is correct (the mrope_section in rope_scaling is unused on this path).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path


def _install_stubs() -> None:
    import contextlib

    import torch.nn.functional as F  # noqa: N812

    # transformers: force flash-attn detection off (avoids its metadata lookup crash)
    import transformers.utils.import_utils as iu
    with contextlib.suppress(Exception):
        iu.PACKAGE_DISTRIBUTION_MAPPING.setdefault("flash_attn", ["flash-attn"])
    for fn in ("is_flash_attn_2_available", "is_flash_attn_3_available",
               "is_flash_attn_4_available"):
        if hasattr(iu, fn):
            setattr(iu, fn, lambda *a, **k: False)
    try:
        import transformers.modeling_flash_attention_utils as mfu
        mfu.flash_attn_supports_top_left_mask = lambda *a, **k: False
    except Exception:
        pass

    # flash_attn stub: SDPA-backed varlen for single-sequence packs
    def _varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                causal=False, **kw):
        h, hkv = q.shape[1], k.shape[1]
        qh, kh, vh = (q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                      v.transpose(0, 1)[None])
        if hkv != h:
            r = h // hkv
            kh, vh = kh.repeat_interleave(r, 1), vh.repeat_interleave(r, 1)
        return F.scaled_dot_product_attention(qh, kh, vh, is_causal=causal)[0].transpose(0, 1)

    fa = types.ModuleType("flash_attn")
    fa.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    fa.__version__ = "2.6.3"
    fa.flash_attn_varlen_func = _varlen
    fa.flash_attn_func = lambda *a, **k: a[0]
    sys.modules["flash_attn"] = fa

    # optional media deps not needed for the forward
    for mod in ("imageio", "imageio.v3", "decord", "av", "moviepy", "moviepy.editor"):
        m = types.ModuleType(mod)
        m.__spec__ = importlib.machinery.ModuleSpec(mod, None)
        sys.modules[mod] = m


def load_reference_qwen2_navit(lance_repo: str):
    """Return the real reference `qwen2_navit` module (CPU-runnable).

    Loads the file directly so the heavy `modeling.lance.__init__` (which pulls
    the full generation pipeline + media deps) is bypassed.
    """
    repo = Path(lance_repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    _install_stubs()
    path = repo / "modeling" / "lance" / "qwen2_navit.py"
    spec = importlib.util.spec_from_file_location("modeling.lance.qwen2_navit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modeling.lance.qwen2_navit"] = mod
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    nav = load_reference_qwen2_navit("reference/Lance")
    names = [n for n in dir(nav) if any(t in n for t in ("Decoder", "Model", "Attention"))]
    print("reference qwen2_navit loaded:", names)
