"""Model-agnostic diffusion feature middle-layer.

The reusable machinery behind SD-WebUI/ComfyUI-style features, extracted so it is NOT
re-implemented per model. A concrete model (Z-Image today, Flux/SD tomorrow) supplies
only a thin adapter: how to resolve a module path to (parent, attr, target), how to
remap checkpoint keys to its module names, and how to tokenize a prompt.

Generic here:
  - parse_prompt_attention   : A1111/ComfyUI (word:1.2)/[word] emphasis → segments.
  - build_token_weights      : align per-segment weights onto tokens via char offsets.
  - apply_prompt_weights     : DIRECTION-based weighting (survives RMSNorm'd captions).
  - parse_lora_tags          : inline <lora:NAME:WEIGHT>.
  - resolve_lora_file        : NAME → .safetensors path (URL-decode aware).
  - DiffusionLoRALinear      : ComfyUI lora_down/up branch wrapping a (quantized) base.
  - load_diffusion_lora      : generic loader given a model + adapter callbacks.

Per-model (NOT here): module naming, ControlNet architecture, tokenizer.
"""

from __future__ import annotations

import re
import urllib.parse

import mlx.core as mx
import mlx.nn as nn

# ── Prompt emphasis (word:weight) — fully model-agnostic ──
_ATTN_RE = re.compile(
    r"""\\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:\s*([+-]?[.\d]+)\s*\)|\)|]|[^\\()\[\]:]+|:""",
    re.X,
)


def parse_prompt_attention(text: str):
    """A1111/ComfyUI emphasis → list of [segment_text, weight]. `(w)`=×1.1, `[w]`=÷1.1,
    nesting, explicit `(w:1.4)`, backslash escapes. No syntax → one weight-1.0 segment."""
    res: list[list] = []
    round_b: list[int] = []
    square_b: list[int] = []
    rb_mult, sb_mult = 1.1, 1.0 / 1.1

    def mul_from(start: int, m: float) -> None:
        for p in range(start, len(res)):
            res[p][1] *= m

    for mt in _ATTN_RE.finditer(text):
        tok, weight = mt.group(0), mt.group(1)
        if tok.startswith("\\"):
            res.append([tok[1:], 1.0])
        elif tok == "(":
            round_b.append(len(res))
        elif tok == "[":
            square_b.append(len(res))
        elif weight is not None and round_b:
            mul_from(round_b.pop(), float(weight))
        elif tok == ")" and round_b:
            mul_from(round_b.pop(), rb_mult)
        elif tok == "]" and square_b:
            mul_from(square_b.pop(), sb_mult)
        else:
            res.append([tok, 1.0])
    for pos in round_b:
        mul_from(pos, rb_mult)
    for pos in square_b:
        mul_from(pos, sb_mult)
    if not res:
        return [["", 1.0]]
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1
    return res


def build_token_weights(parsed, clean, formatted, offset_mapping, num_valid):
    """Map per-segment weights onto tokens by char offset (needs the tokenizer's
    offset_mapping). Tokens outside the clean-prompt span get weight 1.0. Returns a
    numpy float32 array of length num_valid."""
    import numpy as np

    cw: list[float] = []
    for seg, w in parsed:
        cw.extend([w] * len(seg))
    weights = np.ones((num_valid,), dtype=np.float32)
    cidx = formatted.find(clean)
    if cidx < 0 or offset_mapping is None:
        return weights
    for ti in range(num_valid):
        s, e = int(offset_mapping[ti][0]), int(offset_mapping[ti][1])
        if e <= s:
            continue
        rel = (s + e) // 2 - cidx
        if 0 <= rel < len(cw):
            weights[ti] = cw[rel]
    return weights


def apply_prompt_weights(cap_feats, token_weights):
    """DIRECTION-based emphasis: scale each token's deviation from the baseline (mean
    token). Magnitude-scaling (A1111/CLIP) is a no-op when the model RMSNorms the
    caption; scaling the deviation changes post-norm direction → real effect.
    cap_feats [seq, dim], token_weights [seq]."""
    wv = mx.array(token_weights)[:, None]
    baseline = mx.mean(cap_feats, axis=0, keepdims=True)
    return baseline + wv * (cap_feats - baseline)


# ── Inline LoRA <lora:NAME:WEIGHT> — model-agnostic ──
_LORA_TAG_RE = re.compile(r"<lora:\s*([^:>]+?)\s*(?::\s*([+-]?[\d.]+)\s*)?>", re.I)


def parse_lora_tags(prompt: str):
    """Extract <lora:NAME:WEIGHT> tags → (clean_prompt, [(name, weight), ...])."""
    tags = [
        (m.group(1).strip(), float(m.group(2)) if m.group(2) else 1.0)
        for m in _LORA_TAG_RE.finditer(prompt)
    ]
    clean = re.sub(r"\s{2,}", " ", _LORA_TAG_RE.sub("", prompt)).strip()
    return clean, tags


def resolve_lora_file(name: str, search_dir: str = "models") -> str | None:
    """NAME → .safetensors path. Exact stem then substring, matched against the raw
    AND URL-decoded filename (so a Chinese name finds a %-encoded file). Excludes
    ControlNet files."""
    import os

    # Security: this is reachable from an UNTRUSTED image-generation prompt via
    # <lora:NAME:WEIGHT>, so a bare os.path.isfile(name) would load ANY .safetensors on
    # the host — an absolute path (/Volumes/secret/other_tenant.safetensors) or a ..
    # escape — letting any authenticated tenant read another tenant's private adapter or
    # an arbitrary file. Honor a direct path ONLY when it resolves INSIDE search_dir;
    # otherwise fall through to the stem/substring search within search_dir.
    _root = os.path.realpath(search_dir)
    if os.path.isfile(name):
        _rp = os.path.realpath(name)
        if _rp == _root or _rp.startswith(_root + os.sep):
            return name
    if not os.path.isdir(search_dir):
        return None
    nl = name.lower()
    cands = [
        p
        for p in os.listdir(search_dir)
        if p.endswith(".safetensors") and "controlnet" not in p.lower()
    ]

    def forms(p):
        return {p.lower(), urllib.parse.unquote(p).lower()}

    for p in cands:
        if any(os.path.splitext(f)[0] == nl for f in forms(p)):
            return os.path.join(search_dir, p)
    for p in cands:
        if any(nl in f for f in forms(p)):
            return os.path.join(search_dir, p)
    return None


# ── ComfyUI/diffusion-format LoRA — generic wrapper + loader ──
class DiffusionLoRALinear(nn.Module):
    """Wraps a base Linear (possibly quantized) with a ComfyUI LoRA branch:
    y = base(x) + scale * (x @ down^T) @ up^T. down=[rank,in], up=[out,rank]."""

    def __init__(self, base, down, up, scale):
        super().__init__()
        self.base = base
        self.lora_down = down
        self.lora_up = up
        self._scale = float(scale)

    def __call__(self, x):
        d = self.lora_down.astype(x.dtype)
        u = self.lora_up.astype(x.dtype)
        return self.base(x) + self._scale * ((x @ d.T) @ u.T)


def load_diffusion_lora(path, *, resolve_module, key_remap=None, strength=1.0):
    """Generic ComfyUI-LoRA loader. Reads `path` (bf16-safe via mx.load), groups keys
    by module, resolves each via the model-supplied `resolve_module(path)->(target,
    parent,attr)`, wraps matching nn.Linear targets with DiffusionLoRALinear.

    Returns (restore_list, applied, skipped) where restore_list is [(parent,attr,orig)]
    for unloading (caller unwraps in REVERSE for stacked LoRAs). Pure mechanism — the
    model owns module resolution and key naming (key_remap)."""
    raw = mx.load(path)
    remap = key_remap or (lambda k: k)
    groups: dict[str, dict] = {}
    for k, v in raw.items():
        rk = remap(k)
        for suf, slot in (
            (".lora_down.weight", "down"),
            (".lora_up.weight", "up"),
            (".alpha", "alpha"),
        ):
            if rk.endswith(suf):
                mp = rk[: -len(suf)]
                groups.setdefault(mp, {})[slot] = (
                    float(v.reshape(-1)[0]) if slot == "alpha" else v
                )
                break
    restore: list = []
    applied = skipped = 0
    for mp, g in groups.items():
        if "down" not in g or "up" not in g:
            continue
        target, parent, attr = resolve_module(mp)
        # also accept an already-wrapped DiffusionLoRALinear so a SECOND
        # LoRA targeting the same module STACKS onto it (the new wrapper's base = the
        # existing wrapper, which is callable, so y = base(x)+lora2(x) composes both).
        # The old `isinstance(target, nn.Linear)` rejected it → the 2nd adapter was
        # silently dropped (applied=0 → load returns False). The caller already unwraps
        # in REVERSE order, which unwinds the nesting back to the original Linear.
        if target is None or not isinstance(target, (nn.Linear, DiffusionLoRALinear)):
            skipped += 1
            continue
        rank = int(g["down"].shape[0])
        scale = (g.get("alpha", float(rank)) / rank) * strength
        setattr(parent, attr, DiffusionLoRALinear(target, g["down"], g["up"], scale))
        restore.append((parent, attr, target))
        applied += 1
    return restore, applied, skipped
