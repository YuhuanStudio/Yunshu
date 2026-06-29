"""— verify the image-LoRA key remap against the REAL production
Z-Image LoRA (the product's primary feature).

The ComfyUI/Flux/Z-Image attention output projection is named
`...attention.to_out.0` (a Sequential whose [0] is the Linear), but our
_DiTAttention.to_out is a FLAT nn.Linear. Before the un-collapsed `.to_out.0`
path made _resolve_lora_module treat the trailing `0` as a list index →
getattr(linear,"0")=None → the projection was SILENTLY skipped on all 30 DiT
layers, badly weakening every image LoRA with no error.

This checks the remap on the actual on-disk adapter keys WITHOUT loading the
(heavy) diffusion model — so it's safe to run on the 36GB Mac.

Run: PYTHONPATH=. uv run python scripts/verify/verify_image_lora_remap.py
"""
import glob
import sys

from python.yunshu_engine.image_engine import ImageGenEngine


def _module_of(k: str) -> str:
    for suf in (".lora_down.weight", ".lora_up.weight", ".lora_A.weight",
                ".lora_B.weight", ".lora_down", ".lora_up", ".alpha", ".weight"):
        if k.endswith(suf):
            return k[: -len(suf)]
    return k


def main() -> int:
    from safetensors import safe_open

    # The Z-Image style LoRA (filename is URL-encoded CJK); match the *_consistent one.
    candidates = glob.glob("models/*consistent*.safetensors")
    if not candidates:
        print("SKIP: no Z-Image *_consistent*.safetensors LoRA found under models/")
        return 0
    path = candidates[0]
    remap = ImageGenEngine._zimage_lora_key_remap

    with safe_open(path, framework="numpy") as f:
        keys = list(f.keys())

    mods = sorted({_module_of(k) for k in keys})
    to_out_raw = [m for m in mods if "to_out" in m]
    remapped_to_out = [remap(m) for m in to_out_raw]

    # No remapped to_out / adaLN path may still end in a bare list index '.0'
    # (that is the signature of the silent-skip bug).
    dead = [remap(m) for m in mods]
    dead = [r for r in dead if r.endswith(".0") and ("to_out" in r or "adaLN" in r)]

    print(f"LoRA: {path}")
    print(f"  keys={len(keys)} modules={len(mods)} to_out_modules={len(to_out_raw)}")
    if to_out_raw:
        print(f"  raw:     {to_out_raw[0]}")
        print(f"  remapped:{remapped_to_out[0]}")

    if dead:
        print(f"FAIL: {len(dead)} to_out/adaLN paths still end in '.0' (would be SKIPPED)")
        return 1
    if to_out_raw and not all(r.endswith("to_out") for r in remapped_to_out):
        print("FAIL: some to_out paths not collapsed to flat Linear")
        return 1

    print(f"PASS: all {len(to_out_raw)} to_out projections remap to a resolvable "
          f"flat-Linear path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
