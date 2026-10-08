"""CPU-only checkpoint-format check for big MoE MLX packs (config + tensor headers).

Reads ``config.json`` and tensor name/dtype/shape (from a local safetensors directory or
a ``headers.json`` dump); never imports MLX and never reads weight payloads. It reports
what the generic Yunshu VLM loader would do with the pack: model_type support, effective
quantization per module (top-level, per-module override, ``False``), whether packed
shapes agree with the declared bits/group_size, non-standard quantization fields, and
MTP / vision / extra tensors the loader filters out.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import re
import struct
import sys
from pathlib import Path

GIB = 2**30
STD_QUANT_KEYS = {"bits", "group_size", "mode"}
DTYPE_BYTES = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "I64": 8,
    "BOOL": 1,
}


def read_headers(root: Path) -> dict[str, tuple[str, list[int]]]:
    """name -> (dtype, shape) from a ``headers.json`` dump or every safetensors header."""
    dump = root / "headers.json"
    if dump.exists():
        return {k: (v[0], v[1]) for k, v in json.loads(dump.read_text()).items()}
    out: dict[str, tuple[str, list[int]]] = {}
    for path in sorted(root.glob("*.safetensors")):
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        for key, tensor in header.items():
            if key != "__metadata__":
                out[key] = (tensor["dtype"], tensor["shape"])
    return out


def installed_model_types(package: str) -> set[str]:
    """Model directories of an installed package, found without importing it."""
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return set()
    names: set[str] = set()
    for loc in spec.submodule_search_locations:
        models = Path(loc) / "models"
        if models.is_dir():
            names |= {p.name for p in models.iterdir() if p.is_dir()}
    return names


def quant_entries(config: dict) -> tuple[dict, dict, list[str]]:
    """(top-level std fields, per-module overrides, unknown top-level fields)."""
    q = config.get("quantization")
    if not isinstance(q, dict):
        q = {}
    top = {k: v for k, v in q.items() if k in STD_QUANT_KEYS}
    overrides = {k: v for k, v in q.items() if isinstance(v, dict) or v is False}
    unknown = sorted(
        k
        for k, v in q.items()
        if k not in STD_QUANT_KEYS and not (isinstance(v, dict) or v is False)
    )
    return top, overrides, unknown


def module_aliases(module: str) -> list[str]:
    """Config keys that can name a checkpoint tensor's module (post-sanitize paths)."""
    names = [module, module.removeprefix("language_model.")]
    # Glm5-next style: tensors are ``model.language_model.X`` while config keys are the
    # module paths ``language_model.model.X``.
    if module.startswith("model.language_model."):
        tail = module.removeprefix("model.language_model.")
        names.append("language_model.model." + tail)
    return names


def _lookup(overrides: dict, module: str):
    for alias in module_aliases(module):
        if alias in overrides:
            return overrides[alias]
    return None


def analyze(config: dict, tensors: dict, vlm_types=(), lm_types=()) -> dict:
    mtype = config.get("model_type")
    top, overrides, unknown = quant_entries(config)
    notes: list[str] = []
    if unknown:
        notes.append(
            "non-standard quantization fields ignored by stock loaders: "
            + ", ".join(f"{k}={config['quantization'][k]!r}" for k in unknown)
        )
    modules = {n.removesuffix(".scales") for n in tensors if n.endswith(".scales")}
    schemes: collections.Counter = collections.Counter()
    mismatches: list[str] = []
    for module in sorted(modules):
        weight = tensors.get(module + ".weight")
        scales = tensors[module + ".scales"]
        if weight is None:
            mismatches.append(f"{module}: scales without weight")
            continue
        ov = _lookup(overrides, module)
        if ov is False:
            mismatches.append(f"{module}: config says unquantized but scales exist")
            continue
        bits = (ov or top).get("bits")
        group = (ov or top).get("group_size")
        if bits is None or group is None:
            mismatches.append(f"{module}: no bits/group_size in config")
            continue
        wlast, slast = weight[1][-1], scales[1][-1]
        # packed uint32: wlast = in*bits/32 ; affine scales: slast = in/group
        in_dim = wlast * 32 // bits
        if in_dim // group != slast or in_dim % group:
            mismatches.append(
                f"{module}: weight[-1]={wlast} scales[-1]={slast} "
                f"disagree with bits={bits} group={group}"
            )
        schemes[(bits, group)] += 1
    unquantized = [
        n
        for n in tensors
        if n.endswith(".weight") and n.removesuffix(".weight") not in modules
    ]
    total = mtp = vision = 0
    for name, (dtype, shape) in tensors.items():
        n = DTYPE_BYTES.get(dtype, 0)
        for d in shape:
            n *= d
        total += n
        if re.match(r"(model\.)?(mtp|nextn)", name) or ".mtp." in name:
            mtp += n
        elif "vision" in name.split(".")[0] or name.startswith("model.visual"):
            vision += n
    per_expert = len({n for n in tensors if re.search(r"\.experts\.\d+\.", n)})
    if per_expert:
        notes.append(f"{per_expert} per-expert tensors (loader must stack them)")
    return dict(
        model_type=mtype,
        mlx_vlm_supports=mtype in set(vlm_types),
        mlx_lm_supports=mtype in set(lm_types),
        quant_top=top,
        quant_override_count=len(overrides),
        quant_unknown_fields=unknown,
        schemes={f"{b}bit/g{g}": c for (b, g), c in sorted(schemes.items())},
        scale_mismatches=mismatches[:20],
        scale_mismatch_count=len(mismatches),
        unquantized_weight_count=len(unquantized),
        tensor_count=len(tensors),
        total_gib=round(total / GIB, 3),
        mtp_gib=round(mtp / GIB, 3),
        vision_gib=round(vision / GIB, 3),
        base_gib=round((total - mtp - vision) / GIB, 3),
        notes=notes,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="dir with config.json and headers")
    args = parser.parse_args(argv)
    config = json.loads((args.root / "config.json").read_text())
    result = analyze(
        config,
        read_headers(args.root),
        installed_model_types("mlx_vlm"),
        installed_model_types("mlx_lm"),
    )
    print(json.dumps(result, indent=2))
    return 0 if not result["scale_mismatch_count"] else 1


if __name__ == "__main__":
    sys.exit(main())
