"""Published MoE pack formats: shapes decide bits/group when the config disagrees."""

from types import SimpleNamespace

import pytest

from yunshu_engine.moe_pack import (
    infer_quantization,
    tensor_quantization_override,
    unsupported_pack_reason,
)


def T(shape, dtype="uint32"):
    return SimpleNamespace(shape=tuple(shape), dtype=dtype)


def weights_for(path, in_dim, bits, group, *, out=8, biases=True, experts=None):
    lead = (experts,) if experts else ()
    w = {
        f"{path}.weight": T((*lead, out, in_dim * bits // 32)),
        f"{path}.scales": T(
            (*lead, out, in_dim // group), "bfloat16" if biases else "uint8"
        ),
    }
    if biases:
        w[f"{path}.biases"] = T((*lead, out, in_dim // group), "bfloat16")
    return w


@pytest.mark.parametrize(
    "bits,group", [(2, 128), (2, 64), (3, 128), (4, 64), (6, 128), (8, 64), (8, 32)]
)
def test_affine_shapes_round_trip(bits, group):
    w = weights_for("m", 4096, bits, group)
    got = infer_quantization(
        4096, w["m.weight"].shape, w["m.scales"].shape,
        scales_dtype="bfloat16", has_biases=True,
    )  # fmt: skip
    assert got == {"bits": bits, "group_size": group, "mode": "affine"}


def test_two_bit_g128_and_four_bit_g64_differ_only_by_input_width():
    # identical weight/scales shape ratio; the module's real input width decides
    w = weights_for("m", 4096, 4, 64)
    args = (w["m.weight"].shape, w["m.scales"].shape)
    assert infer_quantization(4096, *args, scales_dtype="bf16", has_biases=True) == {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }
    assert infer_quantization(8192, *args, scales_dtype="bf16", has_biases=True) == {
        "bits": 2,
        "group_size": 128,
        "mode": "affine",
    }


def test_ddalcu_late_layer_experts_are_four_bit_despite_two_bit_config():
    # layers.39+ of the mixed-2-3-8bit pack: config says 2-bit g128, tensors are 4-bit g64
    path = "layers.39.ffn.experts.w1"
    w = weights_for(path, 4096, 4, 64, out=2048, experts=256)
    declared = {"bits": 2, "group_size": 128, "mode": "affine"}
    got = tensor_quantization_override(w, path, (256, 2048, 4096), {}, declared)
    assert got == {"bits": 4, "group_size": 64, "mode": "affine"}


def test_pipenetwork_expert_bits_field_is_not_trusted():
    # top-level bits 8 / expert_bits 4 (non-standard): experts have 4-bit tensors
    path = "layers.0.ffn.experts.gate_proj"
    w = weights_for(path, 4096, 4, 64, out=1024, experts=128)
    top = {"bits": 8, "group_size": 64}
    got = tensor_quantization_override(w, path, (128, 1024, 4096), top, None)
    assert got == {"bits": 4, "group_size": 64, "mode": "affine"}


def test_correct_declaration_is_left_alone():
    path = "layers.0.attn.wq"
    w = weights_for(path, 4096, 6, 128)
    for declared, top in (
        ({"bits": 6, "group_size": 128}, {}),
        (None, {"bits": 6, "group_size": 128}),
    ):
        assert tensor_quantization_override(w, path, (8, 4096), top, declared) is None  # fmt: skip


def test_false_override_with_scales_present_is_overruled_by_tensors():
    path = "head"
    w = weights_for(path, 4096, 8, 64)
    got = tensor_quantization_override(w, path, (8, 4096), {"bits": 2}, False)
    assert got == {"bits": 8, "group_size": 64, "mode": "affine"}


def test_native_mx_packs_without_biases():
    w = weights_for("m", 4096, 4, 32, biases=False)
    got = tensor_quantization_override(w, "m", (8, 4096), {}, None)
    assert got == {"bits": 4, "group_size": 32, "mode": "mxfp4"}
    w = weights_for("m", 4096, 8, 32, biases=False)
    assert tensor_quantization_override(w, "m", (8, 4096), {}, None)["mode"] == "mxfp8"
    w = weights_for("m", 4096, 4, 16, biases=False)
    assert tensor_quantization_override(w, "m", (8, 4096), {}, None)["mode"] == "nvfp4"


def test_undecidable_shapes_fall_back_to_config():
    w = weights_for("m", 4096, 4, 64)
    assert tensor_quantization_override(w, "m", None, {}, None) is None
    assert tensor_quantization_override({}, "m", (8, 4096), {}, None) is None
    bad = {"m.weight": T((8, 100)), "m.scales": T((8, 7)), "m.biases": T((8, 7))}
    assert tensor_quantization_override(bad, "m", (8, 4096), {}, None) is None


def test_custom_codec_packs_are_rejected_loudly():
    lemura = {
        "quantization": {
            "bits": 8,
            "routed_expert_bit_plan": {"codec": "mxtq"},
            "mxtq_bits": {"routed_expert": 2},
        }
    }
    assert "custom expert codec" in unsupported_pack_reason(lemura, [])
    names = ["layers.5.ffn.switch_mlp.gate_proj.tq_packed"]
    assert "tq_packed" in unsupported_pack_reason({}, names)
    assert unsupported_pack_reason({"quantization": {"bits": 2}}, ["a.weight"]) is None


def test_wrongcfg_variants_are_definitions_not_truth():
    import sys

    sys.path.insert(0, "scripts/research")
    from bigmoe_wrongcfg import VARIANTS, variant_config

    q = {"bits": 4, "group_size": 32, "mode": "affine", "x": {"bits": 2}}
    base = {"hidden_size": 64, "quantization": {"junk": 1}}
    assert variant_config(base, "truthful", q)["quantization"] is q
    assert variant_config(base, "no_block", q) == {"hidden_size": 64}
    cfgs = {n: variant_config(base, n, q).get("quantization") for n in VARIANTS}
    assert cfgs["expert_bits_style"]["expert_bits"] == 2
    assert "x" not in cfgs["top_only_4bit"]
