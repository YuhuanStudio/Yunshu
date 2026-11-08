"""Device-specific round-driver migration receipts. GPU execution belongs in gpuq.

Tokens and exact float.hex logprobs are kept for each scenario, not rounded or
compared across devices. --reference compares an earlier receipt byte-for-byte
on its deterministic payload. A dry run is never a completed golden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

LAYERS = ("generic", "qwen3_5", "qwen3_5_moe", "flash_next")
GLOBAL_STATES = (
    "ragged_kv._STATE",
    "batch_invariant._STATE",
    "gdn_prefill._STATE",
    "gdn_prefill._PREFILL",
    "gdn_prefill._TAIL",
    "lane_linear.STOCK_ROWS",
    "lane_linear._SUM_POLICY",
    "lane_linear._SUM_REUSE",
)


def digest(payload):
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def global_state():
    from yunshu_engine.kernels import (
        batch_invariant,
        gdn_prefill,
        lane_linear,
        ragged_kv,
    )

    return {
        "ragged_kv._STATE": dict(ragged_kv._STATE),
        "batch_invariant._STATE": dict(batch_invariant._STATE),
        "gdn_prefill._STATE": dict(gdn_prefill._STATE),
        "gdn_prefill._PREFILL": gdn_prefill.in_prefill(),
        "gdn_prefill._TAIL": gdn_prefill.in_tail(),
        "lane_linear.STOCK_ROWS": lane_linear.STOCK_ROWS,
        "lane_linear._SUM_POLICY": lane_linear._SUM_POLICY,
        "lane_linear._SUM_REUSE": lane_linear._SUM_REUSE,
    }


def assert_globals(expected):
    actual = global_state()
    assert actual == expected, f"module state leaked: {expected!r} -> {actual!r}"


def event_record(event):
    assert event.token is not None and event.logprob is not None
    lp = event.logprob["logprob"]
    assert math.isfinite(lp)
    return {"token": int(event.token), "logprob": float(lp).hex()}


def validate(receipt):
    if receipt.get("complete") is not True:
        raise ValueError("missing complete receipt")
    payload = receipt["payload"]
    if receipt.get("sha256") != digest(payload):
        raise ValueError("payload checksum mismatch")
    if payload["layer"] not in LAYERS or not payload.get("device"):
        raise ValueError("missing layer/device")
    if set(payload["globals"]) != set(GLOBAL_STATES):
        raise ValueError("missing module-global assertions")
    cells = payload["cells"]
    expected = {
        (mode, schedule)
        for mode in ("greedy", "seeded")
        for schedule in ("solo", "burst", "stagger", "cancel", "media", "repeat")
    }
    if {(c["mode"], c["schedule"]) for c in cells} != expected or len(cells) != len(
        expected
    ):
        raise ValueError("incomplete scenario matrix")
    for cell in cells:
        if not cell["rows"] or any(
            len(row) != payload["tokens"] for row in cell["rows"]
        ):
            raise ValueError("incomplete token/logprob evidence")
        for row in cell["rows"]:
            for event in row:
                if type(event["token"]) is not int or not math.isfinite(
                    float.fromhex(event["logprob"])
                ):
                    raise ValueError("invalid token/logprob")
    return payload


def compare(reference, candidate):
    a, b = validate(reference), validate(candidate)
    if a != b:
        raise ValueError(
            "golden mismatch (device, configuration, tokens, logprobs or scopes)"
        )


def build(layer, model):
    import mlx.core as mx

    if model:
        if layer == "flash_next":
            from types import SimpleNamespace

            from yunshu_engine.kernels import omlx
            from yunshu_engine.qwen4_mtp import configure_lane, load_native_head
            from yunshu_engine.vlm_engine import VLMEngine

            omlx.apply()
            holder = SimpleNamespace()
            wrapper = VLMEngine._load_vision_model(holder, Path(model))
            lm = wrapper.language_model
            config = json.loads((Path(model) / "config.json").read_text())
            head = load_native_head(
                config,
                holder._qwen4_mtp_weights,
                direct_norms=holder._qwen4_direct_norms,
            )
            configure_lane(lm, head, None)
            from mlx_lm.utils import load_tokenizer

            ids = load_tokenizer(Path(model)).encode(
                "Explain why the sky is blue. " * 30
            )
        else:
            from scripts.research.probe_wide_verify import setup

            wrapper, lm, ids = setup(model)
            from yunshu_engine.kernels import ragged_kv

            ragged_kv.set_dense_lane(False)
            from yunshu_engine.mlxvlm_mtp import _load_drafter_in_memory

            head = _load_drafter_in_memory(model)
        return wrapper, lm, head, [ids[:17], ids[20:53], ids[60:85]]
    if layer == "generic":
        from tests.unit.test_round_driver_generic import PROMPTS, TinyLM

        lm = TinyLM()
        return lm, lm, None, PROMPTS
    if layer == "flash_next":
        from types import SimpleNamespace

        from tests.unit.test_round_driver_flashnext import PROMPTS, make_tiny

        lm, head = make_tiny()
        return SimpleNamespace(language_model=lm), lm, head, PROMPTS
    if layer == "qwen3_5":
        from tests.unit.test_round_driver import _tiny

        lm, head = _tiny()
    else:
        from dataclasses import asdict

        import mlx.nn as nn
        from mlx_vlm.models.qwen3_5_moe.config import TextConfig
        from mlx_vlm.models.qwen3_5_moe.language import LanguageModel
        from scripts.research.phaseb_verify import tiny_config

        from yunshu_engine.kernels import (
            batch_invariant,
            gdn_prefill,
            lane_linear,
            omlx,
            ragged_kv,
        )

        cfg = asdict(tiny_config())
        cfg.pop("intermediate_size")
        cfg.update(
            model_type="qwen3_5_moe",
            num_experts=4,
            num_experts_per_tok=2,
            shared_expert_intermediate_size=256,
            moe_intermediate_size=256,
        )
        omlx.apply()
        mx.random.seed(23)
        lm = LanguageModel(TextConfig.from_dict(cfg))
        lm.set_dtype(mx.bfloat16)
        nn.quantize(lm, group_size=64, bits=4)
        lane_linear.convert(lm)
        batch_invariant.install(lm, packed=False)
        batch_invariant.set_active(True)
        gdn_prefill.install()
        ragged_kv.install()
        ragged_kv.set_dense_lane(False)
        mx.eval(lm.parameters())
        head = None
    return (
        lm,
        lm,
        head,
        [
            [3 + (i * 7 + salt) % 500 for i in range(n)]
            for salt, n in enumerate((17, 33, 25))
        ],
    )


def capture(layer, model=None, tokens=8, device="gpu"):
    import mlx.core as mx

    from yunshu_engine.round_driver.driver import Request, RoundDriver
    from yunshu_engine.vlm_batch_runner import RowParams

    mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    wrapper, lm, head, prompts = build(layer, model)
    baseline = global_state()
    assert not baseline["gdn_prefill._PREFILL"] and not baseline["gdn_prefill._TAIL"]

    def run(indices, mode, schedule, *, spec=False):
        d = RoundDriver(
            wrapper,
            generic_layer=layer == "generic",
            stop_tokens=set(),
            chunk=512,
            drafter=head if spec else None,
            max_draft=1,
        )
        pending = list(indices)
        out = {i: [] for i in indices}
        steps = 0
        while pending or d.busy():
            while pending and (
                schedule != "stagger" or steps >= indices.index(pending[0]) * 3
            ):
                i = pending.pop(0)
                kw = {}
                if layer == "generic":
                    kw["prompt_kwargs"] = {
                        "inputs_embeds": lm.embed(mx.array(prompts[i])[None])
                    }
                elif schedule == "media":
                    kw["embeds"] = lm.model.embed_tokens(mx.array(prompts[i]))
                kw["media"] = schedule == "media"
                params = (
                    RowParams(temperature=0.7, top_p=1.0, top_k=0, min_p=0.0, seed=41)
                    if mode == "seeded"
                    else None
                )
                d.add(
                    Request(
                        prompts[i],
                        tokens,
                        handle=i,
                        draft=spec,
                        logprobs=True,
                        sampling=params,
                        **kw,
                    )
                )
            for event in d.step():
                if event.token is not None:
                    out[event.handle].append(event_record(event))
            steps += 1
            if schedule == "cancel" and steps == 3:
                d.remove(indices[-1])
            assert_globals(baseline)
            assert steps < 1000, "driver did not drain"
        assert not d.rows and not d.batch.rows
        if spec:
            assert d.drafted > 0, "spec requested but no tokens drafted"
        return [out[i] for i in indices if schedule != "cancel" or i != indices[-1]]

    cells = []
    spec_cells = []
    for mode in ("greedy", "seeded"):
        solo = [run([i], mode, "solo")[0] for i in range(3)]
        for schedule in ("solo", "burst", "stagger", "cancel", "media", "repeat"):
            rows = (
                solo
                if schedule == "solo"
                else (
                    [run([i], mode, "solo")[0] for i in range(3)]
                    if schedule == "repeat"
                    else run(list(range(3)), mode, schedule)
                )
            )
            # Stock generic batching may change low logprob bits. Preserve them
            # in each scenario's golden; row invariance is a token identity gate.
            assert [[e["token"] for e in r] for r in rows] == [
                [e["token"] for e in r] for r in solo[: len(rows)]
            ], (mode, schedule)
            cells.append(dict(mode=mode, schedule=schedule, rows=rows))
        if head is not None:
            for schedule in ("solo", "burst", "stagger"):
                rows = (
                    [run([i], mode, "solo", spec=True)[0] for i in range(3)]
                    if schedule == "solo"
                    else run(list(range(3)), mode, schedule, spec=True)
                )
                assert [[e["token"] for e in r] for r in rows] == [
                    [e["token"] for e in r] for r in solo
                ], (mode, schedule, "spec")
                spec_cells.append(dict(mode=mode, schedule=schedule, rows=rows))
    import importlib.metadata

    payload = dict(
        layer=layer,
        model=str(model or "tiny-seed-23"),
        device=str(mx.default_device()),
        hardware=mx.device_info() if device == "gpu" else {"device": "cpu"},
        versions={p: importlib.metadata.version(p) for p in ("mlx", "mlx-vlm")},
        globals=baseline,
        tokens=tokens,
        cells=cells,
        spec_cells=spec_cells,
        capabilities={
            "spec": head is not None,
            "apc": False,
            "replay": False,
            "int8": False,
            "media": "embedding-equivalence",
        },
    )
    receipt = dict(complete=True, payload=payload, sha256=digest(payload))
    validate(receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", choices=LAYERS, required=True)
    p.add_argument("--model")
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.tokens < 4:
        p.error("at least four tokens required to exercise cancellation")
    if a.dry_run:
        print(json.dumps(dict(layer=a.layer, globals=GLOBAL_STATES, complete=False)))
        return
    receipt = capture(a.layer, a.model, a.tokens, a.device)
    if a.reference:
        compare(json.loads(a.reference.read_text()), receipt)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(dict(complete=True, sha256=receipt["sha256"])))


if __name__ == "__main__":
    main()
