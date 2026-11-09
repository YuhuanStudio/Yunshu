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
import struct
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


def execution_records(receipt):
    if "execution_json" in receipt:
        return json.loads(receipt["execution_json"])
    if "execution" in receipt:
        return receipt["execution"]
    payload = (
        json.loads(receipt["payload_json"])
        if "payload_json" in receipt
        else receipt["payload"]
    )
    return payload.get("execution", [])


def validate(receipt):
    if receipt.get("complete") is not True:
        raise ValueError("missing complete receipt")
    payload = (
        json.loads(receipt["payload_json"])
        if "payload_json" in receipt
        else receipt["payload"]
    )
    if receipt.get("sha256") != digest(payload):
        raise ValueError("payload checksum mismatch")
    if payload["layer"] not in LAYERS or not payload.get("device"):
        raise ValueError("missing layer/device")
    if set(payload["globals"]) != set(GLOBAL_STATES):
        raise ValueError("missing module-global assertions")
    if type(payload["tokens"]) is not int or payload["tokens"] < 4:
        raise ValueError("invalid token budget")
    caps = payload["capabilities"]

    def matrix(cells, schedules, count):
        expected = {(m, s) for m in ("greedy", "seeded") for s in schedules}
        if {(c["mode"], c["schedule"]) for c in cells} != expected or len(cells) != len(
            expected
        ):
            raise ValueError("incomplete scenario matrix")
        for cell in cells:
            n = 2 if cell["schedule"] == "cancel" else count
            if len(cell["rows"]) != n or any(
                len(row) != payload["tokens"] for row in cell["rows"]
            ):
                raise ValueError("incomplete token/logprob evidence")
            for row in cell["rows"]:
                for event in row:
                    if type(event["token"]) is not int or not math.isfinite(
                        float.fromhex(event["logprob"])
                    ):
                        raise ValueError("invalid token/logprob")

    matrix(
        payload["cells"], ("solo", "burst", "stagger", "cancel", "media", "repeat"), 3
    )
    matrix(
        payload["spec_cells"], ("solo", "burst", "stagger") if caps["spec"] else (), 3
    )
    matrix(payload["cache_cells"], ("cold", "hit") if caps["apc"] else (), 1)
    for cell in payload["cache_cells"]:
        hits = cell["hits"]
        if len(hits) != 1 or (
            hits[0] != 0 if cell["schedule"] == "cold" else hits[0] <= 0
        ):
            raise ValueError("missing APC hit/cold evidence")
        if (
            caps["replay"]
            and cell["schedule"] == "hit"
            and cell.get("replayed", 0) <= 0
        ):
            raise ValueError("missing replay evidence")
    if not caps["apc"] and not caps.get("apc_unavailable_reason"):
        raise ValueError("APC unavailability needs a reason")
    if caps["spec"]:
        runs = [r for r in execution_records(receipt) if r.get("spec")]
        expected = {
            (mode, schedule, indices)
            for mode in ("greedy", "seeded")
            for schedule, indices in (
                ("solo", (0,)),
                ("solo", (1,)),
                ("solo", (2,)),
                ("burst", (0, 1, 2)),
                ("stagger", (0, 1, 2)),
            )
        }
        if (
            len(runs) != len(expected)
            or {(r["mode"], r["schedule"], tuple(r["indices"])) for r in runs}
            != expected
            or any(r.get("drafted", 0) <= 0 for r in runs)
        ):
            raise ValueError("missing speculative execution evidence")
    int8 = payload["int8_reference"]
    if caps["int8"] and (len(int8) != 3 or any(not r.get("sha256") for r in int8)):
        raise ValueError("missing int8 oracle evidence")
    if not caps["int8"] and int8:
        raise ValueError("unexpected int8 evidence")
    if payload.get("checkpoint") is not None:
        # A real checkpoint is valid only as the served configuration: the kernel stack the server
        # installs, with split-invariant prefill engaged on qwen3_5 (the 27B path).
        kernels = payload.get("serving_kernels")
        if not isinstance(kernels, dict):
            raise ValueError("missing served kernel evidence")
        if (
            payload["layer"] == "qwen3_5"
            and kernels.get("split_prefill") != "split64-v1"
        ):
            raise ValueError(
                "split-invariant prefill is not engaged on the served qwen3_5"
            )
        long_burst = payload.get("long_burst")
        if not isinstance(long_burst, list) or sorted(
            c.get("mode") for c in long_burst
        ) != ["greedy", "seeded"]:
            raise ValueError("missing long-row burst evidence")
        for cell in long_burst:
            if (
                cell.get("schedule") != "burst"
                or len(cell["rows"]) != 3
                or any(len(row) != payload["tokens"] for row in cell["rows"])
            ):
                raise ValueError("incomplete long-row burst evidence")
    # Legacy receipts included scheduler telemetry; it is not arithmetic identity.
    return {key: value for key, value in payload.items() if key != "execution"}


def compare(reference, candidate):
    a, b = validate(reference), validate(candidate)
    if a != b:
        raise ValueError(
            "golden mismatch (device, configuration, tokens, logprobs or scopes)"
        )


def build(layer, model):
    import mlx.core as mx

    if model:
        return _build_checkpoint(layer, model)
    if layer == "generic":
        from tests.unit.test_round_driver_generic import PROMPTS, TinyLM

        lm = TinyLM()
        return lm, lm, None, PROMPTS, None
    if layer == "flash_next":
        from types import SimpleNamespace

        from tests.unit.test_round_driver_flashnext import PROMPTS, make_tiny

        lm, head = make_tiny()
        return SimpleNamespace(language_model=lm), lm, head, PROMPTS, None
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
        from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import Qwen3_5MTPConfig
        from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import (
            Qwen3_5MTPDraftModel,
        )

        mx.random.seed(3)
        head = Qwen3_5MTPDraftModel(
            Qwen3_5MTPConfig(
                text_config={**asdict(tiny_config()), "mtp_num_hidden_layers": 1}
            )
        )
        head.set_dtype(mx.bfloat16)
        mx.eval(head.parameters())
    return (
        lm,
        lm,
        head,
        [
            [3 + (i * 7 + salt) % 500 for i in range(n)]
            for salt, n in enumerate((17, 33, 25))
        ],
        None,
    )


def _build_checkpoint(layer, model):
    """A real checkpoint, loaded and kernel-installed the way VLMEngine serves it.

    The loader, the drafter and the kernel stack are the served ones: VLMEngine._load_vision_model,
    the drafter VLMEngine loads for MTP / Qwen4 heads, and serving_kernels.install (the block
    VLMEngine runs). Returns (wrapper, lm, head, prompts, kernels).
    """
    from types import SimpleNamespace

    from mlx_lm.utils import load_tokenizer

    from yunshu_engine import serving_kernels, settings
    from yunshu_engine.qwen4_family import is_qwen4_family
    from yunshu_engine.vlm_engine import VLMEngine

    holder = SimpleNamespace()
    wrapper = VLMEngine._load_vision_model(holder, Path(model))
    lm = wrapper.language_model
    config = json.loads((Path(model) / "config.json").read_text())
    tokenizer = load_tokenizer(Path(model))
    if layer == "flash_next":
        from yunshu_engine.qwen4_mtp import load_native_head

        head = load_native_head(
            config,
            holder._qwen4_mtp_weights,
            direct_norms=holder._qwen4_direct_norms,
        )
        ids = tokenizer.encode("Explain why the sky is blue. " * 30)
    else:
        from scripts.research.probe_wide_verify import FILLER

        from yunshu_engine.mlxvlm_mtp import _load_drafter_in_memory

        head = optional_mtp_head(_load_drafter_in_memory, model)
        ids = tokenizer.encode(FILLER, add_special_tokens=False)
    kernels, _ = serving_kernels.install(
        wrapper,
        config,
        drafter=head,
        draft_kind="mtp",
        spec_family=VLMEngine.is_spec_model_type(config.get("model_type")),
        model_path=Path(model),
        tokenizer=tokenizer,
        block=settings.get("YUNSHU_MTP_BLOCK_SIZE"),
    )
    if not is_qwen4_family(config):
        # the runner's KV setup for the qwen3_5 round driver (VLMEngine._build_batch_runner)
        from yunshu_engine.kernels import ragged_kv

        if ragged_kv.supports(lm):
            ragged_kv.install()
            ragged_kv.enable(None)
    return wrapper, lm, head, real_prompts(ids, model), kernels


def int8_reference():
    """Keep the existing dequantized-bf16 arithmetic oracle, never int8 == bf16 quality."""
    import mlx.core as mx
    from tests.unit.test_int8_tile_kernel import _int8_cache

    from yunshu_engine import tree_verify as tv
    from yunshu_engine.kernels import int8_tile
    from yunshu_engine.kernels import ragged_attention as ra

    receipts = []
    for hkv, g, tokens, length in ((1, 4, 1, 1300), (2, 4, 4, 1300), (1, 4, 2, 65)):
        cap = -(-length // 64) * 64
        kc, ks, vc, vs = _int8_cache(hkv, cap, seed=length + tokens)
        mx.random.seed(7)
        q = mx.random.normal((1, hkv * g, tokens, 256)).astype(mx.bfloat16)
        fused = tv._fuse_tokens(q, hkv, g, tokens)
        nc = -(-length // tv.CK)
        plan = ra._work_list((length,), tokens, nc, True)
        args = (
            fused,
            kc,
            ks,
            vc,
            vs,
            mx.array([length], dtype=mx.int32),
            mx.array([0], dtype=mx.int32),
            tv._scale_array(256**-0.5),
            plan,
            tokens,
            nc,
        )
        got = int8_tile.tile_partials_q8(*args)
        ref = tv.tile_partials_q8_reference(*args)
        assert all(
            bool(mx.array_equal(a, b)) for a, b in zip(got[:3], ref[:3], strict=True)
        )
        receipts.append(
            dict(
                hkv=hkv,
                group=g,
                tokens=tokens,
                length=length,
                sha256=digest(
                    [
                        {
                            "shape": list(a.shape),
                            "dtype": str(a.dtype),
                            "bits": hashlib.sha256(
                                struct.pack(
                                    f"<{a.size}f",
                                    *a.astype(mx.float32).reshape(-1).tolist(),
                                )
                            ).hexdigest(),
                        }
                        for a in got[:3]
                    ]
                ),
            )
        )
    return receipts


APC_PROBE_MIN_TOKENS = (
    1024  # > checkpoint_interval_tokens (512) so a 64-aligned checkpoint exists
)


def apc_probe_prompt(encode, minimum=APC_PROBE_MIN_TOKENS):
    """A prompt long enough that the APC probe can hit a stored checkpoint.

    Short prompts never reach a checkpoint, so a warm hit cannot occur: the
    real-checkpoint probe needs at least one checkpoint interval of tokens.
    """
    text = "Explain why the sky is blue. "
    reps = 8
    while reps <= 4096:
        ids = list(encode(text * reps))
        if len(ids) >= minimum:
            return ids
        reps *= 2
    raise ValueError("tokenizer produced too few tokens for the APC probe")


def optional_mtp_head(load, model):
    """The MTP head when the checkpoint ships one; None (spec not exercised) otherwise.

    The receipt records spec = (head is not None), so a checkpoint without MTP
    tensors is captured on the non-speculative path instead of failing.
    """
    try:
        return load(model)
    except ValueError as exc:
        if "No MTP tensors found" in str(exc):
            return None
        raise


def apc_probe_sizing(model):
    """(num_blocks, memory_max_gb) for the APC probe.

    The tiny probe keeps its validated sizing (64 blocks = 1024 tokens, 0.1 GB).
    A real checkpoint's 1024-token prompt fills that pool exactly and its hybrid
    checkpoint state exceeds 0.1 GB, so the probe gets room to store checkpoints.
    Sizing only: the arithmetic is unchanged.
    """
    if model is None:
        return 64, 0.1
    return 512, 2.0


def real_prompts(ids, model):
    """Three short cell prompts (as validated) plus one APC probe prompt.

    cells use indices 0-2 and keep their short prefill arithmetic; the APC
    section picks the longest prompt (the probe) via cache_prompt_index.
    """
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(Path(model))
    return [ids[:17], ids[20:53], ids[60:85], apc_probe_prompt(tokenizer.encode)]


def cache_prompt_index(prompts):
    index = max(range(len(prompts)), key=lambda i: len(prompts[i]))
    if len(prompts[index]) <= 16:
        raise ValueError("APC probe needs a prompt past its 16-token floor")
    return index


def _capture(layer, model=None, tokens=8, device="gpu"):
    import mlx.core as mx

    from yunshu_engine.round_driver.driver import Request, RoundDriver
    from yunshu_engine.vlm_batch_runner import RowParams

    mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    wrapper, lm, head, prompts, kernels = build(layer, model)
    baseline = global_state()
    execution = []
    assert not baseline["gdn_prefill._PREFILL"] and not baseline["gdn_prefill._TAIL"]

    def run(indices, mode, schedule, *, spec=False, apc=None, hits=None):
        d = RoundDriver(
            wrapper,
            generic_layer=layer == "generic",
            stop_tokens=set(),
            chunk=512,
            drafter=head if spec else None,
            max_draft=1,
            apc=apc,
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
                hit = d.add(
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
                if hits is not None:
                    hits.append(hit)
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
        if schedule == "cancel":
            assert 0 < len(out[indices[-1]]) < tokens, "cancellation was not exercised"
        execution.append(
            dict(
                indices=indices,
                mode=mode,
                schedule=schedule,
                spec=spec,
                steps=steps,
                drafted=d.drafted,
                accepted=d.accepted,
            )
        )
        return [out[i] for i in indices if schedule != "cancel" or i != indices[-1]]

    cells = []
    spec_cells = []
    cache_cells = []
    long_cells = []
    for mode in ("greedy", "seeded"):
        solo = [run([i], mode, "solo")[0] for i in range(3)]
        if model:
            # The long APC probe row joins the short rows in one burst. Its prefill (and every
            # row's logprobs) must not depend on the rows decoding beside it: solo == burst, bitwise.
            long_index = cache_prompt_index(prompts)
            long_solo = run([long_index], mode, "solo")[0]
            long_rows = run([0, 1, long_index], mode, "burst")
            assert long_rows == [solo[0], solo[1], long_solo], (mode, "long-row burst")
            long_cells.append(dict(mode=mode, schedule="burst", rows=long_rows))
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
            if layer != "generic" or schedule == "repeat":
                assert rows == solo[: len(rows)], (mode, schedule, "logprobs")
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
                assert rows == solo, (mode, schedule, "spec-logprobs")
                spec_cells.append(dict(mode=mode, schedule=schedule, rows=rows))
        if layer != "generic":
            # Use the same APC-enabled cold span plan as the hit: checkpoint seams
            # participate in prefill arithmetic even when lookup found nothing.
            from mlx_vlm.apc import APCManager

            from yunshu_engine.round_driver import replay

            num_blocks, memory_max_gb = apc_probe_sizing(model)
            apc = APCManager(
                num_blocks=num_blocks,
                block_size=16,
                overrides={
                    "memory_max_gb": memory_max_gb,
                    "checkpoint_entries": 4,
                    "checkpoint_interval_tokens": 512,
                },
            )
            replay.clear()
            cache_index = cache_prompt_index(prompts)
            cold_hits, warm_hits = [], []
            cold = run([cache_index], mode, "cold", apc=apc, hits=cold_hits)
            warm = run([cache_index], mode, "hit", apc=apc, hits=warm_hits)
            assert cold_hits == [0] and warm_hits[0] > 0, (cold_hits, warm_hits)
            assert [[e["token"] for e in r] for r in cold] == [
                [e["token"] for e in r] for r in warm
            ]
            assert cold == warm, (mode, "APC logprobs")
            cache_cells.extend(
                [
                    dict(mode=mode, schedule="cold", rows=cold, hits=cold_hits),
                    dict(mode=mode, schedule="hit", rows=warm, hits=warm_hits),
                ]
            )
            replayed = replay.stats()["replayed"]
            if layer != "generic":
                assert replayed > 0, "APC hit did not exercise tail replay"
            cache_cells[-1]["replayed"] = replayed
            replay.clear()
    import importlib.metadata
    import platform
    import subprocess

    int8 = int8_reference() if layer in ("qwen3_5", "qwen3_5_moe") else []
    assert_globals(baseline)
    payload = dict(
        layer=layer,
        model=str(model or f"tiny:{layer}"),
        prompt_sha256=digest(prompts),
        checkpoint={
            "config_sha256": hashlib.sha256(
                (Path(model) / "config.json").read_bytes()
            ).hexdigest(),
            "weight_sizes": {
                p.name: p.stat().st_size
                for p in sorted(Path(model).glob("*.safetensors"))
            },
        }
        if model
        else None,
        seeds={
            "target": 0 if layer == "generic" else 17 if layer == "flash_next" else 23,
            "head": 5 if layer == "flash_next" else 3,
            "sampling": 41,
        }
        if not model
        else {"sampling": 41},
        int8_reference=int8,
        device=str(mx.default_device()),
        hardware=mx.device_info()
        if device == "gpu"
        else {
            "device": "cpu",
            "machine": platform.machine(),
            "chip": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip(),
        },
        versions={p: importlib.metadata.version(p) for p in ("mlx", "mlx-vlm")},
        globals=baseline,
        tokens=tokens,
        cells=cells,
        spec_cells=spec_cells,
        cache_cells=cache_cells,
        capabilities={
            "spec": head is not None,
            "apc": layer != "generic",
            "apc_unavailable_reason": "stock dense generic selects block strategy; current driver admits checkpoints only"
            if layer == "generic"
            else None,
            "replay": layer != "generic",
            "int8": bool(int8),
            "media": "embedding-equivalence",
        },
    )
    if model:
        # Real checkpoints only (the tiny receipts keep their payload bytes): the served kernel
        # stack that serving_kernels.install engaged, and the long-row burst cells.
        payload["serving_kernels"] = {
            key: value
            for key, value in (kernels or {}).items()
            if isinstance(value, (str, int, bool, type(None)))
        }
        payload["long_burst"] = long_cells
    receipt = dict(
        complete=True, payload=payload, sha256=digest(payload), execution=execution
    )
    validate(receipt)
    return receipt


def capture(layer, model=None, tokens=8, device="gpu"):
    import os

    if device == "gpu" and not os.environ.get("GPUQ_JOB_ID"):
        raise RuntimeError("GPU conformance requires gpuq admission")
    import mlx.core as mx

    previous = mx.default_device()
    try:
        return _capture(layer, model, tokens, device)
    finally:
        mx.set_default_device(previous)


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
    # gpuq annotates every dictionary recursively. Seal the canonical payload
    # as text so transport metadata cannot overwrite model state or token data.
    wire = {
        "complete": receipt["complete"],
        "sha256": receipt["sha256"],
        "execution_json": json.dumps(receipt.get("execution", []), sort_keys=True),
        "payload_json": json.dumps(
            receipt["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
    }
    a.out.write_text(json.dumps(wire, sort_keys=True) + "\n")
    print(json.dumps(dict(complete=True, sha256=receipt["sha256"])))


if __name__ == "__main__":
    main()
