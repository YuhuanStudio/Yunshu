"""The VLM serving path end to end on CPU: gateway app -> VLMEngine -> VLMBatchRunner (APC prefix
cache, continuous batching, per-row sampling, constrained JSON) over a tiny RANDOM-weight hybrid
``qwen3_5`` (gated-delta-net + attention layers, vision tower) in fp32, with the real Qwen tokenizer
and the repo's qwen35 chat template.

Outputs are meaningless; identities are exact. Needs the vocab-only GGUF
(``python -m tests.unit.cpu_models``); skipped without it. What cannot run here is in
docs/reports/CPU_VERIFICATION_2026-10-26.md."""

from __future__ import annotations

import json
import threading

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("transformers")
pytest.importorskip("gguf")

from tests.unit.cpu_models import (
    build_dflash2_dir,
    build_qwen35_dir,
    qwen_vocab_gguf,
    session_dir,
)
from tests.unit.cpu_serve import (
    cached_tokens,
    chat,
    install_cpu_bitmask,
    logprob_rows,
    serve,
    sse_events,
    user,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        qwen_vocab_gguf() is None,
        reason="Qwen vocab GGUF absent: run `python -m tests.unit.cpu_models`",
    ),
]

# the APC disk tier is a separate feature with its own e2e tests; keep RAM-only here
WARM = {"YUNSHU_VLM_APC_DISK": 0}
COLD = {"YUNSHU_VLM_APC_DISK": 0, "YUNSHU_VLM_APC_MEMORY_GB": 0}  # prefix cache off


@pytest.fixture(scope="module")
def model_dir():
    return build_qwen35_dir(session_dir("qwen35"))


@pytest.fixture
def cpu_bitmask(monkeypatch):
    install_cpu_bitmask(monkeypatch)


def _tool_turn(i):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": json.dumps({"p": str(i)}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"c{i}",
            "content": f"result {i} " + "lorem ipsum " * 20,
        },
    ]


def agent_conversation(turns: int, note: bool = True) -> list[dict]:
    """System + user, then ``turns`` tool round trips, each followed by a late system note (the
    hook/reminder Claude Code sends after a tool result; kept in place by 5a0a15a7)."""
    msgs = [
        {"role": "system", "content": "You are an agent. " * 8},
        {"role": "user", "content": "Fix the bug in module foo. " * 4},
    ]
    for t in range(turns):
        msgs += _tool_turn(t)
        if note:
            msgs.append({"role": "system", "content": f"hook note {t}"})
    return msgs


def snapshot(resp):
    """Everything a client can compare: text, finish, per-token bytes / logprob / top-k."""
    body = resp.json()["choices"][0]
    rows = [
        (
            e["bytes"],
            e["logprob"],
            [(t["bytes"], t["logprob"]) for t in e["top_logprobs"]],
        )
        for e in logprob_rows(resp)
    ]
    return body["message"]["content"], body["finish_reason"], rows


# fp32 on this CPU stack: a different prefill split (APC checkpoints vs one pass) moves a logprob
# by about one ulp (~1e-6 at |lp| ~ 9). The split-invariant kernels that make it bit-exact are
# Metal-only, so tokens (and their order) are exact and logprobs are held to 1e-5.
LP_ATOL = 1e-5


def assert_same(a, b, msg=""):
    (ta, fa, ra), (tb, fb, rb) = snapshot(a), snapshot(b)
    assert (ta, fa) == (tb, fb), msg
    assert [(x[0], [t[0] for t in x[2]]) for x in ra] == [
        (x[0], [t[0] for t in x[2]]) for x in rb
    ], msg
    for x, y in zip(ra, rb, strict=True):
        assert x[1] == pytest.approx(y[1], abs=LP_ATOL), msg
        for t, u in zip(x[2], y[2], strict=True):
            assert t[1] == pytest.approx(u[1], abs=LP_ATOL), msg


LP = dict(logprobs=True, top_logprobs=3, max_tokens=6)


def run_turns(client, turns=(0, 1, 2), **extra):
    out = []
    for n in turns:
        r = chat(client, agent_conversation(n), **LP, **extra)
        assert r.status_code == 200, r.text
        out.append(r)
    return out


def test_apc_hit_equals_miss_over_an_agent_conversation_with_late_system_notes(
    model_dir,
):
    with serve(model_dir, **WARM) as (client, eng):
        warm = run_turns(client)
        repeat = chat(client, agent_conversation(2), **LP)  # the identical turn again
        # the rendered prompts really are prefix-stable turn to turn (note stays in place)
        ids = [
            eng._tokenizer.encode(
                eng._format_prompt(agent_conversation(n)), add_special_tokens=False
            )
            for n in (0, 1, 2)
        ]
        # the prompt ends with the generation prompt; history replaces it next turn
        gen = len(eng._tokenizer.encode("<|im_start|>assistant\n<think>\n"))
        for a, b in zip(ids, ids[1:], strict=False):
            assert b[: len(a) - gen] == a[: len(a) - gen]
        assert "hook note 1" in eng._format_prompt(agent_conversation(2))
    with serve(model_dir, **COLD) as (client, _):
        cold = run_turns(client)
    assert [cached_tokens(r) for r in cold] == [0, 0, 0]
    assert cached_tokens(warm[0]) == 0
    assert all(cached_tokens(r) > 0 for r in warm[1:]), "later turns reuse the head"
    assert cached_tokens(repeat) > cached_tokens(warm[2]) or cached_tokens(repeat) > 0
    for w, c in zip(warm, cold, strict=True):
        assert_same(w, c, "APC hit != miss")
    assert_same(repeat, warm[2])


def test_stream_equals_non_stream_and_a_cached_repeat(model_dir):
    msgs = agent_conversation(1)
    with serve(model_dir, **WARM) as (client, _):
        plain = chat(client, msgs, **LP)
        streamed = chat(client, msgs, stream=True, **LP)
    assert plain.status_code == streamed.status_code == 200
    events = sse_events(streamed)
    text = "".join(
        e["choices"][0]["delta"].get("content") or "" for e in events if e["choices"]
    )
    lps = [
        t
        for e in events
        if e["choices"] and e["choices"][0].get("logprobs")
        for t in e["choices"][0]["logprobs"]["content"]
    ]
    body = plain.json()["choices"][0]
    assert text == body["message"]["content"] and text
    assert [(t["bytes"], t["logprob"]) for t in lps] == [
        (e["bytes"], e["logprob"]) for e in body["logprobs"]["content"]
    ]


def _argsort_top_logprobs(log_probs, k):
    """What upstream mlx-vlm's GenerationBatch does (sort the whole vocabulary), with the text path's tie order."""
    idx = mx.argsort(-log_probs, axis=-1)[:, :k].astype(
        mx.int32
    )  # stable: ties by ascending id
    return idx, mx.take_along_axis(log_probs, idx, axis=-1)


def test_top_logprobs_device_path_equals_the_upstream_argsort(model_dir, monkeypatch):
    from yunshu_engine import text_logprobs

    assert text_logprobs.device_topk_supported(), "the device path runs on this device"
    msgs = agent_conversation(1)
    with serve(model_dir, **COLD) as (client, _):
        device = chat(client, msgs, logprobs=True, top_logprobs=5, max_tokens=8)
        calls = []

        def spy(lp, k):
            calls.append(k)
            return _argsort_top_logprobs(lp, k)

        monkeypatch.setattr(text_logprobs, "top_logprobs_device", spy)
        upstream = chat(client, msgs, logprobs=True, top_logprobs=5, max_tokens=8)
    assert calls, "the runner asked the (replaced) device path for top-k"
    assert_same(device, upstream)
    assert snapshot(device) == snapshot(upstream), (
        "same config, same prefill: bit-exact"
    )
    assert len(logprob_rows(device)[0]["top_logprobs"]) == 5


def test_batched_row_equals_the_row_alone_and_a_dead_end_fails_only_its_row(
    model_dir, cpu_bitmask, monkeypatch
):
    import importlib

    ar = importlib.import_module("mlx_vlm.generate.ar")
    pa, pb = agent_conversation(1), user("a completely different prompt " * 6)
    # llguidance refuses every unsatisfiable schema when compiling it, so a real runtime dead end
    # cannot be built from a schema; the guide of the one constrained request is made to dead-end
    # on its 3rd mask, which is where a grammar with no valid next token would raise.
    bad = {
        "type": "json_schema",
        "json_schema": {
            "name": "o",
            "schema": {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
            },
        },
    }
    from yunshu_engine import constrained_spec as cs

    masks = []
    real_mask = cs.ConstraintGuide.mask

    def mask(self):
        masks.append(1)
        if len(masks) == 3:
            raise cs.GrammarDeadEndError("Grammar constraint has no valid next token")
        return real_mask(self)

    monkeypatch.setattr(cs.ConstraintGuide, "mask", mask)
    impossible = {
        "type": "json_schema",
        "json_schema": {
            "name": "o",
            "schema": {"type": "integer", "minimum": 10, "maximum": 5},
        },
    }
    with serve(model_dir, **COLD) as (client, _):
        widest = [0]
        step = ar.GenerationBatch._step

        def watching(self):
            widest[0] = max(widest[0], len(self.uids))
            return step(self)

        mp = pytest.MonkeyPatch()
        mp.setattr(ar.GenerationBatch, "_step", watching)
        try:
            alone_a = chat(client, pa, **dict(LP, max_tokens=24))
            alone_b = chat(client, pb, **dict(LP, max_tokens=24))
            assert widest[0] == 1
            results: dict[str, object] = {}
            barrier = threading.Barrier(4)

            def post(name, msgs, **kw):
                barrier.wait()
                results[name] = chat(client, msgs, **kw)

            jobs = {
                "a": (pa, dict(LP, max_tokens=24)),
                "b": (pb, dict(LP, max_tokens=24)),
                "dead_end": (pb, dict(max_tokens=24, response_format=bad)),
                "impossible": (pb, dict(max_tokens=24, response_format=impossible)),
            }
            threads = [
                threading.Thread(target=post, args=(n, m), kwargs=kw)
                for n, (m, kw) in jobs.items()
            ]
            [t.start() for t in threads]
            [t.join(120) for t in threads]
            assert widest[0] >= 2, "the requests really shared a decode batch"
            # the batch can only widen by running the next step with more rows
            after = chat(client, pa, **dict(LP, max_tokens=24))
        finally:
            mp.undo()
    assert results["a"].status_code == results["b"].status_code == 200
    assert results["impossible"].status_code == 400  # refused when compiled
    assert len(masks) >= 3
    assert results["dead_end"].status_code != 200  # failed in the runner, alone
    # same prefill split, so even the logprobs are bit-identical (max |d| measured: 0.0)
    assert snapshot(results["a"]) == snapshot(alone_a), "batched row != row alone"
    assert snapshot(results["b"]) == snapshot(alone_b)
    assert snapshot(after) == snapshot(alone_a), "the engine is healthy afterwards"


def test_json_schema_output_validates_on_the_vlm_path(model_dir, cpu_bitmask):
    jsonschema = pytest.importorskip("jsonschema")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": ["alpha", "beta"]},
            "n": {"type": "integer"},
            "ok": {"type": "boolean"},
        },
        "required": ["name", "n", "ok"],
        "additionalProperties": False,
    }
    with serve(model_dir, **COLD) as (client, _):
        for seed in (1, 2):
            r = chat(
                client,
                user(f"give me json {seed}"),
                max_tokens=80,
                temperature=1.0,
                seed=seed,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "o", "schema": schema, "strict": True},
                },
            )
            assert r.status_code == 200, r.text
            jsonschema.validate(
                json.loads(r.json()["choices"][0]["message"]["content"]), schema
            )


def test_cache_store_off_writes_no_apc_entry_but_still_reuses_the_earlier_prefix(
    model_dir,
):
    off = {"X-Yunshu-Cache-Store": "off"}
    base, longer = agent_conversation(1), agent_conversation(2)
    with serve(model_dir, **WARM) as (client, eng):
        apc = eng._apc_backend

        def entries():
            return sorted(apc.snapshot()["entry_tokens"])

        first = chat(client, base, **LP)
        stored = entries()
        assert stored, "a normal request stores"
        r3 = chat(client, longer, headers=off, **LP)
        assert entries() == stored, "nothing was stored"
        hit = cached_tokens(r3)
        assert hit > 0, "the lookup still ran"
        r4 = chat(client, longer, headers=off, **LP)
        assert cached_tokens(r4) == hit and entries() == stored
        r5 = chat(client, longer, **LP)  # now it stores
        assert entries() != stored and cached_tokens(r5) == hit
        r6 = chat(client, longer, **LP)
        assert cached_tokens(r6) > hit
    for r in (r4, r5, r6):
        assert_same(r, r3)
    assert first.status_code == 200


# ── speculative lanes: spec on == spec off ──


def _speculative(resp):
    return resp.json()["x_yunshu"]["speculative"]


def _spec_on_off(on_dir, on_env, off_env, prompt=None, max_tokens=40):
    msgs = prompt or user("hello there " * 12)
    out = []
    for env in (on_env, off_env):
        with serve(on_dir, YUNSHU_VLM_APC_DISK=0, **env) as (client, _):
            r = chat(client, msgs, logprobs=True, top_logprobs=2, max_tokens=max_tokens)
            assert r.status_code == 200, r.text
            out.append(r)
    return out


@pytest.mark.parametrize("head", ["random", "repeater"])
def test_mtp_lane_runs_on_cpu_and_spec_on_equals_off(head):
    """Random weights never accept a draft (the reject path); the repeater checkpoint's head
    always predicts right (the accept path with its GDN state replay). Both must give exactly
    the plain-decode tokens AND logprobs."""
    path = build_qwen35_dir(
        session_dir(f"qwen35-mtp-{head}"), mtp=True, repeater=head == "repeater"
    )
    on, off = _spec_on_off(path, {"YUNSHU_MTP": 1}, {"YUNSHU_MTP": 0})
    spec = _speculative(on)
    assert spec["mode"] == "mtp" and spec["drafted"] > 0
    if head == "random":
        assert spec["accepted"] == 0
    else:
        assert spec["acceptance_rate"] > 0.9
    assert on.json()["usage"]["completion_tokens"] == 40
    assert off.json()["x_yunshu"].get("speculative") is None
    assert snapshot(on) == snapshot(off), "spec on != spec off"


def test_dflash_lane_runs_on_cpu_and_spec_on_equals_off(model_dir):
    drafter = build_dflash2_dir(session_dir("dflash2"))
    on, off = _spec_on_off(
        model_dir,
        {"YUNSHU_VLM_DRAFT": str(drafter), "YUNSHU_DRAFT_BITS": 0},
        {"YUNSHU_VLM_DRAFT": "off"},
    )
    spec = _speculative(on)
    assert spec["mode"] == "dflash" and spec["drafted"] > 0
    assert snapshot(on) == snapshot(off), "spec on != spec off"


def test_seam_checkpoint_flag_on_or_off_gives_identical_tokens(model_dir):
    """A 10k-token document asked about twice. On this CPU stack prefill is not split-invariant
    (prefill_alignment 0), which is the seam checkpoint's precondition, so the flag is inert here
    (same entries, same reuse); the runner-level test with a 64-aligned manager is
    test_apc_seam_checkpoint_e2e. What this pins: switching the flag never changes the output."""
    import random

    rnd = random.Random(7)
    doc = " ".join(f"tok{rnd.randrange(100000)}" for _ in range(1700))
    results = {}
    for flag in (0, 1):
        with serve(model_dir, YUNSHU_APC_SEAM_CHECKPOINT=flag, **WARM) as (client, eng):
            runs = [
                chat(
                    client,
                    [
                        {"role": "system", "content": "be brief"},
                        {"role": "user", "content": f"{doc}\n\n{q}"},
                    ],
                    logprobs=True,
                    top_logprobs=1,
                    max_tokens=6,
                )
                for q in ("What is the first word?", "Summarise it in one line.")
            ]
            assert all(r.status_code == 200 for r in runs)
            results[flag] = (
                [snapshot(r) for r in runs],
                [cached_tokens(r) for r in runs],
                sorted(eng._apc_backend.snapshot()["entry_tokens"]),
            )
    assert results[0][0] == results[1][0]
    assert results[0][1][0] == 0 and results[0][1][1] >= 8192, (
        "the follow-up reuses the document"
    )
    assert results[0][1:] == results[1][1:]
