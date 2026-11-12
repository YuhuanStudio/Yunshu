"""Orphan replay buffer: compact logprobs per buffered output, replay output unchanged."""

import asyncio
import threading

from yunshu_engine import orphan_adoption
from yunshu_engine.request import RequestOutput


def _lp(token_id, k=20):
    return {
        "token_id": token_id,
        "logprob": -0.125 * (token_id % 7) - 1e-9,
        "top_logprobs": [
            {"token_id": 1000 + j, "logprob": -float(j) / 3.0} for j in range(k)
        ],
    }


def _outputs(n, k=20):
    outs = []
    for i in range(n):
        outs.append(
            RequestOutput(
                request_id="r",
                new_text=f"t{i} ",
                new_token_ids=[i],
                completion_tokens=i + 1,
                logprobs=[_lp(i, k)],
                finished=i == n - 1,
                finish_reason="stop" if i == n - 1 else None,
            )
        )
    return outs


async def _agen(outs):
    for o in outs:
        yield o


def _pump_into_buffer(outs):
    gen = orphan_adoption.Generation("k", threading.Event())
    gen.attached = False  # nobody is reading: the first-token gate returns at once
    asyncio.run(orphan_adoption._pump(gen, _agen(outs)))
    return gen


def test_buffered_outputs_do_not_hold_logprob_dicts():
    gen = _pump_into_buffer(_outputs(40))
    assert len(gen.buffer) == 40
    for out in gen.buffer:
        assert not any(isinstance(entry, dict) for entry in out.logprobs)


def test_pack_unpack_round_trip_is_exact():
    cases = [
        _lp(5),
        _lp(7, k=0),
        {"token_id": 3, "logprob": float("-inf"), "top_logprobs": []},
        {
            "token_id": 9,
            "logprob": -2.5,
            "top_logprobs": [{"token_id": 1, "logprob": float("-inf")}],
        },
    ]
    for lp in cases:
        out = RequestOutput(request_id="r", logprobs=[lp], new_token_ids=[1])
        packed = orphan_adoption.pack_output(out)
        assert orphan_adoption.unpack_output(packed) == out
        assert orphan_adoption.unpack_output(packed).logprobs == [lp]


def test_packed_output_is_under_a_kilobyte_per_token():
    import sys

    def size(o, seen):
        stack, total = [o], 0
        while stack:
            x = stack.pop()
            if id(x) in seen:
                continue
            seen.add(id(x))
            total += sys.getsizeof(x)
            if isinstance(x, dict):
                stack.extend(x.keys())
                stack.extend(x.values())
            elif isinstance(x, (list, tuple)):
                stack.extend(x)
            elif hasattr(x, "__dict__"):
                stack.extend(vars(x).values())
            elif hasattr(x, "__slots__"):
                stack.extend(getattr(x, s) for s in x.__slots__ if hasattr(x, s))
        return total

    seen: set = set()
    total = sum(size(orphan_adoption.pack_output(o), seen) for o in _outputs(200))
    assert total / 200 < 1024


def test_replay_yields_the_same_outputs_as_the_live_stream():
    outs = _outputs(30)

    async def collect():
        reg = orphan_adoption.OrphanRegistry()
        got = []
        async for out in orphan_adoption.run(
            None,
            lambda cancel: _agen(outs),
            threading.Event(),
            0.0,
            10_000,
            registry=reg,
        ):
            got.append(out)
        return got

    got = asyncio.run(collect())
    assert got == outs


def _prompt_entries(n, k=20):
    entries = [
        {"token_id": 0, "logprob": 0.0}
    ]  # first prompt position has no prediction
    for i in range(1, n):
        entry = {"token_id": 100 + i, "logprob": -0.5 * (i % 11) - 1e-7}
        if k > 0:
            entry["top_logprobs"] = [
                {"token_id": 2000 + j, "logprob": -float(j) / 7.0} for j in range(k)
            ]
        entries.append(entry)
    return entries


def _first_output_with_prompt(n, k=20):
    return RequestOutput(
        request_id="r",
        new_token_ids=[1],
        new_text="a",
        prompt_logprobs=_prompt_entries(n, k),
        completion_tokens=1,
    )


def test_buffered_first_output_holds_no_prompt_logprob_dicts():
    first = _first_output_with_prompt(300)
    gen = _pump_into_buffer([first, *_outputs(3)[1:]])
    held = gen.buffer[0].prompt_logprobs
    assert not any(isinstance(entry, dict) for entry in held)


def test_prompt_logprobs_round_trip_exactly_with_and_without_top_k():
    for k in (0, 20):
        out = _first_output_with_prompt(50, k=k)
        packed = orphan_adoption.pack_output(out)
        assert orphan_adoption.unpack_output(packed) == out
