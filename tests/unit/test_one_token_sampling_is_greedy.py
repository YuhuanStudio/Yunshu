# ruff: noqa: F811
"""A request whose filters keep one token (top_k 1, min_p 1) decodes greedily on every path (splash 8d5e05b)."""

import pytest

from tests.unit.test_vlm_runner_batching import FakeGen, runner  # noqa: F401,F811
from yunshu_engine.sampling_defaults import effective_temperature, keeps_one_token

from .tiny_text_engine import make_engine


@pytest.mark.parametrize(
    "top_k, min_p, xtc, expected",
    [
        (1, 0.0, 0.0, True),
        (0, 1.0, 0.0, True),
        (0, 0.99, 0.0, False),
        (2, 0.0, 0.0, False),
        (0, 0.0, 0.0, False),
        (None, None, 0.0, False),
        (1, 0.0, 0.5, False),  # XTC removes top tokens: still a sampled request
    ],
)
def test_keeps_one_token(top_k, min_p, xtc, expected):
    assert keeps_one_token(top_k, min_p, xtc) is expected


def test_effective_temperature():
    assert effective_temperature(0.8, 1, 0.0) == 0.0
    assert effective_temperature(0.8, 0, 1.0) == 0.0
    assert effective_temperature(0.8, 0, 0.0) == 0.8
    assert effective_temperature(0.8, 1, 0.0, 0.2) == 0.8
    assert effective_temperature(0.0, 1, 0.0) == 0.0
    assert effective_temperature(None, 1, 0.0) is None


def _jobs(runner, monkeypatch, **kw):
    seen = []
    submit = runner._submit
    monkeypatch.setattr(runner, "_submit", lambda job: (seen.append(job), submit(job)))
    list(runner.iter_tokens([1, 2, 3], max_tokens=2, prompt_kwargs={}, **kw))
    return seen[0]


def test_vlm_runner_treats_top_k_1_as_greedy(runner, monkeypatch):
    job = _jobs(runner, monkeypatch, temperature=0.8, top_k=1)
    assert job.greedy and job.sampling is None
    job = _jobs(runner, monkeypatch, temperature=0.8, min_p=1.0)
    assert job.greedy and job.sampling is None
    job = _jobs(runner, monkeypatch, temperature=0.8, top_k=40)
    assert not job.greedy and job.sampling is not None


async def test_text_engine_routes_top_k_1_as_greedy():
    eng = make_engine()
    seen = []
    route = eng._spec_route

    def spy(**kw):
        seen.append(kw["temperature"])
        return route(**kw)

    eng._spec_route = spy
    prompt = "the quick brown fox"
    greedy = await eng.generate(prompt=prompt, max_tokens=6, temperature=0.0)
    one = await eng.generate(prompt=prompt, max_tokens=6, temperature=0.9, top_k=1)
    sampled = await eng.generate(prompt=prompt, max_tokens=6, temperature=0.9, top_k=40)
    assert seen[:2] == [0.0, 0.0] and seen[2] == 0.9
    assert one.text == greedy.text and sampled.error is None
    streamed = [
        o
        async for o in eng.stream_generate(
            prompt=prompt, max_tokens=6, temperature=0.9, min_p=1.0
        )
    ]
    assert seen[3] == 0.0 and streamed
