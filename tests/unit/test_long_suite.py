"""The `long` yv suite: decode length fail-closed, needle retrieval, split cells, long stages."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

from test_yv_verify import go, verdict_of, world  # noqa: E402,F401
from verify import analyze, stages, suites  # noqa: E402


@pytest.fixture(scope="module")
def tfb():
    spec = importlib.util.spec_from_file_location(
        "long_tfb", REPO / "scripts/research/tfbench.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_decode_length_is_fail_closed(tfb):
    tfb.check_decode_len({"finish": "length", "ct": 2048}, 2048, "x")
    for bad in (
        {"finish": "length", "ct": 2047},
        {"finish": "stop", "ct": 2048},
        {},
    ):
        with pytest.raises(RuntimeError):
            tfb.check_decode_len(bad, 2048, "x")


def test_decode_tokens_option_default_and_value(tfb):
    base = ["--engine", "yunshu", "--part", "decode", "--out", "o"]
    assert tfb.parse_args(base).decode_tokens == 256
    assert tfb.parse_args(base + ["--decode-tokens", "2048"]).decode_tokens == 2048


def test_part_decode_rejects_short_reply(tfb, monkeypatch, tmp_path):
    a = tfb.parse_args(
        ["--engine", "yunshu", "--part", "decode", "--out", "o", "--only-ctx", "1024"]
        + ["--only-kind", "prose", "--decode-tokens", "64"]
    )
    monkeypatch.setattr(tfb, "load_prompt", lambda n: "p")
    sent = []

    def send(url, body):
        sent.append(body["max_tokens"])
        return {"finish": "length", "ct": 64 if len(sent) < 2 else 63, "_text": "t"}

    monkeypatch.setattr(tfb, "send", send)
    srv = type("S", (), {"url": "u", "model": "m"})()
    with (
        open(tmp_path / "o.jsonl", "w") as out,
        pytest.raises(RuntimeError, match="63"),
    ):
        tfb.part_decode(srv, out, a)
    assert sent == [64, 64]


def test_ngram_repeat_flags_loops(tfb):
    assert tfb.ngram_repeat("a b c d e f g h i j k l") == 0.0
    assert tfb.ngram_repeat("the cat sat down " * 50) > 0.9


def test_needle_items_deterministic_and_unique(tfb):
    a, b = tfb.needle_items(32768), tfb.needle_items(32768)
    assert a == b and len(a) == 10
    assert len({n for n, _ in a}) == 10
    assert tfb.needle_items(65536) != a


def test_needle_haystack_keeps_length_and_every_needle(tfb):
    base = "".join(f"line {i} of filler text\n" for i in range(4000))
    items = tfb.needle_items(1)
    hay = tfb.needle_haystack(base, 1, items)
    assert len(hay) == len(base) - 400
    for nm, code in items:
        assert f"station {nm} is {code}." in hay
    # depths are spread: first needle early, last needle late
    first = hay.index(items[0][1])
    last = hay.index(items[-1][1])
    assert first < len(hay) * 0.2 and last > len(hay) * 0.8


def test_part_needle_scores_answers(tfb, monkeypatch, tmp_path):
    a = tfb.parse_args(
        ["--engine", "yunshu", "--part", "needle", "--out", "o", "--only-ctx", "1024"]
    )
    monkeypatch.setattr(
        tfb, "load_prompt", lambda n: "filler\n" * 3000 + "\n\n---\nask"
    )
    items = tfb.needle_items(1024)
    answers = iter([code for _, code in items[:7]] + ["no idea"] * 3)

    def send(url, body):
        return dict(
            _text=next(answers), ttft_s=1, pt=9, cached=0, finish="length", ct=6
        )

    monkeypatch.setattr(tfb, "send", send)
    srv = type("S", (), {"url": "u", "model": "m"})()
    import json

    with open(tmp_path / "o.jsonl", "w+") as out:
        tfb.part_needle(srv, out, a)
        out.seek(0)
        recs = [json.loads(x) for x in out]
    recs = [r for r in recs if r["part"] == "needle"]
    assert len(recs) == 10 and sum(r["correct"] for r in recs) == 7


def _needle_rows(ctx, bad=0):
    return [
        {"part": "needle", "ctx": ctx, "item": i, "correct": i >= bad}
        for i in range(10)
    ]


def test_needle_compare_plus_minus_one_and_missing():
    ok = analyze.needle_compare(_needle_rows(32768), _needle_rows(32768, bad=1))
    assert ok["ok"] and ok["net"] == -1
    bad = analyze.needle_compare(_needle_rows(32768), _needle_rows(32768, bad=2))
    assert not bad["ok"]
    cut = analyze.needle_compare(_needle_rows(32768), _needle_rows(32768)[:9])
    assert not cut["ok"] and cut["missing"]
    assert not analyze.needle_compare([], [])["ok"]


def test_long_suite_shape():
    cfg = suites.parse_suite("long")
    assert cfg["ctx"] == [32768, 65536, 131072]
    assert cfg["mem_sizes"] == [32768, 131072]
    assert cfg["decode_tokens"] == 2048 and cfg["spec_off"] is True
    for st in ("identity", "apc", "speed", "memory", "longqa", "conc"):
        assert st in cfg["stages"]


@pytest.fixture
def long_world(world):  # noqa: F811
    pr = world.tmp / "prompts"
    for k in ("prose", "code"):
        for c in (32768, 65536, 131072):
            (pr / f"{k}-{c}.txt").write_text("x")
    return world


def long_go(w, **kw):
    kw.setdefault("suite", "long")
    kw.setdefault("reps", 2)
    return go(w, **kw)


def test_long_suite_end_to_end_with_split_cells(long_world, monkeypatch):
    monkeypatch.setattr(stages, "_decode_est_min", lambda *a, **k: 5.0)
    assert long_go(long_world, spec_off=True) == 0
    v, rd = verdict_of(long_world)
    assert [s["status"] for s in v["stages"]][1:] == ["PASS"] * 6
    jobs = (long_world.tmp / "jobs").glob("*.json")
    labels = " ".join(p.read_text() for p in jobs)
    assert "c131072prose" in labels and "c32768code" in labels
    by = {s["name"]: s for s in v["stages"]}
    assert by["longqa"]["numbers"]["items"] == 30
    assert len(by["conc"]["numbers"]["trials"]["cand"]) == 2


def test_long_suite_keeps_going_and_flags_retrieval_loss(long_world, monkeypatch):
    monkeypatch.setattr(stages, "_decode_est_min", lambda *a, **k: 5.0)
    assert long_go(long_world, cand_env=["FAKE_NEEDLE_BAD=3", "FAKE_SLOW=1"]) == 1
    v, _ = verdict_of(long_world)
    by = {s["name"]: s for s in v["stages"]}
    assert (
        by["speed"]["status"] == "FAIL"
    )  # a failed stage does not hide the later ones
    assert (
        by["longqa"]["status"] == "FAIL" and "retrieval" in by["longqa"]["reasons"][0]
    )
    assert by["conc"]["status"] == "FAIL"


def test_send_prints_progress_and_reads_with_a_timeout(tfb, monkeypatch, capsys):
    import json

    def chunk(**d):
        return b"data: " + json.dumps(d).encode() + b"\n"

    lines = [chunk(choices=[{"delta": {"content": "w "}}]) for _ in range(600)]
    lines += [
        chunk(choices=[{"delta": {}, "finish_reason": "length"}]),
        chunk(usage={"completion_tokens": 600, "prompt_tokens": 5}),
        b"data: [DONE]\n",
    ]
    seen = {}

    class Resp:
        def __enter__(self):
            return iter(lines)

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["timeout"] = timeout
        return Resp()

    monkeypatch.setattr(tfb.urllib.request, "urlopen", fake_urlopen)
    r = tfb.send("http://x", {"messages": []})
    assert r["ct"] == 600 and r["finish"] == "length"
    out = capsys.readouterr().out
    assert "streamed 256 chunks" in out and "streamed 512 chunks" in out
    assert seen["timeout"] <= 600  # a silent stream fails within minutes, with evidence
