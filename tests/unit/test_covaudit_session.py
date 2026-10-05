import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / f"scripts/research/{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cs = _load("covaudit_session")

SSE = """event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":5,"cache_read_input_tokens":90}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"Read"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"file_path\\": \\"src/pkg/mod_1.py\\"}"}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}

event: message_stop
data: {"type":"message_stop"}
"""


def req(k, **kw):
    base = {
        "kind": "req",
        "req": k,
        "ended": True,
        "text": "",
        "tool_calls": [],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100},
    }
    base.update(kw)
    return base


def test_session_extends_exactly():
    r = cs.parse_sse(SSE.splitlines())
    b1 = cs.first_body(2, "m")
    b2 = cs.next_body(b1, r, 1, 100)
    assert b2["messages"][:1] == b1["messages"]
    assert b2["messages"][1]["content"][-1]["id"] == "t1"
    assert b2["messages"][2]["content"][0]["tool_use_id"] == "t1"
    assert cs.needle(1) in b2["messages"][2]["content"][0]["content"]
    # no usable tool call: scripted call keeps the session going
    b2b = cs.next_body(b1, dict(r, tool_calls=[]), 1, 100)
    assert b2b["messages"][1]["content"][-1]["input"]["file_path"] == cs.path_of(1)


def test_parse_sse_bytes_and_usage():
    r = cs.parse_sse(SSE.splitlines())
    assert r["ended"] and r["tool_calls"][0]["input"]["file_path"].endswith("mod_1.py")
    assert cs.total_prompt(r["usage"]) == 95
    assert (
        cs.parse_sse([x.encode() + b"\n" for x in SSE.splitlines()]) == r
    )  # urllib yields bytes


def test_judge_pass_and_failures():
    def call(k):
        return [
            {
                "name": "Read",
                "id": "x",
                "input": {"file_path": cs.path_of(k)},
                "raw": "",
            }
        ]

    final = req(
        3,
        text="CODES: " + " ".join(cs.needle(i) for i in (1, 2)),
        stop_reason="end_turn",
        usage={"input_tokens": 10, "cache_read_input_tokens": 190},
    )
    rows = [
        req(1, tool_calls=call(1), usage={"input_tokens": 200}),
        req(
            2,
            tool_calls=call(2),
            usage={"input_tokens": 20, "cache_read_input_tokens": 190},
        ),
        final,
    ]
    assert cs.judge_rows(rows, 2) == []
    bad = cs.judge_rows(
        [
            rows[0],
            dict(
                rows[1],
                tool_calls=[{"name": "Read", "id": "x", "input": None, "raw": "{"}],
            ),
            final,
        ],
        2,
    )
    assert any("no valid Read" in b for b in bad)
    assert any(
        "needles missing" in b
        for b in cs.judge_rows(rows[:2] + [dict(final, text="CODES: nothing")], 2)
    )
    assert any(
        "cached 50" in b
        for b in cs.judge_rows(
            [
                rows[0],
                dict(
                    rows[1], usage={"input_tokens": 150, "cache_read_input_tokens": 50}
                ),
                final,
            ],
            2,
        )
    )
    assert cs.judge_rows([], 2) == ["no request rows"]
    assert cs.judge_rows(rows[:2] + [dict(final, ended=False)], 2)


def test_compare():
    x = req(1, text="a")
    assert cs.compare_rows([x], [dict(x)]) == []
    assert cs.compare_rows([x], [dict(x, text="b")])
    assert cs.compare_rows([x], [req(2)]) == ["no common requests"]


def test_restart_judge():
    def r(cached, text=None):
        return {
            "ended": True,
            "text": text or cs.needle(1),
            "usage": {"input_tokens": 100 - cached, "cache_read_input_tokens": cached},
        }

    assert cs.judge_restart(r(0), r(99), r(90), True, 1) == []
    assert len(cs.judge_restart(r(0), r(99), r(10), False, 1)) == 2
    assert any(
        "differs" in b
        for b in cs.judge_restart(r(0), r(99, "x" + cs.needle(1)), r(90), True, 1)
    )


def test_conc_judge():
    cc = _load("covaudit_conc")
    reqs = cc.make_requests(50, 50, "m")
    checks = {n: v[1] for n, v in reqs.items()}
    ok = {"text": "391", "tool_calls": [], "finish": "stop"}
    solo = {"short": ok}
    assert cc.judge({"solo": solo, "c2": {"short": dict(ok)}}, checks) == []
    bad = cc.judge({"solo": solo, "c2": {"short": dict(ok, text="391 ")}}, checks)
    assert any("differs from solo" in b for b in bad)
    assert cc.judge({"solo": solo, "c2": {"short": {"error": "x"}}}, checks)
    assert cc.judge({"solo": {}, "c2": {}}, checks) == ["no solo phase"]
    assert checks["json"]({"text": '{"answer": 17}'})


def test_stock_judge():
    m = _load("covaudit_stock")
    t = f"The code is {m.needle(5)} and more words follow here."
    assert m.judge(t, t, 10, False) == []
    assert m.judge(t, t[:-3] + "xyz", 30, False) == []
    assert m.judge(t, "The code is " + m.needle(5) + " other", 60, False)
    assert m.judge("", t, 10, False)
    assert m.judge("nothing", t, 10, False)


def test_orphans_summarize():
    m = _load("covaudit_orphans")
    assert m.summarize(0, "a\nPASS\n").startswith("rc=0 | PASS")
    assert m.summarize(None, "").startswith("TIMEOUT")
    assert all(
        (m.ROOT / "scripts/verify" / f"verify_{n}.py").exists() for n in m.DEFAULT
    )
