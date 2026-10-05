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
    assert m.judge(t, t, 500, False)  # identical but too short to mean anything
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


def test_gate_runs_agent_sessions():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from verify import gate

    assert "agent-sessions" in gate.GATE_STAGES
    assert "agent-sessions" in gate.DEFAULT_STAGES
    sh = (ROOT / "scripts/release/gate.sh").read_text()
    assert "has agent-sessions" in sh
    for script in (
        "covaudit_session.py",
        "covaudit_conc.py",
        "covaudit_stock.py",
    ):
        assert script in sh and (ROOT / "scripts/research" / script).exists()


def test_wire_renderings_and_parsers():
    w = _load("covaudit_wire")
    b1 = cs.first_body(2, "m")
    reply = cs.parse_sse(SSE.splitlines())
    b2 = cs.next_body(b1, reply, 1, 50)
    chat = w.to_chat(b2)
    roles = [m["role"] for m in chat["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert chat["messages"][2]["tool_calls"][0]["id"] == "t1"
    assert chat["messages"][3]["tool_call_id"] == "t1"
    rsp = w.to_responses(b2)
    assert [i["type"] for i in rsp["input"]] == [
        "message",
        "function_call",
        "function_call_output",
    ]
    assert rsp["input"][2]["call_id"] == "t1"

    chat_sse = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"Read","arguments":"{\\"file_path\\": "}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"src/pkg/mod_1.py\\"}"}}]},"finish_reason":"tool_calls"}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":90}}}',
        "data: [DONE]",
    ]
    r = w.parse_chat_sse([x.encode() for x in chat_sse])
    assert r["ended"] and r["stop_reason"] == "tool_use"
    assert r["tool_calls"][0]["input"] == {"file_path": "src/pkg/mod_1.py"}
    assert (
        r["usage"]["cache_read_input_tokens"] == 90 and r["usage"]["input_tokens"] == 10
    )

    resp_sse = [
        'data: {"type":"response.output_item.added","item":{"type":"function_call","id":"i1","call_id":"c9","name":"Read","arguments":""}}',
        'data: {"type":"response.function_call_arguments.delta","item_id":"i1","delta":"{\\"file_path\\": \\"src/pkg/mod_2.py\\"}"}',
        'data: {"type":"response.completed","response":{"output":[{"type":"function_call"}],"usage":{"input_tokens":50,"output_tokens":4,"input_tokens_details":{"cached_tokens":40}}}}',
    ]
    r = w.parse_responses_sse(resp_sse)
    assert r["ended"] and r["stop_reason"] == "tool_use"
    assert r["tool_calls"][0]["id"] == "c9" and r["tool_calls"][0]["input"][
        "file_path"
    ].endswith("mod_2.py")
    assert r["usage"]["cache_read_input_tokens"] == 40
    # an incomplete chat stream (no [DONE] / finish) is not "ended"
    assert not w.parse_chat_sse(chat_sse[:1])["ended"]


def test_hol_judge_and_stream_fold():
    h = _load("covaudit_hol")
    ok_call = [{"name": "get_weather", "args": '{"city": "Taipei", "days": 3}'}]
    short = {"ttft_s": 1.0, "total_s": 2.0, "text": "", "calls": ok_call, "done": True}
    rep = {
        "doc": 40,
        "solo": short,
        "short": short,
        "long": {
            "text": cs.needle(40),
            "done": True,
            "ttft_s": 1,
            "total_s": 5,
            "calls": [],
        },
    }
    assert h.judge({"a": [rep]}) == []
    bad = dict(rep, short=dict(short, calls=[], ttft_s=None))
    assert len(h.judge({"a": [bad]})) == 2
    assert h.judge({"a": [dict(rep, long=dict(rep["long"], text="x"))]})
    assert h.judge({"a": []}) == ["a: no reps"]


def test_server_failure_detection_and_timeout(tmp_path):
    line = "2026 ERROR yunshu_gateway.main: FATAL: model 'x' load failed: [Errno 2] ple-store.json"
    assert "load failed" in cs.load_failure("INFO ok\n" + line + "\nINFO next")
    assert cs.load_failure("INFO ok\nGET /v1/models 200") is None
    assert cs.load_timeout(str(tmp_path / "missing")) == 90
    (tmp_path / "a.safetensors").write_bytes(b"x" * 1024)
    assert cs.load_timeout(str(tmp_path)) > 90


def test_arms_expect():
    m = _load("covaudit_arms")
    want = m.parse_expect("base=FAIL,fix=PASS")
    assert m.check_expect({"base": "FAIL", "fix": "PASS"}, want) == (True, [])
    ok, why = m.check_expect({"base": "FAIL", "fix": "FAIL"}, want)
    assert not ok and "fix: got FAIL" in why[0]
    assert not m.check_expect({"base": "ERROR(2)", "fix": "PASS"}, want)[0]
    assert not m.check_expect({"base": "FAIL"}, want)[0]
    assert not m.check_expect({"base": "FAIL", "fix": "PASS", "x": "PASS"}, want)[0]
