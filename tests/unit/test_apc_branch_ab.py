"""CPU checks of the APC branch A/B harness (no server, no GPU)."""

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load():
    path = ROOT / "scripts" / "research" / "apc_branch_ab.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("apc_branch_ab_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_branch_shares_exactly_the_kept_turns():
    m = _load()
    h = [{"role": "system", "content": m.SYSTEM}]
    for t, u in enumerate(m.build_turns(1, 4, 50)):
        h += [{"role": "user", "content": u}, {"role": "assistant", "content": f"a{t}"}]
    b = m.branch_messages(h, 2, 1)
    assert b[:5] == h[:5] and b[5] != h[5] and len(b) == 6


def test_subagents_share_a_prefix_in_user_or_system():
    m = _load()
    for where in ("user", "system"):
        a, b = m.subagent_requests(3, 100, where)
        assert a != b
        text = (
            [a[-1]["content"], b[-1]["content"]]
            if where == "user"
            else [a[0]["content"]] * 2
        )
        assert text[0][:200] == text[1][:200]


def test_scenarios_record_every_step_in_order():
    m = _load()
    rows = []

    def fake(messages, max_tokens):
        n = sum(len(x["content"]) for x in messages)
        return (
            "ok",
            {"prompt_tokens": n, "prompt_tokens_details": {"cached_tokens": 5}},
            0.1,
        )

    m.scenarios(
        fake, lambda s, u, secs, ideal: rows.append((s, u, ideal)), 1, 4, 50, 100
    )
    steps = [r[0] for r in rows]
    assert steps[:4] == [f"build{i}" for i in range(1, 5)]
    assert steps[4:] == [
        "a-linear",
        "b-branch-mid",
        "c-user-first",
        "c-user-second",
        "c-system-first",
        "c-system-second",
    ]
    assert m.cached_of(rows[0][1]) == 5
