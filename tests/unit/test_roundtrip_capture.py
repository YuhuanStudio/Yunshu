from yunshu_engine.roundtrip_capture import RoundTripTracker


def test_whole_and_partial_and_unrelated():
    t = RoundTripTracker()
    assert t.observe_prompt([1, 2, 3]) is None
    t.add([1, 2, 3], [10, 11, 12, 13])
    # next turn re-renders the generation exactly, then continues
    r = t.observe_prompt([1, 2, 3, 10, 11, 12, 13, 99])
    assert (
        r["matched_gen"] == 4 and r["whole_gen_reproduced"] and r["prev_gen_len"] == 4
    )
    # template changes token 3 of the generation (e.g. stripped whitespace)
    r = t.observe_prompt([1, 2, 3, 10, 11, 77, 78, 99])
    assert r["matched_gen"] == 2 and not r["whole_gen_reproduced"]
    assert (
        r["diverge_at_gen"] == 2
        and r["prev_ids"][:1] == [12]
        and r["new_ids"][:1] == [77]
    )
    # a different conversation matches nothing
    assert t.observe_prompt([5, 6, 7, 8]) is None
    # prompt that does not even keep the earlier prompt (compaction) matches nothing
    assert t.observe_prompt([1, 2, 9, 10]) is None


def test_report_summarizes(tmp_path):
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location(
        "rtr",
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "scripts/research/agentic/roundtrip_report.py",
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    f = tmp_path / "c.jsonl"
    f.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"round_trip": None},
                {
                    "round_trip": {
                        "matched_gen": 3,
                        "prev_gen_len": 4,
                        "whole_gen_reproduced": False,
                    }
                },
                {
                    "round_trip": {
                        "matched_gen": 4,
                        "prev_gen_len": 4,
                        "whole_gen_reproduced": True,
                    }
                },
            ]
        )
    )
    s = m.summarize([str(f)])
    assert (s["pairs"], s["whole"], s["matched_gen"], s["prev_gen"]) == (2, 1, 7, 8)
