"""Authored gate fixtures only; none of these are real-world quality evidence."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/websearch_quality_gate.py"
    )
    spec = importlib.util.spec_from_file_location("websearch_quality_gate", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def authored_set():
    m = module()
    gold, pairs, reviews = [], [], []
    for i in range(200):
        key = f"authored-{i}"
        url = "https://fixture.example/reference"
        text = "The authored fixture value is 42."
        category = ("news", "docs", "code", "adversarial", "docs")[i // 40]
        gold.append(
            {
                "id": key,
                "category": category,
                "gold_reviewer": "fixture-author",
                "reference_answer": "42",
                "known_good_urls": [url, "https://second.example/reference"],
                "reference_publishers": {
                    url: "fixture-publisher-a",
                    "https://second.example/reference": "fixture-publisher-b",
                },
                "extracted_pages": {url: text},
            }
        )
        pairs.append(
            {
                "id": key,
                **{
                    arm: {
                        "text": text,
                        "device": "M5",
                        "citations_detail": [{"url": url, "cited_text": text}],
                    }
                    for arm in ("snippets", "research")
                },
            }
        )
        reviews.append(
            {
                "id": key,
                "snapshot_sha256": "frozen-fixture",
                "human_checker": "fixture-human" if i < 20 else None,
                **{
                    arm: {
                        "answer_sha256": m.digest(text),
                        "correct": True,
                        "judge": "fixture-judge",
                        "rationale": "Authored fixture matches reference.",
                        "injection_followed": False,
                        "claims_complete": True,
                        "claims": [
                            {
                                "answer_span": [0, len(text)],
                                "url": url,
                                "quote": text,
                                "supported": True,
                                "rationale": "Authored reference explicitly states value.",
                            }
                        ],
                    }
                    for arm in ("snippets", "research")
                },
            }
        )
    return gold, pairs, reviews


def test_reviewed_fixture_passes_but_substring_is_not_entailment():
    m = module()
    gold, pairs, reviews = authored_set()
    assert m.judge(gold, pairs, reviews, "frozen-fixture")["pass"]
    reviews[0]["research"]["claims"][0]["supported"] = False
    result = m.judge(gold, pairs, reviews, "frozen-fixture")
    assert not result["pass"] and any(
        "unsupported" in reason for reason in result["reasons"]
    )


@pytest.mark.parametrize(
    "case",
    [
        "duplicate",
        "few",
        "hash",
        "answer",
        "device",
        "human",
        "pending",
        "span",
        "quote",
        "news",
        "injection",
        "net",
        "gold",
        "citation",
    ],
)
def test_gate_fails_closed(case):
    m = module()
    gold, pairs, reviews = authored_set()
    if case == "duplicate":
        pairs[1] = copy.deepcopy(pairs[0])
    elif case == "few":
        gold, pairs, reviews = gold[:199], pairs[:199], reviews[:199]
    elif case == "hash":
        reviews[0]["snapshot_sha256"] = "stale"
    elif case == "answer":
        pairs[0]["research"]["text"] = "another answer"
    elif case == "device":
        pairs[0]["research"]["device"] = "M3"
    elif case == "human":
        reviews[19]["human_checker"] = None
    elif case == "pending":
        reviews[0]["research"]["claims_complete"] = False
    elif case == "span":
        reviews[0]["research"]["claims"][0]["answer_span"] = [-1, 2]
    elif case == "quote":
        reviews[0]["research"]["claims"][0]["quote"] = "not in frozen page"
    elif case == "news":
        gold[0]["reference_publishers"] = {}
    elif case == "injection":
        reviews[0]["research"]["injection_followed"] = True
    elif case == "net":
        for row in reviews[:2]:
            row["research"]["correct"] = False
    elif case == "gold":
        gold[0]["gold_reviewer"] = None
    elif case == "citation":
        pairs[0]["research"]["citations_detail"] = []
    assert not m.judge(gold, pairs, reviews, "frozen-fixture")["pass"]


def test_cli_rejects_incomplete_replay(tmp_path):
    m = module()
    gold, pairs, reviews = authored_set()
    paths = {
        name: tmp_path / (name + ".jsonl")
        for name in ("snapshot", "pairs", "reviews", "out")
    }
    for name, rows in (("snapshot", gold), ("pairs", pairs), ("reviews", reviews)):
        paths[name].write_text("\n".join(json.dumps(row) for row in rows))
    assert (
        m.main(
            [arg for name, path in paths.items() for arg in ("--" + name, str(path))]
        )
        == 1
    )
    result = json.loads(paths["out"].read_text())
    assert any("final complete" in reason for reason in result["reasons"])
