"""Token-for-token parity of our record encoding with the reference shipped in a Clef checkpoint.

Runs only where a checkpoint (or its small metadata files) is on disk: YUNSHU_CLEF_DIR, else the
local models directory. The reference module is loaded from the checkpoint itself (it only needs
torch), never copied into the repo."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from yunshu_engine import decision_engine as de
from yunshu_engine.decision_engine import Option, Question

CANDIDATES = [
    os.environ.get("YUNSHU_CLEF_DIR", ""),
    "/Volumes/P5Plus/models/Clef-MLX/Clef-MLX-4bit",
    "/Volumes/P5Plus/models/_clef_meta/Clef-MLX-4bit",
]


def _dir():
    for c in CANDIDATES:
        p = Path(c) if c else None
        if (
            p
            and (p / "joint_schema_model.py").exists()
            and (p / "tokenizer.json").exists()
        ):
            return p
    pytest.skip("no Clef checkpoint metadata on this machine")


def test_encoding_matches_the_reference_token_for_token():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    d = _dir()
    spec = importlib.util.spec_from_file_location(
        "clef_reference", d / "joint_schema_model.py"
    )
    ref = importlib.util.module_from_spec(spec)
    sys.modules["clef_reference"] = ref  # dataclasses look the module up
    spec.loader.exec_module(ref)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(d))

    def tokenize(text):
        return list(tokenizer(text, add_special_tokens=False).input_ids)

    state = "Order 4412 arrived broken. I asked twice for a refund. 請問什麼時候退款？"
    questions = [
        Question(
            "predicate",
            "angry",
            "Is the customer angry?",
            (Option("true", True), Option("false", False)),
        ),
        Question(
            "choice",
            "team",
            "Route the message.",
            (
                Option("support", "support", "broken products"),
                Option("billing", "billing", "refunds"),
                Option("sales", "sales", None),
            ),
        ),
        Question(
            "score",
            "sat",
            "Satisfaction?",
            (
                Option("0", "bad", "terrible"),
                Option("1", "ok", "fine"),
                Option("2", "good", "great"),
            ),
        ),
    ]
    record = {
        "state": state,
        "questions": {
            "angry": {"type": "noul", "instructions": "Is the customer angry?"},
            "team": {
                "type": "choice",
                "instructions": "Route the message.",
                "criteria": {
                    "support": "broken products",
                    "billing": "refunds",
                    "sales": None,
                },
            },
            "sat": {
                "type": "score",
                "instructions": "Satisfaction?",
                "criteria": ["terrible", "fine", "great"],
            },
        },
    }
    want = ref.encode_record(tokenizer, record)
    got = de.encode_record(tokenize, state, questions)
    assert list(got.input_ids) == list(want.input_ids)
    assert len(got.questions) == len(want.questions)
    for g, w in zip(got.questions, want.questions, strict=True):
        assert g.question_type == w.question_type
        assert g.question_span == w.question_span
        assert g.option_spans == w.option_spans
        assert g.option_ids == w.option_ids
