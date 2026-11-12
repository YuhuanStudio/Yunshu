"""Text-only models (mlx-lm fast path) print no VLM runner line: their engaged mode comes from
the spec proposer line, so a spec=off identity cell on Qwen2.5 is not a missing-evidence failure."""

from scripts.research.tfbench import engaged_spec_mode

LINE = (
    "2026-11-12 10:05:01,727 INFO yunshu_engine.batched_engine: Spec proposer initialized: "
    "kind=ngram, max_n=5, k=5, mode=lps, greedy_default={}\n"
)


def test_text_model_without_ngram_default_is_off():
    assert engaged_spec_mode("yunshu", LINE.format("False")) == "off"


def test_text_model_with_ngram_default_is_ngram():
    assert engaged_spec_mode("yunshu", LINE.format("True")) == "ngram"


def test_vlm_runner_line_wins_and_no_evidence_stays_none():
    vlm = "INFO VLM batch runner: rows=8 draft=mtp block=2\n"
    assert engaged_spec_mode("yunshu", LINE.format("False") + vlm) == "mtp"
    assert engaged_spec_mode("yunshu", "nothing here\n") is None
