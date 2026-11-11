"""Length labels are served lengths (scripts/research/prompt_len.py); tokenizer-only, no model weights."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.research import conc_probe
from scripts.research import prompt_len as pl

SMALL = Path("/Volumes/P5Plus/models/Qwen3.5-0.8B-MLX-bf16")
needs_tok = pytest.mark.skipif(not SMALL.exists(), reason="small tokenizer not on disk")
RESEARCH = Path(__file__).resolve().parents[2] / "scripts" / "research"


@pytest.fixture(scope="module")
def tok():
    return pl.load_tokenizer(SMALL)


def test_check_served_length_tolerance_and_turns():
    assert pl.default_tol(1024) == 16 and pl.default_tol(131072) == 1311
    assert pl.check_served_length(1024, {"prompt_tokens": 1034}) == 1034
    with pytest.raises(pl.PromptLengthMismatchError):
        pl.check_served_length(1024, {"prompt_tokens": 1100})
    with pytest.raises(pl.PromptLengthMismatchError):
        pl.check_served_length(
            2048, {"prompt_tokens": 28}
        )  # a "2K" bench that ran 28 tokens
    assert pl.check_served_length(1024, {"prompt_tokens": 5000}, turn=2) is None
    for bad in (None, {}, {"prompt_tokens": None}, {"prompt_tokens": True}):
        with pytest.raises(pl.PromptLengthMismatchError):
            pl.check_served_length(1024, bad)
    with pytest.raises(pl.PromptLengthMismatchError):
        pl.check_served_length(1024, {"prompt_tokens": 1030}, tol=2)


@needs_tok
@pytest.mark.parametrize("target", [64, 300, 1024, 4096])
def test_build_text_hits_served_length_exactly(tok, target):
    src = "Alpha beta gamma delta. " * 10
    text = pl.build_text(tok, src, target, suffix="\nSummarise.")
    assert pl.served_tokens(tok, text) == target
    assert text.endswith("\nSummarise.")


@needs_tok
def test_build_text_accounts_for_media_and_system(tok):
    text = pl.build_text(tok, "word " * 50, 500, media_tokens=70, system="Be brief.")
    assert pl.served_tokens(tok, text, 70, "Be brief.") == 500
    assert pl.served_tokens(tok, text, 0, "Be brief.") == 430
    with pytest.raises(ValueError):
        pl.build_text(tok, "word", 5)


@needs_tok
def test_chars_alias_converts_to_tokens(tok):
    n = pl.tokens_from_chars(tok, 16000)
    assert 2500 < n < 5500  # ~4K tokens, the thing the characters were read as


@needs_tok
def test_conc_probe_body_is_labelled_length(tok, monkeypatch):
    monkeypatch.setattr(conc_probe, "filler", lambda n, salt: pl.filler_source(n))
    b = conc_probe.body("m", 3, 16, None, True, ctx_tokens=700, tok=tok)
    assert pl.served_tokens(tok, b["messages"][0]["content"]) == 700


@needs_tok
def test_conc_probe_ctx_chars_is_a_logged_alias(tok, capsys):
    a = SimpleNamespace(ctx_tokens=None, ctx_chars=16000)
    n = conc_probe.ctx_tokens_from_args(a, tok)
    assert n == pl.tokens_from_chars(tok, 16000)
    assert "deprecated" in capsys.readouterr().out
    assert (
        conc_probe.ctx_tokens_from_args(
            SimpleNamespace(ctx_tokens=None, ctx_chars=None), tok
        )
        == 4096
    )
    with pytest.raises(SystemExit):
        conc_probe.ctx_tokens_from_args(SimpleNamespace(ctx_tokens=8, ctx_chars=8), tok)


def test_conc_probe_run_one_label_check_needs_usage():
    # run_one raises on a mismatched served length before recording the row
    assert pl.check_served_length(4096, {"prompt_tokens": 4100}) == 4100


# --- guard: a script that sends chat prompts AND takes a length argument must use prompt_len ---------------------
SENDS = re.compile(r"chat/completions|generate_stream|/v1/messages")
LENGTH_ARG = re.compile(
    r"""["']--[a-z0-9-]*(?:ctx|size|words|chars|pp|prompt-?len|prompt-?tokens|context|lengths?)[a-z0-9-]*["']"""
)
ALLOW = {
    # the flag name contains "prefill-step-size": an engine prefill chunk size, not a prompt length
    "bench_reference_runtime.py": "--prefill-step-size is the engine's prefill chunk size, not a prompt length",
    # --context-file is a path to a prompt file; the served length is checked by validate_long_usage (>= 32768)
    "docs016_console.py": "--context-file is a path to a prompt file, not a length; served prompt_tokens are checked by validate_long_usage",
    # --text-lengths is a variant count; the prompts are fixed sentence-count sweeps compared token-by-token between arms
    "media_identity.py": "--text-lengths is a variant count per sweep step; the prompts are fixed sentence-count sweeps (65..~2000 tokens) compared between arms, not length-labelled cells",
    # the matched string is the footprint CLI's --swapped flag (a process option), not a length argument
    "swap_probe.py": "the matched string is the footprint CLI's --swapped flag (a process option), not a prompt length",
}


def _flagged():
    out = {}
    for p in sorted(RESEARCH.glob("*.py")):
        text = p.read_text()
        if SENDS.search(text) and LENGTH_ARG.search(text):
            out[p.name] = "prompt_len" in text
    return out


def test_length_argument_scripts_use_the_helper_or_are_allowlisted():
    flagged = _flagged()
    missing = sorted(n for n, uses in flagged.items() if not uses and n not in ALLOW)
    assert not missing, (
        f"{missing}: import prompt_len and check_served_length, or allowlist with a reason"
    )
    stale = sorted(n for n in ALLOW if n not in flagged or flagged[n])
    assert not stale, f"{stale}: allowlisted but no longer needs it (remove the entry)"


MIGRATED = (
    "tfbench.py",
    "conc_probe.py",
    "phaseb_conc.py",
    "multimodal_apc.py",
    "anenative_conc.py",
    "apc_peak_timeline.py",
    "apc_restart_probe.py",
    "apc_restore_identity.py",
    "bench_context_batch.py",
    "bench_mixed_load.py",
    "bqsa_identity.py",
    "convfuse_identity.py",
    "flashnext80_engines.py",
    "flashnext80_footprint.py",
    "lone_decode_bench.py",
    "memadmit_refuse.py",
    "memory_ab.py",
    "phase_probe.py",
    "probe_server_path.py",
    "roundgap_run.py",
    "sampledspec_conc.py",
    "sampledspec_promotion.py",
    "servedev_resident.py",
    "tf110_probe.py",
)


# Harnesses whose one request gets no usage back, so check_served_length has nothing to check. They build the
# length with prompt_len (build_text via conc_probe.body) and say so in their output: tokenizer-served, not
# server-verified. Keep this list to the refusal path only; a successful response must be checked.
TOKENIZER_SERVED = {
    "memadmit_refuse.py": "the 507 refusal carries no usage; the length is the tokenizer's served_tokens, recorded as such",
}


def test_migrated_harnesses_use_the_check():
    for name in MIGRATED:
        text = (RESEARCH / name).read_text()
        if name in TOKENIZER_SERVED:
            assert "import prompt_len" in text and "served_tokens(" in text, name
            assert "check_served_length" not in text, (
                f"{name}: now checks the length, drop the exemption"
            )
        else:
            assert "check_served_length" in text, name


class _WordTok:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)

    def apply_chat_template(self, msgs, tokenize=True, **kw):
        return ["<tpl>"] + [w for m in msgs for w in m["content"].split()]


def test_empty_source_and_missing_corpus_raise_instead_of_spinning(
    monkeypatch, tmp_path
):
    """2026-11-11: in the ci-local sandbox the P5Plus corpus read as empty and build_text looped forever."""
    with pytest.raises(ValueError, match="empty source"):
        pl.build_text(_WordTok(), "", 64)
    monkeypatch.setattr(conc_probe, "CORPUS", tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        conc_probe.filler(100, 0)


class _TemplateTok(_WordTok):
    """A word tokenizer whose chat template adds a fixed overhead (Gemma's rendered prompt added 129 tokens)."""

    def __init__(self, overhead):
        self.overhead = overhead

    def apply_chat_template(self, msgs, tokenize=True, **kw):
        return ["<tpl>"] * self.overhead + [
            w for m in msgs for w in m["content"].split()
        ]


ASK = "\n\n---\nWrite a long essay. Keep writing at length."  # 9 words on _WordTok


def test_fit_text_calibrates_to_a_large_template_overhead():
    """2026-11-11 nax-gemma2 speed cell: a Qwen-sized prompt file served 1153 tokens for a 1024 label."""
    tok = _TemplateTok(129)
    body = " ".join(f"w{i}" for i in range(1500))
    assert (
        pl.served_tokens(tok, body + ASK) == 1500 + 9 + 129
    )  # the unfixed file: label missed by 129
    text = pl.fit_text(tok, body, 1024, suffix=ASK)
    assert pl.served_tokens(tok, text) == 1024
    assert text.endswith(ASK)


def test_fit_text_takes_a_length_inside_tol_when_the_exact_one_is_unreachable(
    monkeypatch,
):
    """2026-11-12 textdriver-id3b-b: Qwen2.5's BPE made 8192 exactly unreachable; the cell refused."""
    tok = _TemplateTok(129)
    body = " ".join(f"w{i}" for i in range(1500))
    real = pl.build_text

    def exact_unreachable(tok, source, target, **kw):
        if target == 1024:
            raise ValueError("cannot construct served length 1024")
        return real(tok, source, target, **kw)

    monkeypatch.setattr(pl, "build_text", exact_unreachable)
    text = pl.fit_text(tok, body, 1024, suffix=ASK)
    assert abs(pl.served_tokens(tok, text) - 1024) <= pl.default_tol(1024)
    assert text.endswith(ASK)


def test_fit_text_keeps_text_within_tolerance_unchanged():
    tok = _TemplateTok(0)
    body = " ".join(f"w{i}" for i in range(1015))  # 1015 + 9 ask words = 1024 served
    assert pl.fit_text(tok, body, 1024, suffix=ASK) == body + ASK


def test_decode_prompt_is_calibrated_to_the_served_template(tmp_path, monkeypatch):
    from scripts.research import tfbench

    monkeypatch.setattr(tfbench, "PROMPT_VARIANT_DIR", tmp_path)
    monkeypatch.delenv("TFB_EXACT_PROMPTS", raising=False)
    body = " ".join(f"w{i}" for i in range(1500))
    monkeypatch.setattr(
        tfbench, "load_variant_prompt", lambda name, variant=0: body + ASK
    )
    tok = _TemplateTok(129)
    assert pl.served_tokens(tok, body + ASK) != 1024  # what the old decode path sent
    text = tfbench.decode_prompt("prose", 1024, False, 0, tok=tok)
    assert pl.served_tokens(tok, text) == 1024
    assert text.endswith(ASK)
    assert (
        tfbench.decode_prompt("prose", 1024, False, 0) == body + ASK
    )  # no tokenizer: file as before
