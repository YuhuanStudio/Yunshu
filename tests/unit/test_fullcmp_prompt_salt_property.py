"""Property test over the REAL manifest and tfbench's prompt builders: within one cell (= one tfbench
invocation = one server process) no two requests share a prompt unless the design says so.

Designed sharing only: conc-same requests of ONE trial (one shared prompt), warm/turn2/idle5/restart repeats of
a cell's own prompt (cache hits are the point of those phases). Everything else must diverge from token 0 of
the content, so no request can hit another request's prefix. The prompt variants are simulated the way the
real smoke variants are made: the same body with a late difference (the case that leaked twice).
CPU only: no model, no server; the tokenizer is replaced by a fixed text stand-in.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))
import fullcmp_manifest as fm  # noqa: E402

HEAD = 40  # first characters of the content that must already diverge (the salt is inside them)


@pytest.fixture(scope="module")
def tb():
    spec = importlib.util.spec_from_file_location("tfbench_prop", ROOT / "scripts" / "research" / "tfbench.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tfbench_prop"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def stand_in(tb, monkeypatch):
    import snapshot_prompts as sp

    # a body shared by every variant, differing only in its last 40 characters (as the smoke variants do)
    def body(name, variant=0):
        return (f"{name} corpus body " * 3000) + f"variant {variant} tail " * 2

    monkeypatch.setattr(tb, "load_variant_prompt", body)
    monkeypatch.setattr(tb, "decode_prompt", lambda kind, ctx, ask, variant=0: body(f"{kind}-{ctx}", variant))
    monkeypatch.setattr(sp, "exact_chat_prompt", lambda src, target, suffix="": f"{src}|{target}")
    return body


def _cell_requests(tb, row):
    """(key, text, designed_share_group) for every request one cell sends, in one process."""
    out = []
    ctx = int(row.get("ctx_tokens") or 1024)
    variant = int(row.get("prompt_variant") or 0)
    turns = row.get("turns")
    if turns == "mixed":
        n = int(row["conc"])
        for trial in range(2):  # the real cell runs conc_trials (2)
            out.append((("mixed-long", trial), tb.mixed_long_text(trial, ctx, variant), None))
            for i in range(1, n):
                # the short requests of the mixed cell (tfbench part_mixed: short_ctx 1024)
                text = tb.mixed_short_text(i, trial, 1024, variant)
                out.append((("mixed-short", trial, i), text, None))
    elif row.get("sweep", "").startswith("conc") and int(row.get("conc", 1)) > 1:
        n = int(row["conc"])
        same = row["sweep"] == "conc-same"
        for trial in range(2):
            for i in range(n):
                text = tb.concurrent_prompt(0 if same else i, trial, ctx, variant, run=f"{n}v{variant}")
                group = ("same", trial) if same else None
                out.append((("conc", trial, i), text, group))
    else:
        # core / out / idle5 / restart / kind cells: the cell's own prompt, repeated by the phases
        text = tb.decode_prompt(row.get("prompt_kind", "code"), ctx, False, variant)
        for phase in ("cold", "warm", "turn2"):
            out.append((("single", phase), text, ("phase-repeat",)))
    return out


def _rows():
    rows = [r for r in fm.build_rows()[0] if r["status"] == "unverified"]
    rows += [dict(r, prompt_variant=v) for r in fm.build_smoke_rows() for v in (0, 1)]
    return rows


def _key_text(text):
    return text[:HEAD]


def test_no_two_requests_in_one_cell_share_a_prompt_head(stand_in, tb):
    leaks = []
    for row in _rows():
        reqs = _cell_requests(tb, row)
        seen = {}
        for key, text, group in reqs:
            head = _key_text(text)
            if head in seen:
                prev_key, prev_group = seen[head]
                designed = (group is not None and group == prev_group) or (
                    group == ("phase-repeat",) and prev_group == ("phase-repeat",)
                )
                if not designed:
                    leaks.append((row["id"], prev_key, key))
            else:
                seen[head] = (key, group)
    assert not leaks, f"{len(leaks)} prompt-head collisions, first: {leaks[:5]}"


def test_conc_same_requests_of_different_trials_diverge_from_token_zero(stand_in, tb):
    for row in _rows():
        if row.get("sweep") != "conc-same":
            continue
        heads = {}
        for key, text, _group in _cell_requests(tb, row):
            if key[0] == "conc" and key[2] == 0:  # request 0 of each trial primes the shared prompt
                heads[key[1]] = _key_text(text)
        assert len(set(heads.values())) == len(heads), row["id"]


def test_the_stand_in_reproduces_a_variant_collision_when_the_salt_omits_the_variant(stand_in, tb, monkeypatch):
    # the detector must see the leak it was written for: the run name without the variant collides
    a = tb.concurrent_prompt(0, 0, 8192, 0, run="4")
    b = tb.concurrent_prompt(0, 0, 8192, 1, run="4")
    assert _key_text(a) == _key_text(b)
