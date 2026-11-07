"""Which request kinds draft on the round driver, and the guard that no new kind silently turns
into a plain row (one token per round) without being listed here with a reason."""

import itertools
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.sample_utils import make_logits_processors  # noqa: E402

from yunshu_engine import constrained_spec as cs  # noqa: E402
from yunshu_engine import tool_call_grammar as tcg  # noqa: E402
from yunshu_engine import vlm_batch_runner as vbr  # noqa: E402
from yunshu_engine.reasoning_gate import ThinkGatedGuide  # noqa: E402
from yunshu_engine.round_driver.driver import RoundDriver  # noqa: E402
from yunshu_engine.text_logit_processors import GeneratedTokenPenalty  # noqa: E402


def _make(cls, **attrs):
    obj = object.__new__(cls)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def _processors():
    mask = vbr.TokenMaskProcessor
    return {
        "none": [],
        "static_ban": [mask(suppress=[1], eos_ids=[2])],
        "ignore_eos": [mask(eos_ids=[2], ignore_eos=True)],
        "logit_bias": [vbr.LogitBiasProcessor({3: 1.0})],
        "ban+bias": [mask(suppress=[1]), vbr.LogitBiasProcessor({3: 1.0})],
        "presence_penalty": [GeneratedTokenPenalty(0.0, 1.5)],
        "penalty+ban": [mask(suppress=[1]), GeneratedTokenPenalty(0.5, 0.5)],
        "penalty+bias": [vbr.LogitBiasProcessor({3: 1.0}), GeneratedTokenPenalty(0, 1)],
        "min_tokens": [mask(eos_ids=[2], min_tokens=5)],
        "top_n_sigma": [mask(top_n_sigma=1.0)],
        "repetition_penalty": make_logits_processors(
            repetition_context_size=20, repetition_penalty=1.1
        ),
        "bias+repetition": [
            vbr.LogitBiasProcessor({3: 1.0}),
            *make_logits_processors(repetition_context_size=20, repetition_penalty=1.1),
        ],
    }


def _guides():
    text = _make(cs.ConstraintGuide)
    in_call = type("InCall", (tcg.ToolCallGuide,), {})
    tool = _make(in_call)
    return {
        "none": None,
        "tool_call": tool,
        "json_schema": text,
        "json_think_gated": _make(ThinkGatedGuide, inner=text),
        "json+tools": _make(cs.CombinedGuide, guides=(text, tool)),
        "unknown_guide": object(),
    }


SAMPLING = ("greedy", "sampled", "xtc")
MODES = ("dflash", "mtp")


def _kinds():
    procs, guides = _processors(), _guides()
    for (pn, p), (gn, g), logprobs, sampling, mode in itertools.product(
        procs.items(), guides.items(), (False, True), SAMPLING, MODES
    ):
        yield SimpleNamespace(
            name=f"{mode}/{sampling}/procs={pn}/guide={gn}/logprobs={int(logprobs)}",
            procs=pn,
            guide=gn,
            logprobs=logprobs,
            sampling=sampling,
            mode=mode,
            _p=p,
            _g=g,
        )


def drafts_on_driver(k) -> bool:
    """The runner's ``allow_draft`` rule and the driver's own ``add`` rule, as built from the
    request: processors as ``iter_tokens`` receives them (reporting adds an Fp32 processor)."""
    processors = list(k._p)
    if k.logprobs:
        processors.append(vbr.Fp32LogitsProcessor())
    dflash = k.mode == "dflash"
    if not vbr.driver_allow_draft(
        True,
        processors,
        k.logprobs,
        k._g,
        dflash_driver=dflash,
        xtc=k.sampling == "xtc",
    ):
        return False
    request_procs = [
        p for p in processors if not (dflash and isinstance(p, vbr.Fp32LogitsProcessor))
    ]
    return bool(
        RoundDriver._processors_draft(None, request_procs, k.logprobs, k._g)
        and (k.sampling == "greedy" or not k.logprobs or dflash)
    )


# kind predicate -> (status, reason). "main-plain": the stock lane does not draft it either (no
# regression). "blocker": main drafts it and the driver does not (a step-1 regression).
PLAIN = [
    (
        "xtc sampling",
        lambda k: k.sampling == "xtc",
        "main-plain",
        "XTC keeps the stateful sampler; the position-keyed draw cannot reproduce it",
    ),
    (
        "unknown guide class",
        lambda k: k.guide == "unknown_guide",
        "main-plain",
        "only the grammar guides have exact per-row masks",
    ),
    (
        "mtp: logprobs, guides",
        lambda k: k.mode == "mtp" and (k.logprobs or k.guide != "none"),
        "blocker",
        "the driver's MTP head rows have no guided / reporting verify; main's MTP lane has",
    ),
    (
        "dflash: logprobs with static processors",
        lambda k: (
            k.mode == "dflash"
            and k.logprobs
            and k.guide == "none"
            and k.procs != "none"
        ),
        "blocker",
        "reporting rows keep fp32 readout and only draft alone (main: chain under exact_verify)",
    ),
    (
        "guide with logprobs",
        lambda k: k.guide != "none" and k.logprobs,
        "blocker",
        "main drafts it on the exact chain (exact_verify); the driver has no reporting guide row",
    ),
    (
        "guide with logprobs and a processor",
        lambda k: k.logprobs and k.guide != "none" and k.procs != "none",
        "blocker",
        "windowed penalty + fp32 reporting: main chain-verifies it",
    ),
    (
        "sampled row with logprobs on mtp",
        lambda k: k.mode == "mtp" and k.sampling != "greedy" and k.logprobs,
        "blocker",
        "needs the native keyed report (Qwen4 only on main)",
    ),
]


def test_every_plain_row_is_listed_with_a_reason():
    unlisted = [
        k.name
        for k in _kinds()
        if not drafts_on_driver(k) and not any(pred(k) for _, pred, _, _ in PLAIN)
    ]
    assert not unlisted, (
        "request kinds that decode as plain rows on the round driver without an entry in "
        f"PLAIN (add a reason, or make them draft): {unlisted[:12]} ... ({len(unlisted)})"
    )


def test_no_allowlist_entry_is_stale():
    stale = [
        name
        for name, pred, _, _ in PLAIN
        if not any(pred(k) and not drafts_on_driver(k) for k in _kinds())
    ]
    assert not stale, f"allowlist entries no kind needs any more: {stale}"


def test_every_processor_class_is_a_known_kind():
    import inspect

    from yunshu_engine import text_logit_processors as tlp

    known = {
        "TokenMaskProcessor",
        "LogitBiasProcessor",
        "GeneratedTokenPenalty",
        "Fp32LogitsProcessor",
        "FrequencyPresencePenalty",
        "ConstraintProcessor",
        "ToolCallProcessor",
        "TokenStreamToolCallProcessor",
    }
    classes = {
        n
        for mod in (vbr, tlp)
        for n, c in vars(mod).items()
        if inspect.isclass(c)
        and c.__module__ == mod.__name__
        and (hasattr(c, "process_last_token") or n.endswith("Processor"))
    }
    assert classes <= known | {"SpecRequest"}, (
        "a new logits-processor class: add it to _processors() / PLAIN so its plain-row status "
        f"is decided: {sorted(classes - known)}"
    )


@pytest.mark.parametrize(
    "procs,guide",
    [
        ("none", "json_schema"),
        ("none", "tool_call"),
        ("none", "json+tools"),
        ("static_ban", "json_schema"),
        ("static_ban", "tool_call"),
        ("logit_bias", "json_schema"),
        ("logit_bias", "none"),
        ("ban+bias", "none"),
        ("presence_penalty", "none"),
        ("penalty+ban", "none"),
        ("penalty+bias", "none"),
        ("repetition_penalty", "none"),
        ("min_tokens", "none"),
        ("top_n_sigma", "none"),
        ("bias+repetition", "none"),
        ("repetition_penalty", "json_schema"),
        ("min_tokens", "tool_call"),
        ("top_n_sigma", "json+tools"),
        ("presence_penalty", "json_schema"),
        ("penalty+ban", "tool_call"),
    ],
)
def test_these_kinds_draft_on_dflash(procs, guide):
    for sampling in ("greedy", "sampled"):
        k = next(
            k
            for k in _kinds()
            if (k.procs, k.guide, k.logprobs, k.sampling, k.mode)
            == (procs, guide, False, sampling, "dflash")
        )
        assert drafts_on_driver(k), k.name


def test_logprobs_alone_drafts_on_dflash_only():
    for mode, expected in (("dflash", True), ("mtp", False)):
        k = next(
            k
            for k in _kinds()
            if (k.procs, k.guide, k.logprobs, k.sampling, k.mode)
            == ("none", "none", True, "greedy", mode)
        )
        assert drafts_on_driver(k) is expected


def test_guided_request_with_a_cache_plan_keeps_the_stock_lane():
    lane = vbr.guided_stock_lane
    # no cache plan: the driver takes it, the stock lane does not
    assert not lane(True, True, True, True, None)
    # an explicit cache plan runs the stock runner, whose lane takes a lone guided request
    assert lane(True, True, True, True, {"points": []})
    assert not lane(True, False, True, True, {"points": []})  # not alone
    assert not lane(False, True, True, True, {"points": []})  # drafting off
    assert not lane(True, True, False, True, {"points": []})  # lane busy
    # no driver at all (non-Qwen3.5 family): the lane as before
    assert lane(True, True, True, False, None)


def test_logit_bias_is_pure_per_row_for_any_leading_shape():
    import mlx.core as mx

    p = vbr.LogitBiasProcessor({3: 2.0, 999: 5.0, -1: 7.0})
    x = mx.zeros((1, 4, 8))
    out = p.pure_at(0, x)
    assert out.shape == x.shape
    assert out[..., 3].tolist() == [[2.0] * 4]
    assert float(out.sum()) == 8.0  # out-of-range ids ignored
    two_d = p(None, mx.zeros((4, 8)))
    assert two_d[:, 3].tolist() == [2.0] * 4


def test_xtc_row_is_not_reported_as_drafting():
    for guide in (None, _make(cs.ConstraintGuide)):
        assert not vbr.driver_allow_draft(
            True, [], False, guide, dflash_driver=True, xtc=True
        )
    assert vbr.driver_allow_draft(True, [], False, None, dflash_driver=True, xtc=False)


def test_mtp_drafts_windowable_processors_but_not_guides_or_logprobs():
    procs = _processors()
    for name in ("repetition_penalty", "min_tokens", "top_n_sigma", "logit_bias"):
        assert vbr.driver_allow_draft(
            True, procs[name], False, None, dflash_driver=False
        )
        assert not vbr.driver_allow_draft(
            True, procs[name], False, _make(cs.ConstraintGuide), dflash_driver=False
        )
