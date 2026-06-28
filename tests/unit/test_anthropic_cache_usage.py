"""o + Anthropic cache-usage accounting respects
input+creation+read==prompt, and cache_creation is bounded by the CACHEABLE PREFIX
(tokens up to the last cache_control breakpoint) — not the entire uncached prompt.

the 3rd arg changed from a bool `has_breakpoints` to an int `cacheable_prefix_tokens`
(0 = no breakpoints). Previously a first cache_control request billed the WHOLE uncached
prompt as cache_creation and reported input_tokens=0; now the post-breakpoint content is
correctly plain input_tokens."""
from yunshu_gateway.routers.anthropic import (
    _anthropic_cache_usage,
    _cacheable_prefix_token_count,
    _extract_cache_control_hints,
)


def test_invariant_holds():
    for prompt, cached, pfx in [(3273, 0, 2000), (3273, 3200, 3200), (3273, 3200, 0),
                                (100, 0, 0), (500, 500, 400), (500, 600, 999)]:
        inp, crt, rd = _anthropic_cache_usage(prompt, cached, pfx)
        assert inp >= 0 and crt >= 0 and rd >= 0
        assert inp + crt + rd == prompt


def test_cold_with_breakpoint_only_prefix_is_creation():
    # prefix of 600 tokens cached → creation=600, the remaining 400 is plain input
    # (was: creation=1000, input=0 — the bug).
    inp, crt, rd = _anthropic_cache_usage(1000, 0, 600)
    assert rd == 0 and crt == 600 and inp == 400


def test_no_breakpoints_all_input():
    inp, crt, rd = _anthropic_cache_usage(1000, 0, 0)
    assert rd == 0 and crt == 0 and inp == 1000


def test_warm_is_read_plus_new_input():
    inp, crt, rd = _anthropic_cache_usage(1000, 900, 0)
    assert rd == 900 and crt == 0 and inp == 100


def test_warm_prefix_already_read_no_new_creation():
    # 900 read, prefix is 900 → nothing new to write; 100 remainder is input
    inp, crt, rd = _anthropic_cache_usage(1000, 900, 900)
    assert rd == 900 and crt == 0 and inp == 100


def test_creation_capped_at_uncached():
    # prefix claims more than the uncached remainder → cap at uncached
    inp, crt, rd = _anthropic_cache_usage(500, 0, 999)
    assert rd == 0 and crt == 500 and inp == 0


def test_cached_clamped_to_prompt():
    inp, crt, rd = _anthropic_cache_usage(500, 600, 999)
    assert rd == 500 and crt == 0 and inp == 0


def test_cacheable_prefix_token_count():
    class _Tok:
        def encode(self, t, add_special_tokens=True):
            return list(range(len(t.split())))
    system = [
        {"type": "text", "text": "alpha beta gamma", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "delta epsilon"},
    ]
    _, offsets = _extract_cache_control_hints(system)
    # prefix = "alpha beta gamma" (up to the breakpoint) = 3 word-tokens
    assert _cacheable_prefix_token_count(system, offsets, _Tok()) == 3
    # no offsets / no tokenizer → 0
    assert _cacheable_prefix_token_count(system, [], _Tok()) == 0
    assert _cacheable_prefix_token_count(system, offsets, None) == 0
    assert _cacheable_prefix_token_count("plain string system", offsets, _Tok()) == 0
