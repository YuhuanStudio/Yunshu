"""omlx 746447e2: the forced thinking close is the generation-side ``\\n</think>\\n\\n``."""

from types import SimpleNamespace

from yunshu_engine.thinking_budget import (
    make_forced_close_criteria,
    resolve_think_close_pattern,
)

ENC = {"\n": [198], "\n\n": [271], "</think>": [901], "<think>": [900]}

QWEN = (
    "{%- if '</think>' in content %}{%- set r = content.split('</think>')[0] %}{%- endif %}"
    "{{- '<think>\\n' + r + '\\n</think>\\n\\n' + content }}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}"
)
STEP = "{{ content.split('</think>')[-1] }}{{ '<think>\\n' + r + '\\n</think>\\n' }}"
GLM = "{% if '</think>' in c %}{{ '<think>' + r + '</think>\\n' }}"
BARE = "{{ c.split('</think>')[-1] }}{{ '<think>' + r + '</think>' }}"


class Tok:
    def __init__(self, tpl):
        self.chat_template = tpl
        self.think_end = "</think>"

    def encode(self, text, add_special_tokens=False):
        return ENC[text]


def test_qwen_skips_history_literal():
    assert resolve_think_close_pattern(Tok(QWEN)) == ([198], [271])


def test_step_and_glm_shapes():
    assert resolve_think_close_pattern(Tok(STEP)) == ([198], [198])
    assert resolve_think_close_pattern(Tok(GLM)) == (None, [198])


def test_bare_template_stays_bare():
    assert resolve_think_close_pattern(Tok(BARE)) == (None, None)


def _crit(tpl, budget=2):
    tok = Tok(tpl)
    return make_forced_close_criteria(
        tok,
        SimpleNamespace(tokenizer=tok),
        budget,
        thinking_end_token="</think>",
        thinking_start_token="<think>",
        enable_thinking=True,
        prompt_preopens_thinking=True,
    )


def _drive(crit):
    """Feed sampled tokens, then each forced token back, as mlx-vlm's batch does."""
    forced, nxt = [], 5
    for _ in range(10):
        crit(nxt)
        f = crit.pop_forced_token_id()
        if f is None:
            if forced:
                break
            nxt = 6
            continue
        forced.append(f)
        nxt = f
    return forced


def test_qwen_forces_newline_close_newlines_once():
    assert _drive(_crit(QWEN)) == [198, 901, 271]


def test_glm_forces_close_then_newline():
    assert _drive(_crit(GLM)) == [901, 198]


def test_bare_template_keeps_upstream_default():
    assert _drive(_crit(BARE)) == [198, 901]


def test_natural_close_forces_nothing():
    c = _crit(QWEN, budget=50)
    for t in [5, 6, 901, 8]:
        c(t)
        assert c.pop_forced_token_id() is None
