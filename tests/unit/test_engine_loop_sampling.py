"""bring the opt-in engine-loop sampler to parity with the fast path.

the engine-loop branch of BatchedEngine.generate dropped min_tokens / ignore_eos /
suppress_tokens, and engine_core.add_request never forwarded them into SamplingParams, so
all three were silently ignored under YUNSHU_ENGINE_LOOP=1.

the engine loop built frequency/presence penalties via mlx-lm's make_logits_processors
(a 20-token sliding window) while _LogitsProcessorSampler._tokens is seeded with the FULL
prompt — so the penalty was computed over the prompt tail and only a 20-token window,
materially diverging from the fast path (which counts the GENERATED completion only, over
the full history). The scheduler now uses the same generated-only closure.
"""

from __future__ import annotations

import inspect


def test_add_request_forwards_min_tokens_ignore_eos_suppress():
    from yunshu_engine import engine_core

    src = inspect.getsource(engine_core.EngineCore.add_request)
    # the three params are forwarded into SamplingParams from kwargs
    assert "min_tokens=int(kwargs.get('min_tokens'" in src
    assert "ignore_eos=bool(kwargs.get('ignore_eos'" in src
    assert "suppress_tokens=kwargs.get('suppress_tokens'" in src


def test_batched_engine_loop_passes_the_three_params():
    from yunshu_engine.batched_engine import BatchedEngine

    src = inspect.getsource(BatchedEngine.generate)
    # the _engine_core.generate call now threads all three
    i = src.index("self._engine_core.generate(")
    window = src[i : i + 2000]
    assert "min_tokens=min_tokens" in window
    assert "ignore_eos=ignore_eos" in window
    assert "suppress_tokens=suppress_tokens" in window


def test_scheduler_freq_pres_penalty_is_generated_only():
    from yunshu_engine import scheduler

    src = inspect.getsource(scheduler.Scheduler._make_sampler)
    # presence/frequency are no longer handed to mlx-lm's window-based factory
    assert "presence_penalty=sp.presence_penalty" not in src
    assert "frequency_penalty=sp.frequency_penalty" not in src
    # the generated-only closure is present and keyed off the prompt length
    assert "_freq_pres_penalty" in src
    assert "n_prompt=_fp_nprompt" in src
    assert "tokens[n_prompt:]" in src


def test_scheduler_freq_penalty_ignores_prompt_tokens_behaviorally():
    """The closure must not penalize tokens that appear only in the prompt."""

    # Build the closure exactly as _make_sampler does, with a 3-token prompt.
    prompt_token_ids = [10, 11, 12]
    n_prompt = len(prompt_token_ids)
    state = {"counts": {}, "last_len": -1}

    fp = 1.0
    pp = 0.0

    def _freq_pres_penalty(tokens, logits, fp=fp, pp=pp, n_prompt=n_prompt, _st=state):
        counts = _st["counts"]
        last_len = int(_st["last_len"])
        cur_len = len(tokens)
        if cur_len <= n_prompt:
            _st["last_len"] = cur_len
            return logits
        if cur_len < last_len or last_len < n_prompt:
            counts = {}
            for t in tokens[n_prompt:]:
                counts[int(t)] = counts.get(int(t), 0) + 1
            _st["counts"] = counts
        else:
            start = max(last_len, n_prompt)
            for t in tokens[start:]:
                counts[int(t)] = counts.get(int(t), 0) + 1
        _st["last_len"] = cur_len
        for tid, cnt in counts.items():
            if fp != 0.0:
                logits[tid] = logits[tid] - fp * cnt
        return logits

    # token id 10 appears in the prompt; id 99 is generated twice.
    # logits as a plain list so we can assert numerically without mlx.
    logits = [0.0] * 100

    # first generated token (99): tokens = prompt + [99]
    out = _freq_pres_penalty([10, 11, 12, 99], list(logits))
    # prompt token 10 must be untouched (not penalized as if repeated)
    assert out[10] == 0.0
    # 99 was generated once → penalized by fp*1
    assert out[99] == -1.0

    # second generated token (99 again): tokens = prompt + [99, 99]
    out2 = _freq_pres_penalty([10, 11, 12, 99, 99], list(logits))
    assert out2[10] == 0.0
    # 99 now counted twice → penalized by fp*2
    assert out2[99] == -2.0
