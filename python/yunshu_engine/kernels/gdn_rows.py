"""Gated-delta recurrence over a batch of rows with per-row window lengths.

The round driver's decode step runs every row's window (its pending token plus
drafts, ``1 .. T`` tokens) through one launch per GatedDeltaNet layer. Rows are
right-padded to ``T`` tokens; ``lengths[b]`` says how many are real.

The per-token arithmetic is the upstream ``gated_delta_step`` kernel's (mlx-vlm
``qwen3_5/gated_delta.py``: decay, ``kv_mem`` by one ``simd_sum``, delta, state
update, output by one ``simd_sum``), one threadgroup column per (row, value
head, value dim) — a row's bits do not depend on the other rows or on ``T``:
a window of ``w`` tokens produces the same ``y`` and states as a call with
``T = w``, which is what makes a verified window equal plain decode.

Outputs: ``y`` [B, T, Hv, Dv] (zeros past a row's length), the state after the
row's last real token, and — for ``T > 1`` — the states after every real token
but the last (``hist[b, j]`` = state after token ``j``, ``j < lengths[b] - 1``,
the rest unwritten), so a row that keeps only part of its window continues from
``hist[b, kept - 1]`` (speculative rollback) without a second pass.
"""

from __future__ import annotations

import mlx.core as mx

_KERNELS: dict = {}


def _source(hist: bool) -> str:
    save = (
        """
          if (t + 1 < L) {
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              hist_[s_idx] = static_cast<StT>(state[i]);
            }
          }
          hist_ += Hv * Dv * Dk;
        """
        if hist
        else ""
    )
    hist_setup = (
        "auto hist_ = hist + (((b_idx * (T - 1)) * Hv + hv_idx) * Dv + dv_idx) * Dk;"
        if hist
        else ""
    )
    return f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        const int L = lens[b_idx];

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;
        {hist_setup}

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if (t < L) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }} else {{
            y[dv_idx] = static_cast<InT>(0);
          }}
          {save}
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """


def _kernel(hist: bool):
    if hist not in _KERNELS:
        _KERNELS[hist] = mx.fast.metal_kernel(
            name="gated_delta_rows" + ("_hist" if hist else ""),
            input_names=["q", "k", "v", "g", "beta", "state_in", "lens", "T"],
            output_names=["y", "state_out"] + (["hist"] if hist else []),
            source=_source(hist),
        )
    return _KERNELS[hist]


def gated_delta_rows(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    lengths: mx.array,
) -> tuple[mx.array, mx.array, mx.array | None]:
    """``q``, ``k`` [B, T, Hk, Dk], ``v`` [B, T, Hv, Dv], ``g`` / ``beta``
    [B, T, Hv] (scalar gating), ``state`` [B, Hv, Dv, Dk], ``lengths`` int32
    [B]. Returns ``(y, state_after, hist)``; ``hist`` [B, T - 1, Hv, Dv, Dk] is
    None for ``T == 1``."""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    hist = T > 1
    outs = _kernel(hist)(
        inputs=[q, k, v, g, beta, state, lengths.astype(mx.int32), T],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape]
        + ([(B, T - 1, Hv, Dv, Dk)] if hist else []),
        output_dtypes=[q.dtype, state.dtype] + ([state.dtype] if hist else []),
    )
    return outs[0], outs[1], (outs[2] if hist else None)


__all__ = ["gated_delta_rows"]
