"""ANE embedding — honor max_seq_length (was hardcoded 128) + benchmark .mlpackage.

(1) The CoreML trace + inference both clamped to min(max_seq_length, 128), so an operator
setting YUNSHU_ANE_MAX_SEQ_LENGTH=512 still got 128-token truncation, and ANE produced worse
vectors than the MLX fallback (which uses the FULL sequence). Now both honor the configured
max_seq_length (and stay equal to each other — the traced-shape invariant).
(2) benchmark_ane_vs_gpu pointed _compiled_path at a .mlmodelc, which ct.models.MLModel can't
load → silent MLX fallback → measured MLX-vs-MLX (bogus speedup≈1.0). Now points at the
.mlpackage that compile_model stores and serving loads.
"""
from __future__ import annotations

import inspect

from yunshu_engine import ane_embedding


def test_seq_length_honors_config_not_hardcoded_128():
    src = inspect.getsource(ane_embedding)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # the hardcoded 128 cap is gone from both trace and inference
    assert "min(self._config.max_seq_length, 128)" not in code
    # both now use the configured length (so trace == inference, and == MLX path)
    assert "seq = int(self._config.max_seq_length)" in code
    assert "_seq = int(self._config.max_seq_length)" in code


def test_benchmark_uses_mlpackage_not_mlmodelc():
    src = inspect.getsource(ane_embedding.benchmark_ane_vs_gpu)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # the benchmark resolves the loadable .mlpackage (not the unloadable .mlmodelc)
    assert 'f"{model_name_safe}.mlpackage"' in code
    assert 'mlmodelc_path = cache_dir' not in code  # the old .mlmodelc target is gone
