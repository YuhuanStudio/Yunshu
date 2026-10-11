"""DIAGNOSTIC ONLY (not lossless): make every PLE lookup return zeros, to measure what the PLE sync costs.

Put this directory first on PYTHONPATH of a server process and set FN80_PLE_ZERO=1.
"""

import os

if os.environ.get("FN80_PLE_ZERO") == "1":
    try:
        import mlx.core as mx
        from mlx_vlm.models.qwen4_exp import ple_storage

        def _zero(self, row_ids):
            shape = row_ids.shape
            return mx.zeros((*shape, self.row_width), dtype=mx.bfloat16)

        ple_storage.QuantizedMMapNGramEmbedding.__call__ = _zero
    except Exception:  # noqa: BLE001
        pass
