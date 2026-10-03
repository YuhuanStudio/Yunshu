"""Kernels vendored from TensorFold (https://github.com/ashhart/TensorFold, MIT).

``lane_qmm.py``, ``lane_widen.py`` and ``inputs.py`` are copied from TensorFold
34bae79 (``src/tensorfold/kernels/qwen/dense/v1/`` and ``kernels/inputs.py``);
imports and cooperative tensor element indexing are adapted for Yunshu. Yunshu uses ``lane_matmul`` through
``yunshu_engine.kernels.int_code_linear``; nothing here patches MLX on import.
"""
