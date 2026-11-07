import mlx.core as mx
from yunshu_engine.kernels import ragged_attention as ra
from yunshu_engine.kernels.tensorfold import lane_qmm
print("device", mx.device_info().get("architecture"))
print("tile_ready", ra.tile_ready(), "lane_ready", lane_qmm.ready(), "variant", lane_qmm._resolve_variant())
