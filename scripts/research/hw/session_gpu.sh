#!/bin/zsh
# GPU capability microbenchmarks in one locked session (run via gpu_run.sh from the repo root).
[ -f scripts/research/local.env ] && source scripts/research/local.env
PY=${YUNSHU_PY:-$PWD/.venv/bin/python}
export PYTHONPATH=$PWD/python:$PWD/scripts/research/hw
H=scripts/research/hw
$PY $H/gpu_compute.py --tag nax
MLX_METAL_GPU_ARCH=applegpu_g16s $PY $H/gpu_compute.py --tag nonax
$PY $H/launch_overhead.py --tag default
MLX_METAL_FAST_SYNCH=1 $PY $H/launch_overhead.py --tag fast_synch
MLX_MAX_OPS_PER_BUFFER=4 $PY $H/launch_overhead.py --tag ops4
MLX_MAX_OPS_PER_BUFFER=200 $PY $H/launch_overhead.py --tag ops200
$PY $H/decode_proxy_step.py
MLX_METAL_FAST_SYNCH=1 $PY $H/decode_proxy_step.py
for n in 8 25 100 400; do
  MLX_MAX_OPS_PER_BUFFER=$n $PY $H/decode_proxy_step.py
done
for mb in 25 200; do
  MLX_MAX_MB_PER_BUFFER=$mb $PY $H/decode_proxy_step.py
done
$PY $H/gpu_stream_overlap.py
