#!/bin/sh
cd /Volumes/P5Plus/yunshu-build/wt-sampdef || exit 1
exec /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python scripts/research/sampdef_modes.py /Volumes/P5Plus/models/Qwen3.5-2B-MLX-bf16 --port 18992 --home /Volumes/P5Plus/yunshu-build/sampdef-home2
