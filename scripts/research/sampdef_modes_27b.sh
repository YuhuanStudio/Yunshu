#!/bin/sh
cd /Volumes/P5Plus/yunshu-build/wt-sampdef || exit 1
exec /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python scripts/research/sampdef_modes.py /Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp --port 18992 --home /Volumes/P5Plus/yunshu-build/sampdef-home2
