#!/bin/bash
set -euo pipefail
cd /Volumes/P5Plus/yunshu-build/wt-codex-turn2prefill
exec /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python scripts/research/turn2prefill_matmul.py --out /Volumes/P5Plus/yunshu-build/verify/runs/turn2prefill-matmul-1021a/result.jsonl
