#!/bin/bash
set -euo pipefail
cd /Volumes/P5Plus/yunshu-build/wt-codex-agentcensus
mkdir -p "$PWD/docs/research/agentcensus/tmp"
export TMPDIR="$PWD/docs/research/agentcensus/tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/python"
export TFB_YUNSHU_SRC="$PWD/python" TFB_PORT_LAST=18996
nice -n 15 /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python -m pytest tests/unit/test_agentcensus_1022.py tests/unit/test_anthropic_thinking_display.py tests/unit/test_coding_agent_api_fixes.py tests/unit/test_agent_compat_stream_check.py -q -p no:cacheprovider
exec /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python scripts/research/agent_compat/census_1022.py --model /Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp --source "$PWD/python" --out /Volumes/P5Plus/yunshu-build/codex/agentcensus-cand-1022b
