#!/bin/bash
# One-time dataset download for paired_eval.py (never the internal disk).
set -e
D=${DATASETS:-/Volumes/P5Plus/datasets}
O=${OMLX_EVAL_DATA:-/Users/yuhuan/Documents/YuhuanStudio/Yunshu/reference/omlx/omlx/eval/data}
mkdir -p $D/gsm8k $D/mmlu_pro $D/ifeval $D/bfcl/possible_answer
[ -s $D/gsm8k/gsm8k_test.jsonl ] || cp $O/gsm8k_test.jsonl $D/gsm8k/
[ -s $D/mmlu_pro/mmlu_pro_test.jsonl ] || cp $O/mmlu_pro_test.jsonl $D/mmlu_pro/
[ -s $D/ifeval/input_data.jsonl ] || curl -fsSL -o $D/ifeval/input_data.jsonl \
  "https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl"
H=https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main
for f in BFCL_v3_simple.json BFCL_v3_parallel.json possible_answer/BFCL_v3_simple.json possible_answer/BFCL_v3_parallel.json; do
  [ -s $D/bfcl/$f ] || curl -fsSL -o $D/bfcl/$f $H/$f
done
wc -l $D/*/*.json* $D/bfcl/possible_answer/*
