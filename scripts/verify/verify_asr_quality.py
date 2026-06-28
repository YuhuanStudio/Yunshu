"""ASR transcription-quality gate.

Upgrades the modality SMOKE (which feeds silence and only checks it runs) to a real
accuracy check: transcribe a known Chinese reference clip and assert the expected
content words appear. Catches regressions that break transcription quality without
crashing. Reference says: "你好，这是一段用于测试语音辨识系统的中文语音档案。"

Run: PYTHONPATH=. uv run python scripts/verify_asr_quality.py
"""
from __future__ import annotations

import os
import sys

MODEL = os.environ.get("YUNSHU_ASR_MODEL", "./models/Qwen3-ASR-1.7B-bf16")
AUDIO = "./testresource/asr_test_zh.wav"
# Stable content words (tolerant of ASR variants like 辨识/识别, 档案/檔案).
EXPECT = ["你好", "测试", "语音", "中文", "系统"]
MIN_HITS = 4  # at least 4 of 5 must appear


def main() -> int:
    if not os.path.exists(MODEL) or not os.path.exists(AUDIO):
        print(f"SKIP: model or audio not available ({MODEL}, {AUDIO})")
        return 0
    from mlx_audio.stt import load
    model = load(MODEL)
    r = model.generate(AUDIO)
    text = (r.text if hasattr(r, "text") else str(r)).strip()
    hits = [w for w in EXPECT if w in text]
    ok = len(hits) >= MIN_HITS
    print(f"transcript: {text!r}")
    print(f"matched {len(hits)}/{len(EXPECT)} keywords: {hits}")
    print(f"RESULT: {'PASS' if ok else 'FAIL'} (need >= {MIN_HITS})")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
