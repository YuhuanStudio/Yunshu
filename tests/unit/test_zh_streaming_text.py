"""Chinese (Simplified + Traditional) streaming text on the real Qwen tokenizer (2026-11-09).

Speculative rounds commit several tokens at once, but the VLM path feeds them to the
per-request detokenizer one by one; a Han character split across byte-level tokens must
never surface as U+FFFD, and a Chinese stop string must cut the text exactly.
"""

from pathlib import Path

import pytest

MODEL = Path("/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp")
TEXT = (
    "今天天气很好，我们去公园散步吧。臺灣的資訊工程師們正在討論語言模型的推論效能，"
    "「繁體中文」與简体中文混合；罕見字：龘、𠮷、鱻。表情符號😀也要正確。END"
)


@pytest.fixture(scope="module")
def tok():
    if not MODEL.exists():
        pytest.skip("27B checkpoint not on this machine")
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    return load_tokenizer(MODEL)


def test_streamed_segments_rebuild_the_text_without_replacement_chars(tok):
    ids = tok.encode(TEXT, add_special_tokens=False)
    detok = tok.detokenizer
    detok.reset()
    out = []
    for t in ids:
        detok.add_token(t)
        seg = detok.last_segment
        assert "�" not in seg, (t, seg)
        out.append(seg)
    detok.finalize()
    out.append(detok.last_segment)
    assert "".join(out) == TEXT


def test_a_chinese_stop_string_cuts_the_stream_exactly(tok):
    from yunshu_engine.text_utils import StopHoldbackBuffer

    stop = "語言模型"
    buf = StopHoldbackBuffer([stop])
    ids = tok.encode(TEXT, add_special_tokens=False)
    detok = tok.detokenizer
    detok.reset()
    emitted = []
    for t in ids:
        detok.add_token(t)
        emitted.append(buf.feed(detok.last_segment))
        if buf.contains_stop():  # the engine's own sequence (vlm_engine streaming loop)
            emitted.append(buf.take_stopped())
            break
    text = "".join(emitted)
    assert text == TEXT[: TEXT.index(stop)]
