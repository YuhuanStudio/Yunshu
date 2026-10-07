import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import egemma2_cases as c  # noqa: E402
import egemma2_reference as r  # noqa: E402


def test_media_and_cases(tmp_path):
    m = c.make_media(str(tmp_path))
    cs = c.cases(m)
    assert {"text_long", "video", "interleaved", "audio_speech"} <= set(cs)
    assert (
        len(r.load_wav_16k(m["speech"])) > 16000
        and len(r.load_wav_16k(m["tone"])) == 48000
    )
    assert all(Path(p).exists() for p in m.values())
    assert len(c.long_text()) > 10000


def test_reference_helpers():
    v = r.truncate([3.0, 4.0, 12.0], 2)
    assert abs(v[0] - 0.6) < 1e-9 and abs(r.cosine(v, [3, 4]) - 1) < 1e-9
    assert r.truncate([1.0, 2.0], None) == [1.0, 2.0]


def test_compare_sets_maps_batch_to_plain_reference():
    import egemma2_parity as p

    refd = {"plain": {"a": [1.0, 0.0]}, "dims256": {"a": [0.0, 1.0]}}
    rows = p.compare_sets(
        refd, {"batch": {"a": [1.0, 0.0]}, "dims256": {"a": [0.0, 1.0]}}
    )
    assert all(abs(c - 1) < 1e-9 for _, _, c in rows) and len(rows) == 2
