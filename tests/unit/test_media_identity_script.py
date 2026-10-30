import importlib.util
from pathlib import Path


def _load():
    p = Path(__file__).resolve().parents[2] / "scripts/research/media_identity.py"
    spec = importlib.util.spec_from_file_location("media_identity", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_images_differ_and_messages_carry_them():
    m = _load()
    assert m.image_url(0).startswith("data:image/png;base64,")
    assert m.image_url(0) != m.image_url(1) != m.image_url(2)
    msg = m.image_msg(1, "q")
    parts = msg[0]["content"]
    assert parts[0]["type"] == "image_url" and parts[1] == {"type": "text", "text": "q"}
