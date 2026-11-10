import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "apc_restart_probe",
    Path(__file__).resolve().parents[2] / "scripts/research/apc_restart_probe.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class _WordTok:
    """One token per word plus a one-token chat template."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)

    def apply_chat_template(self, msgs, tokenize=True, **kw):
        return ["<tpl>"] + [w for msg in msgs for w in msg["content"].split()]


def test_helpers():
    assert m.cached_of({"usage": {"prompt_tokens_details": {"cached_tokens": 5}}}) == 5
    assert m.cached_of({}) == 0
    assert len(m.make_prompt(_WordTok(), 100).split()) == 99  # + 1 template token
