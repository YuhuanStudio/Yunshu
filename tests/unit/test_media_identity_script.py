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


def test_long_image_and_multi_image_messages():
    m = _load()
    assert m.image_url(0, 512, 256) != m.image_url(0, 512, 384)
    msg = m.multi_image_msg([0, 1, 2], "q")
    kinds = [p["type"] for p in msg[0]["content"]]
    assert kinds == ["image_url", "image_url", "image_url", "text"]


def test_wide_compare_separates_near_ties_from_suspects():
    from importlib.util import module_from_spec, spec_from_file_location

    p = Path(__file__).resolve().parents[2] / "scripts/research/media_wide_compare.py"
    spec = spec_from_file_location("media_wide_compare", p)
    m = module_from_spec(spec)
    spec.loader.exec_module(m)

    def steps(tokens, gaps):
        return [
            {"token": t, "logprob": -0.1, "gap": g}
            for t, g in zip(tokens, gaps, strict=True)
        ]

    same = {"steps": steps("abc", [2, 2, 2])}
    out = m.compare([same, same, same], [same, same, same])
    assert out["flips"] == 0 and not out["suspects"]
    near_b = {"steps": steps("abc", [2, 0.1, 2])}
    near_c = {"steps": steps("axc", [2, 0.2, 2])}
    out = m.compare([near_b], [near_c])
    assert out["flips"] == 1 and out["rows"][0]["near_tie"] and not out["suspects"]
    far_c = {"steps": steps("axc", [2, 3.0, 2])}
    out = m.compare([near_b], [far_c])
    assert len(out["suspects"]) == 1  # a confident token in one arm is not a near tie


def test_cli_accepts_every_flag_the_job_scripts_pass():
    """The exact flags phaseb2_mi.sh / phaseb2_miw.sh use must parse (a rejected flag killed a job)."""
    import subprocess
    import sys

    p = Path(__file__).resolve().parents[2] / "scripts/research/media_identity.py"
    help_text = subprocess.run(
        [sys.executable, str(p), "--help"], capture_output=True, text=True, check=True
    ).stdout
    for flag in (
        "--src",
        "--model",
        "--out",
        "--expect-driver-media",
        "--wide",
        "--wide-text",
        "--text-lengths",
        "--probe",
        "--long",
        "--env",
    ):
        assert flag in help_text


def test_paired_stats_use_only_positions_with_the_same_context():
    from importlib.util import module_from_spec, spec_from_file_location

    p = Path(__file__).resolve().parents[2] / "scripts/research/media_wide_compare.py"
    spec = spec_from_file_location("media_wide_compare2", p)
    m = module_from_spec(spec)
    spec.loader.exec_module(m)

    def steps(tokens, lps):
        return [
            {"token": t, "logprob": lp, "gap": 1.0}
            for t, lp in zip(tokens, lps, strict=True)
        ]

    base = [{"steps": steps("abcd", [-1.0, -2.0, -3.0, -4.0])}]
    cand = [{"steps": steps("abxd", [-1.0, -2.5, -9.0, -4.0])}]
    out = m.paired(base, cand)
    # positions 0 and 1 share a context; position 2 diverges (token differs): not paired
    assert out["positions"] == 2 and out["compared"] == 3
    assert out["exactly_zero"] == 1 and out["max"] == 0.5
    assert abs(out["top1_agreement"] - 2 / 3) < 1e-9
    assert out["first_token_n"] == 1 and out["first_token_nonzero"] == 0
