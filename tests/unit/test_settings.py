"""The YUNSHU_* settings registry: parsing, precedence, config file, typo guard,
the experimental budget, generated docs, the no-raw-read rule and `yunshu config`."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from yunshu_engine import settings

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in list(settings.REGISTRY) + ["YUNSHU_MTPP"]:
        monkeypatch.delenv(name, raising=False)
    settings.clear_overrides()
    yield
    settings.clear_overrides()


# ── parsing ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", ["1", "true", "YES", "on"])
def test_bool_true(monkeypatch, text):
    monkeypatch.setenv("YUNSHU_GPU_SAMPLER", text)
    assert settings.get_bool("YUNSHU_GPU_SAMPLER") is True


@pytest.mark.parametrize("text", ["0", "false", "No", "off"])
def test_bool_false(monkeypatch, text):
    monkeypatch.setenv("YUNSHU_MTP", text)
    assert settings.get_bool("YUNSHU_MTP") is False


def test_bad_values_raise(monkeypatch):
    monkeypatch.setenv("YUNSHU_MTP", "maybe")
    monkeypatch.setenv("YUNSHU_DEFAULT_MAX_TOKENS", "lots")
    monkeypatch.setenv("YUNSHU_MTP_BLOCK_SIZE", "1")  # below minimum 2
    with pytest.raises(settings.SettingError) as err:
        settings.validate(warn=False)
    msg = str(err.value)
    assert "YUNSHU_MTP=" in msg and "YUNSHU_DEFAULT_MAX_TOKENS" in msg
    assert "YUNSHU_MTP_BLOCK_SIZE" in msg


def test_typed_values(monkeypatch):
    monkeypatch.setenv("YUNSHU_DEFAULT_MAX_TOKENS", "1024")
    monkeypatch.setenv("YUNSHU_SSD_CACHE_MAX_GB", "2.5")
    monkeypatch.setenv("YUNSHU_TRUSTED_PROXIES", "10.0.0.1, 10.0.0.2")
    monkeypatch.setenv("YUNSHU_MCP_SERVERS", '[{"id": "a"}]')
    monkeypatch.setenv("YUNSHU_LOG_LEVEL", "debug")
    monkeypatch.setenv("YUNSHU_KV_QUANT_BITS", "OFF")
    assert settings.get("YUNSHU_DEFAULT_MAX_TOKENS") == 1024
    assert settings.get("YUNSHU_SSD_CACHE_MAX_GB") == 2.5
    assert settings.get("YUNSHU_TRUSTED_PROXIES") == ("10.0.0.1", "10.0.0.2")
    assert settings.get("YUNSHU_MCP_SERVERS") == [{"id": "a"}]
    assert settings.get("YUNSHU_LOG_LEVEL") == "DEBUG"
    assert settings.get("YUNSHU_KV_QUANT_BITS") == "off"


@pytest.mark.parametrize(
    ("text", "gib"), [("48", 48.0), ("48GB", 48.0), ("12.5gb", 12.5), ("disabled", 0.0)]
)
def test_memory_gb(monkeypatch, text, gib):
    monkeypatch.setenv("YUNSHU_MAX_MEMORY_GB", text)
    assert settings.get("YUNSHU_MAX_MEMORY_GB") == gib


def test_enum_rejects_unknown(monkeypatch):
    monkeypatch.setenv("YUNSHU_OVERLAP", "three_batch")
    with pytest.raises(settings.SettingError):
        settings.get("YUNSHU_OVERLAP")


def test_empty_env_is_unset_except_persona(monkeypatch):
    monkeypatch.setenv("YUNSHU_DEFAULT_MAX_TOKENS", "")
    monkeypatch.setenv("YUNSHU_OMNI_PERSONA", "")
    assert settings.get("YUNSHU_DEFAULT_MAX_TOKENS") == 512
    assert settings.get("YUNSHU_OMNI_PERSONA") == ""


def test_unregistered_name_is_a_key_error():
    with pytest.raises(KeyError):
        settings.get("YUNSHU_NOT_A_SETTING")


# ── precedence and config file ─────────────────────────────────────────


def test_precedence_cli_env_file_default(tmp_path, monkeypatch):
    cfg = tmp_path / "yunshu.toml"
    cfg.write_text(
        "default_max_tokens = 100\n[voice]\nYUNSHU_REALTIME_SILENCE_MS = 400\n"
    )
    monkeypatch.setenv("YUNSHU_CONFIG", str(cfg))
    assert settings.raw("YUNSHU_DEFAULT_MAX_TOKENS") == ("100", "file")
    assert settings.get("YUNSHU_REALTIME_SILENCE_MS") == 400
    assert settings.raw("YUNSHU_BATCH_MAX_ITEMS") == (None, "default")

    monkeypatch.setenv("YUNSHU_DEFAULT_MAX_TOKENS", "200")
    assert settings.get("YUNSHU_DEFAULT_MAX_TOKENS") == 200

    settings.set_override("YUNSHU_DEFAULT_MAX_TOKENS", 300)
    assert settings.raw("YUNSHU_DEFAULT_MAX_TOKENS") == ("300", "cli")


def test_config_file_bools_and_lists(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('mtp = false\ntrusted_proxies = "1.2.3.4"\n')
    assert settings.load_config_file(cfg) == {
        "YUNSHU_MTP": "0",
        "YUNSHU_TRUSTED_PROXIES": "1.2.3.4",
    }


def test_override_rejects_unknown():
    with pytest.raises(KeyError):
        settings.set_override("YUNSHU_NOPE", 1)


# ── typo guard ─────────────────────────────────────────────────────────


def test_unknown_names_suggest_close_match(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNSHU_MTPP", "1")
    warnings = settings.validate(warn=False)
    assert any("YUNSHU_MTPP" in w and "YUNSHU_MTP" in w for w in warnings)


def test_unknown_names_in_config_file(tmp_path, monkeypatch):
    cfg = tmp_path / "c.toml"
    cfg.write_text("defualt_max_tokens = 5\n")
    monkeypatch.setenv("YUNSHU_CONFIG", str(cfg))
    names = dict(settings.unknown_names({}))
    assert "YUNSHU_DEFUALT_MAX_TOKENS" in names
    assert "YUNSHU_DEFAULT_MAX_TOKENS" in names["YUNSHU_DEFUALT_MAX_TOKENS"]


# ── registry discipline ────────────────────────────────────────────────


def test_experimental_budget():
    exp = [s for s in settings.REGISTRY.values() if s.stability == "experimental"]
    assert len(exp) <= settings.MAX_EXPERIMENTAL, [s.name for s in exp]
    for s in exp:
        assert s.decide, f"{s.name} needs decide= (the measurement that settles it)"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", s.added), (
            f"{s.name} needs added=YYYY-MM-DD"
        )


def test_internal_is_tiny():
    internal = [s for s in settings.REGISTRY.values() if s.stability == "internal"]
    assert len(internal) <= 3, [s.name for s in internal]


def test_every_setting_is_well_formed():
    for s in settings.REGISTRY.values():
        assert s.name.startswith("YUNSHU_")
        assert s.stability in ("stable", "experimental", "internal")
        assert s.description.endswith("."), s.name
        if s.stability != "experimental":
            assert not s.decide and not s.added, s.name
        if s.type == "enum":
            assert s.default in s.choices, s.name


def test_docs_in_sync():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_config_docs.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr


# ── nothing reads YUNSHU_* from the environment directly ───────────────

_RAW_READ = re.compile(
    r"""(?:environ\s*\.\s*(?:get|setdefault|pop)|getenv)\s*\(\s*[fr]?["']YUNSHU_"""
    r"""|environ\s*\[\s*[fr]?["']YUNSHU_\w*["']\s*\](?!\s*=[^=])"""
    r"""|["']YUNSHU_\w*["']\s+in\s+(?:\w+\.)*environ\b"""
)


def test_no_raw_yunshu_env_reads():
    offenders = []
    for path in (ROOT / "python").rglob("*.py"):
        if path == ROOT / "python" / "yunshu_engine" / "settings.py":
            continue
        text = path.read_text()
        for m in _RAW_READ.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            offenders.append(f"{path.relative_to(ROOT)}:{line}")
    assert not offenders, "read these through yunshu_engine.settings: " + ", ".join(
        offenders
    )


def test_every_mentioned_name_is_registered():
    """Code and docstrings only name real settings (catches stale flags)."""
    allowed = {"YUNSHU_API_KEY", "YUNSHU_NAME", "YUNSHU_LOGGERS"}
    stale = set()
    for path in (ROOT / "python").rglob("*.py"):
        for name in re.findall(r"YUNSHU_[A-Z0-9_]*[A-Z0-9]", path.read_text()):
            if name not in settings.REGISTRY and name not in allowed:
                stale.add(f"{path.relative_to(ROOT)}: {name}")
    assert not stale, sorted(stale)


# ── CLI ────────────────────────────────────────────────────────────────


def test_cli_config_json(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from yunshu_cli import app

    cfg = tmp_path / "c.toml"
    cfg.write_text("default_max_tokens = 77\n")
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "s3cret")
    r = CliRunner().invoke(app, ["config", "--json", "--config", str(cfg)])
    assert r.exit_code == 0, r.output
    rows = {row["name"]: row for row in json.loads(r.output)["settings"]}
    assert rows["YUNSHU_DEFAULT_MAX_TOKENS"]["value"] == 77
    assert rows["YUNSHU_DEFAULT_MAX_TOKENS"]["source"] == "file"
    assert rows["YUNSHU_AUTH_TOKEN"]["value"] == "***"
    assert "YUNSHU_MTP_ROW_EXACT" not in rows  # experimental: only with --all
    # KV precision is the user's memory/quality choice: a stable setting
    assert rows["YUNSHU_KV_PRECISION"]["value"] == "bf16"


def test_cli_config_all_table():
    from typer.testing import CliRunner

    from yunshu_cli import app

    r = CliRunner().invoke(app, ["config", "--all"])
    assert r.exit_code == 0, r.output
    assert "YUNSHU_MTP_ROW_EXACT" in r.output


def test_cli_serve_set_rejects_unknown_key():
    from typer.testing import CliRunner

    from yunshu_cli import app

    r = CliRunner().invoke(app, ["serve", "--set", "mtpp=1"])
    assert r.exit_code == 2
    assert "YUNSHU_MTP" in r.output
