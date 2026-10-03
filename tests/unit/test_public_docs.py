"""Public quickstarts must link to files and describe the installed CLI surface."""

from __future__ import annotations

import re
import shlex
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
from typer.main import get_command

from yunshu_cli import app
from yunshu_engine import settings

ROOT = Path(__file__).resolve().parents[2]
READMES = [ROOT / name for name in ("README.md", "README.zh-CN.md", "README.zh-TW.md")]


@pytest.mark.parametrize("path", READMES, ids=lambda p: p.name)
def test_readme_local_links_resolve(path):
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
        url = urlsplit(target)
        if url.scheme or not url.path:
            continue
        linked = path.parent / unquote(url.path)
        assert linked.exists(), f"{path.name}: broken link {target}"
        if url.fragment and linked.suffix == ".md":
            headings = re.findall(r"^#{1,6} (.+)$", linked.read_text(), re.MULTILINE)
            anchors = {
                re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings
            }
            assert url.fragment in anchors, f"{path.name}: missing anchor {target}"


@pytest.mark.parametrize("path", READMES, ids=lambda p: p.name)
def test_readme_commands_and_flags_exist(path):
    cli = get_command(app)
    text = path.read_text()
    commands = re.findall(r"`(yunshu [^`\n]+)`", text)
    commands += re.findall(r"^(?:uv run )?(yunshu .+)$", text, re.MULTILINE)
    assert commands
    for example in commands:
        args = shlex.split(example)
        command = cli
        position = 1
        while hasattr(command, "commands") and position < len(args):
            part = args[position]
            if part.startswith("-"):
                break
            positional = any(p.param_type_name == "argument" for p in command.params)
            if part not in command.commands and positional:
                break  # e.g. `yunshu launch claude`: the group's own TOOL argument
            assert part in command.commands, f"Unknown command in {example}"
            command = command.commands[part]
            position += 1
        options = {opt for p in command.params for opt in getattr(p, "opts", [])}
        for arg in args[position:]:
            if arg.startswith("-"):
                assert arg.split("=", 1)[0] in options, f"Unknown flag in {example}"


@pytest.mark.parametrize("path", READMES, ids=lambda p: p.name)
def test_readme_settings_and_install_extras_exist(path):
    text = path.read_text()
    for name in re.findall(r"YUNSHU_[A-Z][A-Z_0-9]+", text):
        assert name in settings.REGISTRY, f"{path.name}: removed setting {name}"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    for extra in re.findall(r"yunshu\[([^\]]+)\]", text):
        assert extra in project["optional-dependencies"]


def test_translated_performance_numbers_match_english():
    def numbers(path):
        tables = re.findall(r"(?:^\|.*\n)+", path.read_text(), re.MULTILINE)
        perf = [t for t in tables if "TensorFold 0.6.1" in t]
        assert len(perf) == 1, f"{path.name}: expected one performance table"
        return re.findall(r"\d+(?:\.\d+)?", perf[0])

    reference = numbers(READMES[0])
    assert reference
    for path in READMES[1:]:
        assert numbers(path) == reference, f"{path.name}: measurement drift"
