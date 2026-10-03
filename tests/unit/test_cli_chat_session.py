"""Chat sessions isolate prior turns and retain instructions after /clear."""

import importlib

import pytest
from typer.testing import CliRunner

from yunshu_cli import app

chat = importlib.import_module("yunshu_cli.chat")


@pytest.fixture(autouse=True)
def clear_history():
    chat.HISTORY.clear()
    yield
    chat.HISTORY.clear()


def test_new_invocation_does_not_send_previous_session(monkeypatch):
    seen = []

    def repl(*args):
        seen.append([dict(m) for m in chat.HISTORY])
        chat.HISTORY.append({"role": "user", "content": "private previous turn"})

    monkeypatch.setattr(chat, "_repl", repl)
    runner = CliRunner()
    assert (
        runner.invoke(
            app, ["chat", "-m", "first", "--system", "first instructions"]
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["chat", "-m", "second"]).exit_code == 0
    assert seen == [[{"role": "system", "content": "first instructions"}], []]


def test_clear_keeps_session_instructions(monkeypatch):
    chat.HISTORY[:] = [
        {"role": "system", "content": "Reply in Traditional Chinese."},
        {"role": "user", "content": "old turn"},
        {"role": "assistant", "content": "old answer"},
    ]
    inputs = iter(["/clear", "new turn", "/quit"])
    monkeypatch.setattr(chat.console, "input", lambda *args: next(inputs))
    requests = []
    monkeypatch.setattr(
        chat,
        "_send_non_stream",
        lambda url, body: requests.append([dict(m) for m in body["messages"]]),
    )
    chat._repl("http://localhost:8000", "model", 0, 16, False, True)
    assert requests == [
        [
            {"role": "system", "content": "Reply in Traditional Chinese."},
            {"role": "user", "content": "new turn"},
        ]
    ]


def test_clear_without_system_removes_all_turns(monkeypatch):
    chat.HISTORY[:] = [{"role": "user", "content": "old turn"}]
    inputs = iter(["/clear", "/quit"])
    monkeypatch.setattr(chat.console, "input", lambda *args: next(inputs))
    chat._repl("http://localhost:8000", "model", 0, 16, False, True)
    assert chat.HISTORY == []
