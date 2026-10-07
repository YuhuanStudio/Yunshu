"""The comparison must fail if isolated HOME hides the requested drafter."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "tfbench", Path(__file__).resolve().parents[2] / "scripts/research/tfbench.py"
)
tfbench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tfbench)


def test_yunshu_passes_drafter_explicitly():
    original = {}
    mode, env = tfbench.spec_request("yunshu", original)
    assert mode == "dflash"
    assert env["YUNSHU_VLM_DRAFT"] == tfbench.D
    assert original == {}


@pytest.mark.parametrize(
    "override,mode", [("mtp", "mtp"), ("off", "off"), ("/draft", "dflash")]
)
def test_explicit_ab_arm_is_preserved(override, mode):
    actual, env = tfbench.spec_request("yunshu", {"YUNSHU_VLM_DRAFT": override})
    assert actual == mode
    assert env["YUNSHU_VLM_DRAFT"] == override


@pytest.mark.parametrize(
    "log",
    [
        "",
        "Speculative decoding: dflash (automatic)",
        "Speculative decoding: dflash (automatic)\nVLM batch runner: apc=off draft=mtp block=6",
    ],
)
def test_dflash_fails_closed_on_missing_or_fallback_runner(tmp_path, log):
    server = object.__new__(tfbench.Srv)
    server.engine = "yunshu"
    server.requested_spec_mode = "dflash"
    server.log = tmp_path / "server.log"
    server.log.write_text(log)
    with pytest.raises(RuntimeError, match="requested spec=dflash"):
        server.verify_spec_mode()


def test_records_actual_runner_mode(tmp_path):
    server = object.__new__(tfbench.Srv)
    server.engine = "yunshu"
    server.requested_spec_mode = "dflash"
    server.log = tmp_path / "server.log"
    server.log.write_text("VLM batch runner: apc=32.0GiB draft=dflash block=8")
    server.verify_spec_mode()
    assert server.engaged_spec_mode == "dflash"


@pytest.mark.parametrize("actual", ["dflash", "mtp", None])
def test_server_launch_checks_fresh_log_and_cleans_up(monkeypatch, tmp_path, actual):
    import io

    monkeypatch.setattr(tfbench, "OUT", tmp_path)
    monkeypatch.setattr(tfbench, "free_port", lambda: 18999)
    log = tmp_path / "out/server-repeat.log"
    log.parent.mkdir()
    log.write_text("VLM batch runner: apc=off draft=dflash block=8\n")
    launched = {}

    class Proc:
        pid = 123456789

        def poll(self):
            return None

    def launch(cmd, **kw):
        launched.update(cmd=cmd, env=kw["env"])
        if actual:
            kw["stdout"].write(
                f"VLM batch runner: apc=off draft={actual} block=8\n".encode()
            )
            kw["stdout"].flush()
        kw["stdout"].close()
        return Proc()

    monkeypatch.setattr(tfbench.subprocess, "Popen", launch)
    monkeypatch.setattr(
        tfbench.urllib.request,
        "urlopen",
        lambda *_a, **_kw: io.BytesIO(b'{"data":[{"id":"test"}]}'),
    )
    killed = []
    monkeypatch.setattr(tfbench.Srv, "kill", lambda s: killed.append(s.proc))
    if actual == "dflash":
        server = tfbench.Srv("yunshu", {}, "repeat")
        assert server.engaged_spec_mode == "dflash"
        assert not killed
    else:
        with pytest.raises(RuntimeError, match="requested spec=dflash"):
            tfbench.Srv("yunshu", {}, "repeat")
        assert len(killed) == 1
    assert launched["env"]["YUNSHU_VLM_DRAFT"] == tfbench.D
    assert launched["env"]["HOME"] == str(tmp_path / "home/repeat")


@pytest.mark.parametrize(
    "tag_args,tag",
    [
        (["--tag", "-dflash"], "-dflash"),
        (["--tag=-dflash"], "-dflash"),
        (["--tag", "plain"], "plain"),
        (["--tag", ""], ""),
    ],
)
def test_queued_tag_suffix_is_a_value(tag_args, tag):
    argv = [
        "--engine",
        "yunshu",
        "--part",
        "decode",
        "--out",
        "result.jsonl",
        *tag_args,
        "--env",
        f"YUNSHU_VLM_DRAFT={tfbench.D}",
    ]
    original = list(argv)
    args = tfbench.parse_args(argv)
    assert args.tag == tag
    assert args.env == [f"YUNSHU_VLM_DRAFT={tfbench.D}"]
    assert argv == original


def test_tag_does_not_consume_next_registered_option():
    with pytest.raises(SystemExit) as exc:
        tfbench.parse_args(
            [
                "--engine",
                "yunshu",
                "--part",
                "decode",
                "--out",
                "x",
                "--tag",
                "--env",
                f"YUNSHU_VLM_DRAFT={tfbench.D}",
            ]
        )
    assert exc.value.code == 2


def test_cpu_preflight_checks_legacy_tag_without_starting_server(
    monkeypatch, tmp_path, capsys
):
    import sys

    (tmp_path / "config.json").write_text('{"model_type":"qwen3_5"}')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tfbench",
            "--engine",
            "yunshu",
            "--part",
            "decode",
            "--out",
            str(tmp_path / "result.jsonl"),
            "--tag",
            "-dflash",
            "--model",
            str(tmp_path),
            "--dry-run",
        ],
    )
    prompts = []
    monkeypatch.setattr(
        tfbench, "load_prompt", lambda name: prompts.append(name) or "prompt"
    )
    monkeypatch.setattr(
        tfbench, "Srv", lambda *_a, **_k: pytest.fail("CPU preflight launched a server")
    )
    tfbench.main()
    assert len(prompts) == 6
    assert '"requested_spec_mode": "dflash"' in capsys.readouterr().out
    assert not (tmp_path / "result.jsonl").exists()


def test_fatal_startup_does_not_spend_queue_time_polling(monkeypatch, tmp_path):
    monkeypatch.setattr(tfbench, "OUT", tmp_path)
    monkeypatch.setattr(tfbench, "free_port", lambda: 18999)

    class Proc:
        def poll(self):
            return None

    def launch(_cmd, **kw):
        kw["stdout"].write(b"FATAL: model load failed: hidden-size mismatch\n")
        kw["stdout"].flush()
        return Proc()

    monkeypatch.setattr(tfbench.subprocess, "Popen", launch)
    monkeypatch.setattr(
        tfbench.urllib.request,
        "urlopen",
        lambda *_a, **_kw: pytest.fail("polled fatal server"),
    )
    killed = []
    monkeypatch.setattr(tfbench.Srv, "kill", lambda s: killed.append(s.proc))
    with pytest.raises(RuntimeError, match="startup failed"):
        tfbench.Srv("yunshu", {}, "fatal")
    assert len(killed) == 1


def test_kill_drops_the_servers_prefix_cache_keeps_logs(tmp_path):
    srv = tfbench.Srv.__new__(tfbench.Srv)
    srv.home = tmp_path / "home" / "t"
    apc = srv.home / ".yunshu" / "cache" / "apc" / "x"
    apc.mkdir(parents=True)
    (apc / "exact.safetensors").write_bytes(b"0" * 1024)
    (srv.home / "server.log").write_text("log")

    class Done:
        pid = 0

        def poll(self):
            return 0

    srv.proc = Done()
    srv.kill()
    assert not (srv.home / ".yunshu" / "cache").exists()
    assert (srv.home / "server.log").read_text() == "log"
