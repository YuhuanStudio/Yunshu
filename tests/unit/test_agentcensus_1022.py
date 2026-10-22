"""CPU validation of the 27B census driver before loading the model."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "census1022",
    Path(__file__).resolve().parents[2]
    / "scripts/research/agent_compat/census_1022.py",
)
driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(driver)


def test_census_requires_real_results_and_rejects_failures():
    assert not driver.summarize({})["pass"]
    assert not driver.summarize({"claude": {"ok": True}, "codex": {"ok": False}})[
        "pass"
    ]
    assert driver.summarize({"claude": {"ok": True}}) == {
        "complete": True,
        "pass": True,
        "results": {"claude": {"ok": True}},
    }


def test_census_parser_covers_all_clients_and_server_tools():
    a = driver.parser().parse_args(
        [
            "--out",
            "/Volumes/P5Plus/test",
            "--model",
            "/Volumes/P5Plus/model",
            "--dry-run",
        ]
    )
    assert a.dry_run
    assert {
        "cc_edit",
        "cx_edit",
        "oc_edit",
        "cc_websearch",
        "cx_mcp",
        "cx_ws",
        "sdk",
    } <= set(a.scenarios.split(","))


def test_census_rejects_missing_scenarios_and_nonzero_client_exit():
    assert not driver.summarize({"claude": {"ok": True}}, ["claude", "codex"])["pass"]
    assert not driver.summarize({"claude": {"ok": True, "rc": 1}})["pass"]
