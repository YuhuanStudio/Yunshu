"""``yunshu statusline`` and the launch helper's status-line settings."""

from __future__ import annotations

import json

from yunshu_cli.integrations.agent_config import claude_statusline_settings
from yunshu_cli.statusline import render


def test_prefill_line_shows_progress_speed_eta_and_cache():
    status = {
        "requests": {
            "items": [
                {
                    "phase": "prefill",
                    "percent": 42.0,
                    "processed_tokens": 3100,
                    "prompt_tokens": 7400,
                    "tokens_per_second": 1500.0,
                    "eta_s": 2.9,
                    "cached_tokens": 1000,
                }
            ]
        }
    }
    line = render(status, {"model": {"display_name": "Qwen3.8 27B"}})
    assert line == "Qwen3.8 27B | prefill 42% 3.1k/7.4k @1.5k/s eta 2.9s (cache 1.0k)"


def test_decode_uses_live_rate_and_context_percentage():
    status = {
        "requests": {"items": [{"phase": "decode"}]},
        "throughput": {"live_decode_tps": 79.6},
    }
    line = render(status, {"context_window": {"used_percentage": 12.4}})
    assert line == "yunshu | decode 80 tok/s | ctx 12%"


def test_queued_line():
    status = {
        "requests": {
            "items": [
                {"phase": "queued", "queue_position": 2, "queue_est_wait_ms": 4200}
            ]
        }
    }
    assert render(status) == "yunshu | queued #2 ~4.2s"


def test_idle_shows_last_request_cache_hit():
    status = {
        "requests": {"items": []},
        "last": {
            "prompt_tokens": 1000,
            "cached_tokens": 910,
            "decode_tps": 78.2,
            "ttft_ms": 400,
        },
    }
    assert render(status) == "yunshu | last 78 tok/s cache 91% ttft 0.4s"
    assert render({"requests": {"items": []}}) == "yunshu | idle"


def test_unreachable_server_never_raises():
    assert render(None) == "yunshu | engine unreachable"


def test_statusline_settings_never_replace_the_users_own():
    cmd = "yunshu statusline --url http://127.0.0.1:8000"
    got = json.loads(claude_statusline_settings(cmd, {}))
    assert got["statusLine"]["command"] == cmd
    assert claude_statusline_settings(cmd, {"statusLine": {"type": "command"}}) is None
