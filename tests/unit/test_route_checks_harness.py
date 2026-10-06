"""Every route check runs against the scripted engine behind the real routers (CPU only), so
harness bugs (a missing api_key, a wrong attribute, a bad import, a typo) show up before an M3 job.
Outcomes that depend on a real model (Fail, API errors, timeouts, sockets) are fine here; a Python
error inside the check itself is not."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import openai
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))

import route_checks as rc  # noqa: E402

from .wire_clients import Clients  # noqa: E402
from .wire_harness import install  # noqa: E402

HARNESS_BUGS = (
    NameError,
    TypeError,
    AttributeError,
    KeyError,
    ImportError,
    ValueError,
    IndexError,
)
SOCKETS = {
    "cancel_live",
    "ws_responses",
    "ws_responses_sdk",
    "ws_stream",
    "ws_realtime",
}


@pytest.mark.parametrize(
    "name",
    [n for n, c in rc.REGISTRY.items() if c.needs == "main" and n not in SOCKETS],
)
def test_check_has_no_harness_bug(name, monkeypatch):
    http, eng = install(monkeypatch)
    cl = Clients(http)
    ctx = rc.Ctx(
        url="http://testserver",
        token="",
        model="scripted",
        kind="text",
        http=http,
        oa=openai.OpenAI(
            base_url="http://testserver/v1",
            api_key="x",
            http_client=http,
            max_retries=0,
        ),
        an=cl.an,
    )
    clock = iter(
        range(0, 10**9, 30)
    )  # a fake clock: every poll loop runs out its deadline at once
    monkeypatch.setattr(
        rc,
        "time",
        types.SimpleNamespace(
            time=lambda: next(clock),
            sleep=lambda s: None,
            monotonic=lambda: next(clock),
        ),
    )
    import signal

    def stuck(*a):
        raise TimeoutError(
            "check blocks on the fake (a socket / stream): not a harness bug"
        )

    signal.signal(signal.SIGALRM, stuck)
    signal.alarm(20)
    try:
        rc.REGISTRY[name].fn(ctx)
    except (rc.Fail, rc.Skip):
        pass
    except openai.OpenAIError as e:
        assert isinstance(e, openai.APIStatusError | openai.APIConnectionError), (
            f"{name}: harness: {e}"
        )
    except HARNESS_BUGS as e:
        # a TypeError/AttributeError raised by the server side under a fake engine is not ours
        import traceback

        tb = traceback.extract_tb(e.__traceback__)
        mine = [
            f
            for f in tb
            if f.filename.endswith(
                (
                    "route_checks.py",
                    "route_checks_tools.py",
                    "route_checks_vision.py",
                    "route_checks_vllm.py",
                )
            )
        ]
        if mine and not any("site-packages" in f.filename for f in tb[-1:]):
            raise AssertionError(
                f"{name}: {type(e).__name__}: {e} at {mine[-1].filename}:{mine[-1].lineno}"
            ) from e
    except Exception:  # noqa: BLE001  timeouts, HTTP errors: model-dependent
        pass
    finally:
        signal.alarm(0)
