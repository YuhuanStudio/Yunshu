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
    "realtime_voice_native",
    "realtime_voice_cascade",
}


@pytest.mark.parametrize(
    "name",
    [
        n
        for n, c in rc.REGISTRY.items()
        if c.needs in ("main", "omni", "native") and n not in SOCKETS
    ],
)
def test_check_has_no_harness_bug(name, monkeypatch):
    http, eng = install(monkeypatch)
    cl = Clients(http)
    ctx = rc.Ctx(
        url="http://testserver",
        token="",
        model="scripted",
        kind="vlm" if name == "vision_media_cache" else "text",
        fixtures={"speech_wav": rc._wav(1.0, 16000)},
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
                    "route_checks_omni.py",
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


def test_omni_helpers():
    import route_checks_omni as om

    pcm = om.wav_pcm16(rc._wav(1.0, 16000), 24000)
    assert len(pcm) == 24000 * 2
    assert om._word_in("The secret word is Pine-apple.")
    assert not om._word_in("banana")
    msgs = om._chat_audio_msg(rc._wav(0.1), "q", rc._png(8))
    assert [p["type"] for p in msgs[0]["content"]] == [
        "text",
        "input_audio",
        "image_url",
    ]
    ev = [{"type": "response.output_audio.delta", "delta": "AAAA"}]
    assert om._audio_bytes(ev) == 3


def test_native_and_cascade_checks_are_served_and_registered():
    served = {n for n, c in rc.REGISTRY.items() if c.served}
    assert {
        "omni_speech_stream",
        "realtime_voice_native",
        "realtime_voice_cascade",
        "omni_audio_in",
        "omni_image_audio_in",
        "omni_video_in",
    } <= served


def test_omni_job_argv_parses_and_plan_has_the_job(tmp_path):
    import importlib.machinery
    import importlib.util

    import m3sweep_jobs as mj

    loader = importlib.machinery.SourceFileLoader(
        "m3sweep_mod", str(ROOT / "scripts/dev/m3sweep")
    )
    spec = importlib.util.spec_from_loader("m3sweep_mod", loader)
    m3 = importlib.util.module_from_spec(spec)
    loader.exec_module(m3)
    jobs = {j["name"]: j for j in m3.plan("abcdef0", tmp_path)}
    j = jobs["omni-gemma"]
    a = mj.build_parser().parse_args(j["cmd"][2:])
    assert (
        a.mode == "small"
        and a.model.endswith("gemma-4-e2b-it-4bit")
        and len(a.asr) == 1
    )
    assert (
        "--mem-gb" in j["submit"]
        and j["submit"][j["submit"].index("--mem-gb") + 1] == "14"
    )
