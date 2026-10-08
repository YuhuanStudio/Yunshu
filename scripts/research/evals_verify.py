"""CPU-testable SDK Evals route smoke; real server execution is only through gpuq/yv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--out", type=Path, required=True)
    return p


def validate_result(result: dict) -> bool:
    return (
        result.get("complete") is True
        and result.get("passed") is True
        and result.get("routes") == 12
        and result.get("invocations") == 3
    )


def main(argv=None):
    a = parser().parse_args(argv)
    import httpx
    import openai
    from covaudit_session import Srv
    from route_checks import Ctx, evals_check

    home = a.out.parent / (a.out.stem + "-home")
    log = a.out.with_suffix(".server.log")
    srv = Srv(
        a.model,
        a.src,
        home,
        log,
        sets=[
            "YUNSHU_OMNI_PRELOAD=0",
            f"YUNSHU_EVALS_DIR={home / 'evals'}",
            f"YUNSHU_FILES_DIR={home / 'files'}",
            f"YUNSHU_CHAT_COMPLETIONS_DIR={home / 'chats'}",
            f"YUNSHU_VLM_APC_DISK_DIR={home / 'apc'}",
        ],
    )
    result = dict(complete=False, passed=False, routes=12, device="M5")
    try:
        srv.wait_ready()
        with httpx.Client(base_url=srv.url, timeout=180) as http:
            sdk = openai.OpenAI(base_url=srv.url + "/v1", api_key="x", http_client=http)
            c = Ctx(
                url=srv.url, token="", model=srv.model_id, kind="vlm", http=http, oa=sdk
            )
            evals_check(c)
            result.update(
                c.notes["evals"],
                complete=True,
                passed=True,
                model=srv.model_id,
                engaged="normal /v1/chat/completions",
                server_log=str(log),
            )
            if not validate_result(result):
                raise AssertionError(result)
        return 0
    except Exception as exc:
        result.update(complete=True, error=str(exc))
        print(srv.log_tail(60), flush=True)
        return 1
    finally:
        srv.kill()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, ensure_ascii=False) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
