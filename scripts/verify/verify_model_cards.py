"""Real-server smoke test: model cards through /v1/models with the openai and anthropic SDKs.

Starts a multi-model server over a scratch models dir (symlinks to a text LLM and an OCR
checkpoint), then checks list / retrieve / a chat request that loads the LLM / the loaded
state, and the Ollama /api/show layer. Run with a python that has `openai` and `anthropic`
installed; the server itself runs with YUNSHU_PYTHON (default: this interpreter).

    YUNSHU_PYTHON=.venv/bin/python $SDK_VENV/bin/python \
        scripts/verify/verify_model_cards.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

PORT = 18997
BASE = f"http://127.0.0.1:{PORT}"
MODELS = Path(os.environ.get("YUNSHU_TEST_MODELS", "~/.yunshu/models")).expanduser()
LLM = os.environ.get("YUNSHU_SMOKE_LLM", "Qwen2.5-3B-Instruct-4bit")


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


def main() -> None:
    import anthropic
    import openai

    tmp = Path(tempfile.mkdtemp(prefix="yunshu-cards-"))
    for name in (LLM, "GLM-OCR-bf16"):
        (tmp / name).symlink_to(MODELS / name)
    env = dict(
        os.environ,
        YUNSHU_MULTI_MODEL="1",
        YUNSHU_MODELS_DIR=str(tmp),
        YUNSHU_AUTH_DISABLED="true",
        YUNSHU_ALLOW_AUTO_LOAD="1",
        NO_PROXY="127.0.0.1,localhost",
        PYTHONPATH=os.path.abspath("python"),
    )
    py = os.environ.get("YUNSHU_PYTHON", sys.executable)
    proc = subprocess.Popen(
        [
            py,
            "-m",
            "uvicorn",
            "yunshu_gateway.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ],
        env=env,
    )
    try:
        for _ in range(240):
            try:
                if httpx.get(f"{BASE}/health/live", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            raise SystemExit("server did not start")

        oa = openai.OpenAI(base_url=f"{BASE}/v1", api_key="x")
        an = anthropic.Anthropic(base_url=BASE, api_key="x")

        # OpenAI SDK: typed parse + the extension fields survive as extras.
        listing = oa.models.list()
        ids = {m.id for m in listing.data}
        check({LLM, "GLM-OCR-bf16"} <= ids, f"openai models.list ids {sorted(ids)}")
        m = oa.models.retrieve(LLM)
        extra = m.model_extra or {}
        check(
            m.object == "model" and m.owned_by == "yunshu",
            "openai retrieve spec fields",
        )
        check(
            extra["context_length"] == 32768 and extra["max_model_len"] == 32768,
            "context_length / max_model_len",
        )
        check(
            extra["architecture"]["input_modalities"] == ["text"],
            "architecture.input_modalities",
        )
        check(
            "tools" in extra["supported_parameters"], "supported_parameters lists tools"
        )
        check(
            extra["yunshu"]["kind"] == "chat"
            and extra["yunshu"]["quantization"]["bits"] == 4,
            "yunshu card nested",
        )
        ocr = oa.models.retrieve("GLM-OCR-bf16").model_extra
        check(
            ocr["yunshu"]["kind"] == "ocr"
            and ocr["architecture"]["output_modalities"] == ["text"],
            "OCR card",
        )

        # Anthropic SDK: ModelInfo parses (capabilities object, max_input_tokens, created_at datetime).
        am = an.models.retrieve(LLM)
        check(
            am.type == "model" and am.display_name == LLM and am.created_at is not None,
            "anthropic ModelInfo",
        )
        check(
            am.max_input_tokens == 32768 and am.max_tokens == 32768,
            "anthropic max_input_tokens / max_tokens",
        )
        check(
            am.capabilities.image_input.supported is False
            and am.capabilities.thinking.supported is False,
            "anthropic capabilities",
        )
        page = list(an.models.list())
        check({x.id for x in page} >= {LLM, "GLM-OCR-bf16"}, "anthropic models.list")

        # Chat request loads the LLM; the card then reports it as loaded.
        r = oa.chat.completions.create(
            model=LLM, messages=[{"role": "user", "content": "Say hi."}], max_tokens=8
        )
        check(bool(r.choices[0].message.content), "chat completion served")
        after = oa.models.retrieve(LLM).model_extra
        check(
            after["state"] == "loaded" and after["yunshu"]["state"]["loaded"] is True,
            "state=loaded after use",
        )
        check(after["yunshu"]["memory"]["weights_bytes"] > 1e9, "memory footprint")

        # Ollama /api/show reads the card.
        show = httpx.post(f"{BASE}/api/show", json={"model": LLM}, timeout=30).json()
        check(
            show["capabilities"] == ["completion", "tools"],
            f"ollama capabilities {show['capabilities']}",
        )
        check(
            show["model_info"]["qwen2.context_length"] == 32768,
            "ollama model_info context_length",
        )

        print(json.dumps(oa.models.retrieve(LLM).model_dump(), indent=1)[:2500])
    finally:
        proc.kill()
        proc.wait()
        subprocess.run(["rm", "-rf", str(tmp)], check=False)


if __name__ == "__main__":
    main()
