"""Real-model smoke for the API-surface gap fixes.

    serve_and_run.sh MODEL PORT smoke_api_gaps.py PORT MODE

MODE: big (27B: logprob check only), vlm (Qwen3.5 via the VLM runner), text (mlx-lm fast path), embed (embedding-only), image.
Uses the openai / anthropic / ollama SDKs where installed and raw httpx otherwise.
Exit code 1 if any check fails.
"""

from __future__ import annotations

import base64
import io
import json
import sys

import httpx

port = sys.argv[1]
mode = sys.argv[2]
B = f"http://localhost:{port}"
FAILED = 0


def chk(name, fn):
    global FAILED
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {e}"
    if not ok:
        FAILED += 1
    print(
        ("PASS " if ok else "FAIL ")
        + name
        + " :: "
        + str(detail)[:220].replace("\n", " "),
        flush=True,
    )


def post(path, body, t=180):
    return httpx.post(B + path, json=body, timeout=t)


mid = httpx.get(B + "/v1/models").json()["data"][0]["id"]
print("MODEL", mid, "MODE", mode, flush=True)


def red_png_b64() -> str:
    from PIL import Image

    b = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(b, "PNG")
    return base64.b64encode(b.getvalue()).decode()


def version():
    v = httpx.get(B + "/version").json()
    return v.get("version") == "0.1.1", v


chk("version 0.1.1", version)


def logprob_consistency():
    r = post(
        "/v1/chat/completions",
        {
            "model": mid,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 12,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 3,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ).json()
    items = r["choices"][0]["logprobs"]["content"]
    bad = []
    for it in items:
        if "\ufffd" in it["token"]:
            continue  # partial UTF-8 byte tokens share one decoded string
        tops = {t["token"]: t["logprob"] for t in it["top_logprobs"]}
        lp = it["logprob"]
        if not (
            lp <= 0.0 and it["token"] in tops and abs(tops[it["token"]] - lp) < 1e-3
        ):
            bad.append((it["token"], lp, tops))
    zeros = sum(
        1
        for it in items
        if it["logprob"] == 0.0
        and len(it["top_logprobs"]) > 1
        and it["top_logprobs"][1]["logprob"] > -8
    )
    return not bad and zeros == 0 and len(items) > 0, (
        len(items),
        bad[:2],
        [round(i["logprob"], 4) for i in items],
    )


if mode in ("vlm", "big"):
    chk(
        "chat logprob == own top_logprobs entry, not rounded to 0.0",
        logprob_consistency,
    )

if mode == "vlm":

    def comp_lp():
        r = post(
            "/v1/completions",
            {
                "model": mid,
                "prompt": "The capital of France is",
                "max_tokens": 6,
                "temperature": 0,
                "logprobs": 3,
            },
        ).json()
        lp = r["choices"][0].get("logprobs")
        ok = (
            bool(lp)
            and len(lp["tokens"]) == len(lp["token_logprobs"]) > 0
            and len(lp["top_logprobs"][0]) >= 1
        )
        return ok, lp

    chk("completions logprobs (VLM runner)", comp_lp)

    def comp_lp_stream():
        n = 0
        with httpx.stream(
            "POST",
            B + "/v1/completions",
            json={
                "model": mid,
                "prompt": "Count: 1 2 3",
                "max_tokens": 6,
                "temperature": 0,
                "logprobs": 2,
                "stream": True,
            },
            timeout=180,
        ) as r:
            for line in r.iter_lines():
                if line.startswith("data:") and "[DONE]" not in line:
                    ch = json.loads(line[5:])["choices"]
                    if ch and ch[0].get("logprobs"):
                        n += 1
        return n > 0, f"{n} chunks with logprobs"

    chk("completions logprobs stream (VLM runner)", comp_lp_stream)

    def comp_nolp():
        r = post(
            "/v1/completions",
            {"model": mid, "prompt": "Hi", "max_tokens": 4, "temperature": 0},
        ).json()
        return r["choices"][0].get("logprobs") in (None, {}), r["choices"][0].get(
            "logprobs"
        )

    chk("completions without logprobs stays null", comp_nolp)

    import ollama

    oc = ollama.Client(host=B, timeout=180)
    SCH = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }

    def fmt_json():
        r = oc.chat(
            model=mid,
            messages=[
                {
                    "role": "user",
                    "content": "Return JSON with name and age of Alice, 30",
                }
            ],
            format="json",
            options={"num_predict": 1500},
        )
        return isinstance(json.loads(r["message"]["content"]), dict), r["message"][
            "content"
        ]

    chk("ollama chat format=json (VLM)", fmt_json)

    def fmt_schema():
        r = oc.chat(
            model=mid,
            messages=[{"role": "user", "content": "Alice is 30"}],
            format=SCH,
            options={"num_predict": 1500},
        )
        return set(json.loads(r["message"]["content"])) == {"name", "age"}, r[
            "message"
        ]["content"]

    chk("ollama chat format=schema (VLM)", fmt_schema)

    def gen():
        r = oc.generate(
            model=mid,
            prompt="The capital of France is",
            options={"num_predict": 400, "temperature": 0},
        )
        return r["done"] and "paris" in (
            r["response"] + (r.get("thinking") or "")
        ).lower(), (r["response"], r["done_reason"])

    chk("ollama generate (VLM)", gen)

    def gen_nothink():
        r = oc.generate(
            model=mid,
            prompt="The capital of France is",
            think=False,
            options={"num_predict": 16, "temperature": 0},
        )
        return "paris" in r["response"].lower(), r["response"]

    chk("ollama generate think=false (VLM)", gen_nothink)

    def gen_stream():
        ch = list(
            oc.generate(
                model=mid,
                prompt="Count to five:",
                think=False,
                stream=True,
                options={"num_predict": 20},
            )
        )
        return ch[-1]["done"] and len(ch) > 2, len(ch)

    chk("ollama generate stream (VLM)", gen_stream)

    def gen_img():
        r = oc.generate(
            model=mid,
            prompt="What color is this image? one word",
            images=[base64.b64decode(red_png_b64())],
            think=False,
            options={"num_predict": 16},
        )
        return "red" in r["response"].lower(), r["response"]

    chk("ollama generate with image (VLM)", gen_img)

if mode == "text":
    png = red_png_b64()

    def anth_img():
        r = post(
            "/v1/messages",
            {
                "model": mid,
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what color"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": png,
                                },
                            },
                        ],
                    }
                ],
            },
        )
        return r.status_code == 400 and r.json()["error"][
            "type"
        ] == "invalid_request_error", (r.status_code, r.text[:200])

    chk("anthropic image block on text model -> 400", anth_img)

    def oai_img():
        r = post(
            "/v1/chat/completions",
            {
                "model": mid,
                "max_tokens": 8,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "x"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + png},
                            },
                        ],
                    }
                ],
            },
        )
        return r.status_code == 400, (r.status_code, r.text[:160])

    chk("openai image_url on text model -> 400", oai_img)

    import ollama

    oc = ollama.Client(host=B, timeout=180)

    def oll_img():
        try:
            oc.chat(
                model=mid,
                messages=[
                    {
                        "role": "user",
                        "content": "what color",
                        "images": [base64.b64decode(png)],
                    }
                ],
            )
        except ollama.ResponseError as e:
            return e.status_code == 400 and "image" in e.error.lower(), (
                e.status_code,
                e.error,
            )
        return False, "no error"

    chk("ollama image on text model -> 400", oll_img)

    TOOL = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather of a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }

    def oll_tools():
        r = oc.chat(
            model=mid,
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[TOOL],
            options={"num_predict": 300, "temperature": 0},
        )
        tc = r["message"].get("tool_calls")
        ok = (
            bool(tc)
            and tc[0]["function"]["name"] == "get_weather"
            and isinstance(tc[0]["function"]["arguments"], dict)
        )
        return ok, tc or r["message"]["content"]

    chk("ollama tool_call (text model)", oll_tools)

    def oll_tools_stream():
        got = []
        for ch in oc.chat(
            model=mid,
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[TOOL],
            stream=True,
            options={"num_predict": 300, "temperature": 0},
        ):
            got += ch["message"].get("tool_calls") or []
        return bool(got) and got[0]["function"]["name"] == "get_weather", got

    chk("ollama tool_call stream (text model)", oll_tools_stream)

    def oll_tool_roundtrip():
        m = [{"role": "user", "content": "What is the weather in Paris?"}]
        r = oc.chat(
            model=mid,
            messages=m,
            tools=[TOOL],
            options={"num_predict": 300, "temperature": 0},
        )
        tc = r["message"]["tool_calls"][0]
        m.append(
            {
                "role": "assistant",
                "content": r["message"].get("content") or "",
                "tool_calls": [
                    {
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        }
                    }
                ],
            }
        )
        m.append({"role": "tool", "content": "sunny 22C", "tool_name": "get_weather"})
        r2 = oc.chat(model=mid, messages=m, tools=[TOOL], options={"num_predict": 200})
        t = r2["message"]["content"].lower()
        return "sunny" in t or "22" in t, t

    chk("ollama tool round trip (text model)", oll_tool_roundtrip)

if mode == "embed":
    import ollama

    oc = ollama.Client(host=B, timeout=180)
    r = post(
        "/v1/chat/completions",
        {
            "model": mid,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        },
    )
    chk(
        "chat on embedding model -> 400",
        lambda: (
            r.status_code == 400 and "/v1/embeddings" in r.text,
            (r.status_code, r.text[:200]),
        ),
    )
    r = post("/v1/completions", {"model": mid, "prompt": "hi", "max_tokens": 8})
    chk(
        "completions on embedding model -> 400",
        lambda: (
            r.status_code == 400 and "/v1/embeddings" in r.text,
            (r.status_code, r.text[:200]),
        ),
    )
    r = post(
        "/v1/messages",
        {
            "model": mid,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    chk(
        "anthropic on embedding model -> 400",
        lambda: (r.status_code == 400, (r.status_code, r.text[:200])),
    )

    def oll_chat():
        try:
            oc.chat(model=mid, messages=[{"role": "user", "content": "hi"}])
        except ollama.ResponseError as e:
            return e.status_code == 400 and "/api/embed" in e.error, (
                e.status_code,
                e.error,
            )
        return False, "no error"

    chk("ollama chat on embedding model -> 400", oll_chat)

    def oll_gen():
        try:
            oc.generate(model=mid, prompt="hi")
        except ollama.ResponseError as e:
            return e.status_code == 400 and "/api/embed" in e.error, (
                e.status_code,
                e.error,
            )
        return False, "no error"

    chk("ollama generate on embedding model -> 400", oll_gen)
    chk(
        "ollama embed still works",
        lambda: (
            lambda e: (
                len(e["embeddings"]) == 1 and len(e["embeddings"][0]) > 8,
                len(e["embeddings"][0]),
            )
        )(oc.embed(model=mid, input="hello")),
    )
    chk(
        "openai embeddings still work",
        lambda: (
            lambda j: (
                len(j["data"][0]["embedding"]) > 8,
                len(j["data"][0]["embedding"]),
            )
        )(post("/v1/embeddings", {"model": mid, "input": "hello"}).json()),
    )

if mode == "image":

    def gen_image():
        r = post(
            "/v1/images/generations",
            {
                "model": mid,
                "prompt": "a red apple on a table",
                "size": "256x256",
                "num_inference_steps": 2,
                "seed": 1,
            },
            t=290,
        )
        if r.status_code != 200:
            return False, (r.status_code, r.text[:200])
        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(r.json()["data"][0]["b64_json"])))
        return img.size == (256, 256), img.size

    chk("image generation (Z-Image)", gen_image)

print("FAILED", FAILED, flush=True)
sys.exit(1 if FAILED else 0)
