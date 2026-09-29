"""Release-gate checks against a running Yunshu server (driven by scripts/release/gate.sh).

Every check appends one row, ``{"check", "status": PASS|FAIL|SKIP, "detail"}``, to
``--results`` and prints it. The subcommands are:

    record NAME STATUS DETAIL          add a row for a check gate.sh ran itself
    sdk    --url U --model M           OpenAI + Anthropic SDKs: chat, streaming, tools
                                       (auto, forced, streamed), JSON schema, stop,
                                       logprobs, reasoning split, image input
    cancel --url U --model M           drop a stream mid-generation; the next request
                                       must still be answered correctly
    long   --url U --model M --tokens N  one N-token prompt with a needle to recall
    family --url U --kinds chat,tools,...  short per-model smoke (see KINDS)
    summary                            the table of every row; exit 1 on any FAIL

``--model auto`` asks the server's /v1/models for its first id. The SDK checks need
``openai`` and ``anthropic``; gate.sh runs them in a throwaway ``uv run --with`` env.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather forecast for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["city", "days"],
        },
    },
}
TOOL_ANTHROPIC = {
    "name": "get_weather",
    "description": "Weather forecast for a city",
    "input_schema": TOOL["function"]["parameters"],
}
WEATHER_Q = "What's the weather in Oslo for the next 3 days? Use the tool."
KINDS = ("chat", "stream", "tools", "schema", "image", "ocr", "asr", "tts", "imagegen")


class Results:
    def __init__(self, path: Path, prefix: str = ""):
        self.path = path
        self.prefix = prefix
        path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, name: str, status: str, detail: str = "") -> bool:
        row = {
            "check": f"{self.prefix}{name}",
            "status": status,
            "detail": str(detail)[:500],
            "t": time.strftime("%H:%M:%S"),
        }
        with self.path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{status:4}  {row['check']}  {row['detail'][:160]}", flush=True)
        return status == "PASS"

    def run(self, name: str, fn) -> bool:
        t0 = time.perf_counter()
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001 - any error is a failed check
            ok, detail = False, f"{type(e).__name__}: {e}"
        detail = f"{detail} ({time.perf_counter() - t0:.1f}s)"
        return self.add(name, "PASS" if ok else "FAIL", detail)


def first_model(url: str) -> str:
    import httpx

    data = httpx.get(f"{url}/v1/models", timeout=30).json()
    return data["data"][0]["id"]


def red_left_png(size: int = 256) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (size, size // 2), "blue")
    ImageDraw.Draw(im).rectangle((0, 0, size // 2 - 1, size // 2 - 1), fill="red")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def text_png(text: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 72)
    except OSError:
        font = ImageFont.load_default()
    im = Image.new("RGB", (900, 200), "white")
    ImageDraw.Draw(im).text((40, 50), text, fill="black", font=font)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def openai_client(url: str):
    from openai import OpenAI

    return OpenAI(base_url=f"{url}/v1", api_key="gate", timeout=900, max_retries=0)


def user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# ── OpenAI / Anthropic SDK suite (tier-1 model) ──────────────────────────────


def sdk(a, res: Results) -> None:
    c = openai_client(a.url)
    m = a.model

    def chat():
        r = c.chat.completions.create(
            model=m,
            messages=user("What is 17 * 3? Reply with only the number."),
            max_tokens=32,
            temperature=0,
            extra_body=NO_THINK,
        )
        text = r.choices[0].message.content or ""
        return "51" in text, repr(text[:80])

    def stream():
        parts, chunks = [], 0
        for ch in c.chat.completions.create(
            model=m,
            messages=user("What is 17 * 3? Reply with only the number."),
            max_tokens=32,
            temperature=0,
            stream=True,
            extra_body=NO_THINK,
        ):
            chunks += 1
            if ch.choices and ch.choices[0].delta.content:
                parts.append(ch.choices[0].delta.content)
        text = "".join(parts)
        return "51" in text and chunks > 1, f"{chunks} chunks, {text[:80]!r}"

    def tool_args(call):
        args = json.loads(call.function.arguments)
        ok = (
            call.function.name == "get_weather"
            and "oslo" in str(args.get("city", "")).lower()
            and args.get("days") == 3
        )
        return ok, f"{call.function.name}({args})"

    def tools_auto():
        r = c.chat.completions.create(
            model=m,
            messages=user(WEATHER_Q),
            tools=[TOOL],
            max_tokens=512,
            temperature=0,
            extra_body=NO_THINK,
        )
        calls = r.choices[0].message.tool_calls or []
        if not calls:
            return False, f"no tool call; content={r.choices[0].message.content!r:.120}"
        return tool_args(calls[0])

    def tools_forced():
        r = c.chat.completions.create(
            model=m,
            messages=user("Oslo, 3 days."),
            tools=[TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            max_tokens=512,
            temperature=0,
            extra_body=NO_THINK,
        )
        calls = r.choices[0].message.tool_calls or []
        if not calls:
            return False, f"no tool call; content={r.choices[0].message.content!r:.120}"
        return tool_args(calls[0])

    def tools_stream():
        name, args, content, first_has_name = "", "", "", None
        for ch in c.chat.completions.create(
            model=m,
            messages=user(WEATHER_Q),
            tools=[TOOL],
            max_tokens=512,
            temperature=0,
            stream=True,
            extra_body=NO_THINK,
        ):
            if not ch.choices:
                continue
            d = ch.choices[0].delta
            content += d.content or ""
            for tc in d.tool_calls or []:
                if first_has_name is None:
                    first_has_name = bool(tc.function and tc.function.name)
                if tc.function and tc.function.name:
                    name = tc.function.name
                if tc.function and tc.function.arguments:
                    args += tc.function.arguments
        parsed = json.loads(args) if args else {}
        ok = (
            name == "get_weather"
            and first_has_name
            and "oslo" in str(parsed.get("city", "")).lower()
            and "<tool_call" not in content
        )
        return (
            ok,
            f"name={name} first_has_name={first_has_name} args={args[:80]} content={content[:60]!r}",
        )

    def schema():
        r = c.chat.completions.create(
            model=m,
            messages=user("What is 17 * 3? Put it in the answer field."),
            max_tokens=128,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "integer"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            },
            extra_body=NO_THINK,
        )
        text = r.choices[0].message.content or ""
        return json.loads(text).get("answer") == 51, repr(text[:80])

    def stop():
        r = c.chat.completions.create(
            model=m,
            messages=user("Count from 1 to 10, separated by spaces. Only the numbers."),
            max_tokens=64,
            temperature=0,
            stop=["5"],
            extra_body=NO_THINK,
        )
        text = r.choices[0].message.content or ""
        ok = "4" in text and "6" not in text and r.choices[0].finish_reason == "stop"
        return ok, f"{text!r:.80} finish={r.choices[0].finish_reason}"

    def logprobs():
        r = c.chat.completions.create(
            model=m,
            messages=user("Say hello."),
            max_tokens=8,
            temperature=0,
            logprobs=True,
            top_logprobs=3,
            extra_body=NO_THINK,
        )
        lp = r.choices[0].logprobs
        content = (lp.content if lp else None) or []
        ok = bool(content) and all(len(t.top_logprobs) == 3 for t in content)
        return ok, f"{len(content)} tokens with logprobs"

    def reasoning():
        r = c.chat.completions.create(
            model=m,
            messages=user("What is 12 + 30? Answer with the number."),
            max_tokens=4096,
            temperature=0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_effort": "low",
            },
        )
        msg = r.choices[0].message
        extra = msg.model_extra or {}
        thought = extra.get("reasoning_content") or extra.get("reasoning") or ""
        text = msg.content or ""
        ok = bool(thought) and "42" in text and "<think>" not in text
        return ok, f"reasoning {len(thought)} chars, content={text[-60:]!r}"

    def image():
        r = c.chat.completions.create(
            model=m,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url(red_left_png())},
                        },
                        {
                            "type": "text",
                            "text": "Which half of the image is red, left or right? One word.",
                        },
                    ],
                }
            ],
            max_tokens=16,
            temperature=0,
            extra_body=NO_THINK,
        )
        text = (r.choices[0].message.content or "").lower()
        return "left" in text and "right" not in text, repr(text[:60])

    from anthropic import Anthropic

    ac = Anthropic(base_url=a.url, api_key="gate", timeout=900, max_retries=0)

    def anth_chat():
        r = ac.messages.create(
            model=m,
            max_tokens=64,
            thinking={"type": "disabled"},
            messages=user("What is 17 * 3? Reply with only the number."),
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        return "51" in text, repr(text[:80])

    def anth_stream():
        with ac.messages.stream(
            model=m,
            max_tokens=64,
            thinking={"type": "disabled"},
            messages=user("What is 17 * 3? Reply with only the number."),
        ) as s:
            text = "".join(s.text_stream)
        return "51" in text, repr(text[:80])

    def anth_tools():
        r = ac.messages.create(
            model=m,
            max_tokens=512,
            thinking={"type": "disabled"},
            tools=[TOOL_ANTHROPIC],
            messages=user(WEATHER_Q),
        )
        uses = [b for b in r.content if b.type == "tool_use"]
        if not uses:
            return False, f"no tool_use; stop={r.stop_reason}"
        u = uses[0]
        ok = u.name == "get_weather" and "oslo" in str(u.input.get("city", "")).lower()
        return (
            ok and r.stop_reason == "tool_use",
            f"{u.name}({u.input}) stop={r.stop_reason}",
        )

    def anth_thinking():
        r = ac.messages.create(
            model=m,
            max_tokens=4096,
            thinking={"type": "enabled", "budget_tokens": 2048},
            messages=user("What is 12 + 30? Answer with the number."),
        )
        kinds = [b.type for b in r.content]
        text = "".join(b.text for b in r.content if b.type == "text")
        return (
            "thinking" in kinds and "42" in text,
            f"blocks={kinds} text={text[-40:]!r}",
        )

    for name, fn in [
        ("openai.chat", chat),
        ("openai.stream", stream),
        ("openai.tools_auto", tools_auto),
        ("openai.tools_forced", tools_forced),
        ("openai.tools_stream", tools_stream),
        ("openai.json_schema", schema),
        ("openai.stop", stop),
        ("openai.logprobs", logprobs),
        ("openai.reasoning_split", reasoning),
        ("openai.image_input", image),
        ("anthropic.chat", anth_chat),
        ("anthropic.stream", anth_stream),
        ("anthropic.tools", anth_tools),
        ("anthropic.thinking", anth_thinking),
    ]:
        res.run(name, fn)


def cancel(a, res: Results) -> None:
    import httpx

    def fn():
        body = {
            "model": a.model,
            "messages": user("Write a very long story about a lighthouse keeper."),
            "max_tokens": 3000,
            "stream": True,
            **NO_THINK,
        }
        got = 0
        with httpx.Client(timeout=600) as h:
            with h.stream("POST", f"{a.url}/v1/chat/completions", json=body) as r:
                for line in r.iter_lines():
                    if line.startswith("data: ") and '"content"' in line:
                        got += 1
                        if got >= 20:
                            break
        time.sleep(2)
        t0 = time.perf_counter()
        c = openai_client(a.url)
        rr = c.chat.completions.create(
            model=a.model,
            messages=user("What is the capital of France? One word."),
            max_tokens=16,
            temperature=0,
            extra_body=NO_THINK,
        )
        wall = time.perf_counter() - t0
        text = (rr.choices[0].message.content or "").lower()
        ok = got >= 20 and "paris" in text and wall < 60
        return (
            ok,
            f"dropped after {got} chunks; next answer {text!r:.40} in {wall:.1f}s",
        )

    res.run("cancel_then_next", fn)


def long(a, res: Results) -> None:
    def fn():
        sentence = "The river flows past the old mill and the fields stay green. "
        filler = sentence * (a.tokens // 13)
        prompt = (
            "Remember this: the secret code is ORCHID-42.\n\n"
            + filler
            + "\n\nWhat is the secret code? Reply with the code only."
        )
        c = openai_client(a.url)
        r = c.chat.completions.create(
            model=a.model,
            messages=user(prompt),
            max_tokens=32,
            temperature=0,
            extra_body=NO_THINK,
        )
        text = r.choices[0].message.content or ""
        n = r.usage.prompt_tokens if r.usage else 0
        return (
            "ORCHID-42" in text and n >= a.tokens * 0.9,
            f"{n} prompt tokens, {text!r:.40}",
        )

    res.run(f"long_prompt_{a.tokens // 1024}k", fn)


# ── per-model family smoke ────────────────────────────────────────────────────


def family(a, res: Results) -> None:
    import httpx

    m = a.model
    c = openai_client(a.url)
    h = httpx.Client(timeout=900)

    def chat():
        r = c.chat.completions.create(
            model=m,
            messages=user("What is the capital of France? Answer in one word."),
            max_tokens=256,
            temperature=0,
            extra_body=NO_THINK,
        )
        text = (r.choices[0].message.content or "").lower()
        return "paris" in text, repr(text[:80])

    def stream():
        parts = []
        for ch in c.chat.completions.create(
            model=m,
            messages=user("What is the capital of France? Answer in one word."),
            max_tokens=256,
            temperature=0,
            stream=True,
            extra_body=NO_THINK,
        ):
            if ch.choices and ch.choices[0].delta.content:
                parts.append(ch.choices[0].delta.content)
        text = "".join(parts).lower()
        return "paris" in text, repr(text[:80])

    def tools():
        r = c.chat.completions.create(
            model=m,
            messages=user(WEATHER_Q),
            tools=[TOOL],
            max_tokens=512,
            temperature=0,
            extra_body=NO_THINK,
        )
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            return False, f"no tool call; content={msg.content!r:.120}"
        args = json.loads(calls[0].function.arguments)
        ok = calls[0].function.name == "get_weather" and "oslo" in str(args).lower()
        return ok, f"{calls[0].function.name}({args})"

    def schema():
        r = c.chat.completions.create(
            model=m,
            messages=user(
                "Which city is the capital of France? Fill in the city field."
            ),
            max_tokens=128,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "city",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            },
            extra_body=NO_THINK,
        )
        text = r.choices[0].message.content or ""
        return "paris" in json.loads(text).get("city", "").lower(), repr(text[:80])

    def image():
        r = c.chat.completions.create(
            model=m,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url(red_left_png())},
                        },
                        {
                            "type": "text",
                            "text": "Which half of the image is red, left or right? One word.",
                        },
                    ],
                }
            ],
            max_tokens=64,
            temperature=0,
            extra_body=NO_THINK,
        )
        text = (r.choices[0].message.content or "").lower()
        return "left" in text and "right" not in text, repr(text[:60])

    def ocr():
        r = h.post(
            f"{a.url}/v1/ocr",
            files={"file": ("text.png", text_png("HELLO 2026"), "image/png")},
            data={"model": m},
        )
        r.raise_for_status()
        text = r.json().get("text", "")
        return "hello" in text.lower() and "2026" in text, repr(text[:80])

    def asr():
        if not a.audio or not Path(a.audio).exists():
            return False, f"no input audio at {a.audio}"
        with open(a.audio, "rb") as f:
            r = h.post(
                f"{a.url}/v1/audio/transcriptions",
                files={"file": ("speech.wav", f.read(), "audio/wav")},
                data={"model": m},
            )
        r.raise_for_status()
        text = r.json().get("text", "").lower()
        return "fox" in text and "dog" in text, repr(text[:80])

    def tts():
        r = h.post(
            f"{a.url}/v1/audio/speech",
            json={
                "model": m,
                "input": "Hello from Yunshu. This is a speech test.",
                "response_format": "wav",
                "instruct": "A calm, clear adult voice.",
            },
        )
        r.raise_for_status()
        body = r.content
        return body[:4] == b"RIFF" and len(
            body
        ) > 20000, f"{len(body)} bytes, head={body[:4]!r}"

    def imagegen():
        from PIL import Image

        r = h.post(
            f"{a.url}/v1/images/generations",
            json={
                "model": m,
                "prompt": "a red apple on a wooden table",
                "size": "512x512",
                "n": 1,
            },
        )
        r.raise_for_status()
        b64 = r.json()["data"][0]["b64_json"]
        im = Image.open(io.BytesIO(base64.b64decode(b64)))
        return im.size == (512, 512), f"image {im.size}"

    fns = {
        "chat": chat,
        "stream": stream,
        "tools": tools,
        "schema": schema,
        "image": image,
        "ocr": ocr,
        "asr": asr,
        "tts": tts,
        "imagegen": imagegen,
    }
    for kind in a.kinds.split(","):
        res.run(kind, fns[kind])


def summary(a) -> int:
    rows = [
        json.loads(line) for line in a.results.read_text().splitlines() if line.strip()
    ]
    width = max((len(r["check"]) for r in rows), default=10)
    print(f"\n{'check':<{width}}  status  detail")
    print("-" * (width + 60))
    for r in rows:
        print(f"{r['check']:<{width}}  {r['status']:<6}  {r['detail'][:100]}")
    counts = {s: sum(r["status"] == s for r in rows) for s in ("PASS", "FAIL", "SKIP")}
    print("-" * (width + 60))
    print(f"PASS {counts['PASS']}  FAIL {counts['FAIL']}  SKIP {counts['SKIP']}")
    if counts["FAIL"]:
        print("GATE: FAIL")
        return 1
    print("GATE: PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--prefix", default="", help="check-name prefix, e.g. 'serve-27b.'")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("name")
    r.add_argument("status", choices=("PASS", "FAIL", "SKIP"))
    r.add_argument("detail", nargs="?", default="")
    for name in ("sdk", "cancel", "long", "family"):
        p = sub.add_parser(name)
        p.add_argument("--url", required=True)
        p.add_argument("--model", default="auto")
        if name == "long":
            p.add_argument("--tokens", type=int, default=32768)
        if name == "family":
            p.add_argument("--kinds", required=True, help=f"comma list of {KINDS}")
            p.add_argument("--audio", help="speech WAV for the asr check")
    sub.add_parser("summary")
    a = ap.parse_args()
    res = Results(a.results, a.prefix)
    if a.cmd == "record":
        res.add(a.name, a.status, a.detail)
        return 0
    if a.cmd == "summary":
        return summary(a)
    if a.model == "auto":
        a.model = first_model(a.url)
    {"sdk": sdk, "cancel": cancel, "long": long, "family": family}[a.cmd](a, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
