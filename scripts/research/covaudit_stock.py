"""Server output against stock mlx-lm / mlx-vlm on the same checkpoint (coverage audit).

Generic (non tier-1-tuned) paths: a text-only mlx-lm model on the single-request fast path, a
dense Qwen3.5-family VLM and a generic mlx-vlm model with an image. Greedy, `--tokens` new tokens,
prompt of about `--doc-tokens` tokens. The server runs first (prefix cache off, isolated HOME,
always kill -9'd), then the stock library generates in this process. PASS: the texts are equal, or
they agree for >= `--min-agree` characters before a near-tie divergence and both answer the needle.

    python covaudit_stock.py run --model PATH --kind lm|vlm [--image] --out FILE.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from covaudit_conc import post  # noqa: E402
from covaudit_session import Srv, file_text, needle  # noqa: E402


def png_b64() -> str:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([40, 40, 260, 260], fill=(200, 30, 30))
    d.ellipse([280, 280, 420, 420], fill=(30, 30, 200))
    b = io.BytesIO()
    im.save(b, "PNG")
    return base64.b64encode(b.getvalue()).decode()


def question(doc_tokens: int, image: bool) -> str:
    q = f"<file>\n{file_text(5, doc_tokens)}\n</file>\n"
    q += "State the SECRET_CODE of the file above, then list the names of the first 12 functions in it, one per line"
    q += ", then describe the image in two sentences." if image else "."
    return q


def agree(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def judge(server: str, stock: str, min_agree: int, image: bool) -> list:
    """Failure reasons (pure; unit-tested)."""
    bad = []
    if not server.strip():
        bad.append("server reply empty")
    if not stock.strip():
        bad.append("stock reply empty")
    if bad:
        return bad
    for tag, t in (("server", server), ("stock", stock)):
        if needle(5) not in t:
            bad.append(f"{tag} did not recall {needle(5)}: {t[:80]!r}")
    n = agree(server, stock)
    if min(len(server), len(stock)) < min_agree and server == stock:
        bad.append(
            f"reply of {len(server)} chars is too short to compare (< {min_agree})"
        )
    if server != stock and n < min_agree:
        bad.append(
            f"diverge at char {n} (< {min_agree}): {server[n : n + 40]!r} vs {stock[n : n + 40]!r}"
        )
    return bad


def stock_lm(path: str, prompt_msgs: list, tokens: int) -> str:
    from mlx_lm import generate, load

    model, tok = load(path)
    prompt = tok.apply_chat_template(
        prompt_msgs, add_generation_prompt=True, tokenize=False
    )
    return generate(model, tok, prompt, max_tokens=tokens, verbose=False)


def stock_vlm(
    path: str, text: str, img_file: str | None, tokens: int, think_kw: dict
) -> str:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    model, proc = load(path)
    prompt = apply_chat_template(
        proc, model.config, text, num_images=1 if img_file else 0, **think_kw
    )
    r = generate(
        model,
        proc,
        prompt,
        image=[img_file] if img_file else None,
        max_tokens=tokens,
        temperature=0.0,
        verbose=False,
    )
    return r.text if hasattr(r, "text") else str(r)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", choices=["lm", "vlm"], required=True)
    ap.add_argument("--image", action="store_true")
    ap.add_argument("--src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--doc-tokens", type=int, default=4000)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--min-agree", type=int, default=150)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = question(a.doc_tokens, a.image)
    img_file = None
    content: object = text
    if a.image:
        img_file = str(out.with_suffix(".png"))
        Path(img_file).write_bytes(base64.b64decode(png_b64()))
        content = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + png_b64()},
            },
            {"type": "text", "text": text},
        ]
    srv = Srv(
        a.model,
        a.src,
        Path(f"/Volumes/P5Plus/yunshu-build/covaudit/home-{out.stem}"),
        out.with_suffix(".server.log"),
        ["YUNSHU_VLM_APC_MEMORY_GB=0", "YUNSHU_VLM_APC_DISK=0"],
    )
    try:
        srv.wait_ready()
        body = {
            "model": srv.model_id,
            "temperature": 0,
            "max_tokens": a.tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": content}],
        }
        r = post(srv.url, body)
        server_text = r["text"]
        print("server", r["finish"], r["usage"], repr(server_text[:80]), flush=True)
    except BaseException as e:
        print(f"FAIL: server {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    finally:
        srv.kill()
    try:
        if a.kind == "lm":
            stock = stock_lm(a.model, [{"role": "user", "content": text}], a.tokens)
        else:
            stock = stock_vlm(
                a.model, text, img_file, a.tokens, {"enable_thinking": False}
            )
    except BaseException as e:
        print(f"FAIL: stock {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    bad = judge(server_text, stock, a.min_agree, a.image)
    out.write_text(
        json.dumps(
            {
                "server": server_text,
                "stock": stock,
                "agree_chars": agree(server_text, stock),
                "bad": bad,
            }
        )
    )
    print("agree chars", agree(server_text, stock), "of", len(server_text), len(stock))
    for b in bad:
        print("JUDGE FAIL:", b)
    print("RESULT", "FAIL" if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
