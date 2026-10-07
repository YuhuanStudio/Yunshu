"""Real-model verification of /v1/decisions and /v1/systemone on a decision checkpoint.

Starts `yunshu serve --model <checkpoint>` on a gpuq port (18990-18999), drives it with the real
`openai` client and raw HTTP, writes a JSON report, and exits nonzero on the first structural
failure (fail closed: a check that did not run is a failure). Semantic checks (the right answer is
the likelier one) are recorded and also required.

    gpuq submit --label decisions-verify -- python scripts/research/decisions_verify.py MODEL OUT.json

The check logic (`run_checks`) takes ready clients, so a CPU unit test drives it against a fake
engine (tests/unit/test_decisions_verify.py, structural checks only).
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import time

PORT = 18990
NOISE_TOL = (
    0.02  # probability units: bf16 backbone noise allowed between identical-token runs
)
BASE = f"http://127.0.0.1:{PORT}"


class CheckError(Exception):
    pass


def expect(cond, msg):
    if not cond:
        raise CheckError(msg)


def png_data_url(color, size=96) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _probs(answer):
    return {p.value: p.probability for p in answer.probabilities}


def run_checks(oa, http, model, semantic=True, timings=None):
    """oa: openai client, http: httpx client with base_url. Returns {check: detail}."""
    out = {}
    timings = timings if timings is not None else {}

    # 1. predicates: a clearly true and a clearly false statement
    state = "The forecast says heavy rain all day; I left the house with an umbrella and wet shoes."
    t0 = time.perf_counter()
    d = oa.decisions.create(
        model=model,
        input=state,
        questions=[
            {"type": "predicate", "instructions": "Is it raining?", "name": "rain"},
            {
                "type": "predicate",
                "instructions": "Is the weather sunny and dry?",
                "name": "dry",
            },
        ],
    )
    timings["two_predicates_s"] = time.perf_counter() - t0
    expect(
        [a.type for a in d.answers] == ["predicate", "predicate"],
        "predicate answer types",
    )
    expect(
        [a.name for a in d.answers] == ["rain", "dry"], "answers not in question order"
    )
    for a in d.answers:
        expect(0.0 <= a.probability <= 1.0, f"probability {a.probability}")
    expect(d.usage.input_tokens > 0 and d.usage.output_tokens == 0, "usage")
    out["predicates"] = {a.name: round(a.probability, 4) for a in d.answers}
    if semantic:
        expect(
            d.answers[0].probability > 0.8,
            f"rain should be likely: {d.answers[0].probability}",
        )
        expect(
            d.answers[1].probability < 0.2,
            f"dry should be unlikely: {d.answers[1].probability}",
        )

    # 2. choice + score in one request; probabilities normalised; typed values
    review_neg = (
        "The blender broke after two uses and support never answered my emails."
    )
    review_pos = "Absolutely love it, works perfectly and arrived a day early."
    qs = [
        {
            "type": "choice",
            "instructions": "Route this message to the right team.",
            "name": "team",
            "choices": [
                {"value": "billing", "description": "payments, invoices, refunds"},
                {
                    "value": "support",
                    "description": "broken products and unanswered requests",
                },
                {"value": "sales", "description": "buying new products"},
            ],
        },
        {
            "type": "score",
            "instructions": "How satisfied is the customer?",
            "name": "sat",
            "levels": [
                {"label": "very unhappy", "description": "furious or hopeless"},
                {"label": "unhappy", "description": "disappointed"},
                {"label": "neutral", "description": "no strong feeling"},
                {"label": "happy", "description": "pleased"},
                {"label": "very happy", "description": "delighted"},
            ],
        },
    ]
    neg = oa.decisions.create(model=model, input=review_neg, questions=qs)
    pos = oa.decisions.create(model=model, input=review_pos, questions=qs)
    for r in (neg, pos):
        expect([a.type for a in r.answers] == ["choice", "score"], "choice/score types")
        ch, sc = r.answers
        expect(
            abs(sum(p.probability for p in ch.probabilities) - 1) < 1e-4,
            "choice probs not normalised",
        )
        expect(
            abs(sum(p.probability for p in sc.probabilities) - 1) < 1e-4,
            "score probs not normalised",
        )
        expect([p.value for p in sc.probabilities] == [0, 1, 2, 3, 4], "score values")
        expect(
            abs(sc.score - sum(p.value * p.probability for p in sc.probabilities))
            < 1e-4,
            "score != weighted mean",
        )
        expect(
            abs(ch.confidence - max(p.probability for p in ch.probabilities)) < 1e-9,
            "confidence",
        )
    out["negative"] = {
        "choice": neg.answers[0].choice,
        "score": round(neg.answers[1].score, 3),
    }
    out["positive"] = {
        "choice": pos.answers[0].choice,
        "score": round(pos.answers[1].score, 3),
    }
    if semantic:
        expect(neg.answers[0].choice == "support", f"routing: {neg.answers[0].choice}")
        expect(
            pos.answers[1].score > neg.answers[1].score + 1.5,
            f"score not monotone: {out}",
        )

    # 3. option order must not move the probabilities: the head sees choices sorted by id, so the
    #    token sequences are identical. Any difference is compute noise, measured and bounded here.
    #    (the head decides ALL questions jointly, so the whole question set is sent again)
    rev = [dict(qs[0], choices=list(reversed(qs[0]["choices"]))), qs[1]]
    r2 = oa.decisions.create(model=model, input=review_neg, questions=rev)
    a, b = _probs(neg.answers[0]), _probs(r2.answers[0])
    expect(a.keys() == b.keys(), f"order changed the values: {a} vs {b}")
    order_delta = max(abs(a[k] - b[k]) for k in a)
    out["order_invariance_max_prob_delta"] = order_delta

    # 4. the same request again: how repeatable is a request?
    r3 = oa.decisions.create(model=model, input=review_neg, questions=qs)
    c = _probs(r3.answers[0])
    repeat_delta = max(abs(a[k] - c[k]) for k in a)
    score_delta = abs(r3.answers[1].score - neg.answers[1].score)
    out["repeat_max_prob_delta"] = repeat_delta
    out["repeat_score_delta"] = score_delta
    expect(
        order_delta < NOISE_TOL,
        f"order moves probabilities by {order_delta}: {a} vs {b}",
    )
    expect(
        repeat_delta < NOISE_TOL and score_delta < NOISE_TOL,
        f"repeat differs: {repeat_delta} {score_delta}",
    )

    # 5. boolean choice values stay booleans
    rb = oa.decisions.create(
        model=model,
        input=review_pos,
        questions=[
            {
                "type": "choice",
                "instructions": "Would this customer recommend the product?",
                "choices": [
                    {"value": True, "description": "yes"},
                    {"value": False, "description": "no"},
                ],
            }
        ],
    )
    expect(
        isinstance(rb.answers[0].choice, bool),
        f"choice type {type(rb.answers[0].choice)}",
    )
    out["bool_choice"] = {
        "choice": rb.answers[0].choice,
        "confidence": round(rb.answers[0].confidence, 4),
    }
    if semantic:
        expect(rb.answers[0].choice is True, "positive review should be recommended")

    # 6. System One wire on the same engine
    body = {
        "model": model,
        "state": review_neg,
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Route this message to the right team.",
                "criteria": {c["value"]: c["description"] for c in qs[0]["choices"]},
            },
            "angry": {"type": "noul", "instructions": "Is the customer angry?"},
            "sat": {
                "type": "score",
                "instructions": "How satisfied is the customer?",
                "criteria": [lv["description"] for lv in qs[1]["levels"]],
            },
        },
    }
    r = http.post("/v1/systemone", json=body, timeout=300)
    expect(r.status_code == 200, f"systemone {r.status_code} {r.text[:200]}")
    j = r.json()
    expect(list(j["answers"]) == ["team", "angry", "sat"], "systemone answer keys")
    expect(
        j["answers"]["angry"]["type"] == "noul"
        and 0 <= j["answers"]["angry"]["noul"] <= 1,
        "noul",
    )
    expect(
        abs(sum(j["answers"]["team"]["probabilities"].values()) - 1) < 2e-3,
        "systemone choice probs",
    )
    out["systemone"] = {
        k: v.get("choice", v.get("noul", v.get("score")))
        for k, v in j["answers"].items()
    }
    if semantic:
        expect(
            j["answers"]["team"]["choice"] == "support",
            f"systemone routing {j['answers']['team']}",
        )

    # 7. images
    colors = [
        {
            "type": "choice",
            "instructions": "What is the main colour of the image?",
            "name": "color",
            "choices": [{"value": "red"}, {"value": "green"}, {"value": "blue"}],
        }
    ]
    img_res = {}
    for color in ("red", "blue"):
        r = http.post(
            "/v1/decisions",
            json={
                "model": model,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe the picture."},
                            {"type": "input_image", "image_url": png_data_url(color)},
                        ],
                    }
                ],
                "questions": colors,
            },
            timeout=300,
        )
        expect(r.status_code == 200, f"image request {r.status_code} {r.text[:300]}")
        img_res[color] = r.json()["answers"][0]["choice"]
        expect(
            r.json()["usage"]["input_tokens"] > 60, "image tokens missing from usage"
        )
    out["images"] = img_res
    if semantic:
        expect(img_res == {"red": "red", "blue": "blue"}, f"colours: {img_res}")

    # 8. errors: the model is not a chat model, bad requests are 400 in the OpenAI shape
    if (
        semantic
    ):  # needs the real server's engine registry; the fake app has no global engine
        r = http.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
            },
            timeout=60,
        )
        expect(
            400 <= r.status_code < 500,
            f"chat on a decision model -> {r.status_code} {r.text[:200]}",
        )
        out["chat_on_decision_model"] = r.status_code
    r = http.post(
        "/v1/decisions",
        json={"model": model, "input": "x", "questions": []},
        timeout=60,
    )
    expect(
        r.status_code == 400 and r.json()["error"]["message"],
        "empty questions not a 400",
    )
    r = http.post(
        "/v1/decisions",
        json={
            "model": model,
            "input": "x",
            "questions": [{"type": "predicate", "instructions": "q"}],
        },
        timeout=300,
    )
    expect(r.status_code == 200, "minimal request")

    # 9. the model card
    if semantic:
        r = http.get("/v1/models", timeout=60)
        expect(r.status_code == 200, "models")
        cards = r.json()["data"]
        expect(any(c["id"] for c in cards), "no models")
        out["model_ids"] = [c["id"] for c in cards]

    # 10. informal latency (CPU contention is not controlled here: indicative only)
    lat = {}
    for n in (1, 5):
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            oa.decisions.create(
                model=model,
                input=review_neg,
                questions=[
                    {
                        "type": "predicate",
                        "instructions": f"Statement {i}: is the customer upset?",
                    }
                    for i in range(n)
                ],
            )
            times.append(time.perf_counter() - t0)
        lat[f"{n}_questions_s"] = [round(t, 3) for t in times]
    out["latency_informal"] = lat
    out["timings"] = timings
    return out


def serve(model_path):
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "yunshu_cli",
            "serve",
            "--model",
            model_path,
            "--port",
            str(PORT),
        ],
        env=dict(os.environ, PYTHONPATH="python"),
        start_new_session=True,
    )


def main():
    import httpx
    import openai

    model_path, report = sys.argv[1], sys.argv[2]
    proc = serve(model_path)
    result = {"model_path": model_path, "port": PORT}
    rc = 1
    try:
        for _ in range(300):
            if proc.poll() is not None:
                raise CheckError(f"server exited early rc={proc.returncode}")
            try:
                if httpx.get(BASE + "/health/ready", timeout=3).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise CheckError("server never became ready")
        http = httpx.Client(base_url=BASE, timeout=300)
        oa = openai.OpenAI(api_key="k", base_url=BASE + "/v1", timeout=300)
        model = http.get("/v1/models").json()["data"][0]["id"]
        result["model"] = model
        result["checks"] = run_checks(oa, http, model)
        result["verdict"] = "PASS"
        rc = 0
    except BaseException as e:  # noqa: BLE001 - report then exit nonzero
        result["verdict"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {e}"
        print("FAIL", result["error"], flush=True)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        with open(report, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(json.dumps(result, indent=2, default=str), flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
