"""Same-checkpoint CPU/Core ML CPU+NE/MLX cross-encoder pilot; run through gpuq/yv.

No model imports at module import time. CPU unit tests exercise scoring metrics and
completion validation before the first job. Core ML CPU+NE excludes GPU but cannot
by itself prove every operation was delegated to ANE; report that limitation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

PAIRS = [
    (
        "What is the capital of France?",
        "The capital of France is Paris.",
        "France has many capitals in finance and venture capital.",
    ),
    (
        "How do I cancel an asyncio task?",
        "Call task.cancel() and handle asyncio.CancelledError.",
        "An asyncio task can run in the background without being cancelled.",
    ),
    (
        "What does HTTP 429 mean?",
        "HTTP status 429 means too many requests; retry after the specified delay.",
        "The server returned HTTP 200 after 429 milliseconds.",
    ),
    (
        "Where are Python virtual environment packages installed?",
        "Python packages are installed under the virtual environment's site-packages directory.",
        "A virtual environment is useful for installing Python itself globally.",
    ),
    (
        "Does BM25 require a neural network?",
        "BM25 is a lexical ranking function and does not require neural network inference.",
        "A neural network can be combined with BM25 for better ranking.",
    ),
    (
        "How can robots.txt disallow all crawling?",
        "Set User-agent: * followed by Disallow: / to disallow all crawling.",
        "A robots.txt file can allow crawling on all pages with Allow: /.",
    ),
    (
        "What is the boiling point of water at sea level?",
        "Water boils at 100 degrees Celsius at standard sea-level pressure.",
        "Water's boiling point changes with sea level and pressure.",
    ),
    (
        "Which HTTP header authenticates a bearer token?",
        "Send Authorization: Bearer followed by the token value.",
        "A token can also appear in an unrelated custom bearer header.",
    ),
    (
        "How do reciprocal rank fusion scores combine lists?",
        "RRF adds reciprocal rank terms 1/(k+rank) from each result list.",
        "The rank of each list can be represented as an integer score.",
    ),
    (
        "What is the difference between GET and POST?",
        "GET retrieves a resource; POST submits data for processing or creation.",
        "GET and POST are both HTTP request methods used by APIs.",
    ),
    (
        "How can I prevent SSRF DNS rebinding?",
        "Validate resolved addresses and pin the request connection to the checked address.",
        "DNS rebinding is a security risk that can bypass hostname-only checks.",
    ),
    (
        "How do I get raw content in Tavily search?",
        "Set include_raw_content to markdown or text to return cleaned full page content.",
        "The content field is a short search excerpt, not necessarily the raw page content.",
    ),
]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    return p


def metrics(scores, seconds):
    if len(scores) != len(PAIRS) or any(
        len(row) != 2 or not all(math.isfinite(value) for value in row)
        for row in scores
    ):
        raise ValueError("Incomplete or nonfinite backend scores")
    correct = sum(row[0] > row[1] for row in scores)
    return {
        "queries": len(scores),
        "correct": correct,
        "accuracy": correct / len(scores),
        "query_p50_ms": statistics.median(seconds) * 1000,
        "query_p95_ms": sorted(seconds)[math.ceil(len(seconds) * 0.95) - 1] * 1000,
    }


def validate(rows):
    if not rows or rows[-1].get("complete") is not True:
        return False, "Missing complete record"
    arms = {row.get("backend"): row for row in rows if "backend" in row}
    required = {"cpu", "coreml_cpu_ne", "mlx"}
    if not required <= arms.keys() or any(
        arms[name].get("status") != "ok" for name in required
    ):
        return False, "Not all backends completed successfully"
    return True, ""


def run(args):
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    rows = []

    def write(row):
        rows.append(row)
        with args.out.open("a") as out:
            out.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    args.out.write_text("")
    if args.dry_run:
        for backend in ("cpu", "coreml_cpu_ne", "mlx"):
            write(
                {
                    "backend": backend,
                    "status": "ok",
                    **metrics([[1, 0]] * len(PAIRS), [0.001] * len(PAIRS)),
                    "fake": True,
                }
            )
        write({"complete": True, "dry_run": True})
        return 0
    if not args.model.is_dir():
        raise ValueError("Checkpoint does not exist")
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            args.model, local_files_only=True, attn_implementation="eager"
        )
        .cpu()
        .eval()
    )

    class Wrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.model = wrapped

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=False,
            )[0]

    wrapper = Wrapper(model)
    encoded = [
        tokenizer(
            [query] * 2,
            [positive, negative],
            padding="max_length",
            max_length=96,
            truncation=True,
            return_tensors="pt",
        )
        for query, positive, negative in PAIRS
    ]
    checksum = hashlib.sha256(
        (args.model / "model.safetensors").read_bytes()
    ).hexdigest()
    scorers = {}
    failures = {}
    scorers["cpu"] = lambda inputs: (
        wrapper(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
        .detach()
        .numpy()
        .reshape(-1)
        .tolist()
    )
    try:
        import coremltools as ct

        example = encoded[0]
        traced = torch.jit.trace(
            wrapper,
            (
                example["input_ids"],
                example["attention_mask"],
                example["token_type_ids"],
            ),
            strict=False,
        )
        package = args.cache / (checksum[:12] + ".mlpackage")
        if not package.exists():
            converted = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(name=name, shape=(2, 96), dtype=np.int32)
                    for name in ("input_ids", "attention_mask", "token_type_ids")
                ],
                convert_to="mlprogram",
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                minimum_deployment_target=ct.target.macOS14,
            )
            converted.save(str(package))
        coreml = ct.models.MLModel(
            str(package), compute_units=ct.ComputeUnit.CPU_AND_NE
        )

        def ne(inputs):
            values = coreml.predict(
                {
                    name: inputs[name].numpy().astype(np.int32)
                    for name in ("input_ids", "attention_mask", "token_type_ids")
                }
            )
            return np.asarray(next(iter(values.values()))).reshape(-1).tolist()

        scorers["coreml_cpu_ne"] = ne
    except Exception as exc:
        failures["coreml_cpu_ne"] = f"{type(exc).__name__}: {exc}"
    engine = None
    loop = asyncio.new_event_loop()
    try:
        from yunshu_engine.scoring_engine import TextScoringEngine

        engine = TextScoringEngine(str(args.model))
        loop.run_until_complete(engine.start())

        # Exact same fixed tokenizer arrays and head, avoiding max-length/padding differences.
        def mlx_score(inputs):
            def score():
                import mlx.core as mx

                values = engine._model(
                    **{
                        name: mx.array(inputs[name].numpy())
                        for name in ("input_ids", "attention_mask", "token_type_ids")
                    }
                )
                mx.eval(values)
                return values.reshape(-1).tolist()

            return loop.run_until_complete(engine._run(score))

        scorers["mlx"] = mlx_score
    except Exception as exc:
        failures["mlx"] = f"{type(exc).__name__}: {exc}"
    samples = {backend: [] for backend in scorers}
    scores = {backend: None for backend in scorers}
    with torch.inference_mode():
        # Validate each first call before the timing matrix; warmups are excluded.
        for backend, scorer in list(scorers.items()):
            try:
                pilot = scorer(encoded[0])
                if len(pilot) != 2 or not all(math.isfinite(value) for value in pilot):
                    raise ValueError("Invalid pilot scores")
            except Exception as exc:
                failures[backend] = f"{type(exc).__name__}: {exc}"
                scorers.pop(backend)
        for _ in range(args.reps):
            for backend, scorer in scorers.items():
                current = []
                for inputs in encoded:
                    start = time.perf_counter()
                    current.append(scorer(inputs))
                    samples[backend].append(time.perf_counter() - start)
                scores[backend] = current
    for backend in ("cpu", "coreml_cpu_ne", "mlx"):
        if backend in failures:
            write({"backend": backend, "status": "failed", "error": failures[backend]})
        else:
            write(
                {
                    "backend": backend,
                    "status": "ok",
                    "scores": scores[backend],
                    **metrics(scores[backend], samples[backend]),
                    "checkpoint_sha256": checksum,
                    "device": "M5",
                    "compute_units": "CPU_AND_NE (GPU excluded; ANE delegation not independently proven)"
                    if backend == "coreml_cpu_ne"
                    else backend,
                }
            )
    loop.close()
    write(
        {
            "complete": True,
            "reps": args.reps,
            "device": "M5",
            "quality_set": "12 authored positive/negative pairs; pilot only, not a 200-item default gate",
        }
    )
    return 0 if validate(rows)[0] else 1


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
