"""Why are two identical-token decision requests not bit-identical? Repeats one request on the
engine directly and reports the max difference of the backbone hidden states and of the head logits.

    gpuq submit --label decisions-probe-... -- python scripts/research/decisions_probe.py MODEL OUT.json
"""

from __future__ import annotations

import asyncio
import json
import sys


def max_abs_diff(a, b) -> float:
    return max((abs(x - y) for x, y in zip(a, b, strict=True)), default=0.0)


def flat(rows) -> list[float]:
    return [v for row in rows for v in row]


async def main(model_path: str, out: str) -> int:
    import mlx.core as mx

    from yunshu_engine import decision_engine as de

    engine = de.DecisionEngine(model_path)
    await engine.start()
    opts = lambda ids: tuple(de.Option(i, i, f"about {i}") for i in ids)  # noqa: E731
    qs = [
        de.Question(
            "predicate",
            "angry",
            "Is the customer angry?",
            (de.Option("true", True), de.Option("false", False)),
        ),
        de.Question(
            "choice",
            "team",
            "Route this message.",
            opts(["billing", "support", "sales"]),
        ),
    ]
    state = "The blender broke after two uses and support never answered my emails."
    rec = de.encode_record(engine._tokenize, state, qs)
    ids = mx.array([list(rec.input_ids)])

    def hidden_and_logits():
        feats = engine._model.get_input_embeddings(ids)
        h = engine._model.language_model.model(
            ids, inputs_embeds=feats.inputs_embeds, position_ids=feats.position_ids
        )[0]
        mx.eval(h)
        lg = de.head_logits(
            engine._head, engine._head_cfg, h, rec, engine._lexical_rows
        )
        return h.astype(mx.float32).reshape(-1).tolist(), flat(lg)

    runs = [await engine._run(hidden_and_logits) for _ in range(4)]
    result = {
        "tokens": len(rec.input_ids),
        "hidden_max_diff_vs_run0": [max_abs_diff(runs[0][0], r[0]) for r in runs[1:]],
        "logit_max_diff_vs_run0": [max_abs_diff(runs[0][1], r[1]) for r in runs[1:]],
        "logits_run0": runs[0][1],
    }
    print(json.dumps(result), flush=True)
    # the head alone on identical hidden states (arrays are built on the MLX thread)
    n = len(rec.input_ids)

    def head_only():
        h0 = mx.array(runs[0][0], dtype=mx.float32).reshape(n, -1)
        return flat(
            de.head_logits(
                engine._head, engine._head_cfg, h0, rec, engine._lexical_rows
            )
        )

    heads = [await engine._run(head_only) for _ in range(3)]
    result["head_only_max_diff"] = [max_abs_diff(heads[0], h) for h in heads[1:]]
    await engine.stop()
    print(json.dumps(result, indent=1), flush=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, "python")
    sys.exit(asyncio.run(main(sys.argv[1], sys.argv[2])))
