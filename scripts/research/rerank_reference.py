"""Independent Transformers CPU oracle for original reranking/classifier checkpoints.

No MLX imports. Qwen follows the Transformers model-card recipe; encoder heads
follow AutoModelForSequenceClassification (sigmoid for one label, softmax otherwise).
"""

from __future__ import annotations

import argparse
import json


def reference(model_dir, pairs=None, texts=None, instruction=None):
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    causal = not any(
        a.endswith("ForSequenceClassification") for a in config.architectures
    )
    cls = AutoModelForCausalLM if causal else AutoModelForSequenceClassification
    model = (
        cls.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager")
        .cpu()
        .eval()
    )
    out = []
    for item in pairs if pairs is not None else texts:
        if causal:
            # Deliberately independent from the production prompt/token helper.
            pre = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            post = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            p = tokenizer.encode(pre, add_special_tokens=False)
            s = tokenizer.encode(post, add_special_tokens=False)
            task = (
                instruction
                if instruction is not None
                else "Given a web search query, retrieve relevant passages that answer the query"
            )
            body = f"<Instruct>: {task}\n<Query>: {item[0]}\n<Document>: {item[1]}"
            ids = tokenizer(body, truncation=True, max_length=8192 - len(p) - len(s))[
                "input_ids"
            ]
            inputs = {"input_ids": torch.tensor([p + ids + s])}
        else:
            inputs = (
                tokenizer(item[0], item[1], truncation=True, return_tensors="pt")
                if pairs is not None
                else tokenizer(item, truncation=True, return_tensors="pt")
            )
        with torch.no_grad():
            logits = model(**inputs).logits.float()
        if causal:
            indices = [tokenizer.convert_tokens_to_ids(t) for t in ("no", "yes")]
            out.append(float(logits[0, -1, indices].softmax(-1)[1]))
        elif pairs is not None:
            if logits.shape[-1] != 1:
                raise ValueError("Scoring requires num_labels=1")
            out.append(float(logits[0, 0].sigmoid()))
        else:
            probs = (
                logits.sigmoid()
                if logits.shape[-1] == 1
                or config.problem_type == "multi_label_classification"
                else logits.softmax(-1)
            )
            out.append(probs[0].tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument(
        "--items",
        required=True,
        help="JSON {pairs: [[query, doc], ...]} or {texts: [...]} ",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.items) as f:
        items = json.load(f)
    scores = reference(args.model_dir, **items)
    with open(args.out, "w") as f:
        json.dump({"scores": scores, "complete": True, "device": "cpu"}, f)


if __name__ == "__main__":
    main()
