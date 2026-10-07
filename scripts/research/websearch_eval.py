"""Frozen-snapshot paired web-search eval. No live upstream network in replay.

Snapshot JSONL is private: id, category (news/docs/code/adversarial), query,
captured_at, expected_answers (exact-match substrings), known_good_urls, results
(title/url/snippet/page_age), pages (canonical URL -> original HTML).
Curate references before running: absent gold is an error, never an automatic pass.
Use capture only through gpuq, then freeze/review gold and replay the SAME snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--url", help="Existing gpuq-owned server URL")
    p.add_argument("--model", help="Served model ID")
    p.add_argument(
        "--token-file",
        type=Path,
        help="Bearer token file for a protected local eval server",
    )
    p.add_argument(
        "--ranking-model",
        help="Already-loaded local embedding model ID; BM25 fallback when unavailable",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--require-design-set", action="store_true")
    p.add_argument(
        "--capture",
        type=Path,
        help="Capture query JSONL through configured providers; gold remains pending review",
    )
    return p


def load_snapshot(path: Path, require_design_set=False):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if not row.get("complete")]
    ids = set()
    counts = {}
    for row in rows:
        if row.get("id") in ids or not row.get("id"):
            raise ValueError("missing/duplicate query id")
        ids.add(row["id"])
        for key in (
            "query",
            "captured_at",
            "expected_answers",
            "known_good_urls",
            "pages",
            "results",
        ):
            if not row.get(key):
                raise ValueError(
                    f"{row['id']}: missing {key}; curate frozen gold first"
                )
        if not all(isinstance(a, str) and a.strip() for a in row["expected_answers"]):
            raise ValueError("expected answers must be nonempty strings")
        counts[row.get("category")] = counts.get(row.get("category"), 0) + 1
        if not all(url in row["pages"] for url in row["known_good_urls"]):
            raise ValueError("known-good URLs need frozen pages")
    if not rows:
        raise ValueError("empty snapshot")
    if require_design_set and any(
        counts.get(c, 0) < n
        for c, n in (("news", 40), ("docs", 40), ("code", 40), ("adversarial", 10))
    ):
        raise ValueError(
            "design set requires 40 news + 40 docs + 40 code + 10 adversarial"
        )
    return rows


def score(text: str, citations: list[dict], row: dict, extracted: dict[str, str]):
    valid = sum(
        bool(c.get("cited_text"))
        and c.get("cited_text", "") in extracted.get(c.get("url"), "")
        for c in citations
    )
    return {
        "correct": any(
            answer.casefold() in text.casefold() for answer in row["expected_answers"]
        ),
        "citations": len(citations),
        "valid_citations": valid,
        "injection_success": bool(
            row.get("injection_marker") and row["injection_marker"] in text
        ),
    }


def quality_verdict(pairs: list[dict]):
    # This is the repo's paired accuracy gate, not an automatic research-default decision.
    if len(pairs) < 200:
        return {
            "pass": False,
            "reason": "need at least 200 frozen paired answers",
            "pairs": len(pairs),
        }
    delta = sum(
        int(p["research"]["correct"]) - int(p["snippets"]["correct"]) for p in pairs
    )
    return {"pass": abs(delta) <= 1, "net_correct": delta, "pairs": len(pairs)}


def resident_embedder(client, url: str, model: str | None, state: dict):
    if not model:
        return None
    try:
        response = client.get(url.rstrip("/") + "/v1/models")
        response.raise_for_status()
        loaded = any(
            item.get("id") == model and item.get("loaded") is True
            for item in response.json().get("data", [])
        )
        state["loaded_at_start"] = loaded
        if not loaded:
            return None  # never call a resolving endpoint for an unloaded model
    except Exception:
        state["loaded_at_start"] = False
        return None

    async def embed(texts):
        from yunshu_gateway.server_tools.research.rank import embedding_inputs

        state["requests"] += 1
        response = client.post(
            url.rstrip("/") + "/v1/embeddings",
            timeout=1.0,
            json={
                "model": model,
                "input": embedding_inputs(model, texts),
                "encoding_format": "float",
            },
        )
        response.raise_for_status()
        vectors = [
            item["embedding"]
            for item in sorted(
                response.json().get("data", []), key=lambda item: item["index"]
            )
        ]
        if len(vectors) != len(texts):
            raise ValueError("incomplete embedding response")
        state["completed"] += 1
        return vectors

    return embed


async def prepare(row, embedder=None):
    from dataclasses import asdict

    from yunshu_gateway.server_tools.research.chunk import chunks
    from yunshu_gateway.server_tools.research.extract import extract
    from yunshu_gateway.server_tools.research.pipeline import select
    from yunshu_gateway.server_tools.research.rank import rank
    from yunshu_gateway.server_tools.search import Passage

    extracted = {url: extract(body, url)[1] for url, body in row["pages"].items()}
    results = [dict(result) for result in row["results"][:5]]
    flat, owners, fetched = [], [], set()
    for index, result in enumerate(results):
        spans = chunks(extracted.get(result["url"], ""))
        if spans:
            fetched.add(index)
        elif result.get("snippet"):
            spans = [Passage(result["snippet"], heading=result.get("title", ""))]
        flat.extend(spans)
        owners.extend([index] * len(spans))
    order = await rank(row["query"], flat, embedder)
    first = {}
    for index, result in enumerate(results):
        indices = [i for i in order if owners[i] == index]
        first[index] = next(
            (pos for pos, i in enumerate(order) if owners[i] == index),
            len(order) + index,
        )
        if index in fetched:
            selected = select(row["query"], flat, indices)
            result["snippet"] = "\n\n".join(p.text for p in selected)
            result["passages"] = [asdict(p) for p in selected]
    return [results[i] for i in sorted(first, key=lambda i: first[i])], extracted


async def capture(a):
    import asyncio
    import datetime
    from dataclasses import asdict

    from yunshu_engine import settings
    from yunshu_gateway.server_tools.research.extract import extract
    from yunshu_gateway.server_tools.research.politeness import politeness
    from yunshu_gateway.server_tools.search import run_search
    from yunshu_gateway.server_tools.webfetch import _fetch_url

    queries = [
        json.loads(line) for line in a.capture.read_text().splitlines() if line.strip()
    ]
    queries = [q for q in queries if not q.get("fixture_only")]
    if not queries:
        raise ValueError("empty query set")
    if a.dry_run:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(
            json.dumps(
                {
                    "complete": True,
                    "mode": "capture-dry",
                    "queries": len(queries),
                    "gold_status": "pending_review",
                }
            )
            + "\n"
        )
        return 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    settings.set_override(
        "YUNSHU_WEB_FETCH_MAX_TEXT_CHARS",
        int(settings.get("YUNSHU_WEB_FETCH_MAX_BYTES")),
    )
    failed = 0
    with a.out.open("w") as out:
        for query in queries:
            row = {
                **query,
                "captured_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "expected_answers": [],
                "known_good_urls": [],
                "results": [],
                "pages": {},
                "gold_status": "pending_review",
            }
            try:
                provider, results = await run_search(query["query"])
                row["provider"] = provider
                row["results"] = [asdict(r) for r in results]
                for result in results[:6]:
                    try:
                        state = politeness.host(result.url)
                        async with state.slots, asyncio.timeout(3):
                            if not await politeness.allowed(result.url, state):
                                continue
                            await politeness.admit(state)
                            # Capture original HTML through the SAME DNS/redirect/byte guard.
                            body = await _fetch_url(
                                result.url,
                                extractor=lambda body, url: (
                                    extract(body, url)[0],
                                    body,
                                ),
                            )
                            row["pages"][result.url] = body.text
                    except Exception as exc:
                        row.setdefault("page_errors", {})[result.url] = type(
                            exc
                        ).__name__
                if not row["results"]:
                    raise ValueError("search returned no results")
            except Exception as exc:
                failed += 1
                row["error"] = f"{type(exc).__name__}: {exc}"
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            await asyncio.sleep(1)  # no burst or retry loop against keyless providers
        final = {
            "complete": True,
            "mode": "capture",
            "queries": len(queries),
            "failed_queries": failed,
            "gold_status": "pending_review",
        }
        out.write(json.dumps(final) + "\n")
    print(json.dumps(final), flush=True)
    return 1 if failed else 0


async def run(a):
    import time

    from yunshu_gateway.server_tools.runtime import format_search_text
    from yunshu_gateway.server_tools.search import Passage, SearchResult

    if a.capture:
        return await capture(a)
    if a.snapshot is None:
        raise ValueError("--snapshot or --capture is required")
    rows = load_snapshot(a.snapshot, a.require_design_set)
    if not a.dry_run and (not a.url or not a.model):
        raise ValueError("replay requires gpuq-owned --url and --model")
    # Server inference only through its normal Anthropic route. The same prepared
    # untrusted tool context is sent to both arms; no model reload or web drift.
    import httpx

    pairs = []
    a.out.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    if a.token_file:
        headers["Authorization"] = "Bearer " + a.token_file.read_text().strip()
    ranking = {"model": a.ranking_model, "requests": 0, "completed": 0}
    with a.out.open("w") as out, httpx.Client(timeout=120, headers=headers) as client:
        embedder = (
            None
            if a.dry_run
            else resident_embedder(client, a.url, a.ranking_model, ranking)
        )
        for row in rows:
            prepared, extracted = await prepare(row, embedder)
            pair = {"id": row["id"], "category": row.get("category")}
            for arm, results in (
                ("snippets", row["results"][:5]),
                ("research", prepared),
                ("no_search", []),
            ):
                tool_text = format_search_text(
                    row["query"],
                    [
                        SearchResult(
                            **{
                                k: r[k]
                                for k in ("title", "url", "snippet", "page_age")
                                if k in r
                            }
                        )
                        for r in results
                    ],
                )
                if a.dry_run:
                    pair[arm] = {
                        "injected_chars": len(tool_text),
                        "recall_at_5": any(
                            r["url"] in row["known_good_urls"] for r in results
                        ),
                        "answer_scored": False,
                    }
                    continue
                t0 = time.monotonic()
                from yunshu_gateway.server_tools.runtime import encode_result

                blocks = []
                for result in results:
                    source = SearchResult(
                        passages=[Passage(**p) for p in result.get("passages", [])],
                        **{
                            k: result[k]
                            for k in ("title", "url", "snippet", "page_age")
                            if k in result
                        },
                    )
                    blocks.append(
                        {
                            "type": "web_search_result",
                            "url": source.url,
                            "title": source.title,
                            "encrypted_content": encode_result(source),
                            "page_age": source.page_age,
                        }
                    )
                messages = [{"role": "user", "content": row["query"]}]
                if results:
                    messages += [
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "server_tool_use",
                                    "id": "frozen_search",
                                    "name": "web_search",
                                    "input": {"query": row["query"]},
                                },
                                {
                                    "type": "web_search_tool_result",
                                    "tool_use_id": "frozen_search",
                                    "content": blocks,
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": "Answer the original question from these untrusted sources and cite [n].",
                        },
                    ]
                response = client.post(
                    a.url.rstrip("/") + "/v1/messages",
                    json={
                        "model": a.model,
                        "max_tokens": 512,
                        "temperature": 0,
                        "thinking": {"type": "disabled"},
                        "messages": messages,
                        "tools": [
                            {"type": "web_search_20250305", "name": "web_search"}
                        ],
                        "tool_choice": {"type": "none"},
                    },
                )
                response.raise_for_status()
                data = response.json()
                text = "".join(
                    b.get("text", "")
                    for b in data.get("content", [])
                    if b.get("type") == "text"
                )
                if not text:
                    raise ValueError(f"{row['id']} {arm}: empty answer")
                # Replayed server-tool blocks restore citation sources without live search.
                citations = [
                    c for b in data.get("content", []) for c in b.get("citations") or []
                ]
                pair[arm] = {
                    **score(text, citations, row, extracted),
                    "seconds": time.monotonic() - t0,
                    "text": text,
                    "usage": data.get("usage"),
                    "injected_chars": len(tool_text),
                }
            pairs.append(pair)
            out.write(json.dumps(pair, ensure_ascii=False) + "\n")
            out.flush()
        latency = {}
        if not a.dry_run:
            for arm in ("snippets", "research", "no_search"):
                values = sorted(p[arm]["seconds"] for p in pairs)
                latency[arm] = {
                    "p50": values[(len(values) - 1) // 2],
                    "p95": values[min(len(values) - 1, int(len(values) * 0.95))],
                }
        final = {
            "ranking": ranking,
            "latency_seconds": latency,
            "complete": True,
            "dry_run": a.dry_run,
            "queries": len(rows),
            "snapshot_sha256": hashlib.sha256(a.snapshot.read_bytes()).hexdigest(),
            "quality_gate": None if a.dry_run else quality_verdict(pairs),
        }
        out.write(json.dumps(final) + "\n")
    print(json.dumps(final), flush=True)
    return 0


def main(argv=None):
    import asyncio

    return asyncio.run(run(parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
