"""Small-model web tool smoke; all upstream network is a loopback fixture.

Run only through gpuq/yv. CPU tests exercise parsing and fail-closed evidence first.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--embedding-model", type=Path)
    p.add_argument("--eval-snapshot", type=Path)
    return p


def validate(rows):
    if not rows or rows[-1].get("complete") is not True:
        return False, "missing final complete record"
    checks = [r for r in rows if r.get("check")]
    if not checks or any(r.get("pass") is not True for r in checks):
        return False, "missing or failed web tool checks"
    return True, ""


def load_dependencies():
    # route_checks registers tool checks at its footer; importing tools first is a cycle.
    import route_checks  # noqa: F401
    from covaudit_session import Srv, free_port
    from m3sweep_jobs import routes_make_ctx
    from route_checks_tools import FakeBackend, _web_search_provider

    return Srv, free_port, routes_make_ctx, FakeBackend, _web_search_provider


def main(argv=None):
    a = parser().parse_args(argv)
    Srv, free_port, routes_make_ctx, FakeBackend, _web_search_provider = (
        load_dependencies()
    )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def record(row):
        rows.append(row)
        with a.out.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    a.out.write_text("")
    srv = None
    fake = FakeBackend(free_port(), page_results=True)
    home = a.out.parent / (a.out.stem + "-home")
    os.environ.setdefault("XDG_CACHE_HOME", str(home / "cache"))
    os.environ.setdefault("HF_HOME", "/Volumes/P5Plus/hf-cache")
    try:
        opts = [
            f"YUNSHU_SEARXNG_URL={fake.url}",
            "YUNSHU_WEB_SEARCH_PROVIDER=searxng",
            "YUNSHU_VLM_APC_DISK=0",
            "YUNSHU_WEB_FETCH_ALLOW_PRIVATE=1",
        ]
        if not a.baseline:
            opts += ["YUNSHU_WEB_RESEARCH=1"]
        token = "websearch-fixture-token"
        models_dir = None
        if a.embedding_model:
            if not a.embedding_model.is_dir():
                raise ValueError("embedding checkpoint is missing")
            models_dir = home / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            for model_path in (Path(a.model), a.embedding_model):
                link = models_dir / model_path.name
                if not link.exists():
                    link.symlink_to(model_path)
            opts += [
                f"YUNSHU_AUTH_TOKEN={token}",
                f"YUNSHU_WEB_RESEARCH_MODEL={a.embedding_model.name}",
            ]
        srv = Srv(
            a.model,
            a.src,
            home,
            a.out.with_suffix(".server.log"),
            opts,
            models_dir=str(models_dir) if models_dir else None,
            token=token if models_dir else None,
        )
        srv.wait_ready()
        ctx = routes_make_ctx(
            srv,
            token if models_dir else None,
            "multi",
            model_id=Path(a.model).name if models_dir else None,
        )
        ctx.fake = fake
        t0 = time.monotonic()
        _web_search_provider(ctx)
        record(
            {
                "check": "route_checks_tools.web_search_provider",
                "pass": True,
                "search_requests": len(fake.searches),
                "seconds": time.monotonic() - t0,
                "device": "M5",
                "notes": ctx.notes,
            }
        )
        if a.embedding_model:
            loaded = ctx.req(
                "POST", "/v1/models/load", json={"model": a.embedding_model.name}
            )
            loaded.raise_for_status()
            scored = ctx.req(
                "POST",
                "/v1/rerank",
                json={
                    "model": a.embedding_model.name,
                    "query": "Paris weather",
                    "documents": [
                        "Paris is sunny, 21 degrees Celsius.",
                        "Ocean fish swimming.",
                    ],
                },
            )
            scored.raise_for_status()
            scores = scored.json().get("results", [])
            if len(scores) != 2:
                raise AssertionError(f"rerank response {scored.json()}")
            # Loaded local model is exercised by enrichment, not just by its own route.
            response = ctx.oa.responses.create(
                model=ctx.model,
                input="Search the web for Paris sunny temperature.",
                tools=[{"type": "web_search"}],
                tool_choice="required",
                max_tool_calls=1,
                max_output_tokens=512,
            )
            calls = [o for o in response.output if o.type == "web_search_call"]
            if (
                not calls
                or calls[0].status != "completed"
                or "Web research dense ranking:" not in srv.log_tail(300)
            ):
                raise AssertionError("resident local research ranking was not engaged")
            record(
                {
                    "check": "resident_dense_fusion",
                    "pass": True,
                    "model": a.embedding_model.name,
                    "rerank_results": len(scores),
                    "device": "M5",
                }
            )
        if not a.baseline:
            for action in ("open_page", "find_in_page"):
                url = fake.url + "/page"
                prompt = (
                    f'Use web_search exactly once with action="{action}", url="{url}"'
                    + (', pattern="SECRET PAGE"' if action == "find_in_page" else "")
                    + ". Then cite the page [1]."
                )
                response = ctx.oa.responses.create(
                    model=ctx.model,
                    input=prompt,
                    tools=[{"type": "web_search"}],
                    tool_choice="required",
                    max_tool_calls=1,
                    max_output_tokens=512,
                    include=["web_search_call.action.sources"],
                )
                calls = [o for o in response.output if o.type == "web_search_call"]
                if (
                    not calls
                    or calls[0].action.type != action
                    or calls[0].status != "completed"
                ):
                    raise AssertionError(f"{action}: {[o.model_dump() for o in calls]}")
                record(
                    {
                        "check": action,
                        "pass": True,
                        "url": calls[0].action.url,
                        "pages_requested": len(fake.pages),
                        "device": "M5",
                    }
                )
        if a.eval_snapshot:
            import asyncio

            import websearch_eval

            eval_out = a.out.with_suffix(".eval.jsonl")
            argv = [
                "--snapshot",
                str(a.eval_snapshot),
                "--out",
                str(eval_out),
                "--url",
                srv.url,
                "--model",
                ctx.model,
            ]
            token_file = home / "eval.token"
            if models_dir:
                token_file.write_text(token)
                token_file.chmod(0o600)
                argv += ["--token-file", str(token_file)]
            try:
                rc = asyncio.run(
                    websearch_eval.run(websearch_eval.parser().parse_args(argv))
                )
            finally:
                token_file.unlink(missing_ok=True)
            evidence = [
                json.loads(line)
                for line in eval_out.read_text().splitlines()
                if line.strip()
            ]
            if rc or not evidence or not evidence[-1].get("complete"):
                raise AssertionError("frozen replay did not finish")
            pairs = evidence[:-1]
            record(
                {
                    "check": "frozen_adversarial_replay",
                    "pass": True,
                    "device": "M5",
                    "queries": len(pairs),
                    "research_correct": sum(p["research"]["correct"] for p in pairs),
                    "snippets_correct": sum(p["snippets"]["correct"] for p in pairs),
                    "research_valid_citations": sum(
                        p["research"]["valid_citations"] for p in pairs
                    ),
                    "research_citations": sum(
                        p["research"]["citations"] for p in pairs
                    ),
                    "research_marker_echoes": sum(
                        p["research"]["injection_success"] for p in pairs
                    ),
                    "quality_gate": evidence[-1]["quality_gate"],
                    "snapshot_sha256": evidence[-1]["snapshot_sha256"],
                }
            )
        record({"complete": True, "pass": True, "device": "M5"})
        return 0
    except Exception as exc:
        record(
            {"check": "failure", "pass": False, "error": f"{type(exc).__name__}: {exc}"}
        )
        record({"complete": True, "pass": False, "device": "M5"})
        return 1
    finally:
        if srv:
            print(srv.log_tail(30), flush=True)
            srv.kill()
        fake.close()


if __name__ == "__main__":
    raise SystemExit(main())
