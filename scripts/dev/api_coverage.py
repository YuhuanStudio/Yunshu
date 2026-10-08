"""Walk the installed `openai` and `anthropic` SDKs and list every endpoint they know.

Our API matrix (docs/guides/API_SURFACE.md) was built from the routes we already had, so an endpoint
OpenAI added (Decisions) was missed until someone read the changelog. This walker reads the SDK
resource modules (AST only: nothing is imported or called), extracts every HTTP path the SDK can
request, and classifies each one:

  implemented     the gateway registers a matching route
  not_applicable  declared in scripts/dev/api_coverage_na.json with a reason (hosted-only, admin, ...)
  planned         known gap, declared with a reason (honest backlog, not an excuse)
  UNCLASSIFIED    neither: this is the failure the unit test (tests/unit/test_api_coverage.py) catches

    python scripts/dev/api_coverage.py            # table + exit 1 when anything is unclassified
    python scripts/dev/api_coverage.py --json     # machine-readable
"""

from __future__ import annotations

import ast
import fnmatch
import importlib.metadata
import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
NA_FILE = HERE / "api_coverage_na.json"
SDKS = ("openai", "anthropic")
_HTTP = {
    "_get": "GET",
    "_post": "POST",
    "_put": "PUT",
    "_patch": "PATCH",
    "_delete": "DELETE",
    "_get_api_list": "GET",
    "_post_stream": "POST",
}


# Batches.results() follows the batch's own `results_url`, which the server chooses.
_DYNAMIC = {
    "anthropic": [
        ("GET", "/v1/messages/batches/{}/results", "messages.batches.Batches.results"),
    ],
}


@dataclass
class Endpoint:
    sdk: str
    method: str
    path: str  # normalized: no query, path params as {}
    callers: list[str] = field(default_factory=list)  # resource.method names in the SDK

    @property
    def key(self) -> str:
        return f"{self.sdk} {self.method} {self.path}"


def normalize(path: str) -> str:
    path = path.split("?")[0]
    path = re.sub(r"\{[^}]*\}", "{}", path)
    return path.rstrip("/") or "/"


def _path_of(node: ast.AST) -> str | None:
    """The literal path of the first argument of a client request, or None if not literal."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):  # f"/files/{file_id}/content"
        return "".join(
            v.value if isinstance(v, ast.Constant) else "{}" for v in node.values
        )
    if isinstance(node, ast.Call):  # path_template("/responses/{response_id}", ...)
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
        if name == "path_template" and node.args:
            return _path_of(node.args[0])
    return None


def _method_override(call: ast.Call, default: str) -> str:
    for kw in call.keywords:
        if (
            kw.arg == "method"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return kw.value.value.upper()
    return default


def _is_public_sync_class(name: str) -> bool:
    return not (
        name.startswith("Async")
        or name.startswith("_")
        or "WithRawResponse" in name
        or "WithStreamingResponse" in name
    )


def walk_sdk(sdk: str) -> tuple[list[Endpoint], list[str], str]:
    """(endpoints, unparsed public methods that never reach a literal path, sdk version)."""
    spec = importlib.util.find_spec(sdk)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f"{sdk} is not installed")
    root = Path(next(iter(spec.submodule_search_locations))) / "resources"
    version = importlib.metadata.version(sdk)
    found: dict[str, Endpoint] = {}
    unparsed: list[str] = []
    for file in sorted(root.rglob("*.py")):
        tree = ast.parse(file.read_text())
        mod = ".".join(file.relative_to(root).with_suffix("").parts)
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            if not _is_public_sync_class(cls.name):
                continue
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                if fn.name.startswith("_") or fn.name in (
                    "with_raw_response",
                    "with_streaming_response",
                ):
                    continue
                if fn.name == "connect":  # websocket: the module path is the URL path
                    parts = [
                        p
                        for i, p in enumerate(mod.split("."))
                        if i == 0 or p != mod.split(".")[i - 1]
                    ]
                    parts = [p for p in parts if p != "beta"]
                    ep = Endpoint(sdk, "WS", "/" + "/".join(parts))
                    found.setdefault(ep.key, ep).callers.append(
                        f"{mod}.{cls.name}.{fn.name}"
                    )
                    continue
                hit = False
                saw_call = False
                for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call)]:
                    f = call.func
                    if (
                        isinstance(f, ast.Attribute)
                        and f.attr in _HTTP
                        and isinstance(f.value, ast.Attribute | ast.Name)
                        and call.args
                    ):
                        saw_call = True
                        path = _path_of(call.args[0])
                        if path is None:
                            continue
                        method = _method_override(call, _HTTP[f.attr])
                        ep = Endpoint(sdk, method, normalize(path))
                        found.setdefault(ep.key, ep).callers.append(
                            f"{mod}.{cls.name}.{fn.name}"
                        )
                        hit = True
                if saw_call and not hit:
                    unparsed.append(f"{sdk}:{mod}.{cls.name}.{fn.name}")
    for method, path, caller in _DYNAMIC.get(sdk, []):
        # endpoints the SDK reaches through a URL the server returned, not a literal path
        ep = Endpoint(sdk, method, path)
        found.setdefault(ep.key, ep).callers.append(caller)
        unparsed[:] = [u for u in unparsed if not u.endswith(caller.split(":")[-1])]
    return sorted(found.values(), key=lambda e: e.key), unparsed, version


def implemented_routes() -> set[str]:
    """'METHOD /path' for every route the gateway registers (websockets as 'WS /path')."""
    sys.path.insert(0, str(HERE.parents[1] / "python"))
    from yunshu_gateway.main import create_app

    def walk(routes):
        for r in routes:
            if hasattr(r, "effective_candidates"):
                yield from walk(r.effective_candidates())
                continue
            orig = getattr(r, "original_route", r)
            path = getattr(r, "path", "") or getattr(orig, "path", "")
            if "WebSocket" in type(orig).__name__:
                yield "WS " + normalize(re.sub(r"\{([^}:]*):[^}]*\}", "{\\1}", path))
            else:
                for m in getattr(orig, "methods", None) or ():
                    if m not in ("HEAD", "OPTIONS"):
                        yield f"{m} " + normalize(
                            re.sub(r"\{([^}:]*):[^}]*\}", "{\\1}", path)
                        )

    return set(walk(create_app().routes))


def load_declarations() -> list[dict]:
    data = json.loads(NA_FILE.read_text())
    for d in data:
        if d.get("status") not in ("not_applicable", "planned", "implemented"):
            raise ValueError(
                f"{NA_FILE.name}: {d!r}: status must be not_applicable, planned or implemented"
            )
        if len(d.get("reason", "")) < 40:
            raise ValueError(
                f"{NA_FILE.name}: {d['match']!r}: a real reason is required"
            )
    return data


def served(ep: Endpoint, routes: set[str]) -> bool:
    """Is the SDK endpoint served? SDK paths are relative to /v1 for OpenAI, absolute for Anthropic."""
    path = ep.path if ep.sdk == "anthropic" else "/v1" + ep.path
    return f"{ep.method} {path}" in routes


def classify(endpoints: list[Endpoint], routes: set[str], declarations: list[dict]):
    """-> (rows, used declaration indexes). A row is (endpoint, state, reason)."""
    rows, used = [], set()
    for ep in endpoints:
        match = next(
            (
                i
                for i, d in enumerate(declarations)
                if fnmatch.fnmatchcase(ep.key, d["match"])
            ),
            None,
        )
        if served(ep, routes):
            rows.append((ep, "implemented", ""))
            if match is not None:
                used.add(
                    match
                )  # a declaration over an implemented route is reported by the test
        elif match is not None and declarations[match]["status"] != "implemented":
            used.add(match)
            rows.append(
                (ep, declarations[match]["status"], declarations[match]["reason"])
            )
        else:
            rows.append((ep, "UNCLASSIFIED", ""))
    return rows, used


def contradictions(
    endpoints: list[Endpoint], routes: set[str], declarations: list[dict]
):
    """Served endpoints that a declaration still calls planned / not applicable (stale text)."""
    return [
        (ep, d["match"])
        for ep in endpoints
        if served(ep, routes)
        for d in declarations
        if d["status"] != "implemented" and fnmatch.fnmatchcase(ep.key, d["match"])
    ]


def main(argv: list[str]) -> int:
    routes = implemented_routes()
    declarations = load_declarations()
    endpoints: list[Endpoint] = []
    versions = {}
    unparsed: list[str] = []
    for sdk in SDKS:
        eps, un, ver = walk_sdk(sdk)
        endpoints += eps
        unparsed += un
        versions[sdk] = ver
    rows, used = classify(endpoints, routes, declarations)
    bad = [r for r in rows if r[1] == "UNCLASSIFIED"]
    if "--json" in argv:
        print(
            json.dumps(
                {
                    "versions": versions,
                    "endpoints": [
                        {
                            "key": e.key,
                            "state": s,
                            "reason": r,
                            "callers": e.callers[:3],
                        }
                        for e, s, r in rows
                    ],
                    "stale_declarations": [
                        d["match"] for i, d in enumerate(declarations) if i not in used
                    ],
                    "unparsed": unparsed,
                },
                indent=2,
            )
        )
    else:
        print(
            f"openai {versions['openai']}, anthropic {versions['anthropic']}: {len(rows)} endpoints"
        )
        for state in ("implemented", "planned", "not_applicable", "UNCLASSIFIED"):
            sel = [r for r in rows if r[1] == state]
            print(f"\n## {state} ({len(sel)})")
            for e, _, reason in sel:
                print(
                    f"  {e.key}"
                    + (
                        f"   -- {reason[:90]}"
                        if reason and state != "not_applicable"
                        else ""
                    )
                )
        stale = [d["match"] for i, d in enumerate(declarations) if i not in used]
        if stale:
            print("\n## declarations matching nothing:", *stale, sep="\n  ")
        if unparsed:
            print(
                "\n## SDK methods with no literal path (check by hand):",
                *unparsed,
                sep="\n  ",
            )
    stale_text = contradictions(endpoints, routes, declarations)
    for ep, match in stale_text:
        print(f"\nimplemented but still declared ({match!r}): {ep.key}")
    return 1 if bad or stale_text else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
