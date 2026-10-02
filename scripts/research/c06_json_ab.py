"""CPU A/B of the in-house JSON-schema mask vs llguidance on a real tokenizer.

Semantic differential (accept/reject of valid and mutated token paths, and the
allowed-token sets along valid paths), compile time, per-token mask time, memory.

    PYTHONPATH=python python scripts/research/c06_json_ab.py --tokenizer DIR \
        [--corpus-dir RUNS_DIR] [--out out.json] [--llg-mode {raw,preprocessed}]
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import random
import resource
import statistics
import time
from pathlib import Path
from typing import Any

# ── corpus ──────────────────────────────────────────────────────────────────

_OBJ = "object"
OPENAI: dict[str, dict] = {
    "math_steps": {
        "type": _OBJ,
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": _OBJ,
                    "properties": {
                        "explanation": {"type": "string"},
                        "output": {"type": "string"},
                    },
                    "required": ["explanation", "output"],
                    "additionalProperties": False,
                },
            },
            "final_answer": {"type": "string"},
        },
        "required": ["steps", "final_answer"],
        "additionalProperties": False,
    },
    "extraction": {
        "type": _OBJ,
        "properties": {
            "name": {"type": "string"},
            "date": {"type": "string"},
            "participants": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "date", "participants"],
        "additionalProperties": False,
    },
    "moderation": {
        "type": _OBJ,
        "properties": {
            "violates": {"type": "boolean"},
            "violation_categories": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["violence", "sexual", "self_harm", "other"],
                },
            },
            "violation_reason": {"type": ["string", "null"]},
        },
        "required": ["violates", "violation_categories", "violation_reason"],
        "additionalProperties": False,
    },
    "nullable_anyof": {
        "type": _OBJ,
        "properties": {
            "item": {
                "anyOf": [
                    {
                        "type": _OBJ,
                        "properties": {
                            "name": {"type": "string"},
                            "n": {"type": "integer"},
                        },
                        "required": ["name", "n"],
                        "additionalProperties": False,
                    },
                    {"type": "null"},
                ]
            },
            "kind": {"enum": ["a", "b", 3, None, True]},
        },
        "required": ["item", "kind"],
        "additionalProperties": False,
    },
    "numbers": {
        "type": _OBJ,
        "properties": {
            "count": {"type": "integer"},
            "price": {"type": "number"},
            "tags": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["count", "price", "tags"],
        "additionalProperties": False,
    },
    "optional_props": {
        "type": _OBJ,
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "integer"},
            "c": {"type": "boolean"},
        },
        "required": ["a"],
    },
    "no_additional_keyword": {
        "type": _OBJ,
        "properties": {"x": {"type": "string"}, "y": {"type": "number"}},
        "required": ["x", "y"],
    },
    "open_object": {
        "type": _OBJ,
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
        "additionalProperties": True,
    },
    "map_of_ints": {
        "type": _OBJ,
        "additionalProperties": {"type": "integer"},
    },
    "defs_ref": {
        "type": _OBJ,
        "properties": {
            "pet": {"$ref": "#/$defs/Pet"},
            "owners": {"type": "array", "items": {"$ref": "#/$defs/Owner"}},
        },
        "required": ["pet", "owners"],
        "$defs": {
            "Pet": {
                "type": _OBJ,
                "properties": {
                    "name": {"type": "string"},
                    "species": {"enum": ["cat", "dog"]},
                },
                "required": ["name", "species"],
            },
            "Owner": {
                "type": _OBJ,
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name"],
            },
        },
    },
    "pydantic_model": {
        "$defs": {
            "Address": {
                "properties": {
                    "street": {"title": "Street", "type": "string"},
                    "zip": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "title": "Zip",
                    },
                },
                "required": ["street"],
                "title": "Address",
                "type": _OBJ,
            }
        },
        "properties": {
            "name": {"title": "Name", "type": "string"},
            "age": {"title": "Age", "type": "integer"},
            "scores": {"items": {"type": "number"}, "title": "Scores", "type": "array"},
            "address": {"$ref": "#/$defs/Address"},
            "role": {"enum": ["admin", "user"], "title": "Role", "type": "string"},
        },
        "required": ["name", "age", "address"],
        "title": "Person",
        "type": _OBJ,
    },
    "oneof_union": {
        "type": _OBJ,
        "properties": {
            "shape": {
                "oneOf": [
                    {
                        "type": _OBJ,
                        "properties": {
                            "kind": {"const": "circle"},
                            "r": {"type": "number"},
                        },
                        "required": ["kind", "r"],
                    },
                    {
                        "type": _OBJ,
                        "properties": {
                            "kind": {"const": "rect"},
                            "w": {"type": "number"},
                            "h": {"type": "number"},
                        },
                        "required": ["kind", "w", "h"],
                    },
                ]
            }
        },
        "required": ["shape"],
    },
    "top_array": {
        "type": "array",
        "items": {
            "type": _OBJ,
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
        },
    },
    "top_string_enum": {"type": "string", "enum": ["red", "green", "blue"]},
    "nested_depth": {
        "type": _OBJ,
        "properties": {
            "a": {
                "type": _OBJ,
                "properties": {
                    "b": {
                        "type": _OBJ,
                        "properties": {
                            "c": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                            }
                        },
                        "required": ["c"],
                    }
                },
                "required": ["b"],
            }
        },
        "required": ["a"],
    },
    "allof_merge": {
        "allOf": [
            {"type": _OBJ, "properties": {"a": {"type": "string"}}, "required": ["a"]},
            {"type": _OBJ, "properties": {"b": {"type": "integer"}}, "required": ["b"]},
        ]
    },
    "string_escapes": {
        "type": _OBJ,
        "properties": {"text": {"type": "string"}, "code": {"type": "string"}},
        "required": ["text", "code"],
        "additionalProperties": False,
    },
}


def tool_schemas(runs_dir: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(glob.glob(f"{runs_dir}/**/*-req.json", recursive=True)):
        try:
            body = json.loads(Path(p).read_text())
        except Exception:
            continue
        for t in body.get("tools") or []:
            fn = t.get("function") or {}
            sch = t.get("input_schema") or fn.get("parameters") or t.get("parameters")
            name = t.get("name") or fn.get("name")
            if isinstance(sch, dict) and name:
                out.setdefault(f"tool:{name}", sch)
    return out


# ── instance generation / mutation ──────────────────────────────────────────


def resolve(node: Any, root: dict) -> Any:
    while isinstance(node, dict) and "$ref" in node:
        cur: Any = root
        for seg in node["$ref"][2:].split("/"):
            cur = cur[seg]
        node = cur
    return node


def gen(node: Any, root: dict, rng: random.Random, depth: int = 0) -> Any:
    node = resolve(node, root)
    if not isinstance(node, dict):
        return rng.choice([1, "x", None])
    if "const" in node:
        return node["const"]
    if "enum" in node:
        return rng.choice(node["enum"])
    for k in ("anyOf", "oneOf"):
        if k in node:
            return gen(rng.choice(node[k]), root, rng, depth + 1)
    if "allOf" in node:
        merged: dict = {"type": _OBJ, "properties": {}, "required": []}
        for sub in node["allOf"]:
            sub = resolve(sub, root)
            merged["properties"].update(sub.get("properties", {}))
            merged["required"] += sub.get("required", [])
        return gen(merged, root, rng, depth + 1)
    t = node.get("type")
    if isinstance(t, list):
        t = rng.choice(t)
    if t is None and "properties" in node:
        t = _OBJ
    if t == _OBJ:
        props = node.get("properties", {})
        req = node.get("required", [])
        o: dict = {}
        for k, v in props.items():
            if k in req or (rng.random() < 0.5 and depth < 4):
                o[k] = gen(v, root, rng, depth + 1)
        ap = node.get("additionalProperties")
        if isinstance(ap, dict) and not props:
            for k in rng.sample(["k1", "k2", "k3"], rng.randint(0, 2)):
                o[k] = gen(ap, root, rng, depth + 1)
        return o
    if t == "array":
        n = rng.randint(0, 3) if depth < 4 else 0
        return [gen(node.get("items", {}), root, rng, depth + 1) for _ in range(n)]
    if t == "string":
        return rng.choice(
            [
                "",
                "hello",
                'qu"ote',
                "line\nbreak",
                "tab\there",
                "é中文😀",
                "back\\slash",
                "a" * 40,
                "</tool>",
                "x y",
            ]
        )
    if t == "integer":
        return rng.choice([0, 1, -1, 42, -17, 1000000, 10**15, 7])
    if t == "number":
        return rng.choice([0, 1, -1, 3.14, -0.5, 1e10, 2.5e-3, 100, 1.0])
    if t == "boolean":
        return rng.choice([True, False])
    if t == "null":
        return None
    return rng.choice([1, "x", None, True])


def dumps(v: Any, rng: random.Random) -> str:
    style = rng.choice(["default", "compact", "indent2", "indent4", "ascii"])
    if style == "compact":
        return json.dumps(v, separators=(",", ":"), ensure_ascii=False)
    if style == "indent2":
        return json.dumps(v, indent=2, ensure_ascii=False)
    if style == "indent4":
        return json.dumps(v, indent=4, ensure_ascii=False)
    if style == "ascii":
        return json.dumps(v, ensure_ascii=True)
    return json.dumps(v, ensure_ascii=False)


def mutations(v: Any, rng: random.Random) -> list[tuple[str, str]]:
    """(label, text) variants that are mostly invalid."""
    out: list[tuple[str, str]] = []
    base = json.dumps(v, ensure_ascii=False)
    out.append(("truncated", base[: max(1, len(base) // 2)]))
    out.append(("trailing_junk", base + " x"))
    out.append(
        ("trailing_comma", base[:-1] + ",}" if base.endswith("}") else base + ",")
    )
    out.append(("single_quote", base.replace('"', "'")))
    out.append(("bare_word", "hello"))
    out.append(("num_float_for_int", base.replace("1", "1.0", 1)))
    out.append(("leading_zero", base.replace("1", "01", 1)))
    out.append(("exp_int", base.replace("1", "1e2", 1)))
    out.append(("neg_zero", base.replace("0", "-0", 1)))
    out.append(("true_cap", base.replace("true", "True")))
    if isinstance(v, dict) and v:
        k = rng.choice(list(v))
        d = copy.deepcopy(v)
        d.pop(k)
        out.append(("drop_key", json.dumps(d)))
        d = copy.deepcopy(v)
        d["zz_extra"] = 1
        out.append(("extra_key", json.dumps(d)))
        d = {kk: v[kk] for kk in reversed(list(v))}
        out.append(("reversed_order", json.dumps(d)))
        d = copy.deepcopy(v)
        d[k] = [d[k]] if not isinstance(d[k], list) else "s"
        out.append(("wrong_type", json.dumps(d)))
        out.append(("dup_key", base[:-1] + ", " + json.dumps(k) + ": null}"))
    for ws in ("\t", "   ", "\n\n\n", " \n ", "\r\n"):
        out.append(
            (
                f"ws:{ws!r}",
                base.replace(", ", "," + ws, 1).replace('": ', '":' + ws, 1),
            )
        )
        out.append((f"ws_lead:{ws!r}", ws + base))
        out.append((f"ws_trail:{ws!r}", base + ws))
    out.append(("ws_before_colon", base.replace('":', '" :', 1)))
    out.append(("ws_inside_brace", base.replace("{", "{ ", 1)))
    return out


# ── engines ─────────────────────────────────────────────────────────────────


def llg_schema_prep(schema: dict, mode: str) -> dict:
    """Preprocessing so llguidance follows our documented conventions."""
    if mode == "raw":
        return schema
    s = copy.deepcopy(schema)

    def walk(n: Any) -> None:
        if isinstance(n, dict):
            if (
                (n.get("type") == _OBJ or "properties" in n)
                and "additionalProperties" not in n
                and n.get("properties")
            ):
                n["additionalProperties"] = False
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(s)
    return s


def make_inhouse(schema, tok):
    from yunshu_engine.json_schema import (
        JsonSchemaConstraint,
        validate_supported_schema,
    )

    validate_supported_schema(schema)
    return JsonSchemaConstraint(schema)


def make_llg(schema, tok, mode, ws):
    from yunshu_engine.grammar_constraint import LlgJsonSchemaConstraint

    c = LlgJsonSchemaConstraint(llg_schema_prep(schema, mode), tok)
    if ws is not None:
        c._make_grammar = lambda: c._LLMatcher.grammar_from_json_schema(  # type: ignore[method-assign]
            c._schema_json, defaults=ws
        )
    c._bind(tok)
    return c


def eos_in(c, tok, eos) -> bool:
    return any(e in set(c.get_allowed_tokens(tok, [])) for e in eos)


def walk_text(make, tok, text, eos):
    """Feed the real tokenization of text; (accepted, reject_index, n_tokens)."""
    ids = tok.encode(text, add_special_tokens=False)
    c = make()
    for i, tid in enumerate(ids):
        if tid not in set(c.get_allowed_tokens(tok, [])):
            return (
                False,
                i,
                len(ids),
                tok.decode([tid]),
                tok.decode(ids[max(0, i - 5) : i]),
            )
        c.advance(tok.decode([tid]))
    return eos_in(c, tok, eos), len(ids), len(ids), "", ""


def valid_walk_sets(make, tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    c = make()
    sets = []
    for tid in ids:
        a = set(c.get_allowed_tokens(tok, []))
        sets.append(a)
        if tid not in a:
            break
        c.advance(tok.decode([tid]))
    return ids, sets


def truth(schema: dict, text: str, closed: bool) -> bool:
    """Ground truth under our conventions (objects that declare properties are
    closed unless additionalProperties is given); whitespace must be one compact run."""
    import jsonschema

    try:
        inst = json.loads(text)
    except ValueError:
        return False
    sch = llg_schema_prep(schema, "preprocessed") if closed else schema
    return jsonschema.Draft202012Validator(sch).is_valid(inst)


def _bad_ws(text: str) -> bool:
    """True when structural whitespace is outside ' ?|\\n {0,16}' (the documented run)."""
    import re

    out = []
    in_str = False
    esc = False
    run = ""
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch in " \t\r\n":
            run += ch
            continue
        if run and not re.fullmatch(r" ?|\n {0,16}", run):
            out.append(run)
        run = ""
        if ch == '"':
            in_str = True
    if run and not re.fullmatch(r" ?|\n {0,16}", run):
        out.append(run)
    return bool(out) or text[:1] in " \t\r\n"


def _q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus-dir", default="")
    ap.add_argument("--out", default="c06.json")
    ap.add_argument(
        "--llg-mode", default="preprocessed", choices=["raw", "preprocessed"]
    )
    ap.add_argument("--ws", default="run", help="none | run | compact | json:<obj>")
    ap.add_argument("--n-valid", type=int, default=6)
    ap.add_argument("--sets", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from yunshu_engine.constraint_eos import normalize_eos_ids
    from yunshu_engine.json_schema import UnsupportedSchemaError

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = list(normalize_eos_ids(tok))
    ws = None
    if args.ws == "run":
        ws = {"whitespace_pattern": r"[ ]?|\n[ ]{0,16}"}
    elif args.ws == "compact":
        ws = {"whitespace_flexible": False}
    elif args.ws.startswith("json:"):
        ws = json.loads(args.ws[5:])
    corpus = dict(OPENAI)
    if args.corpus_dir:
        corpus.update(tool_schemas(args.corpus_dir))
    rng = random.Random(args.seed)
    report: dict[str, Any] = {"schemas": {}, "llg_mode": args.llg_mode, "ws": ws}
    warm = {"type": _OBJ, "properties": {"a": {"type": "string"}}}
    make_llg(warm, tok, args.llg_mode, ws)
    make_inhouse(warm, tok).get_allowed_tokens(tok, [])
    totals = {"paths": 0, "agree": 0, "disagree": 0, "set_steps": 0, "set_equal": 0}
    for name, schema in corpus.items():
        rec: dict[str, Any] = {}
        t0 = time.perf_counter()
        try:
            make_inhouse(schema, tok)
        except UnsupportedSchemaError as e:
            rec["inhouse"] = f"unsupported: {str(e)[:80]}"
        else:
            rec["inhouse_compile_ms"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        try:
            make_llg(schema, tok, args.llg_mode, ws)
        except Exception as e:  # noqa: BLE001
            rec["llg"] = f"error: {str(e)[:80]}"
        else:
            rec["llg_compile_ms"] = (time.perf_counter() - t0) * 1e3
        if "inhouse" in rec or "llg" in rec:
            report["schemas"][name] = rec
            continue

        def mk_in(s=schema):
            return make_inhouse(s, tok)

        def mk_llg(s=schema):
            return make_llg(s, tok, args.llg_mode, ws)

        diffs: list[dict] = []
        paths = agree = 0
        texts: list[tuple[str, str]] = []
        for _ in range(args.n_valid):
            v = gen(schema, schema, rng)
            texts.append(("valid", dumps(v, rng)))
            texts += mutations(v, rng)
        for label, text in texts:
            paths += 1
            a = walk_text(mk_in, tok, text, eos)
            b = walk_text(mk_llg, tok, text, eos)
            tr = truth(schema, text, True) and not _bad_ws(text)
            for eng, r in (("inhouse", a), ("llg", b)):
                key = (
                    "false_accept"
                    if r[0] and not tr
                    else "false_reject"
                    if tr and not r[0]
                    else "ok"
                )
                rec.setdefault(f"{eng}_vs_truth", {}).setdefault(key, 0)
                rec[f"{eng}_vs_truth"][key] += 1
            if a[:3] == b[:3]:
                agree += 1
            else:
                kind = (
                    "llg_stricter"
                    if a[0] and not b[0]
                    else (
                        "inhouse_stricter"
                        if b[0] and not a[0]
                        else "both_reject_at_different_token"
                    )
                )
                diffs.append(
                    {"label": label, "kind": kind, "text": text, "inhouse": a, "llg": b}
                )
        rec["paths"] = paths
        rec["agree"] = agree
        rec["path_diffs"] = diffs
        set_steps = set_eq = 0
        set_diffs = []
        for _ in range(args.sets):
            v = gen(schema, schema, rng)
            text = dumps(v, rng)
            ids, sa = valid_walk_sets(mk_in, tok, text)
            _, sb = valid_walk_sets(mk_llg, tok, text)
            for i, (x, y) in enumerate(zip(sa, sb, strict=False)):
                set_steps += 1
                if x == y:
                    set_eq += 1
                elif len(set_diffs) < 6:
                    set_diffs.append(
                        {
                            "step": i,
                            "prefix": tok.decode(ids[:i])[-60:],
                            "n_only_inhouse": len(x - y),
                            "n_only_llg": len(y - x),
                            "only_inhouse": sorted(
                                tok.decode([t]) for t in list(x - y)[:6]
                            ),
                            "only_llg": sorted(
                                tok.decode([t]) for t in list(y - x)[:6]
                            ),
                        }
                    )
        rec["set_steps"] = set_steps
        rec["set_equal"] = set_eq
        rec["set_diffs"] = set_diffs
        v = gen(schema, schema, random.Random(1))
        text = json.dumps(v, ensure_ascii=False)
        for eng, mk in (("inhouse", mk_in), ("llg", mk_llg)):
            ids = tok.encode(text, add_special_tokens=False)
            c = mk()
            ts = []
            for tid in ids:
                t0 = time.perf_counter()
                c.get_allowed_tokens(tok, [])
                ts.append((time.perf_counter() - t0) * 1e3)
                c.advance(tok.decode([tid]))
            rec[f"{eng}_mask_ms"] = ts
        report["schemas"][name] = rec
        totals["paths"] += paths
        totals["agree"] += agree
        totals["set_steps"] += set_steps
        totals["set_equal"] += set_eq
    totals["disagree"] = totals["paths"] - totals["agree"]
    report["totals"] = totals
    for eng in ("inhouse", "llg"):
        allms = [
            x for r in report["schemas"].values() for x in r.get(f"{eng}_mask_ms", [])
        ]
        cm = [
            r[f"{eng}_compile_ms"]
            for r in report["schemas"].values()
            if f"{eng}_compile_ms" in r
        ]
        report[f"{eng}_summary"] = {
            "mask_n": len(allms),
            "mask_median_ms": statistics.median(allms) if allms else None,
            "mask_p90_ms": _q(allms, 0.9),
            "mask_max_ms": max(allms) if allms else None,
            "compile_median_ms": statistics.median(cm) if cm else None,
            "compile_max_ms": max(cm) if cm else None,
        }
    report["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    keys = ("totals", "inhouse_summary", "llg_summary", "maxrss_mb")
    print(json.dumps({k: report[k] for k in keys}, indent=1))
    for n, r in report["schemas"].items():
        if "path_diffs" in r:
            print(
                f"{n:28s} paths {r['agree']}/{r['paths']}  "
                f"sets {r['set_equal']}/{r['set_steps']}"
            )
        else:
            print(f"{n:28s} {r.get('inhouse') or r.get('llg')}")


if __name__ == "__main__":
    main()
