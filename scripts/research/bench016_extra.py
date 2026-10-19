"""Pure launch recipes for additional fresh-snapshot engines; no GPU imports."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path


def install(be, manifest: dict) -> None:
    recipes = manifest.get("extra_engines", {})
    original_launch = be.build_launch
    original_detect = be.detect_mode
    original_paths = be.required_paths
    evidence = {}
    for name, cfg in recipes.items():
        base = be.ENGINES["fn-yunshu-main" if cfg.get("flash_next") else "yunshu-main"]
        be.ENGINES[name] = dataclasses.replace(
            base,
            id=name,
            kind="external",
            product=cfg["product"],
            checkpoint=cfg["checkpoint"],
            expected_mode=cfg["mode"],
            drafter=cfg.get("drafter"),
            weights=cfg.get("weights", "same"),
        )

    def launch(engine, port, home, environ, ctx_tokens=4096, parallel=1):
        if engine not in recipes:
            return original_launch(engine, port, home, environ, ctx_tokens, parallel)
        if not 18990 <= port <= 18999:
            raise ValueError("benchmark server outside port pool")
        cfg = recipes[engine]
        values = {"port": str(port), "home": str(home), "ctx": str(ctx_tokens)}
        files = {}
        if cfg.get("strata"):
            data = dict(cfg["strata"])
            data.update(
                port=port, host="127.0.0.1", log=str(Path(home) / "strata-engine.log")
            )
            data["args"] = [s.format(**values) for s in data["args"]]
            files[str(Path(home) / "strata.json")] = json.dumps(data)
            evidence[engine] = data["log"]
            files[data["log"]] = ""
        return be.Launch(
            [arg.format(**values) for arg in cfg["argv"]],
            be.scrubbed_env(
                environ,
                {
                    "HOME": str(home),
                    "TMPDIR": str(home),
                    "XDG_CACHE_HOME": str(Path(home) / "cache"),
                    "HF_HOME": "/Volumes/P5Plus/hf-cache",
                    "HF_HUB_OFFLINE": "1",
                    **cfg.get("env", {}),
                },
            ),
            cfg.get("flags", {}),
            files,
            {},
            probe=cfg.get("probe"),
        )

    def detect(engine, log, probe_text=None):
        if engine not in recipes:
            return original_detect(engine, log, probe_text)
        if engine in evidence:
            log += Path(evidence[engine]).read_text(errors="replace")
        cfg = recipes[engine]
        return (
            cfg["mode"]
            if re.search(cfg["engaged_regex"], log + (probe_text or ""), re.I)
            else None
        )

    def paths(engine):
        if engine not in recipes:
            return original_paths(engine)
        cfg = recipes[engine]
        return [cfg["argv"][0], *cfg.get("required_paths", [])]

    be.build_launch = launch
    be.detect_mode = detect
    be.required_paths = paths
