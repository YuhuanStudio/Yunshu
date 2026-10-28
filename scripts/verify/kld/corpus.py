"""The fixed KLD corpus: real long documents cut into 8K and 32K token windows.

The text lives under /Volumes/P5Plus/yunshu-build/kld-corpus/<version>/ (never in the repo); the repo carries
``manifest.json`` with every window's id, group, token count and sha256, and the corpus sha that cache keys
use. Building is deterministic given the sources and the tokenizer; a built corpus is verified against the
manifest and a mismatch fails closed.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

VERSION = "v1"
CORPUS_ROOT = Path("/Volumes/P5Plus/yunshu-build/kld-corpus")
MANIFEST = Path(__file__).with_name("manifest.json")
REPO = Path("/Users/yuhuan/Documents/YuhuanStudio/Yunshu")
AGENT_RUNS = REPO / "docs/research/runs"

# (group, window tokens, count): 12 x 8K + 3 x 32K = 196,608 tokens
PLAN = (
    ("prose", 8192, 4),
    ("code", 8192, 4),
    ("agent", 8192, 4),
    ("prose", 32768, 1),
    ("code", 32768, 1),
    ("agent", 32768, 1),
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _files(group: str) -> list[Path]:
    if group == "prose":
        files = sorted(REPO.glob("docs/**/*.md")) + sorted(
            REPO.glob("frontend/docs/guides/*.mdx")
        )
        files = [
            f for f in files if "research" not in f.parts and "archive" not in f.parts
        ]
    elif group == "code":
        files = sorted(REPO.glob("python/**/*.py"))
    else:  # the fullest recorded request of each agent-client capture (it holds the whole conversation)
        best: dict[Path, Path] = {}
        for f in sorted(AGENT_RUNS.glob("*/artifacts/*/bodies/*-req.json")):
            if f.parent not in best or f.stat().st_size > best[f.parent].stat().st_size:
                best[f.parent] = f
        files = sorted(best.values())
    return [f for f in files if f.is_file() and 1500 < f.stat().st_size < 300_000]


def _render(group: str, path: Path) -> str:
    text = path.read_text(errors="ignore")
    if group != "agent":
        return f"\n\n### {path.name}\n{text}"
    body = json.loads(text)
    parts = []
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                str(c.get("text", c)) if isinstance(c, dict) else str(c)
                for c in content
            )
        parts.append(f"[{message.get('role', '?')}]\n{content}")
    return "\n\n".join(parts)


def build_texts(tokenizer, plan=PLAN, seed: int = 20261028) -> list[dict]:
    """Windows of exactly their token budget (cut at a token boundary), each from distinct source files."""
    out = []
    for group in dict.fromkeys(g for g, _, _ in plan):
        files = _files(group)
        random.Random(seed).shuffle(files)
        cursor = 0
        for g, n_tokens, count in plan:
            if g != group:
                continue
            for index in range(count):
                text, ids = "", []
                while len(ids) < n_tokens:
                    if cursor >= len(files):
                        raise RuntimeError(
                            f"not enough {group} source text for {n_tokens}-token windows"
                        )
                    text += _render(group, files[cursor])
                    cursor += 1
                    ids = tokenizer.encode(text)
                window = tokenizer.decode(ids[:n_tokens])
                out.append(
                    dict(
                        id=f"{group}-{n_tokens // 1024}k-{index}",
                        group=group,
                        tokens=len(tokenizer.encode(window)),
                        text=window,
                    )
                )
    return out


def corpus_sha(windows: list[dict]) -> str:
    h = hashlib.sha256()
    for w in sorted(windows, key=lambda w: w["id"]):
        h.update(f"{w['id']}:{w['sha256']}:{w['tokens']}\n".encode())
    return h.hexdigest()


def write(
    windows: list[dict],
    root: Path = CORPUS_ROOT,
    manifest: Path = MANIFEST,
    tokenizer_name: str = "",
) -> dict:
    directory = root / VERSION
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for w in windows:
        (directory / f"{w['id']}.txt").write_text(w["text"], encoding="utf-8")
        entries.append(
            dict(
                id=w["id"],
                group=w["group"],
                tokens=w["tokens"],
                sha256=sha256_text(w["text"]),
            )
        )
    doc = dict(
        version=VERSION,
        tokenizer=tokenizer_name,
        total_tokens=sum(e["tokens"] for e in entries),
        corpus_sha=corpus_sha(entries),
        windows=entries,
    )
    manifest.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def load_manifest(manifest: Path = MANIFEST) -> dict:
    doc = json.loads(manifest.read_text())
    if doc["corpus_sha"] != corpus_sha(doc["windows"]):
        raise RuntimeError("manifest corpus_sha does not match its window list")
    return doc


def load_texts(root: Path = CORPUS_ROOT, manifest: Path = MANIFEST) -> list[dict]:
    """The manifest's windows with their text; fails closed when a file is missing or its sha differs."""
    doc = load_manifest(manifest)
    out = []
    for entry in doc["windows"]:
        path = root / doc["version"] / f"{entry['id']}.txt"
        if not path.is_file():
            raise RuntimeError(f"corpus window missing: {path}")
        text = path.read_text(encoding="utf-8")
        if sha256_text(text) != entry["sha256"]:
            raise RuntimeError(
                f"corpus window {entry['id']} does not match the manifest sha"
            )
        out.append({**entry, "text": text})
    return out


if __name__ == "__main__":
    import sys

    from transformers import AutoTokenizer

    model = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(model)
    doc = write(build_texts(tok), tokenizer_name=Path(model).name)
    print(json.dumps({k: v for k, v in doc.items() if k != "windows"}))
