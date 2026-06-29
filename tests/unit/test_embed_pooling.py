"""embedding pooling auto-detection was dead for HF-repo-id models.

_resolve_embedding_pooling() looked for sentence-transformers' 1_Pooling/config.json by joining
it onto the RAW model_name. For the common HF-repo-id deployment (e.g. "BAAI/bge-base-en-v1.5")
that produced "BAAI/bge-base-en-v1.5/1_Pooling/config.json" — a relative path that never exists
on disk → isfile False → every repo-id-served model fell through to MEAN. CLS-trained BGE models
were silently MEAN-pooled (wrong embedding space, degraded retrieval) — the exact bug wrote
the function to fix, defeated for anything but an explicit local dir. Fix: resolve the repo id to
its local snapshot (hf_repo_to_path) before the join, mirroring the engine's own load path.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile

from yunshu_engine.batched_engine import BatchedEngine


def _engine_with_model(model_name):
    eng = BatchedEngine.__new__(BatchedEngine)
    eng.model_name = model_name
    eng._embed_pooling = None
    return eng


def _pooling_dir(flag: str) -> str:
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "1_Pooling"))
    with open(os.path.join(d, "1_Pooling", "config.json"), "w") as f:
        json.dump({flag: True}, f)
    return d


def test_cls_and_last_detected_from_local_dir():
    assert (
        _engine_with_model(
            _pooling_dir("pooling_mode_cls_token")
        )._resolve_embedding_pooling()
        == "CLS"
    )
    assert (
        _engine_with_model(
            _pooling_dir("pooling_mode_lasttoken")
        )._resolve_embedding_pooling()
        == "LAST"
    )


def test_unknown_repo_id_falls_back_to_mean_no_crash():
    assert (
        _engine_with_model("NonExistent/repo-xyz-999")._resolve_embedding_pooling()
        == "MEAN"
    )


def test_plain_model_dir_without_pooling_config_is_mean():
    d = tempfile.mkdtemp()  # no 1_Pooling
    assert _engine_with_model(d)._resolve_embedding_pooling() == "MEAN"


def test_source_resolves_repo_id_before_lookup():
    src = inspect.getsource(BatchedEngine._resolve_embedding_pooling)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # the raw-name join is now preceded by a repo-id → local-path resolution
    assert "hf_repo_to_path" in code
    assert "_os.path.isdir(cand)" in code
