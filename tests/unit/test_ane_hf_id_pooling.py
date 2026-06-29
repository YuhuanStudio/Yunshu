"""(HIGH): the ANE CLS/LAST pooling-refusal guard was bypassed for every HF id.

_model_is_mean_pooled read a LOCAL model_path/1_Pooling/config.json. The default + documented
ANE embedding models are HF ids (intfloat/e5-small-v2, BAAI/bge-small-en-v1.5), not local dirs,
so os.path.isfile() was always False → the guard returned True (assume mean) → a CLS-pooled
model (bge) was compiled+served with MEAN pooling: the wrong embedding space claimed to
close, on the HF-id sibling the local-tmp-only test never covered. Now non-local ids
resolve the pooling spec from the hub; only an explicit non-mean spec refuses.
"""
from __future__ import annotations

import json
import sys
import types

from yunshu_engine.ane_embedding import _model_is_mean_pooled


def _local_pool_dir(tmp_path, mode_key):
    d = tmp_path / "1_Pooling"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({mode_key: True}))
    return str(tmp_path)


def test_local_cls_still_refused(tmp_path):
    assert _model_is_mean_pooled(_local_pool_dir(tmp_path, "pooling_mode_cls_token")) is False


def test_local_mean_still_mean(tmp_path):
    assert _model_is_mean_pooled(_local_pool_dir(tmp_path, "pooling_mode_mean_tokens")) is True


def _patch_hub(monkeypatch, tmp_path, pool_dict_or_exc):
    """Install a fake huggingface_hub.hf_hub_download that writes pool_dict and returns its
    path, or raises if pool_dict_or_exc is an Exception."""
    cfg = tmp_path / "hub_config.json"

    def _dl(repo_id, filename, *a, **k):
        if isinstance(pool_dict_or_exc, Exception):
            raise pool_dict_or_exc
        cfg.write_text(json.dumps(pool_dict_or_exc))
        return str(cfg)

    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = _dl
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


def test_hf_id_cls_now_refused(monkeypatch, tmp_path):
    # the BUG: a CLS HF id used to slip through as mean; now it resolves the spec → refuse
    _patch_hub(monkeypatch, tmp_path, {"pooling_mode_cls_token": True})
    assert _model_is_mean_pooled("BAAI/bge-small-en-v1.5") is False


def test_hf_id_mean_stays_on_ane(monkeypatch, tmp_path):
    _patch_hub(monkeypatch, tmp_path, {"pooling_mode_mean_tokens": True})
    assert _model_is_mean_pooled("intfloat/e5-small-v2") is True


def test_hf_id_no_pooling_spec_defaults_mean(monkeypatch, tmp_path):
    # repo with no 1_Pooling (raw encoder) / offline → documented "assume mean" preserved
    _patch_hub(monkeypatch, tmp_path, FileNotFoundError("no 1_Pooling"))
    assert _model_is_mean_pooled("some/raw-encoder") is True
