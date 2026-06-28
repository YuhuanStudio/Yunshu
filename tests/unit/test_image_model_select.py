"""the image routes had the W801/W818/W823 wrong-model keystone. The
img2img-family routes (variation/edit/inpaint/controlnet/depth) and the t2i fallbacks
grabbed the FIRST loaded ImageGenEngine, ignoring req.model. With ≥2 image models loaded
that served the WRONG model — and since every image route gates can_access_model(req.model)
at the top but then served first-of-type, a key authorized for B could be served A (which
it may not access). _select_image_engine now matches by model_id."""
from __future__ import annotations

import inspect
import types

from yunshu_gateway.routers import (
    images as IMG,  # noqa: N812  # intentional short module alias
)
from yunshu_gateway.routers.images import _select_image_engine


class _ImgEngine:
    def __init__(self, tag):
        self.tag = tag


def _entry(model_id, engine, loaded=True):
    return types.SimpleNamespace(model_id=model_id, engine=engine, is_loaded=loaded)


def _mgr(entries, monkeypatch):
    # _select_image_engine imports ImageGenEngine and isinstance-checks against it;
    # make our stubs pass by patching the symbol the helper imports.
    import yunshu_engine.image_engine as ie
    monkeypatch.setattr(ie, "ImageGenEngine", _ImgEngine)
    return types.SimpleNamespace(list_entries=lambda: entries)


def test_selects_matching_model(monkeypatch):
    a, b = _ImgEngine("A"), _ImgEngine("B")
    mgr = _mgr([_entry("img-A", a), _entry("img-B", b)], monkeypatch)
    assert _select_image_engine(mgr, "img-B") is b
    assert _select_image_engine(mgr, "img-A") is a


def test_case_insensitive(monkeypatch):
    a = _ImgEngine("A")
    mgr = _mgr([_entry("Z-Image-Turbo", a)], monkeypatch)
    assert _select_image_engine(mgr, "z-image-turbo") is a


def test_no_match_multi_model_returns_none(monkeypatch):
    # ≥2 loaded + no match → None (wrong-model protection: don't guess among many)
    a, b = _ImgEngine("A"), _ImgEngine("B")
    mgr = _mgr([_entry("img-A", a), _entry("img-B", b)], monkeypatch)
    assert _select_image_engine(mgr, "img-Z") is None


def test_no_match_single_model_served(monkeypatch):
    # exactly one loaded + no match → serve it (single-model deployments are
    # unambiguous; req.model defaults to a hardcoded id that may not match the loaded one).
    a = _ImgEngine("A")
    mgr = _mgr([_entry("my-custom-flux", a)], monkeypatch)
    assert _select_image_engine(mgr, "Z-Image-Turbo-MLX-4bit") is a


def test_empty_model_first_of_type(monkeypatch):
    a, b = _ImgEngine("A"), _ImgEngine("B")
    mgr = _mgr([_entry("img-A", a), _entry("img-B", b)], monkeypatch)
    assert _select_image_engine(mgr, "") is a


def test_all_image_routes_use_selector():
    src = inspect.getsource(IMG)
    # no route still grabs the first-of-type engine
    assert "img_engine = entry.engine" not in src
    # every route resolves through the shared selector
    assert src.count("_select_image_engine(manager, req.model)") >= 7
