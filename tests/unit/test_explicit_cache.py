"""Unit tests for the Gemini-style explicit context cache store ."""

import time

from yunshu_gateway.explicit_cache import ExplicitContextCache


def _msgs():
    return [{"role": "system", "content": "long doc"}]


def test_create_get_use_delete():
    s = ExplicitContextCache()
    e = s.create("m", _msgs(), token_count=42, ttl_seconds=60)
    assert e.name.startswith("cachedContents/")
    assert e.token_count == 42
    assert s.get(e.name) is not None
    got = s.use(e.name)
    assert got.read_count == 1
    assert s.use(e.name).read_count == 2
    assert s.delete(e.name) is True
    assert s.get(e.name) is None


def test_ttl_expiry():
    s = ExplicitContextCache()
    e = s.create("m", _msgs(), token_count=1, ttl_seconds=1.0)
    # force expiry by rewinding expire_at
    e.expire_at = time.time() - 1
    assert s.get(e.name) is None  # evicted on access
    assert s.use(e.name) is None


def test_update_ttl():
    s = ExplicitContextCache()
    e = s.create("m", _msgs(), token_count=1, ttl_seconds=1.0)
    e.expire_at = time.time() - 1  # would be expired...
    # update before the next access reads it (update doesn't evict)
    s2 = ExplicitContextCache()
    e2 = s2.create("m", _msgs(), token_count=1, ttl_seconds=60)
    upd = s2.update_ttl(e2.name, 3600)
    assert upd is not None
    assert upd.expire_at > time.time() + 3000


def test_capacity_cap():
    s = ExplicitContextCache(max_entries=3)
    names = [
        s.create("m", _msgs(), token_count=1, ttl_seconds=3600).name for _ in range(5)
    ]
    listed = s.list()
    assert len(listed) <= 3
    # the most-recently-created (latest expiry) survive
    assert any(n in {e.name for e in listed} for n in names[-3:])


def test_to_api_shape():
    s = ExplicitContextCache()
    e = s.create("m", _msgs(), token_count=7, ttl_seconds=120, display_name="dn")
    api = e.to_api()
    for k in (
        "name",
        "model",
        "displayName",
        "createTime",
        "expireTime",
        "ttl",
        "usageMetadata",
    ):
        assert k in api
    assert api["usageMetadata"]["totalTokenCount"] == 7
    assert api["displayName"] == "dn"


def test_missing_returns_none():
    s = ExplicitContextCache()
    assert s.get("cachedContents/nope") is None
    assert s.use("cachedContents/nope") is None
    assert s.delete("cachedContents/nope") is False
    assert s.update_ttl("cachedContents/nope", 60) is None
