"""A request whose insert fails never became a row: its APC descriptor and console id are released."""

import collections
import importlib
from types import SimpleNamespace

import pytest

from tests.unit.test_vlm_runner_batching import FakeGen
from yunshu_engine import vlm_batch_runner as vbr


class FakeAPC:
    def __init__(self):
        self.manager = SimpleNamespace(_console_requests=collections.OrderedDict())
        self.set = []
        self.released = []

    def set_request(self, ids, plan):
        self.set.append(tuple(ids))

    def release_request(self, ids, plan):
        self.released.append(tuple(ids))


class FailingInsertGen(FakeGen):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.apc = FakeAPC()

    def insert(self, *a, **k):
        raise RuntimeError("insert exploded")


@pytest.fixture
def failing_runner(monkeypatch):
    ar = importlib.import_module("mlx_vlm.generate.ar")
    FakeGen.instances = []
    monkeypatch.setattr(ar, "BatchGenerator", FailingInsertGen)
    return vbr.VLMBatchRunner(SimpleNamespace(language_model=object()), processor=None)


def test_failed_insert_releases_the_apc_request(failing_runner):
    with pytest.raises(RuntimeError, match="insert exploded"):
        list(failing_runner.iter_tokens([1, 2, 3], max_tokens=3, prompt_kwargs={}))
    (gen,) = FakeGen.instances
    assert gen.apc.set == [(1, 2, 3)]
    assert gen.apc.released == [(1, 2, 3)]
    assert not gen.apc.manager._console_requests
    assert not failing_runner.busy()
