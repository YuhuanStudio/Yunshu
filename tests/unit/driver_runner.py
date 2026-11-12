"""A real ``VLMBatchRunner`` on the generic round-driver layer over a tiny real model: the path every family takes
now that the stock runner is gone. Same call shape as the ``VLMBatchRunner`` constructor the old tests used."""

from yunshu_engine.round_driver.driver import RoundDriver
from yunshu_engine.vlm_batch_runner import VLMBatchRunner


def driver_runner(
    model, processor=None, apc_manager=None, apc_semantic_hash=None, **kw
):
    runner = VLMBatchRunner(
        model,
        processor,
        apc_manager=apc_manager,
        apc_semantic_hash=apc_semantic_hash,
        **kw,
    )
    from yunshu_engine.round_driver.layer import GenericLayer, language_model

    runner.driver = RoundDriver(
        model,
        layer=GenericLayer(language_model(model)),
        apc=apc_manager,
        chunk=2048,
        stop_tokens=set(),
    )
    return runner
