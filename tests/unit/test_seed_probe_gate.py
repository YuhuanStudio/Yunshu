"""Seed identity requires successful nonempty generations in every path."""

from types import SimpleNamespace

from scripts.research.probe_seed_path_identity import consume_tokens, identity_complete


def results(tokens):
    return {mode: (list(tokens), None) for mode in ("L", "S", "M")}


def test_empty_matching_outputs_are_not_parity():
    assert not identity_complete(results([]), [])
    assert identity_complete(results([1, 2]), [])


def test_thread_failure_after_matching_partial_tokens_fails_the_gate():
    def broken(*_a, **_kw):
        yield 1
        raise RuntimeError("generation failed")

    sink, errors = [], []
    consumer = consume_tokens(
        SimpleNamespace(iter_tokens=broken), [0], {}, 42, True, 16, sink, None, errors
    )
    consumer.join(timeout=5)
    assert not consumer.is_alive()
    assert sink == [1]
    assert errors[0]["seed"] == 42
    assert "generation failed" in errors[0]["error"]
    assert not identity_complete(results(sink), errors)


def test_successful_thread_records_tokens_without_errors():
    sink, errors = [], []
    runner = SimpleNamespace(iter_tokens=lambda *_a, **_kw: iter([1, 2]))
    consume_tokens(runner, [0], {}, 42, False, 2, sink, None, errors).join(timeout=5)
    assert sink == [1, 2]
    assert not errors
    paired = results(sink)
    paired["M"] = ([1, 3], None)
    assert not identity_complete(paired, errors)
