import mlx.core as mx

from yunshu_engine import dflash_copy


def test_first_round_without_target_hidden_states_does_not_crash():
    """A row promoted onto the lane has no hidden states until its first verify (release gate
    on 77466c75: AttributeError 'NoneType' has no attribute 'shape' in _hidden)."""
    from types import SimpleNamespace

    from yunshu_engine.dflash_context import neutral_context

    drafter = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[1, 2, 3, 4, 5], hidden_size=16)
    )
    context = neutral_context(drafter)
    assert tuple(context.shape) == (1, 1, 80) and float(context.sum().item()) == 0.0

    seen = {}

    class Draft:
        accept_lens: list = []

        def draft_block_greedy(self, bonus, hidden, cache, block, sampler, dtype):
            seen["hidden"] = hidden
            return mx.array([[7]], dtype=dtype)

    proxy = dflash_copy.CopyDraft(Draft(), [1, 2, 3], 4, 4)
    proxy._next_copy = [9, 9]
    out = proxy.draft_block_greedy(4, None, [], 3, None)  # copy island, no hidden yet
    assert out.tolist() == [[9, 9]] and proxy._pending_hidden is None
    proxy._next_copy = []
    proxy.draft_block_greedy(4, None, [], 3, None)  # model round: passes None through
    assert seen["hidden"] is None
