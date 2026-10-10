from yunshu_engine import penalty_context


class _Batch:
    """Stands in for mlx-vlm's PromptProcessingBatch: context = the tokens it prefills."""

    def __init__(self, suffix, meta):
        self._token_context = [list(suffix)]
        self._apc_meta = [meta]


def test_apc_hit_context_is_the_full_prompt_like_a_miss():
    penalty_context.install(_Batch)
    full = [1, 2, 3, 4, 5, 6]
    miss = _Batch(full, None)  # cold: the prefill sees everything
    hit = _Batch(full[4:], {"prefix_len": 4, "full_input_ids": full})
    assert hit._token_context == miss._token_context == [full]


def test_cold_rows_and_rows_without_context_are_left_alone():
    penalty_context.install(_Batch)
    cold = _Batch([7, 8], {"prefix_len": 0, "full_input_ids": [7, 8]})
    assert cold._token_context == [[7, 8]]
    none = _Batch([], None)
    none._token_context = []
    penalty_context.full_context(none)
    assert none._token_context == []
