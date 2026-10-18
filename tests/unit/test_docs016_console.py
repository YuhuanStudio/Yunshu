"""CPU contract for the real-load screenshot receipt."""

from scripts.research.docs016_console import validate


def test_capture_requires_identical_inventory_and_real_load():
    images = {"en/overview.webp": [2880, 1800]}
    assert validate(images, images, 3) == []
    assert validate(images, {}, 3)
    assert validate(images, images, 2)


def test_long_session_requires_real_depth_cache_and_complete_replies():
    from scripts.research.docs016_console import validate_long_usage

    usage = {
        "prompt_tokens": 32768,
        "completion_tokens": 2048,
        "prompt_tokens_details": {"cached_tokens": 32768},
    }
    assert validate_long_usage(usage, True, "length") == []
    assert validate_long_usage({**usage, "completion_tokens": 100}, True, "stop")
    assert validate_long_usage({**usage, "prompt_tokens": 1024}, False, "length")
    assert validate_long_usage({**usage, "prompt_tokens_details": {}}, True, "length")


def test_capture_swap_reading_fails_closed():
    import pytest
    from scripts.research.docs016_console import swap_mb

    assert swap_mb("used = 10.25M") == 10.25
    assert swap_mb("used = 1.25G") == 1280
    with pytest.raises(ValueError):
        swap_mb("unknown")


def test_sticky_swap_allowed_only_without_growth_or_pageouts():
    import pytest
    from scripts.research.docs016_console import pageouts, validate_memory_window

    before = {"swap_mb": 9600, "pageouts": 100}
    assert validate_memory_window(before, dict(before)) == []
    # Background noise seen on 10-18 (6 host pageouts in a 6-minute 128K cell) must not reject it.
    assert (
        validate_memory_window(before, {**before, "swap_mb": 9601, "pageouts": 106})
        == []
    )
    assert validate_memory_window(before, {**before, "swap_mb": 9600 + 65})
    assert validate_memory_window(before, {**before, "pageouts": 100 + 4097})
    assert pageouts("Pageouts: 123.\n") == 123
    with pytest.raises(ValueError):
        pageouts("unknown")


def test_agent_followups_advance_to_distinct_implementation_tasks():
    from scripts.research.docs016_console import followup_task

    tasks = [followup_task(i) for i in range(1, 7)]
    assert len(set(tasks)) == 6
    assert "journal" in tasks[0]
    assert "cancellation" in tasks[1]
    assert all("full Python implementation" in task for task in tasks)


def test_natural_long_eos_is_complete_but_unterminated_output_fails():
    from scripts.research.docs016_console import validate_long_usage

    usage = {
        "prompt_tokens": 69818,
        "completion_tokens": 1886,
        "prompt_tokens_details": {"cached_tokens": 67676},
    }
    assert validate_long_usage(usage, True, "stop") == []
    assert validate_long_usage(usage, True)
    assert validate_long_usage(usage, True, "error")
    assert validate_long_usage({**usage, "completion_tokens": 1023}, True, "stop")
