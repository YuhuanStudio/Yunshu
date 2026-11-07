"""Per-request drafted / accepted counts of a drafting driver row, as the stats API reports them."""

from types import SimpleNamespace

from yunshu_engine.vlm_batch_runner import RunStats
from yunshu_gateway import x_yunshu


def test_spec_counters_reach_the_stats_api():
    stats = RunStats()
    # what ``RoundDriver._observe`` adds per round (test_round_driver covers the row side):
    # two rounds of the request's own, 10 drafted, 5 accepted
    stats.spec_rounds, stats.spec_drafted, stats.spec_accepted = 2, 10, 5
    stats.spec_mode = "mtp"
    info = x_yunshu.RequestInfo("r", "POST", "/v1/chat/completions")
    info.gen = SimpleNamespace(stats=stats)
    usage = {"prompt_tokens": 1, "completion_tokens": 2}
    spec = x_yunshu.build_stats(info, usage)["speculative"]
    assert spec == {
        "mode": "mtp",
        "per_depth": [],
        "position_basis": "depth",
        "drafted": 10,
        "accepted": 5,
        "acceptance_rate": 0.5,
        "rounds": 2,
    }
