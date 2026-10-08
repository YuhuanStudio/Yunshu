"""Per-request drafted / accepted counts of the single-row speculative lane."""

from types import SimpleNamespace

from yunshu_engine.vlm_batch_runner import RunStats, _note_spec, _spec_counters
from yunshu_gateway import x_yunshu


def test_spec_counters_diff_per_request():
    drafter = SimpleNamespace(
        speculative_total_rounds=10,
        speculative_total_accepted=30.0,
        speculative_total_drafted=50,
    )
    job = SimpleNamespace(stats=RunStats(), spec_base=_spec_counters(drafter))
    # two rounds of the request's own: 10 drafted, 5 accepted
    drafter.speculative_total_rounds += 2
    drafter.speculative_total_accepted += 5.0
    drafter.speculative_total_drafted += 10
    _note_spec(drafter, job)
    assert (job.stats.spec_drafted, job.stats.spec_accepted) == (10, 5)
    job.stats.spec_mode = "mtp"
    info = x_yunshu.RequestInfo("r", "POST", "/v1/chat/completions")
    info.gen = SimpleNamespace(stats=job.stats)
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
    _note_spec(None, job)  # no drafter: leaves the numbers alone
    assert job.stats.spec_drafted == 10
