from scripts.research.replay_spare_rows import replay


def round_record(**updates):
    row = {
        "b": 0,
        "bs": 6,
        "copy": False,
        "acc": 1,
        "drafted": [1, 2, 3, 4, 5],
        "tgt": [1, 20, 30, 40, 50, 60],
        "alts": [[1, 11], [2, 20], [3, 30], [4, 40], [5, 50]],
        "lp": [[-0.01, -5], [-4, -0.1], [-0.01, -0.1], [-0.01, -0.1], [-0.01, -0.1]],
    }
    row.update(updates)
    return row


def test_replay_only_credits_the_first_rejection_not_invalid_tail_targets():
    records = [round_record(), round_record(b=20, copy=True)]
    result = replay(records, 5)
    assert result["extra_committed_lower_bound"] == 1


def test_replay_does_not_spend_rows_using_future_target_ids():
    records = [
        round_record(tgt=[1, 999, 30, 40, 50, 60]),
        round_record(b=999, copy=True),
    ]
    result = replay(records, 5)
    assert result["extra_committed_lower_bound"] == 0


def test_replay_excludes_unobserved_final_budget():
    assert replay([round_record()], 5)["extra_committed_lower_bound"] == 0


def test_replay_excludes_actual_draft_instead_of_assuming_full_head_top1():
    record = round_record(acc=0, drafted=[11, 2, 3, 4, 5], tgt=[1, 20, 30, 40, 50, 60])
    result = replay([record, round_record(b=1, copy=True)], 1)
    assert result["extra_committed_lower_bound"] == 1
    assert result["top1_differs_from_draft"] == 1
