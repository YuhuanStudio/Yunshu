"""Always-on rules use footprint, reset on restart, and alert only on transitions."""

from yunshu_console.poller import DeploymentWatch


def test_footprint_slope_and_restart():
    watch = DeploymentWatch()
    for minute in range(16):
        result = watch.update(minute * 60, 1, (), 20 + minute / 30, [])
    assert abs(result["slope_gib_h"] - 2) < 1e-9
    assert len(result["raised"]) == 1
    assert not watch.update(960, 1, (), 20 + 16 / 30, [])["raised"]
    assert watch.update(1020, 2, (), 30, [])["slope_gib_h"] is None
    assert not watch.state["alerts"]


def test_error_alert_minimum_and_recovery():
    watch = DeploymentWatch()
    rows = [{"status": 200}] * 19 + [{"status": 500}]
    assert not watch.update(0, 1, (), 20, rows[:19])["alerts"]
    assert len(watch.update(1, 1, (), 20, rows)["raised"]) == 1
    assert not watch.update(2, 1, (), 20, rows)["raised"]
    assert not watch.update(3, 1, (), 20, [{"status": 200}] * 20)["alerts"]
    assert watch.update(4, 1, (), 20, rows)["raised"]


def test_missing_footprint_and_model_change_reset():
    watch = DeploymentWatch()
    for minute in range(16):
        watch.update(minute * 60, 1, (), 20, [])
    assert watch.state["slope_gib_h"] == 0
    assert watch.update(960, 1, (), None, [])["slope_gib_h"] is None
    assert watch.update(1020, 1, (("new", True),), 40, [])["slope_gib_h"] is None


def test_fastest_poller_cadence_keeps_the_full_detection_window():
    watch = DeploymentWatch()
    for quarter_second in range(3601):
        t = quarter_second / 4
        result = watch.update(t, 1, (), 20 + t / 1800, [])
    assert abs(result["slope_gib_h"] - 2) < 1e-9
    assert result["raised"]
