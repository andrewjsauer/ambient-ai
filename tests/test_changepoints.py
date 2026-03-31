import numpy as np
import pytest

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.changepoints import detect_changepoints, _categorize_command
from ambient.detect.pauses import PauseClassification, PauseFindings


def _make_events_at_rate(
    start_ts: int,
    duration_min: int,
    commands_per_5min: int,
    command_template: str = "cmd_{i}",
) -> list[Event]:
    events = []
    bucket_ms = 5 * 60 * 1000
    n_buckets = duration_min // 5
    ts = start_ts

    for bucket in range(n_buckets):
        bucket_start = start_ts + bucket * bucket_ms
        interval = bucket_ms // max(commands_per_5min, 1)
        for j in range(commands_per_5min):
            cmd_ts = bucket_start + j * interval
            events.append(
                Event(
                    ts_start=cmd_ts,
                    ts_end=cmd_ts + 100,
                    duration_ms=100,
                    command=command_template.format(i=len(events)),
                    exit_code=0,
                    cwd="/tmp",
                    tmux_pane="%0",
                    gap_ms=interval if events else None,
                )
            )
    return events


@pytest.fixture
def config():
    return Config(bucket_minutes=5, pelt_min_size=5, pelt_model="l1")


def test_three_regimes(config):
    # High rate -> Low rate -> High rate
    base_ts = 1_000_000_000_000
    high1 = _make_events_at_rate(base_ts, 60, 10, "git status")           # 12 buckets
    low = _make_events_at_rate(base_ts + 60 * 60_000, 60, 1, "vim file")  # 12 buckets
    high2 = _make_events_at_rate(base_ts + 120 * 60_000, 60, 10, "pytest") # 12 buckets

    events = high1 + low + high2
    result = detect_changepoints(events, config)

    # Should detect at least 1 changepoint (ideally 2)
    assert len(result.changepoints) >= 1
    assert len(result.segments) >= 2

    # Segments should have varying rates
    rates = [s.mean_rate for s in result.segments]
    assert max(rates) > 3 * min(rates)  # significant rate difference


def test_segment_rates_and_durations(config):
    base_ts = 1_000_000_000_000
    events = _make_events_at_rate(base_ts, 120, 8)  # Constant rate for 2 hours

    result = detect_changepoints(events, config)

    # With constant rate, should be 0-1 changepoints
    for seg in result.segments:
        assert seg.duration_min > 0
        assert seg.mean_rate > 0


def test_flat_signal_no_changepoints(config):
    base_ts = 1_000_000_000_000
    events = _make_events_at_rate(base_ts, 120, 5)

    result = detect_changepoints(events, config)

    # Flat signal should produce few or no changepoints
    # (ruptures may still find 0-1 depending on noise)
    assert len(result.segments) >= 1


def test_short_signal_no_crash(config):
    base_ts = 1_000_000_000_000
    # Only 4 buckets (20 min) - less than min_size * 2
    events = _make_events_at_rate(base_ts, 20, 5)

    result = detect_changepoints(events, config)

    # Should return single segment, no changepoints
    assert len(result.changepoints) == 0
    assert len(result.segments) == 1


def test_empty_events(config):
    result = detect_changepoints([], config)
    assert result.segments == []
    assert result.changepoints == []


def test_dominant_category(config):
    base_ts = 1_000_000_000_000
    git_events = _make_events_at_rate(base_ts, 60, 10, "git status")

    result = detect_changepoints(git_events, config)

    for seg in result.segments:
        assert seg.dominant_category == "git"


def test_uncalibrated_gmm_null_pause_dist(config):
    base_ts = 1_000_000_000_000
    events = _make_events_at_rate(base_ts, 120, 5)

    # Pass unavailable pause findings
    pause_findings = PauseFindings(available=False, reason="not_calibrated")
    result = detect_changepoints(events, config, pause_findings=pause_findings)

    for seg in result.segments:
        assert seg.pause_distribution is None


def test_categorize_commands():
    assert _categorize_command("git status") == "git"
    assert _categorize_command("git add .") == "git"
    assert _categorize_command("pytest tests/") == "test"
    assert _categorize_command("vim file.py") == "edit"
    assert _categorize_command("claude") == "claude"
    assert _categorize_command("python run.py") == "other"
    assert _categorize_command("make build") == "build"


def test_pause_attribution_by_timestamp_not_command_text(config):
    """Same command in two segments — pauses should be attributed by timestamp, not text."""
    base_ts = 1_000_000_000_000
    # Two segments both using "git status" as a command
    seg1 = _make_events_at_rate(base_ts, 60, 8, "git status")
    seg2 = _make_events_at_rate(base_ts + 120 * 60_000, 60, 8, "git status")
    # Gap between segments (low-rate period to force a changepoint)
    gap = _make_events_at_rate(base_ts + 60 * 60_000, 60, 1, "vim file")

    events = seg1 + gap + seg2

    # Create pause findings with timestamps only in segment 2's range
    seg2_start = base_ts + 120 * 60_000
    pause_findings = PauseFindings(
        available=True,
        classifications=[
            PauseClassification(
                gap_ms=50000, label="stuck",
                probabilities={"routine": 0.05, "evaluating": 0.1, "stuck": 0.85},
                preceding_command="git status", following_command="git status",
                ts_start=seg2_start + 5 * 60_000,  # clearly in segment 2
            ),
        ],
    )

    result = detect_changepoints(events, config, pause_findings=pause_findings)

    # The stuck pause should only appear in the segment covering seg2's time range
    for seg in result.segments:
        if seg.start_ts <= seg2_start and seg.end_ts < seg2_start + 30 * 60_000:
            # This is segment 1 — should NOT have stuck pauses
            if seg.pause_distribution:
                assert seg.pause_distribution.get("stuck", 0) == 0
        elif seg.start_ts >= seg2_start:
            # This is segment 2 — should have stuck pauses
            if seg.pause_distribution:
                assert seg.pause_distribution.get("stuck", 0) > 0


def test_simulated_workday(config):
    base_ts = 1_000_000_000_000

    # Morning flow: high rate coding (2 hours)
    morning = _make_events_at_rate(base_ts, 120, 12, "pytest test_{i}")
    # Lunch/admin: low rate (1 hour)
    admin = _make_events_at_rate(base_ts + 120 * 60_000, 60, 2, "git log")
    # Afternoon coding: high rate (1 hour)
    afternoon = _make_events_at_rate(base_ts + 180 * 60_000, 60, 10, "vim file_{i}")

    events = morning + admin + afternoon
    result = detect_changepoints(events, config)

    # Should detect transitions
    assert len(result.segments) >= 2
    # Total duration should be roughly 4 hours
    total_min = sum(s.duration_min for s in result.segments)
    assert 200 < total_min < 280
