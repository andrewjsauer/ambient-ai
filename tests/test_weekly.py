import json
import logging
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from ambient.config import Config
from ambient.present.narrator import narrate_weekly, load_batch_analyses
from ambient.present.prompts import WEEKLY_SYSTEM, build_weekly_prompt


@pytest.fixture
def config(tmp_path):
    return Config(base_dir=tmp_path)


def _write_analysis(config, date_str, data=None):
    """Write a fake analysis JSONL entry for a given date."""
    if data is None:
        data = {
            "timestamp": f"{date_str}T12:00:00",
            "compression": {
                "sequences": [{"sequence": ["git add", "git commit"], "count": 3,
                               "total_time_ms": 5000, "compression_gain": 5}],
                "compression_ratio": 0.72,
            },
            "pauses": {
                "available": True,
                "classifications": [
                    {"label": "routine", "gap_ms": 2000,
                     "preceding_command": "ls", "following_command": "vim"},
                    {"label": "stuck", "gap_ms": 60000,
                     "preceding_command": "pytest", "following_command": "vim"},
                ],
            },
            "project_allocation": {
                "allocations": [
                    {"project": "ambient-ai", "total_ms": 3600000, "event_count": 50,
                     "session_count": 2},
                ],
            },
            "analysis": {"work_phase": {"current": "coding"}},
        }
    config.ensure_dirs()
    path = config.analysis_path(date_str)
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


def _populate_weeks(config, num_weeks=3):
    """Populate analysis files for N weeks ending last Sunday."""
    today = datetime.now()
    # Find last Sunday
    days_since_sunday = (today.weekday() + 1) % 7
    last_sunday = today - timedelta(days=days_since_sunday)

    for w in range(num_weeks):
        week_end = last_sunday - timedelta(weeks=w)
        # Write data for 3 days in each week
        for d in [0, 2, 4]:
            day = week_end - timedelta(days=d)
            _write_analysis(config, day.strftime("%Y-%m-%d"))


def test_weekly_prompt_has_required_sections():
    """WEEKLY_SYSTEM prompt has all required sections."""
    sections = [
        "## Week Overview",
        "## Project Allocation Trends",
        "## Pattern Trends",
        "## Stuck Episode Trends",
        "## Recommendation Adoption",
        "## Key Shifts",
    ]
    for section in sections:
        assert section in WEEKLY_SYSTEM, f"Missing section: {section}"


def test_build_weekly_prompt_with_data():
    """build_weekly_prompt formats multi-week data correctly."""
    weeks = [
        {
            "date_range": "2026-03-31 to 2026-04-06",
            "days": [
                {
                    "date": "2026-04-01",
                    "compression": {"compression_ratio": 0.7, "sequences": [{"seq": 1}]},
                    "pauses": {"classifications": [
                        {"label": "stuck", "gap_ms": 60000},
                        {"label": "routine", "gap_ms": 2000},
                    ]},
                },
            ],
        },
        {
            "date_range": "2026-03-24 to 2026-03-30",
            "days": [
                {
                    "date": "2026-03-25",
                    "compression": {"compression_ratio": 0.8, "sequences": []},
                    "pauses": {"classifications": []},
                },
            ],
        },
    ]
    prompt = build_weekly_prompt(weeks, ["Current week", "Week -1"])
    assert "Current week" in prompt
    assert "Week -1" in prompt
    assert "2026-04-01" in prompt
    assert "0.700" in prompt


def test_build_weekly_prompt_empty_week():
    """Empty weeks are handled gracefully."""
    weeks = [
        {"date_range": "2026-03-31 to 2026-04-06", "days": []},
    ]
    prompt = build_weekly_prompt(weeks, ["Current week"])
    assert "No analysis data" in prompt


@patch("ambient.present.narrator._call_api")
def test_narrate_weekly_happy_path(mock_api, config):
    """3 weeks of data produces a weekly digest."""
    mock_api.return_value = "## Week Overview\nYou had a productive week..."

    _populate_weeks(config, num_weeks=3)

    # Build the weekly data
    today = datetime.now()
    weekly_analyses = []
    week_labels = []
    for w in range(3):
        week_end = today - timedelta(weeks=w)
        week_start = week_end - timedelta(days=6)
        date_range = f"{week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}"

        days_data = []
        current_day = week_start.date()
        while current_day <= week_end.date():
            ds = current_day.strftime("%Y-%m-%d")
            analyses = load_batch_analyses(config, ds)
            if analyses:
                merged = {"date": ds}
                for a in analyses:
                    if "compression" in a:
                        merged["compression"] = a["compression"]
                    if "pauses" in a:
                        merged["pauses"] = a["pauses"]
                days_data.append(merged)
            current_day += timedelta(days=1)

        weekly_analyses.append({"date_range": date_range, "days": days_data})
        week_labels.append("Current week" if w == 0 else f"Week -{w}")

    result = narrate_weekly(weekly_analyses, week_labels, config, date_str="2026-04-06")

    assert "productive week" in result
    mock_api.assert_called_once()
    # Verify file was written
    assert config.weekly_summary_path("2026-04-06").exists()


@patch("ambient.present.narrator._call_api")
def test_narrate_weekly_only_one_week(mock_api, config, caplog):
    """Only 1 week of data -- narrate_weekly still works (gating is in tick)."""
    mock_api.return_value = "## Week Overview\nLimited data..."

    _populate_weeks(config, num_weeks=1)

    weekly_analyses = [
        {"date_range": "2026-03-31 to 2026-04-06", "days": [{"date": "2026-04-01"}]},
    ]
    week_labels = ["Current week"]

    result = narrate_weekly(weekly_analyses, week_labels, config, date_str="2026-04-06")
    assert "Limited data" in result
    assert config.weekly_summary_path("2026-04-06").exists()


@patch("ambient.present.narrator._call_api")
def test_narrate_weekly_api_failure(mock_api, config):
    """API failure returns None and writes nothing — a placeholder at the
    canonical path would satisfy the daemon's gate and block the retry."""
    mock_api.side_effect = Exception("API quota exceeded")

    weekly_analyses = [
        {"date_range": "2026-03-31 to 2026-04-06", "days": [{"date": "2026-04-01"}]},
        {"date_range": "2026-03-24 to 2026-03-30", "days": [{"date": "2026-03-25"}]},
    ]
    week_labels = ["Current week", "Week -1"]

    result = narrate_weekly(weekly_analyses, week_labels, config, date_str="2026-04-06")
    assert result is None
    assert not config.weekly_summary_path("2026-04-06").exists()


@patch("ambient.present.narrator._call_api")
def test_check_weekly_summary_failure_does_not_advance(mock_api, config):
    """A failed Sunday generation leaves last_weekly_summary_date unset."""
    from ambient.daemon.state import DaemonState
    from ambient.daemon.tick import _check_weekly_summary

    mock_api.side_effect = Exception("outage")
    state = DaemonState()
    _write_analysis(config, "2026-04-01")  # current week (Mar 30 - Apr 5)
    _write_analysis(config, "2026-03-25")  # week -1

    with patch("ambient.daemon.tick.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 4, 5, 12, 0)  # Sunday
        mock_dt.strptime = datetime.strptime
        _check_weekly_summary(config, state)

    assert state.last_weekly_summary_date == ""
    assert not config.weekly_summary_path("2026-04-05").exists()


@patch("ambient.present.narrator._call_api")
def test_check_weekly_summary_overdue_retry_after_sunday(mock_api, config):
    """A weekly summary lost to a Sunday outage is regenerated on the next
    tick once the last success is 8+ days old, even on a Monday."""
    from ambient.daemon.state import DaemonState
    from ambient.daemon.tick import _check_weekly_summary

    mock_api.return_value = "recovered weekly narrative"
    # Last success was the PREVIOUS Sunday; this week's Sunday (Apr 5) failed.
    state = DaemonState(last_weekly_summary_date="2026-03-29")
    _write_analysis(config, "2026-04-01")
    _write_analysis(config, "2026-03-25")

    with patch("ambient.daemon.tick.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 4, 6, 12, 0)  # Monday, 8 days after
        mock_dt.strptime = datetime.strptime
        _check_weekly_summary(config, state)

    # The cadence anchors to the MISSED Sunday (Apr 5), not the Monday it
    # fired — otherwise the next real Sunday sees days_since=6 and skips,
    # drifting the schedule one weekday per outage.
    assert state.last_weekly_summary_date == "2026-04-05"
    assert config.weekly_summary_path("2026-04-06").exists()

    # And the following Sunday (Apr 12, days_since=7) fires on schedule.
    with patch("ambient.daemon.tick.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 4, 12, 12, 0)  # next Sunday
        mock_dt.strptime = datetime.strptime
        _check_weekly_summary(config, state)
    assert state.last_weekly_summary_date == "2026-04-12"


@patch("ambient.present.narrator._call_api")
def test_check_weekly_summary_not_overdue_at_seven_days(mock_api, config):
    """Non-Sunday with days_since exactly 7 (the normal post-Sunday state)
    must NOT fire — only 8+ days means a Sunday was missed."""
    from ambient.daemon.state import DaemonState
    from ambient.daemon.tick import _check_weekly_summary

    state = DaemonState(last_weekly_summary_date="2026-03-30")  # Monday
    _write_analysis(config, "2026-04-01")
    _write_analysis(config, "2026-03-25")

    with patch("ambient.daemon.tick.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 4, 6, 12, 0)  # Monday, 7 days
        mock_dt.strptime = datetime.strptime
        _check_weekly_summary(config, state)

    mock_api.assert_not_called()
    assert state.last_weekly_summary_date == "2026-03-30"


@patch("ambient.present.narrator._call_api")
def test_check_weekly_summary_sunday_skips_when_recent(mock_api, config):
    """Sunday with days_since=6 (generated mid-week, e.g. an anchor bug)
    must skip — the < 7 suppression guards double generation."""
    from ambient.daemon.state import DaemonState
    from ambient.daemon.tick import _check_weekly_summary

    state = DaemonState(last_weekly_summary_date="2026-03-30")  # prior Monday
    _write_analysis(config, "2026-04-01")
    _write_analysis(config, "2026-03-25")

    with patch("ambient.daemon.tick.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 4, 5, 12, 0)  # Sunday, 6 days
        mock_dt.strptime = datetime.strptime
        _check_weekly_summary(config, state)

    mock_api.assert_not_called()
    assert state.last_weekly_summary_date == "2026-03-30"


def test_weekly_summary_path(config):
    """Config.weekly_summary_path returns correct path."""
    path = config.weekly_summary_path("2026-04-06")
    assert path.name == "weekly-2026-04-06.md"
    assert path.parent == config.analysis_dir


@patch("ambient.present.narrator._call_api")
def test_check_weekly_summary_not_sunday(mock_api, config):
    """_check_weekly_summary skips if not Sunday."""
    from ambient.daemon.state import DaemonState
    from ambient.daemon.tick import _check_weekly_summary

    state = DaemonState()

    # Patch datetime to a Monday
    with patch("ambient.daemon.tick.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 4, 6)  # Monday
        mock_dt.strptime = datetime.strptime
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        _check_weekly_summary(config, state)

    mock_api.assert_not_called()
    assert state.last_weekly_summary_date == ""


@patch("ambient.present.narrator._call_api")
def test_check_weekly_summary_insufficient_data(mock_api, config, caplog):
    """_check_weekly_summary skips with log when not enough weeks of data."""
    from ambient.daemon.state import DaemonState
    from ambient.daemon.tick import _check_weekly_summary

    state = DaemonState()

    # Sunday with no analysis files
    with patch("ambient.daemon.tick.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 4, 5, 12, 0)  # Sunday
        mock_dt.strptime = datetime.strptime
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        with caplog.at_level(logging.INFO):
            _check_weekly_summary(config, state)

    mock_api.assert_not_called()
    assert "skipped" in caplog.text.lower() or state.last_weekly_summary_date == ""
