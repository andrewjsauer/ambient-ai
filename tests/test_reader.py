import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ambient.capture.reader import Event, read_events
from ambient.config import Config


def _make_event(ts_start: int, command: str = "echo hi", gap_ms: int | None = None, **kwargs) -> dict:
    return {
        "ts_start": ts_start,
        "ts_end": ts_start + 100,
        "duration_ms": 100,
        "command": command,
        "exit_code": 0,
        "cwd": "/tmp",
        "tmux_pane": "%0",
        "gap_ms": gap_ms,
        **kwargs,
    }


def _write_events(path: Path, events: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


@pytest.fixture
def tmp_config(tmp_path):
    config = Config(base_dir=tmp_path)
    config.ensure_dirs()
    return config


def test_read_events_from_date(tmp_config):
    events = [_make_event(1000 * i) for i in range(10)]
    _write_events(tmp_config.events_path("2026-03-30"), events)

    result = read_events(tmp_config, date_str="2026-03-30")
    assert len(result) == 10
    assert all(isinstance(e, Event) for e in result)


def test_read_events_time_range(tmp_config):
    # Create events spanning a range
    base_ts = int(datetime(2026, 3, 30, 14, 0).timestamp() * 1000)
    events = [_make_event(base_ts + i * 60_000, command=f"cmd_{i}") for i in range(10)]
    _write_events(tmp_config.events_path("2026-03-30"), events)

    # Read only middle 4 events (minutes 3-6)
    start = datetime(2026, 3, 30, 14, 3)
    end = datetime(2026, 3, 30, 14, 6)
    result = read_events(tmp_config, start=start, end=end)
    assert len(result) == 4
    assert result[0].command == "cmd_3"
    assert result[-1].command == "cmd_6"


def test_malformed_lines_skipped(tmp_config):
    path = tmp_config.events_path("2026-03-30")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(_make_event(1000)) + "\n")
        f.write("not valid json\n")
        f.write(json.dumps(_make_event(2000)) + "\n")

    result = read_events(tmp_config, date_str="2026-03-30")
    assert len(result) == 2


def test_empty_file(tmp_config):
    path = tmp_config.events_path("2026-03-30")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")

    result = read_events(tmp_config, date_str="2026-03-30")
    assert result == []


def test_missing_file(tmp_config):
    result = read_events(tmp_config, date_str="2026-03-30")
    assert result == []


def test_midnight_crossing(tmp_config):
    # Events near midnight on day 1
    day1_ts = int(datetime(2026, 3, 30, 23, 55).timestamp() * 1000)
    day1_events = [_make_event(day1_ts + i * 60_000) for i in range(3)]
    _write_events(tmp_config.events_path("2026-03-30"), day1_events)

    # Events just after midnight on day 2
    day2_ts = int(datetime(2026, 3, 31, 0, 2).timestamp() * 1000)
    day2_events = [_make_event(day2_ts + i * 60_000) for i in range(3)]
    _write_events(tmp_config.events_path("2026-03-31"), day2_events)

    # Read across midnight
    start = datetime(2026, 3, 30, 23, 54)
    end = datetime(2026, 3, 31, 0, 5)
    result = read_events(tmp_config, start=start, end=end)
    assert len(result) == 6


def test_claude_session_event_round_trips(tmp_config):
    """Claude session events with all new fields should round-trip through JSONL."""
    events = [
        {
            **_make_event(1000, command="claude: fix tests"),
            "type": "claude_session",
            "claude_session_id": "abc-123",
            "claude_prompts": ["fix the failing tests", "yes, apply that"],
            "claude_tools": [{"name": "Edit", "files": ["src/foo.py"]}],
            "claude_files": ["src/foo.py"],
            "claude_project": "my-project",
            "claude_prompt_count": 2,
            "claude_is_error_count": 0,
        }
    ]
    _write_events(tmp_config.events_path("2026-04-08"), events)

    result = read_events(tmp_config, date_str="2026-04-08")
    assert len(result) == 1
    e = result[0]
    assert e.type == "claude_session"
    assert e.claude_session_id == "abc-123"
    assert e.claude_prompts == ["fix the failing tests", "yes, apply that"]
    assert e.claude_tools == [{"name": "Edit", "files": ["src/foo.py"]}]
    assert e.claude_files == ["src/foo.py"]
    assert e.claude_project == "my-project"
    assert e.claude_prompt_count == 2
    assert e.claude_is_error_count == 0


def test_shell_event_claude_fields_default_none(tmp_config):
    """Shell command events should have all Claude fields as None."""
    events = [_make_event(1000)]
    _write_events(tmp_config.events_path("2026-04-08"), events)

    result = read_events(tmp_config, date_str="2026-04-08")
    e = result[0]
    assert e.type == "command"
    assert e.claude_session_id is None
    assert e.claude_prompts is None
    assert e.claude_tools is None
    assert e.claude_files is None
    assert e.claude_project is None
    assert e.claude_prompt_count is None
    assert e.claude_is_error_count is None


def test_partial_claude_fields(tmp_config):
    """Events with some Claude fields present and others missing."""
    events = [
        {
            **_make_event(1000),
            "type": "claude_session",
            "claude_session_id": "xyz-789",
            "claude_project": "partial-project",
            # other Claude fields missing
        }
    ]
    _write_events(tmp_config.events_path("2026-04-08"), events)

    result = read_events(tmp_config, date_str="2026-04-08")
    e = result[0]
    assert e.claude_session_id == "xyz-789"
    assert e.claude_project == "partial-project"
    assert e.claude_prompts is None
    assert e.claude_tools is None


def test_extra_unknown_fields_ignored(tmp_config):
    """Events with unknown extra fields should not raise."""
    events = [
        {
            **_make_event(1000),
            "some_future_field": "value",
            "another_unknown": 42,
        }
    ]
    _write_events(tmp_config.events_path("2026-04-08"), events)

    result = read_events(tmp_config, date_str="2026-04-08")
    assert len(result) == 1


def test_session_boundary_preserved(tmp_config):
    events = [
        _make_event(1000, gap_ms=None),
        _make_event(2000, gap_ms=700000, session_boundary=True),
    ]
    _write_events(tmp_config.events_path("2026-03-30"), events)

    result = read_events(tmp_config, date_str="2026-03-30")
    assert result[0].session_boundary is False
    assert result[1].session_boundary is True
    assert result[1].gap_ms == 700000
