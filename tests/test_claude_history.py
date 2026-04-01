import json
from datetime import datetime

import pytest

from ambient.daemon.claude_history import (
    filter_completed_sessions,
    group_into_sessions,
    read_new_history_entries,
    session_to_event,
)


def _make_entry(display, session_id, timestamp, project="/Users/test/project"):
    return {
        "display": display,
        "pastedContents": {},
        "timestamp": timestamp,
        "project": project,
        "sessionId": session_id,
    }


def _write_history(path, entries):
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


class TestReadNewHistoryEntries:
    def test_reads_from_start_line(self, tmp_path):
        path = tmp_path / "history.jsonl"
        entries = [
            _make_entry("old prompt", "s1", 1000),
            _make_entry("new prompt", "s2", 2000),
            _make_entry("newer prompt", "s2", 3000),
        ]
        _write_history(path, entries)

        result, line_count = read_new_history_entries(path, start_line=1)
        assert len(result) == 2
        assert result[0]["display"] == "new prompt"
        assert line_count == 3

    def test_skips_entries_without_session_id(self, tmp_path):
        path = tmp_path / "history.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"display": "/login", "timestamp": 1000}) + "\n")
            f.write(json.dumps(_make_entry("real prompt", "s1", 2000)) + "\n")

        result, _ = read_new_history_entries(path, start_line=0)
        assert len(result) == 1
        assert result[0]["display"] == "real prompt"

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "history.jsonl"
        with open(path, "w") as f:
            f.write("not json\n")
            f.write(json.dumps(_make_entry("good", "s1", 1000)) + "\n")

        result, _ = read_new_history_entries(path, start_line=0)
        assert len(result) == 1

    def test_empty_file(self, tmp_path):
        path = tmp_path / "history.jsonl"
        path.write_text("")
        result, line_count = read_new_history_entries(path, start_line=0)
        assert result == []
        assert line_count == 0

    def test_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.jsonl"
        result, line_count = read_new_history_entries(path, start_line=0)
        assert result == []
        assert line_count == 0


class TestGroupIntoSessions:
    def test_groups_by_session_id(self):
        entries = [
            _make_entry("prompt 1", "s1", 1000),
            _make_entry("prompt 2", "s1", 2000),
            _make_entry("prompt 3", "s2", 3000),
        ]
        sessions = group_into_sessions(entries)
        assert len(sessions) == 2

        s1 = next(s for s in sessions if s["session_id"] == "s1")
        assert s1["prompt_count"] == 2
        assert s1["ts_start"] == 1000
        assert s1["ts_end"] == 2000

    def test_single_entry_session(self):
        entries = [_make_entry("solo", "s1", 5000)]
        sessions = group_into_sessions(entries)
        assert len(sessions) == 1
        assert sessions[0]["prompt_count"] == 1
        assert sessions[0]["ts_start"] == 5000
        assert sessions[0]["ts_end"] == 5000

    def test_truncates_prompts_to_100_chars(self):
        long_prompt = "x" * 200
        entries = [_make_entry(long_prompt, "s1", 1000)]
        sessions = group_into_sessions(entries)
        assert len(sessions[0]["prompts"][0]) == 100


class TestFilterCompletedSessions:
    def test_separates_completed_from_in_progress(self):
        now_ms = 100_000_000
        sessions = [
            {"session_id": "old", "ts_start": 1000, "ts_end": 2000, "prompts": [], "prompt_count": 1, "project": ""},
            {"session_id": "recent", "ts_start": now_ms - 1000, "ts_end": now_ms - 500, "prompts": [], "prompt_count": 1, "project": ""},
        ]
        completed, in_progress = filter_completed_sessions(sessions, now_ms=now_ms)
        assert len(completed) == 1
        assert completed[0]["session_id"] == "old"
        assert len(in_progress) == 1
        assert in_progress[0]["session_id"] == "recent"


class TestSessionToEvent:
    def test_produces_correct_event_dict(self):
        session = {
            "session_id": "abc-123",
            "project": "/Users/test/myproject",
            "prompts": ["fix the auth bug in middleware"],
            "ts_start": 1000,
            "ts_end": 5000,
            "prompt_count": 1,
        }
        event = session_to_event(session)
        assert event["type"] == "claude_session"
        assert event["ts_start"] == 1000
        assert event["ts_end"] == 5000
        assert event["duration_ms"] == 4000
        assert event["command"] == "claude: fix the auth bug in middleware"
        assert event["cwd"] == "/Users/test/myproject"
        assert event["claude_session_id"] == "abc-123"
        assert event["claude_prompt_count"] == 1
        assert event["claude_prompts"] == ["fix the auth bug in middleware"]
        assert event["gap_ms"] is None
        assert event["exit_code"] == 0

    def test_zero_duration_single_prompt(self):
        session = {
            "session_id": "s1",
            "project": "/test",
            "prompts": ["quick question"],
            "ts_start": 1000,
            "ts_end": 1000,
            "prompt_count": 1,
        }
        event = session_to_event(session)
        assert event["duration_ms"] == 0

    def test_empty_prompts(self):
        session = {
            "session_id": "s1",
            "project": "/test",
            "prompts": [],
            "ts_start": 1000,
            "ts_end": 2000,
            "prompt_count": 0,
        }
        event = session_to_event(session)
        assert event["command"] == "claude: (session)"
