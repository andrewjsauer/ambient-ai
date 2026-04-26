"""Tests for the project ledger detector."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.project_ledger import (
    ProjectLedger,
    ProjectLedgerEntry,
    detect_project_ledger,
)


# ---------- helpers ----------

def _user_line(ts: str, content_text: str, session_id: str = "sess-1") -> str:
    obj = {
        "type": "user",
        "sessionId": session_id,
        "timestamp": ts,
        "message": {"content": content_text},
    }
    return json.dumps(obj) + "\n"


def _make_projects_dir(tmp_path: Path, layout: dict[str, list[str]]) -> Path:
    root = tmp_path / "projects"
    root.mkdir()
    for slug, lines in layout.items():
        slug_dir = root / slug
        slug_dir.mkdir()
        (slug_dir / f"{slug}-session.jsonl").write_text("".join(lines), encoding="utf-8")
    return root


def _config(tmp_path: Path, **overrides) -> Config:
    cfg = Config(base_dir=tmp_path / ".ambient")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


_FIXED_TS_MS = int(datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc).timestamp() * 1000)


def _claude_session_event(
    project_path: str, duration_ms: int, files: list[str] | None = None
) -> Event:
    return Event(
        ts_start=_FIXED_TS_MS,
        ts_end=_FIXED_TS_MS + duration_ms,
        duration_ms=duration_ms,
        command="claude: doing stuff",
        exit_code=0,
        cwd=project_path,
        tmux_pane=None,
        gap_ms=None,
        type="claude_session",
        claude_project=project_path,
        claude_files=files or [],
    )


def _shell_event(cwd: str, duration_ms: int) -> Event:
    return Event(
        ts_start=_FIXED_TS_MS,
        ts_end=_FIXED_TS_MS + duration_ms,
        duration_ms=duration_ms,
        command="ls -la",
        exit_code=0,
        cwd=cwd,
        tmux_pane=None,
        gap_ms=None,
        type="command",
    )


WINDOW_START = datetime(2026, 4, 20, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 4, 27, tzinfo=timezone.utc)
T_INSIDE = "2026-04-22T15:00:00Z"


# ---------- Smoke / shape ----------

class TestLedgerShape:
    def test_returns_project_ledger_dataclass(self, tmp_path):
        cfg = _config(tmp_path)
        result = detect_project_ledger(
            [], tmp_path, WINDOW_START, WINDOW_END, cfg, skip_summaries=True
        )
        assert isinstance(result, ProjectLedger)
        assert result.entries == []
        assert result.time_basis == "command_span"

    def test_window_iso_strings_populated(self, tmp_path):
        cfg = _config(tmp_path)
        result = detect_project_ledger(
            [], tmp_path, WINDOW_START, WINDOW_END, cfg, skip_summaries=True
        )
        assert result.window_start_iso.startswith("2026-04-20")
        assert result.window_end_iso.startswith("2026-04-27")


# ---------- Time floor + sessions + top files ----------

class TestEntryAggregation:
    def test_time_floor_filters_quiet_projects(self, tmp_path):
        # min_active_ms default = 600_000 (10 min)
        events = [
            _claude_session_event("/a/ambient-ai", duration_ms=15 * 60_000),
            _claude_session_event("/b/quiet-proj", duration_ms=2 * 60_000),
        ]
        cfg = _config(tmp_path)
        result = detect_project_ledger(
            events, tmp_path, WINDOW_START, WINDOW_END, cfg, skip_summaries=True
        )
        names = [e.project for e in result.entries]
        assert "ambient-ai" in names
        assert "quiet-proj" not in names

    def test_session_count_reflects_only_claude_sessions(self, tmp_path):
        events = [
            _claude_session_event("/a/proj", duration_ms=15 * 60_000),
            _claude_session_event("/a/proj", duration_ms=15 * 60_000),
            _shell_event("/a/proj", duration_ms=5 * 60_000),
        ]
        cfg = _config(tmp_path)
        result = detect_project_ledger(
            events, tmp_path, WINDOW_START, WINDOW_END, cfg, skip_summaries=True
        )
        proj_entry = next(e for e in result.entries if e.project == "proj")
        assert proj_entry.session_count == 2  # only claude_sessions

    def test_top_files_orders_by_frequency(self, tmp_path):
        events = [
            _claude_session_event(
                "/a/proj", duration_ms=15 * 60_000,
                files=["/repo/a.py", "/repo/a.py", "/repo/b.py"],
            ),
            _claude_session_event(
                "/a/proj", duration_ms=15 * 60_000,
                files=["/repo/a.py", "/repo/c.py"],
            ),
        ]
        cfg = _config(tmp_path)
        result = detect_project_ledger(
            events, tmp_path, WINDOW_START, WINDOW_END, cfg, skip_summaries=True
        )
        proj_entry = next(e for e in result.entries if e.project == "proj")
        assert proj_entry.top_files[0] == "a.py"  # most frequent
        assert "b.py" in proj_entry.top_files
        assert "c.py" in proj_entry.top_files

    def test_top_files_n_caps_results(self, tmp_path):
        events = [
            _claude_session_event(
                "/a/proj", duration_ms=15 * 60_000,
                files=[f"/repo/file{i}.py" for i in range(10)],
            ),
        ]
        cfg = _config(tmp_path, project_ledger_top_files_n=3)
        result = detect_project_ledger(
            events, tmp_path, WINDOW_START, WINDOW_END, cfg, skip_summaries=True
        )
        proj_entry = next(e for e in result.entries if e.project == "proj")
        assert len(proj_entry.top_files) == 3

    def test_entries_sorted_by_active_ms_desc(self, tmp_path):
        events = [
            _claude_session_event("/a/small-proj", duration_ms=11 * 60_000),
            _claude_session_event("/b/big-proj", duration_ms=60 * 60_000),
            _claude_session_event("/c/medium-proj", duration_ms=30 * 60_000),
        ]
        cfg = _config(tmp_path)
        result = detect_project_ledger(
            events, tmp_path, WINDOW_START, WINDOW_END, cfg, skip_summaries=True
        )
        names = [e.project for e in result.entries]
        assert names == ["big-proj", "medium-proj", "small-proj"]


# ---------- Prompts aggregation ----------

class TestPromptsAggregation:
    def test_prompts_truncated_to_config_limit(self, tmp_path):
        long_prompt = "x" * 1000
        layout = {
            "-Users-andrew-proj": [_user_line(T_INSIDE, long_prompt)],
        }
        root = _make_projects_dir(tmp_path, layout)
        events = [_claude_session_event("/Users/andrew/proj", duration_ms=15 * 60_000)]
        cfg = _config(tmp_path, project_ledger_summary_truncate_chars=120)
        result = detect_project_ledger(
            events, root, WINDOW_START, WINDOW_END, cfg, skip_summaries=True
        )
        proj_entry = next(e for e in result.entries if e.project == "proj")
        assert len(proj_entry.representative_prompts[0]) == 120

    def test_prompts_capped_at_max(self, tmp_path):
        layout = {
            "-Users-andrew-proj": [
                _user_line(T_INSIDE, f"prompt {i}") for i in range(50)
            ],
        }
        root = _make_projects_dir(tmp_path, layout)
        events = [_claude_session_event("/Users/andrew/proj", duration_ms=15 * 60_000)]
        cfg = _config(tmp_path, project_ledger_summary_max_prompts=10)
        result = detect_project_ledger(
            events, root, WINDOW_START, WINDOW_END, cfg, skip_summaries=True
        )
        proj_entry = next(e for e in result.entries if e.project == "proj")
        assert len(proj_entry.representative_prompts) == 10

    def test_prompts_most_recent_first(self, tmp_path):
        layout = {
            "-Users-andrew-proj": [
                _user_line("2026-04-22T08:00:00Z", "early"),
                _user_line("2026-04-22T15:00:00Z", "later"),
                _user_line("2026-04-22T20:00:00Z", "latest"),
            ],
        }
        root = _make_projects_dir(tmp_path, layout)
        events = [_claude_session_event("/Users/andrew/proj", duration_ms=15 * 60_000)]
        cfg = _config(tmp_path)
        result = detect_project_ledger(
            events, root, WINDOW_START, WINDOW_END, cfg, skip_summaries=True
        )
        proj_entry = next(e for e in result.entries if e.project == "proj")
        assert proj_entry.representative_prompts[0] == "latest"
        assert proj_entry.representative_prompts[-1] == "early"

    def test_tool_output_echoes_excluded_from_prompts(self, tmp_path):
        layout = {
            "-Users-andrew-proj": [
                _user_line(T_INSIDE, "<bash-stdout>build output</bash-stdout>"),
                _user_line(T_INSIDE, "real prompt"),
            ],
        }
        root = _make_projects_dir(tmp_path, layout)
        events = [_claude_session_event("/Users/andrew/proj", duration_ms=15 * 60_000)]
        cfg = _config(tmp_path)
        result = detect_project_ledger(
            events, root, WINDOW_START, WINDOW_END, cfg, skip_summaries=True
        )
        proj_entry = next(e for e in result.entries if e.project == "proj")
        assert proj_entry.representative_prompts == ["real prompt"]


# ---------- Summary call (mocked) ----------

class TestSummaryCall:
    def test_summary_populated_from_api_response(self, tmp_path):
        with patch(
            "ambient.detect.project_ledger._api_available", return_value=True
        ), patch("ambient.present.api.call_api") as mock_call:
            mock_call.return_value = "Phase 2 git activity detector with subprocess hardening."
            layout = {
                "-Users-andrew-proj": [_user_line(T_INSIDE, "ship the git activity detector")],
            }
            root = _make_projects_dir(tmp_path, layout)
            events = [_claude_session_event("/Users/andrew/proj", duration_ms=15 * 60_000)]
            cfg = _config(tmp_path)
            result = detect_project_ledger(events, root, WINDOW_START, WINDOW_END, cfg)
            proj = next(e for e in result.entries if e.project == "proj")
            assert proj.summary == "Phase 2 git activity detector with subprocess hardening."
            mock_call.assert_called_once()

    def test_summary_none_when_api_raises(self, tmp_path):
        with patch(
            "ambient.detect.project_ledger._api_available", return_value=True
        ), patch("ambient.present.api.call_api", side_effect=RuntimeError("rate limited")):
            layout = {
                "-Users-andrew-proj": [_user_line(T_INSIDE, "hi")],
            }
            root = _make_projects_dir(tmp_path, layout)
            events = [_claude_session_event("/Users/andrew/proj", duration_ms=15 * 60_000)]
            cfg = _config(tmp_path)
            result = detect_project_ledger(events, root, WINDOW_START, WINDOW_END, cfg)
            proj = next(e for e in result.entries if e.project == "proj")
            assert proj.summary is None
            # Entry still rendered with time + prompts intact
            assert proj.active_ms == 15 * 60_000
            assert proj.representative_prompts == ["hi"]

    def test_skip_summaries_flag_skips_api_call(self, tmp_path):
        with patch("ambient.present.api.call_api") as mock_call:
            layout = {
                "-Users-andrew-proj": [_user_line(T_INSIDE, "hi")],
            }
            root = _make_projects_dir(tmp_path, layout)
            events = [_claude_session_event("/Users/andrew/proj", duration_ms=15 * 60_000)]
            cfg = _config(tmp_path)
            detect_project_ledger(
                events, root, WINDOW_START, WINDOW_END, cfg, skip_summaries=True
            )
            mock_call.assert_not_called()

    def test_summary_skipped_when_no_prompts(self, tmp_path):
        with patch(
            "ambient.detect.project_ledger._api_available", return_value=True
        ), patch("ambient.present.api.call_api") as mock_call:
            # No JSONL files → no prompts → summary stays None, no API call
            events = [_claude_session_event("/Users/andrew/proj", duration_ms=15 * 60_000)]
            cfg = _config(tmp_path)
            result = detect_project_ledger(events, tmp_path, WINDOW_START, WINDOW_END, cfg)
            proj = next(e for e in result.entries if e.project == "proj")
            assert proj.summary is None
            mock_call.assert_not_called()


# ---------- Time-basis flag ----------

class TestTimeBasis:
    def test_default_time_basis_is_command_span(self, tmp_path):
        cfg = _config(tmp_path)
        result = detect_project_ledger(
            [], tmp_path, WINDOW_START, WINDOW_END, cfg, skip_summaries=True
        )
        assert result.time_basis == "command_span"
