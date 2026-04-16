import json
import os
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from ambient.config import Config
from ambient.daemon.state import DaemonState
from ambient.daemon.tick import daemon_tick


@pytest.fixture
def config(tmp_path):
    c = Config(
        base_dir=tmp_path,
        claude_history_path=tmp_path / "claude_history.jsonl",
        claude_projects_dir=tmp_path / "claude_projects",
    )
    c.ensure_dirs()
    return c


def _write_events(config, date_str, events):
    path = config.events_path(date_str)
    with open(path, "a") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _make_event(ts_start, command="ls", gap_ms=1000):
    return {
        "ts_start": ts_start,
        "ts_end": ts_start + 500,
        "duration_ms": 500,
        "command": command,
        "exit_code": 0,
        "cwd": "/tmp",
        "tmux_pane": None,
        "gap_ms": gap_ms,
        "session_boundary": False,
    }


def _make_claude_session_event(ts_start, session_id="sess-1", duration_ms=60_000):
    return {
        "type": "claude_session",
        "ts_start": ts_start,
        "ts_end": ts_start + duration_ms,
        "duration_ms": duration_ms,
        "command": f"claude: backfilled session {session_id}",
        "exit_code": 0,
        "cwd": "/tmp",
        "tmux_pane": None,
        "gap_ms": None,
        "claude_session_id": session_id,
        "claude_prompts": ["hello"],
        "claude_tools": [],
        "claude_files": [],
        "claude_project": "test",
        "claude_prompt_count": 1,
        "claude_is_error_count": 0,
    }


def _write_analysis(config, date_str, count=3):
    path = config.analysis_path(date_str)
    with open(path, "a") as f:
        for i in range(count):
            f.write(json.dumps({
                "timestamp": f"{date_str}T10:{i:02d}:00",
                "compression": {},
                "pauses": {},
                "analysis": {"summary": f"batch {i}"},
            }) + "\n")


class TestDaemonTick:
    @patch.dict(os.environ, {}, clear=True)
    def test_skips_when_no_api_key(self, config):
        daemon_tick(config)
        # Should not create any analysis files
        assert not config.state_path.exists() or DaemonState.load(config.state_path).last_analyzed_ts == 0

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_skips_when_no_events(self, config):
        daemon_tick(config)
        state = DaemonState.load(config.state_path)
        assert state.last_analyzed_ts == 0

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("ambient.daemon.tick._run_analysis")
    def test_runs_analysis_when_events_exist(self, mock_analysis, config):
        mock_analysis.return_value = {"analysis": {"summary": "test"}}
        today = datetime.now().strftime("%Y-%m-%d")
        now_ms = int(datetime.now().timestamp() * 1000)
        _write_events(config, today, [
            _make_event(now_ms - 5000),
            _make_event(now_ms - 3000),
        ])
        daemon_tick(config)
        mock_analysis.assert_called_once()
        state = DaemonState.load(config.state_path)
        assert state.last_analyzed_ts == now_ms - 3000 + 1  # exclusive cursor

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("ambient.daemon.tick._run_analysis")
    def test_skips_already_analyzed_events(self, mock_analysis, config):
        mock_analysis.return_value = {"analysis": {}}
        today = datetime.now().strftime("%Y-%m-%d")
        now_ms = int(datetime.now().timestamp() * 1000)
        _write_events(config, today, [_make_event(now_ms - 5000)])

        # First tick processes events
        daemon_tick(config)
        assert mock_analysis.call_count == 1

        # Second tick finds no new events
        daemon_tick(config)
        assert mock_analysis.call_count == 1  # not called again

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("ambient.daemon.tick._run_analysis")
    def test_updates_events_since_calibration(self, mock_analysis, config):
        mock_analysis.return_value = {"analysis": {}}
        today = datetime.now().strftime("%Y-%m-%d")
        now_ms = int(datetime.now().timestamp() * 1000)
        _write_events(config, today, [
            _make_event(now_ms - 5000),
            _make_event(now_ms - 3000),
            _make_event(now_ms - 1000),
        ])
        daemon_tick(config)
        state = DaemonState.load(config.state_path)
        assert state.events_since_calibration == 3

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("ambient.daemon.tick.acquire_lock", return_value=False)
    def test_skips_when_locked(self, mock_lock, config):
        today = datetime.now().strftime("%Y-%m-%d")
        now_ms = int(datetime.now().timestamp() * 1000)
        _write_events(config, today, [_make_event(now_ms - 1000)])
        daemon_tick(config)
        # State should not be updated
        state = DaemonState.load(config.state_path)
        assert state.last_analyzed_ts == 0

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("ambient.daemon.tick._run_analysis", side_effect=Exception("API error"))
    def test_releases_lock_on_error(self, mock_analysis, config):
        today = datetime.now().strftime("%Y-%m-%d")
        now_ms = int(datetime.now().timestamp() * 1000)
        _write_events(config, today, [_make_event(now_ms - 1000)])
        with pytest.raises(Exception, match="API error"):
            daemon_tick(config)
        # Lock should be released
        assert not config.lock_path.exists()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("ambient.daemon.tick._run_analysis")
    def test_first_run_processes_all_events(self, mock_analysis, config):
        mock_analysis.return_value = {"analysis": {}}
        today = datetime.now().strftime("%Y-%m-%d")
        now_ms = int(datetime.now().timestamp() * 1000)
        events = [_make_event(now_ms - i * 1000) for i in range(10)]
        _write_events(config, today, events)
        daemon_tick(config)
        # Should process all 10 events
        call_args = mock_analysis.call_args
        assert len(call_args[0][1]) == 10


class TestCursorWatermark:
    """Cursor advance uses only command-event timestamps to avoid rewind from
    backfilled claude_session events whose ts_start can be days old."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("ambient.daemon.tick._run_analysis")
    def test_cursor_ignores_backfilled_claude_sessions(self, mock_analysis, config):
        """Tick with one recent command + backfilled old claude_sessions advances cursor to command ts, not old session ts."""
        mock_analysis.return_value = {"analysis": {}}
        today = datetime.now().strftime("%Y-%m-%d")
        now_ms = int(datetime.now().timestamp() * 1000)
        old_ts = now_ms - (3 * 24 * 60 * 60 * 1000)  # 3 days ago

        _write_events(config, today, [
            _make_event(now_ms - 1000),  # recent command
            _make_claude_session_event(old_ts, session_id="old-1"),
            _make_claude_session_event(old_ts + 1000, session_id="old-2"),
        ])
        daemon_tick(config)
        state = DaemonState.load(config.state_path)
        # Cursor advances to recent command, not ancient session ts
        assert state.last_analyzed_ts == (now_ms - 1000) + 1

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("ambient.daemon.tick._run_analysis")
    def test_cursor_unchanged_when_only_claude_sessions(self, mock_analysis, config):
        """Tick processing only claude_session events (no commands) leaves cursor unchanged."""
        mock_analysis.return_value = {"analysis": {}}
        today = datetime.now().strftime("%Y-%m-%d")
        now_ms = int(datetime.now().timestamp() * 1000)
        old_ts = now_ms - (2 * 24 * 60 * 60 * 1000)

        # Seed state with a prior cursor
        prior_cursor = now_ms - 10_000
        state = DaemonState(last_analyzed_ts=prior_cursor)
        state.save(config.state_path)

        _write_events(config, today, [
            _make_claude_session_event(old_ts, session_id="old-1"),
        ])
        daemon_tick(config)
        reloaded = DaemonState.load(config.state_path)
        # Cursor must not rewind to ancient session ts
        assert reloaded.last_analyzed_ts == prior_cursor

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("ambient.daemon.tick._run_analysis")
    def test_cursor_advances_normally_on_command_events(self, mock_analysis, config):
        """Baseline: pure-command events still advance cursor to latest command ts."""
        mock_analysis.return_value = {"analysis": {}}
        today = datetime.now().strftime("%Y-%m-%d")
        now_ms = int(datetime.now().timestamp() * 1000)
        _write_events(config, today, [
            _make_event(now_ms - 5000),
            _make_event(now_ms - 2000),
        ])
        daemon_tick(config)
        state = DaemonState.load(config.state_path)
        assert state.last_analyzed_ts == (now_ms - 2000) + 1


class TestSummaryCatchup:
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("ambient.daemon.tick._run_analysis")
    @patch("ambient.present.narrator._call_api", return_value='{"summary": "daily"}')
    def test_generates_summary_for_yesterday(self, mock_api, mock_analysis, config):
        mock_analysis.return_value = {"analysis": {}}
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        # Create events and analysis for yesterday
        yesterday_ts = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        _write_events(config, yesterday, [_make_event(yesterday_ts)])
        _write_analysis(config, yesterday)

        # Create events for today so tick runs
        now_ms = int(datetime.now().timestamp() * 1000)
        _write_events(config, today, [_make_event(now_ms - 1000)])

        daemon_tick(config)

        # Yesterday's summary should exist
        assert config.summary_path(yesterday).exists()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("ambient.daemon.tick._run_analysis")
    def test_skips_summary_for_today(self, mock_analysis, config):
        mock_analysis.return_value = {"analysis": {}}
        today = datetime.now().strftime("%Y-%m-%d")
        now_ms = int(datetime.now().timestamp() * 1000)
        _write_events(config, today, [_make_event(now_ms - 1000)])
        _write_analysis(config, today)

        daemon_tick(config)

        # Today's summary should NOT exist (day not complete)
        assert not config.summary_path(today).exists()


class TestRecalibration:
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("ambient.daemon.tick._run_analysis")
    @patch("ambient.detect.pauses.calibrate")
    def test_skips_when_not_enough_events(self, mock_calibrate, mock_analysis, config):
        mock_analysis.return_value = {"analysis": {}}
        today = datetime.now().strftime("%Y-%m-%d")
        now_ms = int(datetime.now().timestamp() * 1000)
        _write_events(config, today, [_make_event(now_ms - 1000)])

        # Set state: 7+ days since cal but only 50 events
        state = DaemonState(
            last_calibration_date=(datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
            events_since_calibration=50,
        )
        state.save(config.state_path)

        daemon_tick(config)
        mock_calibrate.assert_not_called()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("ambient.daemon.tick._run_analysis")
    @patch("ambient.detect.pauses.calibrate")
    def test_skips_when_not_enough_days(self, mock_calibrate, mock_analysis, config):
        mock_analysis.return_value = {"analysis": {}}
        today = datetime.now().strftime("%Y-%m-%d")
        now_ms = int(datetime.now().timestamp() * 1000)
        _write_events(config, today, [_make_event(now_ms - 1000)])

        # Set state: only 3 days since cal but 300 events
        state = DaemonState(
            last_calibration_date=(datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
            events_since_calibration=300,
        )
        state.save(config.state_path)

        daemon_tick(config)
        mock_calibrate.assert_not_called()
