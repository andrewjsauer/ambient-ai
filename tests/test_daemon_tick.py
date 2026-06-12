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
        # Exclusive cursor on the latest command's ts_end (+500ms duration)
        assert state.last_analyzed_ts == (now_ms - 3000 + 500) + 1

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
    @patch("ambient.daemon.tick._ingest_claude_sessions")
    @patch("ambient.daemon.tick.acquire_lock", return_value=False)
    def test_no_ingestion_when_locked(self, mock_lock, mock_ingest, config):
        """Regression: ingestion appends events and saves state, so it must not
        run while another tick holds the lock. Pre-fix, ingestion ran before
        lock acquisition and concurrent ticks double-ingested sessions."""
        daemon_tick(config)
        mock_ingest.assert_not_called()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("ambient.daemon.tick._ingest_claude_sessions", side_effect=Exception("boom"))
    def test_lock_released_when_ingestion_fails(self, mock_ingest, config):
        """Ingestion failures are logged, the tick continues, and the lock is
        released on the way out."""
        daemon_tick(config)
        mock_ingest.assert_called_once()
        assert not config.lock_path.exists()

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
        # Cursor advances to recent command's ts_end, not ancient session ts
        assert state.last_analyzed_ts == (now_ms - 1000 + 500) + 1

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
        assert state.last_analyzed_ts == (now_ms - 2000 + 500) + 1

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("ambient.daemon.tick._run_analysis")
    def test_long_running_command_not_skipped(self, mock_analysis, config):
        """Regression: a command started before the cursor but finished after
        it is appended at completion with an old ts_start. The ts_start
        watermark dropped it forever; the ts_end watermark picks it up."""
        mock_analysis.return_value = {"analysis": {}}
        today = datetime.now().strftime("%Y-%m-%d")
        now_ms = int(datetime.now().timestamp() * 1000)

        # Tick 1: a quick command finishes; cursor advances past its ts_end.
        _write_events(config, today, [_make_event(now_ms - 600_000)])
        daemon_tick(config)
        cursor = DaemonState.load(config.state_path).last_analyzed_ts
        assert cursor == (now_ms - 600_000 + 500) + 1

        # A long build that STARTED before that cursor finishes only now and
        # is appended to the log with its old ts_start.
        long_cmd = _make_event(now_ms - 900_000, command="make build")
        long_cmd["ts_end"] = now_ms - 1000
        long_cmd["duration_ms"] = long_cmd["ts_end"] - long_cmd["ts_start"]
        _write_events(config, today, [long_cmd])

        # Tick 2 must analyze it and advance the cursor past its ts_end.
        daemon_tick(config)
        assert mock_analysis.call_count == 2
        second_batch = mock_analysis.call_args[0][1]
        assert any(e.command == "make build" for e in second_batch)
        assert DaemonState.load(config.state_path).last_analyzed_ts == (now_ms - 1000) + 1


class TestSummaryRetry:
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("ambient.present.narrator._call_api")
    def test_failed_summary_retries_next_tick(self, mock_api, config):
        """End-to-end retry: outage tick writes nothing and leaves state
        unadvanced; the next healthy tick generates the real summary."""
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        _write_analysis(config, yesterday)

        mock_api.side_effect = Exception("transient outage")
        daemon_tick(config)
        assert not config.summary_path(yesterday).exists()
        assert DaemonState.load(config.state_path).last_summary_date == ""

        mock_api.side_effect = None
        mock_api.return_value = "recovered summary"
        daemon_tick(config)
        assert config.summary_path(yesterday).exists()
        assert "recovered summary" in config.summary_path(yesterday).read_text()
        assert DaemonState.load(config.state_path).last_summary_date == yesterday

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch("ambient.present.narrator._call_api")
    def test_failed_day_blocks_advance_past_it(self, mock_api, config):
        """The catch-up loop stops at the first failed day: a later success in
        the same tick must not advance last_summary_date past the failure."""
        d1 = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        d2 = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        _write_analysis(config, d1)
        _write_analysis(config, d2)

        mock_api.side_effect = Exception("outage")
        daemon_tick(config)
        assert not config.summary_path(d1).exists()
        assert not config.summary_path(d2).exists()
        assert DaemonState.load(config.state_path).last_summary_date == ""

        mock_api.side_effect = None
        mock_api.return_value = "ok"
        daemon_tick(config)
        assert config.summary_path(d1).exists()
        assert config.summary_path(d2).exists()
        assert DaemonState.load(config.state_path).last_summary_date == d2


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
