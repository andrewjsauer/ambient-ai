"""Tests for verification-gap detector."""

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.verification import (
    VerificationGapFindings,
    detect_verification_gaps,
)


def _session(
    ts_start=10_000,
    duration_ms=60_000,
    session_id="sess-1",
    project="auth",
    cwd="/home/user/auth",
    tools=None,
    files=None,
):
    return Event(
        ts_start=ts_start,
        ts_end=ts_start + duration_ms,
        duration_ms=duration_ms,
        command="claude: fix",
        exit_code=0,
        cwd=cwd,
        tmux_pane=None,
        gap_ms=None,
        type="claude_session",
        claude_session_id=session_id,
        claude_prompts=["fix"],
        claude_tools=tools or [{"name": "Edit", "files": files or ["auth.py"]}],
        claude_files=files or ["auth.py"],
        claude_project=project,
        claude_prompt_count=1,
        claude_is_error_count=0,
    )


def _cmd(command="pytest", exit_code=0, ts_start=1000, duration_ms=5000,
         cwd="/home/user/auth"):
    return Event(
        ts_start=ts_start,
        ts_end=ts_start + duration_ms,
        duration_ms=duration_ms,
        command=command,
        exit_code=exit_code,
        cwd=cwd,
        tmux_pane=None,
        gap_ms=None,
        type="command",
    )


def _config(**overrides):
    return Config(**overrides)


class TestDetectVerificationGaps:
    def test_verified_fix_not_in_gaps(self):
        """Session with Edit followed by pytest in same cwd within 5 min → not a gap."""
        session = _session(ts_start=10_000, duration_ms=60_000)
        events = [
            session,
            _cmd("pytest", ts_start=session.ts_end + 30_000, cwd="/home/user/auth"),
        ]
        result = detect_verification_gaps(events, _config())
        assert result.total_fix_sessions == 1
        assert len(result.gaps) == 0

    def test_unverified_fix_is_gap(self):
        """Session with Edit, no test command within window → appears in gaps."""
        session = _session(ts_start=10_000, duration_ms=60_000)
        events = [session]  # no follow-up command at all
        result = detect_verification_gaps(events, _config())
        assert result.total_fix_sessions == 1
        assert len(result.gaps) == 1
        assert result.gaps[0].session_id == "sess-1"
        assert "auth.py" in result.gaps[0].edited_files

    def test_cwd_mismatch_is_gap(self):
        """Test command ran in different cwd → still a gap."""
        session = _session(ts_start=10_000, duration_ms=60_000, cwd="/home/user/auth")
        events = [
            session,
            _cmd("pytest", ts_start=session.ts_end + 30_000, cwd="/home/user/frontend"),
        ]
        result = detect_verification_gaps(events, _config())
        assert len(result.gaps) == 1

    def test_window_expired_is_gap(self):
        """Test command ran after window → still a gap."""
        session = _session(ts_start=10_000, duration_ms=60_000)
        events = [
            session,
            # 10 min later, outside 5-min default window
            _cmd("pytest", ts_start=session.ts_end + 600_000, cwd="/home/user/auth"),
        ]
        result = detect_verification_gaps(events, _config())
        assert len(result.gaps) == 1

    def test_no_edit_not_counted(self):
        """Session with only Read tools → not a fix session."""
        session = _session(
            ts_start=10_000,
            tools=[{"name": "Read", "files": ["a.py"]}],
        )
        events = [session]
        result = detect_verification_gaps(events, _config())
        assert result.total_fix_sessions == 0
        assert result.gaps == []

    def test_multiedit_counts_as_fix(self):
        """MultiEdit tool also counts as a fix."""
        session = _session(
            ts_start=10_000,
            tools=[{"name": "MultiEdit", "files": ["a.py"]}],
        )
        events = [session]
        result = detect_verification_gaps(events, _config())
        assert result.total_fix_sessions == 1
        assert len(result.gaps) == 1

    def test_low_sample_no_gap_rate(self):
        """Under min_fix_sessions floor → gap_rate is None and low_sample True."""
        # Create 5 fix sessions (below default floor of 10), all unverified
        events = []
        for i in range(5):
            events.append(_session(
                ts_start=10_000 + i * 200_000,
                duration_ms=60_000,
                session_id=f"s{i}",
            ))
        result = detect_verification_gaps(events, _config())
        assert result.total_fix_sessions == 5
        assert result.low_sample is True
        assert result.gap_rate is None
        # gaps list still populated for transparency
        assert len(result.gaps) == 5

    def test_gap_rate_when_enough_samples(self):
        """>=10 fix sessions → gap_rate is emitted."""
        events = []
        # Space sessions 10 min apart (well beyond the 5-min window) so a pytest
        # for one session can't bleed into the verification window of another.
        spacing = 600_000
        for i in range(12):
            session = _session(
                ts_start=10_000 + i * spacing,
                duration_ms=60_000,
                session_id=f"s{i}",
            )
            events.append(session)
            # Half get a test run, half don't
            if i % 2 == 0:
                events.append(_cmd(
                    "pytest",
                    ts_start=session.ts_end + 30_000,
                    cwd="/home/user/auth",
                ))
        result = detect_verification_gaps(events, _config())
        assert result.total_fix_sessions == 12
        assert result.low_sample is False
        assert result.gap_rate == 6 / 12

    def test_empty_events(self):
        result = detect_verification_gaps([], _config())
        assert result.total_fix_sessions == 0
        assert result.gaps == []

    def test_custom_test_patterns(self):
        """Config can extend verification_test_command_patterns."""
        config = _config(verification_test_command_patterns=["just test"])
        session = _session(ts_start=10_000, duration_ms=60_000)
        events = [
            session,
            _cmd("just test", ts_start=session.ts_end + 30_000, cwd="/home/user/auth"),
        ]
        result = detect_verification_gaps(events, config)
        assert len(result.gaps) == 0

    def test_case_insensitive_matching(self):
        """Test command patterns match case-insensitively."""
        session = _session(ts_start=10_000, duration_ms=60_000)
        events = [
            session,
            _cmd("PYTEST -x", ts_start=session.ts_end + 30_000, cwd="/home/user/auth"),
        ]
        result = detect_verification_gaps(events, _config())
        assert len(result.gaps) == 0

    def test_aggregate_integration(self, tmp_path):
        """aggregate_coaching_data populates data.verification_gaps."""
        import json
        from datetime import datetime

        from ambient.present.insights import aggregate_coaching_data

        config = _config(base_dir=tmp_path)
        today = datetime.now().strftime("%Y-%m-%d")
        events_path = config.events_path(today)
        events_path.parent.mkdir(parents=True, exist_ok=True)

        now_ms = int(datetime.now().timestamp() * 1000)
        session = {
            "type": "claude_session",
            "ts_start": now_ms - 600_000,
            "ts_end": now_ms - 540_000,
            "duration_ms": 60_000,
            "command": "claude: fix",
            "exit_code": 0,
            "cwd": "/home/user/proj",
            "tmux_pane": None,
            "gap_ms": None,
            "claude_session_id": "sess-1",
            "claude_prompts": ["fix"],
            "claude_tools": [{"name": "Edit", "files": ["proj/main.py"]}],
            "claude_files": ["proj/main.py"],
            "claude_project": "proj",
            "claude_prompt_count": 1,
            "claude_is_error_count": 0,
        }
        with open(events_path, "w") as f:
            f.write(json.dumps(session) + "\n")

        data = aggregate_coaching_data(config, window_days=7, compare=False)
        assert isinstance(data.verification_gaps, VerificationGapFindings)
        assert data.verification_gaps.total_fix_sessions == 1
        assert len(data.verification_gaps.gaps) == 1
