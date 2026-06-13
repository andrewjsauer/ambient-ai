"""Tests for verification-gap detector."""

import json

import pytest

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.project_capabilities import clear_capability_cache
from ambient.detect.verification import (
    VerificationGapFindings,
    detect_verification_gaps,
)


@pytest.fixture(autouse=True)
def _reset_capability_cache():
    """Per-test isolation for the project-capability cache."""
    clear_capability_cache()
    yield
    clear_capability_cache()


@pytest.fixture
def has_tests_cwd(tmp_path):
    """A cwd that detect_capabilities classifies as has_tests."""
    (tmp_path / "Makefile").write_text("test:\n\tpytest\n")
    return str(tmp_path)


@pytest.fixture
def has_typecheck_cwd(tmp_path):
    """A cwd that detect_capabilities classifies as has_typecheck only."""
    (tmp_path / "tsconfig.json").write_text("{}")
    return str(tmp_path)


@pytest.fixture
def neither_cwd(tmp_path):
    """A cwd with no test/typecheck capability."""
    return str(tmp_path)


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
    def test_verified_fix_not_in_gaps(self, has_tests_cwd):
        """Session with Edit followed by pytest in same cwd within 5 min → not a gap."""
        session = _session(ts_start=10_000, duration_ms=60_000, cwd=has_tests_cwd)
        events = [
            session,
            _cmd("pytest", ts_start=session.ts_end + 30_000, cwd=has_tests_cwd),
        ]
        result = detect_verification_gaps(events, _config())
        assert result.total_fix_sessions == 1
        assert len(result.gaps) == 0

    def test_in_session_verification_not_a_gap(self, has_tests_cwd):
        """A fix session that ran its own test via Claude's Bash tool is
        verified even with no follow-up shell command — the real-data case
        that made this detector report ~100% false gaps."""
        session = _session(ts_start=10_000, duration_ms=60_000, cwd=has_tests_cwd)
        session.claude_ran_verification = True
        result = detect_verification_gaps([session], _config())
        assert result.total_fix_sessions == 1
        assert len(result.gaps) == 0

    def test_unverified_fix_is_gap(self, has_tests_cwd):
        """Session with Edit, no test command within window → appears in gaps."""
        session = _session(ts_start=10_000, duration_ms=60_000, cwd=has_tests_cwd)
        events = [session]  # no follow-up command at all
        result = detect_verification_gaps(events, _config())
        assert result.total_fix_sessions == 1
        assert len(result.gaps) == 1
        assert result.gaps[0].session_id == "sess-1"
        assert "auth.py" in result.gaps[0].edited_files
        assert result.gaps[0].bucket == "has_tests"

    def test_cwd_mismatch_is_gap(self, has_tests_cwd, tmp_path):
        """Test command ran in different cwd → still a gap."""
        # Build a sibling cwd that also has tests so both are has_tests bucket
        # but the session's verification check requires same cwd.
        sibling = tmp_path / "frontend"
        sibling.mkdir()
        (sibling / "Makefile").write_text("test:\n\tpytest\n")
        session = _session(ts_start=10_000, duration_ms=60_000, cwd=has_tests_cwd)
        events = [
            session,
            _cmd("pytest", ts_start=session.ts_end + 30_000, cwd=str(sibling)),
        ]
        result = detect_verification_gaps(events, _config())
        assert len(result.gaps) == 1

    def test_window_expired_is_gap(self, has_tests_cwd):
        """Test command ran after window → still a gap."""
        session = _session(ts_start=10_000, duration_ms=60_000, cwd=has_tests_cwd)
        events = [
            session,
            # 10 min later, outside 5-min default window
            _cmd("pytest", ts_start=session.ts_end + 600_000, cwd=has_tests_cwd),
        ]
        result = detect_verification_gaps(events, _config())
        assert len(result.gaps) == 1

    def test_no_edit_not_counted(self, has_tests_cwd):
        """Session with only Read tools → not a fix session."""
        session = _session(
            ts_start=10_000,
            cwd=has_tests_cwd,
            tools=[{"name": "Read", "files": ["a.py"]}],
        )
        events = [session]
        result = detect_verification_gaps(events, _config())
        assert result.total_fix_sessions == 0
        assert result.gaps == []

    def test_multiedit_counts_as_fix(self, has_tests_cwd):
        """MultiEdit tool also counts as a fix."""
        session = _session(
            ts_start=10_000,
            cwd=has_tests_cwd,
            tools=[{"name": "MultiEdit", "files": ["a.py"]}],
        )
        events = [session]
        result = detect_verification_gaps(events, _config())
        assert result.total_fix_sessions == 1
        assert len(result.gaps) == 1

    def test_low_sample_no_gap_rate(self, has_tests_cwd):
        """Under min_fix_sessions floor → gap_rate is None and low_sample True."""
        # Create 5 fix sessions (below default floor of 10), all unverified
        events = []
        for i in range(5):
            events.append(_session(
                ts_start=10_000 + i * 200_000,
                duration_ms=60_000,
                session_id=f"s{i}",
                cwd=has_tests_cwd,
            ))
        result = detect_verification_gaps(events, _config())
        assert result.total_fix_sessions == 5
        assert result.low_sample is True
        assert result.gap_rate is None
        # gaps list still populated for transparency
        assert len(result.gaps) == 5

    def test_gap_rate_when_enough_samples(self, has_tests_cwd):
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
                cwd=has_tests_cwd,
            )
            events.append(session)
            # Half get a test run, half don't
            if i % 2 == 0:
                events.append(_cmd(
                    "pytest",
                    ts_start=session.ts_end + 30_000,
                    cwd=has_tests_cwd,
                ))
        result = detect_verification_gaps(events, _config())
        assert result.total_fix_sessions == 12
        assert result.low_sample is False
        assert result.gap_rate == 6 / 12

    def test_empty_events(self):
        result = detect_verification_gaps([], _config())
        assert result.total_fix_sessions == 0
        assert result.gaps == []

    def test_custom_test_patterns(self, has_tests_cwd):
        """Config can extend verification_test_command_patterns."""
        config = _config(verification_test_command_patterns=["just test"])
        session = _session(ts_start=10_000, duration_ms=60_000, cwd=has_tests_cwd)
        events = [
            session,
            _cmd("just test", ts_start=session.ts_end + 30_000, cwd=has_tests_cwd),
        ]
        result = detect_verification_gaps(events, config)
        assert len(result.gaps) == 0

    def test_case_insensitive_matching(self, has_tests_cwd):
        """Test command patterns match case-insensitively."""
        session = _session(ts_start=10_000, duration_ms=60_000, cwd=has_tests_cwd)
        events = [
            session,
            _cmd("PYTEST -x", ts_start=session.ts_end + 30_000, cwd=has_tests_cwd),
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


class TestProjectAwareBucketing:
    def test_neither_bucket_excluded_from_test_pattern_matching(self, neither_cwd):
        """A project with no detected capability cannot be verified by pytest."""
        session = _session(ts_start=10_000, duration_ms=60_000, cwd=neither_cwd)
        events = [
            session,
            # pytest in the same cwd; would have verified before, now does not
            _cmd("pytest", ts_start=session.ts_end + 30_000, cwd=neither_cwd),
        ]
        result = detect_verification_gaps(events, _config())
        assert len(result.gaps) == 1
        assert result.gaps[0].bucket == "neither"
        assert result.total_fix_sessions_by_bucket["neither"] == 1
        assert result.gaps_by_bucket["neither"] == 1

    def test_typecheck_bucket_uses_typecheck_patterns(self, has_typecheck_cwd):
        """A typecheck-only project counts tsc as verification, pytest as gap."""
        session_with_tsc = _session(
            ts_start=10_000, duration_ms=60_000,
            session_id="s-tsc", cwd=has_typecheck_cwd,
        )
        session_with_pytest = _session(
            ts_start=1_000_000, duration_ms=60_000,
            session_id="s-py", cwd=has_typecheck_cwd,
        )
        events = [
            session_with_tsc,
            _cmd("tsc --noEmit", ts_start=session_with_tsc.ts_end + 30_000,
                 cwd=has_typecheck_cwd),
            session_with_pytest,
            _cmd("pytest", ts_start=session_with_pytest.ts_end + 30_000,
                 cwd=has_typecheck_cwd),
        ]
        result = detect_verification_gaps(events, _config())
        assert result.total_fix_sessions == 2
        assert result.total_fix_sessions_by_bucket["has_typecheck"] == 2
        # The pytest session is unverified because pytest doesn't count for typecheck bucket
        assert result.gaps_by_bucket["has_typecheck"] == 1
        gap_session_ids = {g.session_id for g in result.gaps}
        assert gap_session_ids == {"s-py"}

    def test_has_tests_bucket_uses_test_patterns(self, has_tests_cwd):
        """A test-bearing project counts pytest as verification."""
        session = _session(ts_start=10_000, duration_ms=60_000, cwd=has_tests_cwd)
        events = [
            session,
            _cmd("pytest", ts_start=session.ts_end + 30_000, cwd=has_tests_cwd),
        ]
        result = detect_verification_gaps(events, _config())
        assert result.gaps_by_bucket["has_tests"] == 0
        assert result.total_fix_sessions_by_bucket["has_tests"] == 1

    def test_per_bucket_low_sample_independent_of_total(
        self, has_tests_cwd, has_typecheck_cwd
    ):
        """Each bucket gets its own low-sample gating — total can be high enough
        for the global rate while a bucket is still below the floor."""
        events = []
        # 10 unverified test-bucket sessions (above floor)
        for i in range(10):
            events.append(_session(
                ts_start=100_000 + i * 600_000,
                duration_ms=60_000,
                session_id=f"t{i}",
                cwd=has_tests_cwd,
            ))
        # 2 unverified typecheck-bucket sessions (below floor)
        for i in range(2):
            events.append(_session(
                ts_start=100_000_000 + i * 600_000,
                duration_ms=60_000,
                session_id=f"y{i}",
                cwd=has_typecheck_cwd,
            ))
        result = detect_verification_gaps(events, _config())
        # Global: 12 fix sessions, all gaps -> rate 1.0
        assert result.total_fix_sessions == 12
        assert result.gap_rate == 1.0
        # has_tests bucket: 10 sessions -> rate published
        assert result.low_sample_by_bucket["has_tests"] is False
        assert result.gap_rate_by_bucket["has_tests"] == 1.0
        # has_typecheck bucket: 2 sessions -> low sample, no rate
        assert result.low_sample_by_bucket["has_typecheck"] is True
        assert result.gap_rate_by_bucket["has_typecheck"] is None

    def test_bucket_keys_always_present_even_when_empty(self, has_tests_cwd):
        """All three bucket keys appear in the dicts so downstream code can
        index without KeyError."""
        session = _session(ts_start=10_000, duration_ms=60_000, cwd=has_tests_cwd)
        events = [session]
        result = detect_verification_gaps(events, _config())
        for bucket in ("has_tests", "has_typecheck", "neither"):
            assert bucket in result.total_fix_sessions_by_bucket
            assert bucket in result.gaps_by_bucket
            assert bucket in result.gap_rate_by_bucket
            assert bucket in result.low_sample_by_bucket

    def test_capability_probe_failure_does_not_kill_detector(self, monkeypatch, has_tests_cwd):
        """A pathological cwd that makes detect_capabilities raise must not
        propagate out of _bucket_for — the session falls into 'neither' and
        the rest of the run continues."""
        from ambient.detect import verification as v
        called = {"n": 0}
        original = v.detect_capabilities

        def flaky_probe(cwd):
            called["n"] += 1
            if cwd == "/explosive/cwd":
                raise RuntimeError("simulated probe explosion")
            return original(cwd)

        monkeypatch.setattr(v, "detect_capabilities", flaky_probe)
        events = [
            _session(ts_start=10_000, session_id="bad", cwd="/explosive/cwd"),
            _session(ts_start=20_000, session_id="good", cwd=has_tests_cwd),
        ]
        # Should not raise
        result = detect_verification_gaps(events, _config())
        assert result.total_fix_sessions == 2
        # Bad cwd lands in neither (probe failure fallback)
        assert result.total_fix_sessions_by_bucket["neither"] == 1
        # Good cwd lands in has_tests
        assert result.total_fix_sessions_by_bucket["has_tests"] == 1
        assert called["n"] >= 2

    def test_neither_bucket_gap_rate_is_always_none(self, neither_cwd):
        """`neither` projects have no verification capability by definition,
        so the per-bucket rate is never published — even with high session
        counts. Downstream consumers (terminal summary, baseline store) must
        not see a misleading 1.0."""
        events = []
        for i in range(15):
            events.append(_session(
                ts_start=100_000 + i * 600_000,
                duration_ms=60_000,
                session_id=f"n{i}",
                cwd=neither_cwd,
            ))
        result = detect_verification_gaps(events, _config())
        assert result.total_fix_sessions_by_bucket["neither"] == 15
        assert result.gap_rate_by_bucket["neither"] is None
        # low_sample_by_bucket still reflects raw count for symmetry with
        # other buckets; neither just unconditionally suppresses the rate.
        assert result.low_sample_by_bucket["neither"] is False
