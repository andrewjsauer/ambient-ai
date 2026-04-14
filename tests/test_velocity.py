"""Tests for resolution velocity tracker."""

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.velocity import (
    ResolutionChain,
    compute_velocity_metrics,
    detect_resolution_chains,
)


def _cmd(command="pytest", exit_code=0, ts_start=1000, duration_ms=5000, cwd="/home/user/auth"):
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


def _session(ts_start=2000, duration_ms=300_000, project="auth", session_id="sess-1",
             prompt_count=5, error_count=0):
    return Event(
        ts_start=ts_start,
        ts_end=ts_start + duration_ms,
        duration_ms=duration_ms,
        command="claude: fix the test",
        exit_code=0,
        cwd=f"/home/user/{project}",
        tmux_pane=None,
        gap_ms=None,
        type="claude_session",
        claude_session_id=session_id,
        claude_prompts=["fix the test"],
        claude_tools=[{"name": "Edit", "files": ["test.py"]}],
        claude_files=["test.py"],
        claude_project=project,
        claude_prompt_count=prompt_count,
        claude_is_error_count=error_count,
    )


def _config(**overrides):
    return Config(**overrides)


class TestDetectResolutionChains:
    def test_happy_path_resolved(self):
        """Failed pytest → Claude session → successful pytest → chain detected."""
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5000),
            _session(ts_start=10000, duration_ms=300_000, session_id="s1"),
            _cmd("pytest", exit_code=0, ts_start=320000, duration_ms=5000),
        ]
        chains = detect_resolution_chains(events, _config())
        assert len(chains) == 1
        assert chains[0].resolved is True
        assert chains[0].project == "auth"
        assert chains[0].initial_command == "pytest"
        assert chains[0].resolution_command == "pytest"
        assert "s1" in chains[0].claude_session_ids
        # Active time = failed cmd (5s) + session (300s) + success cmd (5s)
        assert chains[0].active_time_ms == 5000 + 300_000 + 5000

    def test_multiple_projects(self):
        """Chains across 2 projects → separate chains."""
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, cwd="/home/user/auth"),
            _session(ts_start=10000, project="auth", session_id="s1"),
            _cmd("pytest", exit_code=0, ts_start=320000, cwd="/home/user/auth"),
            _cmd("npm test", exit_code=1, ts_start=1000, cwd="/home/user/frontend"),
            _session(ts_start=10000, project="frontend", session_id="s2"),
            _cmd("npm test", exit_code=0, ts_start=320000, cwd="/home/user/frontend"),
        ]
        chains = detect_resolution_chains(events, _config())
        assert len(chains) == 2
        projects = {c.project for c in chains}
        assert projects == {"auth", "frontend"}
        assert all(c.resolved for c in chains)

    def test_no_claude_session_no_chain(self):
        """Failed command with no subsequent Claude session → no chain."""
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000),
            _cmd("pytest", exit_code=0, ts_start=10000),
        ]
        chains = detect_resolution_chains(events, _config())
        # Chain exists but unresolved (no Claude involvement before success)
        resolved = [c for c in chains if c.resolved]
        assert len(resolved) == 0

    def test_idle_break(self):
        """20 min gap between Claude session and success → chain broken."""
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5000),
            _session(ts_start=10000, duration_ms=60_000, session_id="s1"),
            # 20 min gap (> 15 min idle threshold)
            _cmd("pytest", exit_code=0, ts_start=70000 + 1_200_000, duration_ms=5000),
        ]
        chains = detect_resolution_chains(events, _config())
        assert len(chains) >= 1
        assert not any(c.resolved for c in chains)

    def test_different_base_command_no_match(self):
        """Failed pytest → Claude → successful git commit → no resolution (different command)."""
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5000),
            _session(ts_start=10000, duration_ms=60_000, session_id="s1"),
            _cmd("git commit -m 'fix'", exit_code=0, ts_start=80000, duration_ms=2000),
        ]
        chains = detect_resolution_chains(events, _config())
        resolved = [c for c in chains if c.resolved]
        assert len(resolved) == 0

    def test_benign_nonzero_no_chain(self):
        """grep with non-zero exit → no chain opened."""
        events = [
            _cmd("grep foo bar.txt", exit_code=1, ts_start=1000),
            _session(ts_start=10000, session_id="s1"),
        ]
        chains = detect_resolution_chains(events, _config())
        assert len(chains) == 0

    def test_multiple_sessions_in_chain(self):
        """Two Claude sessions in one chain → all session IDs collected."""
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5000),
            _session(ts_start=10000, duration_ms=60_000, session_id="s1"),
            _session(ts_start=80000, duration_ms=60_000, session_id="s2"),
            _cmd("pytest", exit_code=0, ts_start=150000, duration_ms=5000),
        ]
        chains = detect_resolution_chains(events, _config())
        assert len(chains) == 1
        assert chains[0].resolved
        assert set(chains[0].claude_session_ids) == {"s1", "s2"}

    def test_outcome_from_session_outcomes(self):
        """Worst session outcome used for chain."""
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5000),
            _session(ts_start=10000, duration_ms=60_000, session_id="s1"),
            _session(ts_start=80000, duration_ms=60_000, session_id="s2"),
            _cmd("pytest", exit_code=0, ts_start=150000, duration_ms=5000),
        ]
        outcomes = {"s1": "productive", "s2": "friction"}
        chains = detect_resolution_chains(events, _config(), session_outcomes=outcomes)
        assert chains[0].outcome == "friction"

    def test_empty_events(self):
        chains = detect_resolution_chains([], _config())
        assert chains == []


class TestComputeVelocityMetrics:
    def test_basic_metrics(self):
        chains = [
            ResolutionChain(
                initial_failure_ts=0, initial_command="pytest",
                claude_session_ids=["s1"], resolution_ts=100000,
                resolution_command="pytest", active_time_ms=60_000,
                wall_time_ms=100000, project="auth", outcome="productive", closure_reason="matched_success",
            ),
            ResolutionChain(
                initial_failure_ts=0, initial_command="pytest",
                claude_session_ids=["s2"], resolution_ts=200000,
                resolution_command="pytest", active_time_ms=120_000,
                wall_time_ms=200000, project="auth", outcome="productive", closure_reason="matched_success",
            ),
        ]
        metrics = compute_velocity_metrics(chains)
        assert metrics.total_chains == 2
        assert metrics.resolved_count == 2
        assert metrics.avg_ms == 90_000  # (60k + 120k) / 2
        assert metrics.median_ms == 90_000

    def test_per_project_breakdown(self):
        chains = [
            ResolutionChain(
                initial_failure_ts=0, initial_command="pytest",
                claude_session_ids=["s1"], resolution_ts=100000,
                resolution_command="pytest", active_time_ms=60_000,
                wall_time_ms=100000, project="auth", outcome="productive", closure_reason="matched_success",
            ),
            ResolutionChain(
                initial_failure_ts=0, initial_command="npm test",
                claude_session_ids=["s2"], resolution_ts=200000,
                resolution_command="npm test", active_time_ms=30_000,
                wall_time_ms=200000, project="frontend", outcome="productive", closure_reason="matched_success",
            ),
        ]
        metrics = compute_velocity_metrics(chains)
        assert "auth" in metrics.by_project
        assert "frontend" in metrics.by_project
        assert metrics.by_project["auth"].avg_ms == 60_000
        assert metrics.by_project["frontend"].avg_ms == 30_000

    def test_unresolved_excluded_from_metrics(self):
        chains = [
            ResolutionChain(
                initial_failure_ts=0, initial_command="pytest",
                claude_session_ids=[], resolution_ts=100000,
                resolution_command="", active_time_ms=60_000,
                wall_time_ms=100000, project="auth", outcome="productive", closure_reason="end_of_window",
            ),
        ]
        metrics = compute_velocity_metrics(chains)
        assert metrics.total_chains == 1
        assert metrics.resolved_count == 0
        assert metrics.avg_ms == 0

    def test_empty_chains(self):
        metrics = compute_velocity_metrics([])
        assert metrics.total_chains == 0
        assert metrics.avg_ms == 0


class TestClosureReason:
    def test_matched_success(self):
        """Fail → Claude → matching success → closure_reason == matched_success."""
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5000),
            _session(ts_start=10000, duration_ms=300_000, session_id="s1"),
            _cmd("pytest", exit_code=0, ts_start=320000, duration_ms=5000),
        ]
        chains = detect_resolution_chains(events, _config())
        assert len(chains) == 1
        assert chains[0].closure_reason == "matched_success"
        assert chains[0].resolved is True

    def test_idle_break(self):
        """Gap > velocity_idle_break_ms closes open chain with idle_break."""
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5000),
            _session(ts_start=10000, duration_ms=60_000, session_id="s1"),
            # 20 min gap > 15 min idle, then an unrelated event in same project
            _cmd("pytest", exit_code=1, ts_start=70000 + 1_200_000, duration_ms=5000),
        ]
        chains = detect_resolution_chains(events, _config())
        assert len(chains) >= 1
        idle_chains = [c for c in chains if c.closure_reason == "idle_break"]
        assert len(idle_chains) == 1
        assert idle_chains[0].resolved is False

    def test_end_of_window(self):
        """Open chain at end of events closes with end_of_window."""
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5000),
            _session(ts_start=10000, duration_ms=60_000, session_id="s1"),
        ]
        chains = detect_resolution_chains(events, _config())
        assert len(chains) == 1
        assert chains[0].closure_reason == "end_of_window"
        assert chains[0].resolved is False

    def test_by_reason_sums_to_total(self):
        """compute_velocity_metrics.by_reason counts sum to total_chains."""
        chains = [
            ResolutionChain(
                initial_failure_ts=0, initial_command="pytest",
                claude_session_ids=["s1"], resolution_ts=100000,
                resolution_command="pytest", active_time_ms=60_000,
                wall_time_ms=100000, project="auth", outcome="productive",
                closure_reason="matched_success",
            ),
            ResolutionChain(
                initial_failure_ts=0, initial_command="pytest",
                claude_session_ids=[], resolution_ts=100000,
                resolution_command="", active_time_ms=60_000,
                wall_time_ms=100000, project="auth", outcome="productive",
                closure_reason="idle_break",
            ),
            ResolutionChain(
                initial_failure_ts=0, initial_command="pytest",
                claude_session_ids=[], resolution_ts=100000,
                resolution_command="", active_time_ms=60_000,
                wall_time_ms=100000, project="auth", outcome="productive",
                closure_reason="end_of_window",
            ),
        ]
        metrics = compute_velocity_metrics(chains)
        assert sum(metrics.by_reason.values()) == metrics.total_chains == 3
        assert metrics.by_reason["matched_success"] == 1
        assert metrics.by_reason["idle_break"] == 1
        assert metrics.by_reason["end_of_window"] == 1

    def test_resolved_property_back_compat(self):
        """resolved property returns True only for matched_success."""
        c1 = ResolutionChain(
            initial_failure_ts=0, initial_command="pytest",
            claude_session_ids=[], resolution_ts=0, resolution_command="",
            active_time_ms=0, wall_time_ms=0, project="p", outcome="productive",
            closure_reason="matched_success",
        )
        c2 = ResolutionChain(
            initial_failure_ts=0, initial_command="pytest",
            claude_session_ids=[], resolution_ts=0, resolution_command="",
            active_time_ms=0, wall_time_ms=0, project="p", outcome="productive",
            closure_reason="idle_break",
        )
        c3 = ResolutionChain(
            initial_failure_ts=0, initial_command="pytest",
            claude_session_ids=[], resolution_ts=0, resolution_command="",
            active_time_ms=0, wall_time_ms=0, project="p", outcome="productive",
            closure_reason="end_of_window",
        )
        assert c1.resolved is True
        assert c2.resolved is False
        assert c3.resolved is False
