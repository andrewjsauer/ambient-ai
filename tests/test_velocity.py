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
        # Real ingestion writes claude_project as the session's FULL cwd path
        # (session_parser captures cwd verbatim) — never a bare project name.
        claude_project=f"/home/user/{project}",
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

    def test_in_session_resolution_standalone(self):
        """A session that fixed its own failing test (red→green via Bash),
        with no shell fail/success to bracket it, is a resolved chain — the
        common Claude-Code workflow the shell-only detector missed."""
        sess = _session(ts_start=10000, duration_ms=600_000, session_id="s1")
        sess.claude_verification_resolved = True
        chains = detect_resolution_chains([sess], _config())
        assert len(chains) == 1
        assert chains[0].resolved is True
        assert chains[0].closure_reason == "in_session"
        assert "s1" in chains[0].claude_session_ids

    def test_in_session_resolution_closes_open_shell_chain(self):
        """Shell fail → session that resolves in-session → chain resolved even
        with no matching shell success."""
        fail = _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5000)
        sess = _session(ts_start=10000, duration_ms=300_000, session_id="s1")
        sess.claude_verification_resolved = True
        chains = detect_resolution_chains([fail, sess], _config())
        resolved = [c for c in chains if c.resolved]
        assert len(resolved) == 1
        assert resolved[0].closure_reason == "in_session"
        # The session is not also double-counted as a standalone chain.
        assert len(chains) == 1

    def test_in_session_no_double_count_when_shell_resolves(self):
        """A normal shell fail→session→shell-success chain still counts once,
        even if the session also resolved in-session."""
        sess = _session(ts_start=10000, duration_ms=300_000, session_id="s1")
        sess.claude_verification_resolved = True
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5000),
            sess,
            _cmd("pytest", exit_code=0, ts_start=320000, duration_ms=5000),
        ]
        chains = detect_resolution_chains(events, _config())
        assert len([c for c in chains if c.resolved]) == 1

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

    def test_full_path_claude_project_groups_with_commands(self):
        """Regression: claude_project is a full path on real data; the session
        must land in the same project group as the failed/success commands or
        the chain never stitches (the pre-fix code returned the path verbatim,
        so no real chain ever resolved)."""
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, cwd="/Users/x/proj"),
            _session(ts_start=10000, session_id="s1"),
            _cmd("pytest", exit_code=0, ts_start=320000, cwd="/Users/x/proj"),
        ]
        # Override the fixture's path to share only the basename with cwd.
        events[1].claude_project = "/Users/x/proj"
        chains = detect_resolution_chains(events, _config())
        assert len(chains) == 1
        assert chains[0].resolved is True
        assert chains[0].project == "proj"
        assert "s1" in chains[0].claude_session_ids

    def test_session_without_claude_project_falls_back_to_cwd(self):
        """A session missing claude_project groups by its cwd basename."""
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, cwd="/home/user/auth"),
            _session(ts_start=10000, session_id="s1"),
            _cmd("pytest", exit_code=0, ts_start=320000, cwd="/home/user/auth"),
        ]
        events[1].claude_project = None
        chains = detect_resolution_chains(events, _config())
        assert len(chains) == 1
        assert chains[0].project == "auth"

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


class TestFirstClaudePrompt:
    def test_populated_from_first_session(self):
        """first_claude_prompt captures the first prompt of the first claude session in the chain."""
        session = _session(ts_start=10000, duration_ms=60_000, session_id="s1")
        session.claude_prompts = ["fix the failing test in auth_test.py"]
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5000),
            session,
            _cmd("pytest", exit_code=0, ts_start=80000, duration_ms=5000),
        ]
        chains = detect_resolution_chains(events, _config())
        assert len(chains) == 1
        assert chains[0].first_claude_prompt == "fix the failing test in auth_test.py"

    def test_truncated_to_max_length(self):
        """Long prompt is truncated to FIRST_PROMPT_MAX_LENGTH."""
        from ambient.detect.velocity import FIRST_PROMPT_MAX_LENGTH
        long_prompt = "x" * 500
        session = _session(ts_start=10000, session_id="s1")
        session.claude_prompts = [long_prompt]
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5000),
            session,
            _cmd("pytest", exit_code=0, ts_start=320000, duration_ms=5000),
        ]
        chains = detect_resolution_chains(events, _config())
        assert len(chains[0].first_claude_prompt) == FIRST_PROMPT_MAX_LENGTH

    def test_only_first_session_prompt_captured(self):
        """When chain has multiple claude sessions, only the first session's first prompt is stored."""
        s1 = _session(ts_start=10000, duration_ms=60_000, session_id="s1")
        s1.claude_prompts = ["first session first prompt"]
        s2 = _session(ts_start=80000, duration_ms=60_000, session_id="s2")
        s2.claude_prompts = ["second session first prompt"]
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5000),
            s1,
            s2,
            _cmd("pytest", exit_code=0, ts_start=150000, duration_ms=5000),
        ]
        chains = detect_resolution_chains(events, _config())
        assert chains[0].first_claude_prompt == "first session first prompt"

    def test_empty_prompts_list_no_crash(self):
        """Claude session with empty claude_prompts yields empty string, not None."""
        session = _session(ts_start=10000, session_id="s1")
        session.claude_prompts = []
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5000),
            session,
            _cmd("pytest", exit_code=0, ts_start=320000, duration_ms=5000),
        ]
        chains = detect_resolution_chains(events, _config())
        assert chains[0].first_claude_prompt == ""

    def test_no_claude_session_empty_prompt(self):
        """Chain that never hits a claude_session has empty first_claude_prompt."""
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5000),
        ]
        chains = detect_resolution_chains(events, _config())
        assert len(chains) == 1
        assert chains[0].first_claude_prompt == ""
        assert chains[0].closure_reason == "end_of_window"


class TestActiveTimeCap:
    """Per-event contribution to active_time_ms is capped so long-running
    foreground processes (dev servers) cannot dominate the metric."""

    def test_command_over_cap_clamped(self):
        """A make-dev style command running for hours contributes at most the cap."""
        # Chain: failed pytest (5 s) -> claude (60 s) -> make dev (8 h) success
        eight_hours_ms = 8 * 60 * 60 * 1000
        events = [
            _cmd("pytest", exit_code=1, ts_start=1_000_000, duration_ms=5_000),
            _session(ts_start=1_010_000, duration_ms=60_000, session_id="s1"),
            _cmd("pytest", exit_code=0, ts_start=1_080_000, duration_ms=eight_hours_ms),
        ]
        chains = detect_resolution_chains(events, _config())
        assert len(chains) == 1
        chain = chains[0]
        assert chain.resolved is True
        # Expected active time: 5 s + 60 s + 600 s (cap) = 665 s
        expected = 5_000 + 60_000 + 600_000
        assert chain.active_time_ms == expected
        # Wall time is unaffected (still reflects real elapsed time)
        assert chain.wall_time_ms > chain.active_time_ms

    def test_session_over_cap_clamped(self):
        """A 4-hour Claude session contributes at most the session cap (60 min)."""
        four_hours_ms = 4 * 60 * 60 * 1000
        events = [
            _cmd("pytest", exit_code=1, ts_start=1_000_000, duration_ms=5_000),
            _session(ts_start=1_010_000, duration_ms=four_hours_ms, session_id="s1"),
            _cmd("pytest", exit_code=0, ts_start=1_010_000 + four_hours_ms + 1_000,
                 duration_ms=3_000),
        ]
        chains = detect_resolution_chains(events, _config())
        assert len(chains) == 1
        chain = chains[0]
        # Expected: 5 s + 3600 s (session cap) + 3 s = 3608 s
        expected = 5_000 + 3_600_000 + 3_000
        assert chain.active_time_ms == expected

    def test_initial_failed_command_also_capped(self):
        """The seeding command for a chain is subject to the same cap."""
        # A failed `make dev` that ran for 3 hours before finally exiting non-zero
        three_hours_ms = 3 * 60 * 60 * 1000
        events = [
            _cmd("make dev", exit_code=1, ts_start=1_000_000, duration_ms=three_hours_ms),
            _session(ts_start=1_000_000 + three_hours_ms + 1_000,
                     duration_ms=60_000, session_id="s1"),
            _cmd("make dev", exit_code=0,
                 ts_start=1_000_000 + three_hours_ms + 70_000, duration_ms=2_000),
        ]
        chains = detect_resolution_chains(events, _config())
        assert len(chains) == 1
        # Expected: 600_000 (initial cap) + 60_000 + 2_000 = 662_000
        assert chains[0].active_time_ms == 662_000

    def test_under_cap_unchanged(self):
        """Events well under the cap contribute their full duration."""
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5_000),
            _session(ts_start=10_000, duration_ms=300_000, session_id="s1"),
            _cmd("pytest", exit_code=0, ts_start=320_000, duration_ms=5_000),
        ]
        chains = detect_resolution_chains(events, _config())
        assert chains[0].active_time_ms == 5_000 + 300_000 + 5_000

    def test_custom_caps_config(self):
        """Cap values come from Config, not hardcoded."""
        config = _config(
            velocity_max_command_contribution_ms=1_000,  # 1 s cap
            velocity_max_session_contribution_ms=2_000,  # 2 s cap
        )
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=10_000),  # capped to 1 s
            _session(ts_start=15_000, duration_ms=10_000, session_id="s1"),  # capped to 2 s
            _cmd("pytest", exit_code=0, ts_start=30_000, duration_ms=10_000),  # capped to 1 s
        ]
        chains = detect_resolution_chains(events, config)
        assert chains[0].active_time_ms == 1_000 + 2_000 + 1_000


class TestAbandonmentReasonTaxonomy:
    """closure_reason carries specific idle-break semantics: interrupt_mid_thought,
    context_rot, or given_up — in addition to matched_success and end_of_window."""

    def _base_events(self, session):
        """Shared helper: failed command → session → idle gap."""
        return [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5_000),
            session,
            _cmd("pytest", exit_code=1,
                 ts_start=session.ts_end + 1_200_000,  # 20 min gap
                 duration_ms=5_000),
        ]

    def test_interrupt_mid_thought(self):
        """Last tool is AskUserQuestion → interrupt_mid_thought."""
        session = _session(ts_start=10_000, duration_ms=60_000, session_id="s1")
        session.claude_tools = [
            {"name": "Read", "files": ["a.py"]},
            {"name": "AskUserQuestion", "files": []},
        ]
        chains = detect_resolution_chains(self._base_events(session), _config())
        interrupted = [c for c in chains if c.closure_reason == "interrupt_mid_thought"]
        assert len(interrupted) == 1

    def test_context_rot(self):
        """>=5 Read/Grep/ToolSearch calls with 0 Edit/Write → context_rot."""
        session = _session(ts_start=10_000, duration_ms=60_000, session_id="s1")
        session.claude_tools = [
            {"name": "Read", "files": []},
            {"name": "Grep", "files": []},
            {"name": "Glob", "files": []},
            {"name": "Read", "files": []},
            {"name": "ToolSearch", "files": []},
        ]
        chains = detect_resolution_chains(self._base_events(session), _config())
        rotted = [c for c in chains if c.closure_reason == "context_rot"]
        assert len(rotted) == 1

    def test_given_up_with_writes(self):
        """Session had at least one Edit/Write → given_up."""
        session = _session(ts_start=10_000, duration_ms=60_000, session_id="s1")
        session.claude_tools = [
            {"name": "Read", "files": []},
            {"name": "Edit", "files": ["x.py"]},
        ]
        chains = detect_resolution_chains(self._base_events(session), _config())
        given_up = [c for c in chains if c.closure_reason == "given_up"]
        assert len(given_up) == 1

    def test_matched_success_still_wins(self):
        """matched_success takes priority over all idle classifications."""
        events = [
            _cmd("pytest", exit_code=1, ts_start=1_000, duration_ms=5_000),
            _session(ts_start=10_000, duration_ms=60_000, session_id="s1"),
            _cmd("pytest", exit_code=0, ts_start=80_000, duration_ms=5_000),
        ]
        chains = detect_resolution_chains(events, _config())
        assert len(chains) == 1
        assert chains[0].closure_reason == "matched_success"
        assert chains[0].resolved is True

    def test_end_of_window_unchanged(self):
        """Chain still open at end of events → end_of_window, not classified."""
        session = _session(ts_start=10_000, duration_ms=60_000, session_id="s1")
        events = [
            _cmd("pytest", exit_code=1, ts_start=1_000, duration_ms=5_000),
            session,
        ]
        chains = detect_resolution_chains(events, _config())
        assert chains[0].closure_reason == "end_of_window"

    def test_shell_only_chain_given_up(self):
        """Chain with no claude_session at all → defaults to given_up on idle_break."""
        events = [
            _cmd("pytest", exit_code=1, ts_start=1_000, duration_ms=5_000),
            _cmd("ls", exit_code=0, ts_start=10_000, duration_ms=500,
                 cwd="/home/user/auth"),
            _cmd("pytest", exit_code=1, ts_start=1_300_000, duration_ms=5_000),
        ]
        chains = detect_resolution_chains(events, _config())
        idle_chains = [c for c in chains
                       if c.closure_reason not in ("matched_success", "end_of_window")]
        assert len(idle_chains) == 1
        assert idle_chains[0].closure_reason == "given_up"

    def test_context_rot_threshold_configurable(self):
        """velocity_context_rot_min_tool_calls controls the threshold."""
        config = _config(velocity_context_rot_min_tool_calls=2)
        session = _session(ts_start=10_000, duration_ms=60_000, session_id="s1")
        session.claude_tools = [
            {"name": "Read", "files": []},
            {"name": "Grep", "files": []},
        ]
        chains = detect_resolution_chains(self._base_events(session), config)
        rotted = [c for c in chains if c.closure_reason == "context_rot"]
        assert len(rotted) == 1

    def test_resolved_property_back_compat(self):
        """All new reason codes keep resolved=False (only matched_success is True)."""
        session = _session(ts_start=10_000, duration_ms=60_000, session_id="s1")
        session.claude_tools = [{"name": "AskUserQuestion", "files": []}]
        chains = detect_resolution_chains(self._base_events(session), _config())
        assert chains[0].closure_reason == "interrupt_mid_thought"
        assert chains[0].resolved is False


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

    def test_idle_break_given_up(self):
        """Gap > velocity_idle_break_ms closes open chain; session had Edit so classified as given_up."""
        events = [
            _cmd("pytest", exit_code=1, ts_start=1000, duration_ms=5000),
            _session(ts_start=10000, duration_ms=60_000, session_id="s1"),
            # 20 min gap > 15 min idle, then an unrelated event in same project
            _cmd("pytest", exit_code=1, ts_start=70000 + 1_200_000, duration_ms=5000),
        ]
        chains = detect_resolution_chains(events, _config())
        assert len(chains) >= 1
        # Session had Edit tool (real work) so reason is given_up, not interrupt/rot
        given_up = [c for c in chains if c.closure_reason == "given_up"]
        assert len(given_up) == 1
        assert given_up[0].resolved is False

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
