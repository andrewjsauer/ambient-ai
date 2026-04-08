from ambient.capture.reader import Event
from ambient.detect.correlator import correlate_signals


def _cmd(ts_start, ts_end, command="make build", exit_code=0):
    return Event(
        ts_start=ts_start,
        ts_end=ts_end,
        duration_ms=ts_end - ts_start,
        command=command,
        exit_code=exit_code,
        cwd="/tmp",
        tmux_pane=None,
        gap_ms=None,
        type="command",
    )


def _claude(ts_start, ts_end):
    return Event(
        ts_start=ts_start,
        ts_end=ts_end,
        duration_ms=ts_end - ts_start,
        command="claude session",
        exit_code=0,
        cwd="/tmp",
        tmux_pane=None,
        gap_ms=None,
        type="claude_session",
        claude_session_id="sess-1",
    )


class TestHappyPath:
    def test_all_three_patterns_detected(self):
        """Failed command -> Claude session within 5 min -> successful retry."""
        events = [
            _cmd(1000, 2000, "make build", exit_code=1),   # failed command
            _claude(3000, 10000),                            # Claude session within 5 min
            _cmd(11000, 12000, "make build", exit_code=0),  # successful retry
        ]
        findings = correlate_signals(events)

        types = {p.pattern_type for p in findings.patterns}
        assert "error_then_claude" in types
        assert "claude_then_retry" in types
        assert "claude_then_success" in types
        assert findings.total_correlations == 3

    def test_multiple_error_then_claude_sequences(self):
        """Multiple error-then-claude sequences counted correctly."""
        events = [
            _cmd(1000, 2000, "cargo build", exit_code=1),
            _claude(3000, 10000),
            _cmd(100_000, 101_000, "npm test", exit_code=2),
            _claude(102_000, 110_000),
        ]
        findings = correlate_signals(events)

        etc = next(p for p in findings.patterns if p.pattern_type == "error_then_claude")
        assert etc.count == 2


class TestEdgeCases:
    def test_claude_more_than_5min_after_failure_not_correlated(self):
        """Claude session starts more than 5 minutes after failed command."""
        events = [
            _cmd(1000, 2000, "make build", exit_code=1),
            _claude(400_000, 500_000),  # 398 seconds after cmd end, > 300s window
        ]
        findings = correlate_signals(events)

        types = {p.pattern_type for p in findings.patterns}
        assert "error_then_claude" not in types

    def test_no_shell_commands_only_claude_sessions(self):
        """All Claude sessions, no shell commands -- no crash, no correlations."""
        events = [
            _claude(1000, 5000),
            _claude(10000, 15000),
        ]
        findings = correlate_signals(events)

        assert findings.patterns == []
        assert findings.total_correlations == 0

    def test_grep_nonzero_exit_not_treated_as_error(self):
        """grep with non-zero exit code is benign, not an error."""
        events = [
            _cmd(1000, 2000, "grep -r foo .", exit_code=1),
            _claude(3000, 10000),
        ]
        findings = correlate_signals(events)

        types = {p.pattern_type for p in findings.patterns}
        assert "error_then_claude" not in types

    def test_empty_event_list(self):
        """Empty event list returns empty findings."""
        findings = correlate_signals([])

        assert findings.patterns == []
        assert findings.total_correlations == 0

    def test_diff_nonzero_exit_not_treated_as_error(self):
        """diff with exit code 1 is benign."""
        events = [
            _cmd(1000, 2000, "diff a.txt b.txt", exit_code=1),
            _claude(3000, 10000),
        ]
        findings = correlate_signals(events)

        types = {p.pattern_type for p in findings.patterns}
        assert "error_then_claude" not in types

    def test_test_command_nonzero_exit_not_treated_as_error(self):
        """test -f with exit code 1 is benign."""
        events = [
            _cmd(1000, 2000, "test -f /tmp/foo", exit_code=1),
            _claude(3000, 10000),
        ]
        findings = correlate_signals(events)

        types = {p.pattern_type for p in findings.patterns}
        assert "error_then_claude" not in types

    def test_claude_session_exit_code_not_false_match(self):
        """Claude sessions have exit_code=0 but should not match as shell success."""
        # Only commands (type=command) after Claude should count for claude_then_success
        events = [
            _cmd(1000, 2000, "make build", exit_code=1),
            _claude(3000, 10000),
            # No shell commands after Claude -- only another Claude session
            _claude(11000, 15000),
        ]
        findings = correlate_signals(events)

        types = {p.pattern_type for p in findings.patterns}
        # error_then_claude should match (failed cmd -> first Claude)
        assert "error_then_claude" in types
        # claude_then_success should NOT match (no shell command after Claude)
        assert "claude_then_success" not in types
