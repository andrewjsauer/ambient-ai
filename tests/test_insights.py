"""Tests for coaching insights module."""

import json
from unittest.mock import patch

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.coaching import CoachingFindings, SessionOutcome, StuckPattern, StuckPatternFindings
from ambient.detect.velocity import ResolutionChain, VelocityMetrics
from ambient.detect.prompt_patterns import PromptPattern, PromptPatternFindings
from ambient.present.insights import (
    CoachingData,
    PeriodComparison,
    aggregate_coaching_data,
    build_insights_prompt,
    compute_period_comparison,
    format_terminal_summary,
    generate_insights_report,
)


def _config(**overrides):
    return Config(**overrides)


def _sample_data(
    total_sessions=5,
    stuck_sessions=2,
    resolved_chains=3,
    avg_velocity_ms=120_000,
):
    outcomes = [
        SessionOutcome("s1", "productive", 0.1, "auth", 300_000, 10, 1, [], []),
        SessionOutcome("s2", "friction", 0.8, "auth", 600_000, 6, 5, [], []),
        SessionOutcome("s3", "quick", None, "frontend", 30_000, 2, 0, [], []),
        SessionOutcome("s4", "abandoned", 0.6, "auth", 400_000, 8, 3, [], []),
        SessionOutcome("s5", "productive", 0.2, "frontend", 500_000, 12, 2, [], []),
    ][:total_sessions]

    findings = CoachingFindings(
        outcomes=outcomes,
        count_by_classification={"productive": 2, "friction": 1, "quick": 1, "abandoned": 1},
        avg_thrash_score=0.43,
    )

    stuck = StuckPatternFindings(
        patterns=[
            StuckPattern("auth", ["src/auth.py"], ["Bash"], 2, 0.7, 1_000_000, ["s2", "s4"]),
        ],
        total_stuck_sessions=stuck_sessions,
    )

    chains = [
        ResolutionChain(0, "pytest", ["s1"], 100000, "pytest", avg_velocity_ms, 200000, "auth", "productive", True),
    ] * resolved_chains

    velocity = VelocityMetrics(
        avg_ms=avg_velocity_ms,
        median_ms=avg_velocity_ms,
        p90_ms=avg_velocity_ms + 60_000,
        total_chains=resolved_chains + 1,
        resolved_count=resolved_chains,
        by_project={"auth": VelocityMetrics(avg_ms=avg_velocity_ms, resolved_count=resolved_chains)},
    )

    return CoachingData(
        coaching_findings=findings,
        stuck_patterns=stuck,
        velocity_metrics=velocity,
        chains=chains,
        window_days=7,
        date_range="2026-04-01 to 2026-04-08",
    )


class TestBuildInsightsPrompt:
    def test_includes_session_outcomes(self):
        prompt = build_insights_prompt(_sample_data())
        assert "productive" in prompt
        assert "friction" in prompt
        assert "5 sessions" in prompt

    def test_includes_velocity(self):
        prompt = build_insights_prompt(_sample_data())
        assert "RESOLUTION VELOCITY" in prompt
        assert "3 resolved" in prompt

    def test_includes_stuck_patterns(self):
        prompt = build_insights_prompt(_sample_data())
        assert "STUCK PATTERNS" in prompt
        assert "auth" in prompt
        assert "Bash" in prompt

    def test_no_resolved_chains(self):
        data = _sample_data(resolved_chains=0)
        data.velocity_metrics = VelocityMetrics(total_chains=1, resolved_count=0)
        prompt = build_insights_prompt(data)
        assert "No resolved chains" in prompt

    def test_no_stuck_patterns(self):
        data = _sample_data(stuck_sessions=0)
        data.stuck_patterns = StuckPatternFindings()
        prompt = build_insights_prompt(data)
        assert "No stuck patterns detected" in prompt


class TestFormatTerminalSummary:
    def test_includes_velocity(self):
        summary = format_terminal_summary(_sample_data())
        assert "Resolution velocity" in summary
        assert "min avg" in summary

    def test_includes_stuck_count(self):
        summary = format_terminal_summary(_sample_data())
        assert "Stuck episodes" in summary
        assert "2" in summary

    def test_includes_top_finding(self):
        summary = format_terminal_summary(_sample_data())
        assert "auth" in summary

    def test_no_resolved_chains(self):
        data = _sample_data(resolved_chains=0)
        data.velocity_metrics = VelocityMetrics()
        summary = format_terminal_summary(data)
        assert "no resolved chains" in summary

    def test_no_stuck_patterns(self):
        data = _sample_data(stuck_sessions=0)
        data.stuck_patterns = StuckPatternFindings()
        summary = format_terminal_summary(data)
        assert "No significant stuck patterns" in summary


class TestGenerateInsightsReport:
    def test_writes_report_file(self, tmp_path):
        config = _config(base_dir=tmp_path)
        data = _sample_data()

        with patch("ambient.present.api.call_api", return_value="# Coaching Report\nGreat work!"):
            narrative = generate_insights_report(data, config)

        assert narrative is not None
        assert "Coaching Report" in narrative
        # Check file was written
        insight_files = list((tmp_path / "insights").glob("*.md"))
        assert len(insight_files) == 1

    def test_returns_none_on_api_failure(self, tmp_path):
        config = _config(base_dir=tmp_path)
        data = _sample_data()

        with patch("ambient.present.api.call_api", side_effect=Exception("API error")):
            narrative = generate_insights_report(data, config)

        assert narrative is None


def _write_events_for_aggregate(config, date_str, event_dicts):
    """Append Event-shaped dicts to the daily events log."""
    path = config.events_path(date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for e in event_dicts:
            f.write(json.dumps(e) + "\n")


class TestAggregateCoachingData:
    """aggregate_coaching_data runs every detector and returns one aggregate."""

    def test_extended_fields_populated_on_empty_window(self, tmp_path):
        config = _config(base_dir=tmp_path)
        data = aggregate_coaching_data(config, window_days=7)
        # No events → every detector returns an empty-but-shaped result, no None
        assert data.prompt_patterns is not None
        assert data.compression is not None
        assert data.correlations is not None
        assert data.prompt_patterns.patterns == []
        assert data.compression.sequences == []
        assert data.correlations.patterns == []

    def test_correlator_is_invoked_with_real_data(self, tmp_path):
        """Fail-then-Claude event pair → correlator emits a pattern."""
        from datetime import datetime
        config = _config(base_dir=tmp_path)
        today = datetime.now().strftime("%Y-%m-%d")
        now_ms = int(datetime.now().timestamp() * 1000)

        # Shell failure 10s before claude session — should match error_then_claude
        shell_fail = {
            "ts_start": now_ms - 120_000,
            "ts_end": now_ms - 115_000,
            "duration_ms": 5_000,
            "command": "pytest",
            "exit_code": 1,
            "cwd": "/home/user/proj",
            "tmux_pane": None,
            "gap_ms": None,
            "type": "command",
        }
        claude = {
            "ts_start": now_ms - 100_000,
            "ts_end": now_ms - 10_000,
            "duration_ms": 90_000,
            "command": "claude: fix the test",
            "exit_code": 0,
            "cwd": "/home/user/proj",
            "tmux_pane": None,
            "gap_ms": None,
            "type": "claude_session",
            "claude_session_id": "sess-1",
            "claude_prompts": ["fix the test"],
            "claude_tools": [],
            "claude_files": [],
            "claude_project": "proj",
            "claude_prompt_count": 1,
            "claude_is_error_count": 0,
        }
        _write_events_for_aggregate(config, today, [shell_fail, claude])

        data = aggregate_coaching_data(config, window_days=7)
        assert data.correlations.total_correlations >= 1
        pattern_types = {p.pattern_type for p in data.correlations.patterns}
        assert "error_then_claude" in pattern_types

    def test_detectors_failure_does_not_crash_aggregate(self, tmp_path):
        """If a detector raises, aggregate still returns with the default empty."""
        config = _config(base_dir=tmp_path)
        with patch(
            "ambient.present.insights.detect_prompt_patterns",
            side_effect=RuntimeError("boom"),
        ):
            data = aggregate_coaching_data(config, window_days=7, compare=False)
        assert data.prompt_patterns.patterns == []


def _make_coaching_data(
    resolved_count=10,
    avg_ms=120_000,
    stuck_sessions=5,
    avg_thrash=0.4,
    top_patterns=(),
    date_range="2026-04-01 to 2026-04-07",
):
    velocity = VelocityMetrics(
        avg_ms=avg_ms,
        median_ms=avg_ms,
        p90_ms=avg_ms,
        total_chains=resolved_count,
        resolved_count=resolved_count,
    )
    stuck = StuckPatternFindings(patterns=[], total_stuck_sessions=stuck_sessions)
    findings = CoachingFindings(
        outcomes=[],
        count_by_classification={},
        avg_thrash_score=avg_thrash,
    )
    prompt_patterns = PromptPatternFindings(
        patterns=[
            PromptPattern(
                normalized_prompt=norm,
                raw_examples=[norm],
                count=count,
                projects=["p"],
                scope="within_session",
            )
            for norm, count in top_patterns
        ],
        total_prompts=sum(c for _, c in top_patterns),
    )
    return CoachingData(
        coaching_findings=findings,
        stuck_patterns=stuck,
        velocity_metrics=velocity,
        chains=[],
        window_days=7,
        date_range=date_range,
        prompt_patterns=prompt_patterns,
    )


class TestComputePeriodComparison:
    def test_happy_path_velocity_delta(self):
        current = _make_coaching_data(resolved_count=10, avg_ms=120_000, stuck_sessions=5)
        prior = _make_coaching_data(resolved_count=10, avg_ms=180_000, stuck_sessions=8,
                                    date_range="2026-03-25 to 2026-03-31")
        cmp = compute_period_comparison(current, prior, Config())
        assert cmp.insufficient_data_reason is None
        # Current faster than prior → negative delta
        assert cmp.velocity_delta_ms == -60_000
        assert cmp.stuck_delta == -3

    def test_insufficient_current_chains(self):
        current = _make_coaching_data(resolved_count=3, avg_ms=120_000, stuck_sessions=5)
        prior = _make_coaching_data(resolved_count=10, avg_ms=180_000, stuck_sessions=8)
        cmp = compute_period_comparison(current, prior, Config())
        assert cmp.insufficient_data_reason is not None
        assert "resolved chains" in cmp.insufficient_data_reason
        assert cmp.velocity_delta_ms is None

    def test_insufficient_prior_stuck(self):
        current = _make_coaching_data(resolved_count=10, avg_ms=120_000, stuck_sessions=5)
        prior = _make_coaching_data(resolved_count=10, avg_ms=180_000, stuck_sessions=1)
        cmp = compute_period_comparison(current, prior, Config())
        assert cmp.insufficient_data_reason is not None
        assert "stuck sessions" in cmp.insufficient_data_reason

    def test_pattern_churn_new_and_dropped(self):
        current = _make_coaching_data(
            resolved_count=10, avg_ms=120_000, stuck_sessions=5,
            top_patterns=[("commit this", 6), ("fix the test", 4)],
        )
        prior = _make_coaching_data(
            resolved_count=10, avg_ms=180_000, stuck_sessions=5,
            top_patterns=[("plan the feature", 5), ("commit this", 4)],
            date_range="2026-03-25 to 2026-03-31",
        )
        cmp = compute_period_comparison(current, prior, Config())
        assert "fix the test" in cmp.new_patterns
        assert "plan the feature" in cmp.dropped_patterns
        assert "commit this" not in cmp.new_patterns

    def test_thrash_delta_skipped_when_either_is_none(self):
        current = _make_coaching_data(resolved_count=10, avg_thrash=None, stuck_sessions=5)
        prior = _make_coaching_data(resolved_count=10, avg_thrash=0.5, stuck_sessions=5)
        cmp = compute_period_comparison(current, prior, Config())
        assert cmp.thrash_delta is None

    def test_prior_date_range_always_set(self):
        current = _make_coaching_data(resolved_count=3, stuck_sessions=5)
        prior = _make_coaching_data(resolved_count=3, stuck_sessions=5,
                                    date_range="2026-03-25 to 2026-03-31")
        cmp = compute_period_comparison(current, prior, Config())
        assert cmp.prior_date_range == "2026-03-25 to 2026-03-31"


class TestAggregateCompareFlag:
    def test_compare_false_skips_prior_window_read(self, tmp_path):
        config = _config(base_dir=tmp_path)
        data = aggregate_coaching_data(config, window_days=7, compare=False)
        assert data.comparison is None

    def test_compare_true_runs_prior_aggregate(self, tmp_path):
        """When compare=True, comparison is populated (insufficient reason is fine)."""
        config = _config(base_dir=tmp_path)
        data = aggregate_coaching_data(config, window_days=7, compare=True)
        assert data.comparison is not None
        # Empty window → both sides fail the gate
        assert data.comparison.insufficient_data_reason is not None
