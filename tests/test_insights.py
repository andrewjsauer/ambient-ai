"""Tests for coaching insights module."""

from unittest.mock import patch

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.coaching import CoachingFindings, SessionOutcome, StuckPattern, StuckPatternFindings
from ambient.detect.velocity import ResolutionChain, VelocityMetrics
from ambient.present.insights import (
    CoachingData,
    build_insights_prompt,
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
