"""Tests for v4 Phase 1 Unit 5 — insights wiring + --by-day rendering.

Covers:
- New `_section_*` functions (project_ledger, command_mix, freeform_fraction)
- `INSIGHTS_SYSTEM` references the new section names
- `format_terminal_summary` includes the worked-on + freeform-fraction lines
- `--by-day` rendering produces a daily timeline
- `--by-day` falls back to aggregate when window < 2 days
- All sections omit cleanly when their data is None
"""

from datetime import datetime, timezone
from dataclasses import replace

from ambient.capture.reader import Event
from ambient.detect.coaching import (
    CoachingFindings,
    SessionOutcome,
    StuckPatternFindings,
)
from ambient.detect.command_mix import CommandMixFindings, ProjectMix
from ambient.detect.freeform_fraction import FreeformFraction
from ambient.detect.project_ledger import (
    ProjectLedger,
    ProjectLedgerEntry,
)
from ambient.detect.velocity import ResolutionChain, VelocityMetrics
from ambient.present.insights import (
    CoachingData,
    INSIGHTS_SYSTEM,
    build_insights_prompt,
    format_terminal_summary,
)


def _empty_data(by_day: bool = False, events: list | None = None) -> CoachingData:
    findings = CoachingFindings(
        outcomes=[],
        count_by_classification={},
        avg_thrash_score=None,
    )
    stuck = StuckPatternFindings(patterns=[], total_stuck_sessions=0)
    velocity = VelocityMetrics(0, 0, 0, 0, 0)
    return CoachingData(
        coaching_findings=findings,
        stuck_patterns=stuck,
        velocity_metrics=velocity,
        chains=[],
        window_days=7,
        date_range="2026-04-20 to 2026-04-27",
        by_day=by_day,
        events=events,
    )


def _ledger(entries):
    return ProjectLedger(
        entries=entries,
        window_start_iso="2026-04-20T00:00:00+00:00",
        window_end_iso="2026-04-27T00:00:00+00:00",
        time_basis="command_span",
    )


def _command_mix(per_project=None, overall=None):
    return CommandMixFindings(
        per_project=per_project or {},
        overall=overall or ProjectMix(),
        window_start_iso="2026-04-20T00:00:00+00:00",
        window_end_iso="2026-04-27T00:00:00+00:00",
    )


def _freeform_fraction(overall_pct, prior_pct=None, total=100, per_project=None):
    delta = (overall_pct - prior_pct) if prior_pct is not None else None
    return FreeformFraction(
        overall_pct=overall_pct,
        prior_window_pct=prior_pct,
        delta_pct=delta,
        per_project=per_project or {},
        total_prompts=total,
        prior_total_prompts=total if prior_pct is not None else 0,
        window_start_iso="2026-04-20T00:00:00+00:00",
        window_end_iso="2026-04-27T00:00:00+00:00",
    )


# ---------- INSIGHTS_SYSTEM smoke ----------

class TestInsightsSystemPrompt:
    def test_mentions_project_ledger_section(self):
        assert "PROJECT LEDGER" in INSIGHTS_SYSTEM
        assert "What You Worked On" in INSIGHTS_SYSTEM

    def test_mentions_command_mix_section(self):
        assert "COMMAND MIX" in INSIGHTS_SYSTEM
        assert "Command Mix" in INSIGHTS_SYSTEM

    def test_omit_when_absent_rule_present(self):
        # The system prompt must instruct the model to drop sections when their input is missing.
        assert "omit this section" in INSIGHTS_SYSTEM


# ---------- _section_project_ledger ----------

class TestSectionProjectLedger:
    def test_renders_entries_with_summary(self):
        data = _empty_data()
        data.project_ledger = _ledger([
            ProjectLedgerEntry(
                project="ambient-ai",
                active_ms=4 * 60 * 60_000,  # 4h
                session_count=23,
                top_files=["tick.py", "git_activity.py"],
                representative_prompts=["..."],
                summary="Phase 2 git activity detector with subprocess hardening.",
            ),
        ])
        prompt = build_insights_prompt(data)
        assert "PROJECT LEDGER" in prompt
        assert "ambient-ai" in prompt
        assert "4.0h" in prompt
        assert "23 session" in prompt
        assert "tick.py" in prompt
        assert "Phase 2 git activity detector with subprocess hardening." in prompt

    def test_renders_command_span_label_by_default(self):
        data = _empty_data()
        data.project_ledger = _ledger([
            ProjectLedgerEntry(project="x", active_ms=15 * 60_000, session_count=1),
        ])
        prompt = build_insights_prompt(data)
        assert "command-span time" in prompt

    def test_renders_active_label_when_attention_weighted(self):
        data = _empty_data()
        data.project_ledger = ProjectLedger(
            entries=[
                ProjectLedgerEntry(project="x", active_ms=15 * 60_000, session_count=1),
            ],
            time_basis="attention_weighted",
        )
        prompt = build_insights_prompt(data)
        assert "active time" in prompt

    def test_omits_when_ledger_is_none(self):
        prompt = build_insights_prompt(_empty_data())
        assert "PROJECT LEDGER" not in prompt

    def test_omits_when_entries_empty(self):
        data = _empty_data()
        data.project_ledger = _ledger([])
        prompt = build_insights_prompt(data)
        assert "PROJECT LEDGER" not in prompt

    def test_renders_minutes_for_short_sessions(self):
        data = _empty_data()
        data.project_ledger = _ledger([
            ProjectLedgerEntry(project="quick", active_ms=20 * 60_000, session_count=1),
        ])
        prompt = build_insights_prompt(data)
        assert "20min" in prompt


# ---------- _section_command_mix ----------

class TestSectionCommandMix:
    def test_renders_overall_and_per_project(self):
        data = _empty_data()
        data.command_mix = _command_mix(
            overall=ProjectMix(planning_count=10, execution_count=20, review_count=10, freeform_count=60),
            per_project={"sample-app": ProjectMix(planning_count=5, execution_count=10, freeform_count=15)},
        )
        prompt = build_insights_prompt(data)
        assert "COMMAND MIX" in prompt
        assert "sample-app" in prompt
        assert "Overall:" in prompt
        # Spot-check a percentage rendered (60 freeform of 100 → 60%)
        assert "60%" in prompt

    def test_omits_when_command_mix_is_none(self):
        prompt = build_insights_prompt(_empty_data())
        assert "COMMAND MIX" not in prompt

    def test_omits_when_total_is_zero(self):
        data = _empty_data()
        data.command_mix = _command_mix()  # all zeros
        prompt = build_insights_prompt(data)
        assert "COMMAND MIX" not in prompt


# ---------- _section_freeform_fraction ----------

class TestSectionFreeformFraction:
    def test_renders_overall_with_no_delta(self):
        data = _empty_data()
        data.freeform_fraction = _freeform_fraction(overall_pct=0.785, total=4087)
        prompt = build_insights_prompt(data)
        assert "FREEFORM FRACTION" in prompt
        assert "78.5%" in prompt
        assert "4087" in prompt

    def test_renders_delta_when_prior_present(self):
        data = _empty_data()
        data.freeform_fraction = _freeform_fraction(
            overall_pct=0.70, prior_pct=0.90, total=100
        )
        prompt = build_insights_prompt(data)
        assert "FREEFORM FRACTION" in prompt
        # 0.70 - 0.90 = -0.20 → -20.0pp
        assert "-20.0pp" in prompt

    def test_renders_positive_delta_with_sign(self):
        data = _empty_data()
        data.freeform_fraction = _freeform_fraction(
            overall_pct=0.85, prior_pct=0.78, total=100
        )
        prompt = build_insights_prompt(data)
        # 0.85 - 0.78 = 0.07 → +7.0pp
        assert "+7.0pp" in prompt

    def test_renders_per_project_breakdown(self):
        data = _empty_data()
        data.freeform_fraction = _freeform_fraction(
            overall_pct=0.78, total=100,
            per_project={"sample-app": 0.85, "ambient-ai": 0.70},
        )
        prompt = build_insights_prompt(data)
        assert "sample-app" in prompt
        assert "ambient-ai" in prompt

    def test_omits_when_none(self):
        prompt = build_insights_prompt(_empty_data())
        assert "FREEFORM FRACTION" not in prompt

    def test_omits_when_total_zero(self):
        data = _empty_data()
        data.freeform_fraction = _freeform_fraction(overall_pct=0.0, total=0)
        prompt = build_insights_prompt(data)
        assert "FREEFORM FRACTION" not in prompt


# ---------- format_terminal_summary additions ----------

class TestTerminalSummaryAdditions:
    def test_includes_worked_on_lines(self):
        data = _empty_data()
        data.project_ledger = _ledger([
            ProjectLedgerEntry(
                project="ambient-ai",
                active_ms=4 * 60 * 60_000,
                session_count=23,
                top_files=["tick.py"],
                summary="Phase 2 git activity detector.",
            ),
        ])
        out = format_terminal_summary(data)
        assert "Worked on:" in out
        assert "ambient-ai" in out
        assert "Phase 2 git activity detector." in out

    def test_caps_worked_on_at_three_entries(self):
        data = _empty_data()
        data.project_ledger = _ledger([
            ProjectLedgerEntry(project=f"proj{i}", active_ms=60 * 60_000, session_count=1)
            for i in range(5)
        ])
        out = format_terminal_summary(data)
        assert out.count("Worked on:") == 3

    def test_includes_freeform_line(self):
        data = _empty_data()
        data.freeform_fraction = _freeform_fraction(overall_pct=0.78, total=100)
        out = format_terminal_summary(data)
        assert "Freeform fraction" in out
        assert "78%" in out

    def test_omits_phase1_lines_when_data_absent(self):
        out = format_terminal_summary(_empty_data())
        assert "Worked on:" not in out
        assert "Freeform fraction" not in out


# ---------- --by-day rendering ----------

def _shell_event(day_iso: str, project_path: str, duration_ms: int) -> Event:
    ts_ms = int(datetime.fromisoformat(day_iso).timestamp() * 1000)
    return Event(
        ts_start=ts_ms,
        ts_end=ts_ms + duration_ms,
        duration_ms=duration_ms,
        command="ls",
        exit_code=0,
        cwd=project_path,
        tmux_pane=None,
        gap_ms=None,
        type="command",
    )


class TestByDayRendering:
    def test_renders_per_day_timeline(self):
        events = [
            _shell_event("2026-04-21T10:00:00", "/repo/ambient-ai", 90 * 60_000),  # 1.5h
            _shell_event("2026-04-22T10:00:00", "/repo/sample-app", 30 * 60_000),
            _shell_event("2026-04-22T15:00:00", "/repo/ambient-ai", 20 * 60_000),
        ]
        data = _empty_data(by_day=True, events=events)
        out = format_terminal_summary(data)
        assert "--by-day" in out
        assert "ambient-ai" in out
        assert "sample-app" in out
        # Day labels: "Tue Apr 21" then "Wed Apr 22" — order matters
        idx_21 = out.find("Apr 21")
        idx_22 = out.find("Apr 22")
        assert idx_21 < idx_22

    def test_skips_projects_under_5_minute_floor(self):
        events = [
            _shell_event("2026-04-21T10:00:00", "/repo/quick", 60_000),  # 1 min, below floor
            _shell_event("2026-04-21T10:00:00", "/repo/real", 30 * 60_000),
        ]
        data = _empty_data(by_day=True, events=events)
        out = format_terminal_summary(data)
        assert "real" in out
        assert "quick" not in out

    def test_falls_back_when_window_too_short(self):
        data = _empty_data(by_day=True, events=[])
        data.window_days = 1
        out = format_terminal_summary(data)
        assert "falls back to aggregate" in out

    def test_no_events_message(self):
        data = _empty_data(by_day=True, events=[])
        out = format_terminal_summary(data)
        assert "No events" in out

    def test_aggregate_render_when_by_day_false(self):
        # The default render path must NOT emit the by-day banner.
        events = [_shell_event("2026-04-21T10:00:00", "/repo/x", 30 * 60_000)]
        data = _empty_data(by_day=False, events=events)
        out = format_terminal_summary(data)
        assert "--by-day" not in out

    def test_year_boundary_sorts_chronologically(self):
        # Window spanning Dec 30 → Jan 2 must sort chronologically, not by
        # calendar-day-of-current-year. Earlier implementation used
        # datetime.now().year on every day, putting Dec 30 AFTER Jan 2.
        events = [
            _shell_event("2026-01-02T10:00:00", "/repo/p", 30 * 60_000),
            _shell_event("2025-12-30T10:00:00", "/repo/p", 30 * 60_000),
            _shell_event("2025-12-31T10:00:00", "/repo/p", 30 * 60_000),
            _shell_event("2026-01-01T10:00:00", "/repo/p", 30 * 60_000),
        ]
        data = _empty_data(by_day=True, events=events)
        out = format_terminal_summary(data)
        idx_dec30 = out.find("Dec 30")
        idx_dec31 = out.find("Dec 31")
        idx_jan01 = out.find("Jan 01")
        idx_jan02 = out.find("Jan 02")
        # All four days must appear, in chronological order.
        assert idx_dec30 != -1 and idx_dec31 != -1 and idx_jan01 != -1 and idx_jan02 != -1
        assert idx_dec30 < idx_dec31 < idx_jan01 < idx_jan02


class TestWalkCoalescing:
    """Lock in the contract that aggregate_coaching_data walks each window
    exactly once for Phase 1 detectors, not once per detector. Regression guard
    for the perf review's PERF-1 finding (8x walks → 2x).
    """

    def test_phase1_walks_once_per_window(self, tmp_path, monkeypatch):
        from unittest.mock import patch
        from ambient.config import Config
        from ambient.present import insights as insights_mod

        cfg = Config(base_dir=tmp_path / ".ambient")
        cfg.ensure_dirs()
        # Empty Claude projects + empty events file path
        cfg.claude_projects_dir = tmp_path / "projects"
        cfg.claude_projects_dir.mkdir()

        call_log = []

        original_walk = insights_mod._walk_prompts_for_window

        def tracking_walk(projects_dir, start, end):
            call_log.append((start.isoformat(), end.isoformat()))
            return original_walk(projects_dir, start, end)

        with patch.object(insights_mod, "_walk_prompts_for_window", tracking_walk):
            insights_mod.aggregate_coaching_data(cfg, window_days=7, compare=True)

        # Expect exactly two walks: one for prior window, one for current window.
        # No detector should walk independently.
        assert len(call_log) == 2, (
            f"Expected 2 prompt walks (1 current + 1 prior), got {len(call_log)}: {call_log}"
        )

    def test_phase1_detectors_skipped_on_prior_window(self, tmp_path):
        from unittest.mock import patch
        from ambient.config import Config
        from ambient.present import insights as insights_mod

        cfg = Config(base_dir=tmp_path / ".ambient")
        cfg.ensure_dirs()
        cfg.claude_projects_dir = tmp_path / "projects"
        cfg.claude_projects_dir.mkdir()

        with patch.object(insights_mod, "detect_command_mix") as mock_cm, \
             patch.object(insights_mod, "detect_freeform_fraction") as mock_ff, \
             patch.object(insights_mod, "detect_project_ledger") as mock_pl:
            mock_cm.return_value = None
            mock_ff.return_value = None
            mock_pl.return_value = None
            insights_mod.aggregate_coaching_data(cfg, window_days=7, compare=True)

        # Each Phase 1 detector should run exactly ONCE — for the current window.
        # The prior aggregation must skip them (skip_phase1=True).
        assert mock_cm.call_count == 1, f"command_mix called {mock_cm.call_count}x"
        assert mock_ff.call_count == 1, f"freeform_fraction called {mock_ff.call_count}x"
        assert mock_pl.call_count == 1, f"project_ledger called {mock_pl.call_count}x"
