"""Tests for coaching insights module."""

import json
from unittest.mock import patch

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.coaching import (
    CoachingFindings,
    FileClusterStuckPattern,
    SessionOutcome,
    StuckPattern,
    StuckPatternFindings,
    ToolStuckPattern,
)
from ambient.detect.compression import CompressionFindings, RepeatedSequence
from ambient.detect.correlator import CorrelationFindings, CorrelationPattern
from ambient.detect.prompt_patterns import PromptPattern, PromptPatternFindings
from ambient.detect.velocity import ResolutionChain, VelocityMetrics
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
        """v3 behavior: empty stuck-patterns section is omitted entirely."""
        data = _sample_data(stuck_sessions=0)
        data.stuck_patterns = StuckPatternFindings()
        prompt = build_insights_prompt(data)
        assert "STUCK PATTERNS" not in prompt


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

    def test_resets_capability_cache_each_run(self, tmp_path):
        """aggregate_coaching_data must clear the per-run capability cache so
        capability changes between runs (new package.json, removed Makefile)
        are picked up. Without this, stale capabilities bleed across back-to-
        back insights runs."""
        from ambient.detect import project_capabilities as pc
        from pathlib import Path

        config = _config(base_dir=tmp_path)

        # Pre-seed the cache with a value that detect_capabilities would not
        # produce on its own; the first thing aggregate_coaching_data does is
        # clear the cache, so this entry must not survive.
        pc._capability_cache["/sentinel/path"] = pc.Capabilities(
            has_tests=True, evidence={"has_tests": "sentinel"}
        )
        assert "/sentinel/path" in pc._capability_cache

        aggregate_coaching_data(config, window_days=7, compare=False)

        assert "/sentinel/path" not in pc._capability_cache

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


def _fully_populated_data():
    """CoachingData with at least one item in every detector output for prompt-shape tests."""
    outcomes = [
        SessionOutcome("s1", "productive", 0.1, "auth", 300_000, 10, 1,
                       [{"name": "Edit"}], ["src/auth.py"]),
        SessionOutcome("s2", "friction", 0.8, "auth", 600_000, 6, 5,
                       [{"name": "Edit"}, {"name": "Bash"}], ["src/auth.py"]),
        SessionOutcome("s3", "friction", 0.7, "auth", 500_000, 7, 5,
                       [{"name": "Edit"}, {"name": "Bash"}], ["src/login.py"]),
    ]
    findings = CoachingFindings(
        outcomes=outcomes,
        count_by_classification={"productive": 1, "friction": 2},
        avg_thrash_score=0.53,
    )
    stuck = StuckPatternFindings(
        patterns=[StuckPattern("auth", ["src/auth.py", "src/login.py"],
                               ["Edit", "Bash"], 2, 0.75, 1_100_000, ["s2", "s3"])],
        total_stuck_sessions=2,
        tool_level_patterns=[ToolStuckPattern("Edit", 2, ["auth"], None, 1_100_000)],
        file_cluster_patterns=[FileClusterStuckPattern("src/", 2, ["auth"],
                                                        ["Edit", "Bash"], 1_100_000)],
    )
    chains = [
        ResolutionChain(
            initial_failure_ts=1000, initial_command="pytest auth_test.py",
            claude_session_ids=["s1"], resolution_ts=200000,
            resolution_command="pytest auth_test.py", active_time_ms=120_000,
            wall_time_ms=199_000, project="auth", outcome="productive",
            closure_reason="matched_success",
            first_claude_prompt="fix the failing auth test",
        ),
        ResolutionChain(
            initial_failure_ts=1000, initial_command="npm test",
            claude_session_ids=["s2"], resolution_ts=300000,
            resolution_command="", active_time_ms=420_000,
            wall_time_ms=299_000, project="frontend", outcome="friction",
            closure_reason="idle_break",
            first_claude_prompt="figure out why the snapshot test is broken",
        ),
    ]
    velocity = VelocityMetrics(
        avg_ms=120_000, median_ms=120_000, p90_ms=180_000,
        total_chains=2, resolved_count=1,
        by_reason={"matched_success": 1, "idle_break": 1},
    )
    prompt_patterns = PromptPatternFindings(
        patterns=[
            PromptPattern("commit this", ["commit this"], 7, ["auth"], "within_session"),
            PromptPattern("plan it -> ship it", ["plan it -> ship it"], 4,
                          ["frontend"], "cross_session"),
        ],
        total_prompts=30,
    )
    compression = CompressionFindings(
        sequences=[
            RepeatedSequence(("pytest -x", "git add"), 14, 30_000, 28),
        ],
        compression_ratio=0.6,
    )
    correlations = CorrelationFindings(
        patterns=[
            CorrelationPattern(
                pattern_type="error_then_claude",
                count=8,
                examples=[
                    {"command": "pytest auth_test.py", "exit_code": 1,
                     "claude_session_start": 2000, "gap_ms": 15_000},
                ],
            ),
        ],
        total_correlations=8,
    )
    comparison = PeriodComparison(
        velocity_delta_ms=-30_000,
        stuck_delta=-1,
        thrash_delta=-0.05,
        prior_date_range="2026-04-03 to 2026-04-09",
    )
    return CoachingData(
        coaching_findings=findings,
        stuck_patterns=stuck,
        velocity_metrics=velocity,
        chains=chains,
        window_days=7,
        date_range="2026-04-10 to 2026-04-16",
        prompt_patterns=prompt_patterns,
        compression=compression,
        correlations=correlations,
        comparison=comparison,
        pending_recommendations=[
            {"id": "skill-commit-this", "type": "skill", "title": "Commit this skill",
             "rationale": "repeated 7x"},
        ],
    )


class TestRichBuildInsightsPrompt:
    def test_every_new_section_appears(self):
        prompt = build_insights_prompt(_fully_populated_data())
        assert "RECURRING PROMPTS" in prompt
        assert "RECURRING COMMAND SEQUENCES" in prompt
        assert "SHELL ↔ CLAUDE CORRELATIONS" in prompt
        assert "TOP RESOLUTION CHAINS" in prompt
        assert "STUCK PATTERNS — BY PROJECT" in prompt
        assert "STUCK PATTERNS — BY FAILING TOOL" in prompt
        assert "STUCK PATTERNS — BY FILE CLUSTER" in prompt
        assert "PERIOD COMPARISON" in prompt
        assert "PENDING RECOMMENDATIONS" in prompt

    def test_concrete_examples_embedded(self):
        """Prompt embeds verbatim prompts, commands, files, and first_claude_prompt strings."""
        prompt = build_insights_prompt(_fully_populated_data())
        # Verbatim prompt text
        assert '"commit this"' in prompt
        assert '"plan it -> ship it"' in prompt
        # Shell command sequence
        assert "pytest -x" in prompt and "git add" in prompt
        # Resolution chain opener
        assert "fix the failing auth test" in prompt
        assert "pytest auth_test.py" in prompt
        # Cluster + tool grouping
        assert "src/" in prompt
        assert "Edit" in prompt

    def test_within_and_cross_prompts_deduped(self):
        """A pattern with identical text in both scopes is only rendered once (cross wins)."""
        data = _fully_populated_data()
        data.prompt_patterns = PromptPatternFindings(
            patterns=[
                PromptPattern("commit this", ["commit this"], 7,
                              ["auth"], "within_session"),
                PromptPattern("commit this", ["commit this"], 3,
                              ["auth"], "cross_session"),
            ],
            total_prompts=10,
        )
        prompt = build_insights_prompt(data)
        assert prompt.count('"commit this"') == 1

    def test_system_prompt_requires_verbatim_citation(self):
        """The system prompt instructs Sonnet to quote specific examples."""
        from ambient.present.insights import INSIGHTS_SYSTEM
        assert "verbatim" in INSIGHTS_SYSTEM.lower()
        # Must forbid generic phrasing
        assert "insufficient" in INSIGHTS_SYSTEM.lower()

    def test_system_prompt_has_surprise_section(self):
        """Surprise of the Week directive is present; v3 omits empty sections
        entirely instead of using a verbatim escape-hatch phrase."""
        from ambient.present.insights import INSIGHTS_SYSTEM
        assert "Surprise of the Week" in INSIGHTS_SYSTEM
        # v3 instructs Sonnet to omit the section rather than echo a
        # placeholder phrase. Confirm the omit instruction is present.
        assert "omit this section" in INSIGHTS_SYSTEM.lower()

    def test_system_prompt_drops_anti_pattern_callout(self):
        """v3 removes the Anti-Pattern Callout in favor of Diagnostic Questions.
        The 2026-04-25 critique showed the callout pathologized normal
        continuation prompts ('go for it', 'yes lets fix these') based on
        thin data."""
        from ambient.present.insights import INSIGHTS_SYSTEM
        assert "Anti-Pattern Callout" not in INSIGHTS_SYSTEM

    def test_system_prompt_has_diagnostic_questions(self):
        """v3 ends the report with three Socratic questions instead of
        prescribing recommendations. Forces the developer's own synthesis."""
        from ambient.present.insights import INSIGHTS_SYSTEM
        assert "Diagnostic Questions" in INSIGHTS_SYSTEM
        # Must require exactly three, and forbid prescriptive directives
        assert "exactly three" in INSIGHTS_SYSTEM
        assert "Do not issue directives" in INSIGHTS_SYSTEM

    def test_system_prompt_drops_developer_prompt_grading(self):
        """v3 explicitly tells the model not to grade continuation prompts
        like 'go for it' as risk signals — those are normal acceptance
        responses, not anti-patterns."""
        from ambient.present.insights import INSIGHTS_SYSTEM
        assert "go for it" in INSIGHTS_SYSTEM
        assert "agent's planning posture" in INSIGHTS_SYSTEM

    def test_system_prompt_flags_100_percent_as_suspect(self):
        """v3 instructs the model to treat 100% rates as detector-suspect
        rather than confident findings — direct response to the false
        100% verification gap in the v2 report."""
        from ambient.present.insights import INSIGHTS_SYSTEM
        assert "100%" in INSIGHTS_SYSTEM
        assert "suspect" in INSIGHTS_SYSTEM.lower()

    def test_system_prompt_vocabulary_glossary(self):
        """Glossary names industry-standard terms for consistent naming.
        v3 drops 'vague framing' from the user-facing glossary because that
        framing pathologized normal continuation prompts; the vague-framing
        detector remains internal as a stuck-pattern grouping signal."""
        from ambient.present.insights import INSIGHTS_SYSTEM
        for term in ("prompt debt", "verification gap", "context rot",
                     "cognitive debt"):
            assert term in INSIGHTS_SYSTEM.lower()

    def test_empty_sections_are_omitted_entirely(self):
        """v3 behavior: sections with no input data are omitted from the
        prompt entirely, not rendered with 'no data' filler. The narrative
        layer is instructed to skip them rather than acknowledge them."""
        data = _fully_populated_data()
        data.prompt_patterns = PromptPatternFindings(patterns=[], total_prompts=0)
        data.compression = CompressionFindings(sequences=[], compression_ratio=1.0)
        data.correlations = CorrelationFindings()
        prompt = build_insights_prompt(data)
        assert "RECURRING PROMPTS" not in prompt
        assert "RECURRING COMMAND SEQUENCES" not in prompt
        assert "SHELL ↔ CLAUDE CORRELATIONS" not in prompt
        # The placeholder text from v2 should never appear
        assert "None detected" not in prompt


class TestResearchBackedSections:
    """Unit 6 wire-up: verification gaps + abandonment reasons + opening prompts
    all surface in the prompt built from a populated fixture."""

    def _rich_data_with_new_signals(self):
        from ambient.detect.verification import VerificationGap, VerificationGapFindings
        data = _fully_populated_data()
        # Stuff the stuck pattern with opening prompts + vague framing
        data.stuck_patterns.patterns[0].opening_prompts = [
            "figure out why the auth test is broken",
            "fix this login flow",
        ]
        data.stuck_patterns.patterns[0].vague_framing_fraction = 0.67
        # Add a tool-level pattern with prompts
        data.stuck_patterns.tool_level_patterns[0].opening_prompts = [
            "debug this edit issue",
        ]
        # Add a file cluster with prompts
        data.stuck_patterns.file_cluster_patterns[0].opening_prompts = [
            "why is src/auth broken",
        ]
        # Verification gaps — has_tests bucket dominates the rich fixture
        data.verification_gaps = VerificationGapFindings(
            gaps=[
                VerificationGap(
                    session_id="sess-gap-1", project="auth",
                    session_end_ts=123456, session_cwd="/tmp/auth",
                    edited_files=["src/auth.py"],
                    bucket="has_tests",
                ),
            ],
            total_fix_sessions=12,
            gap_rate=1 / 12,
            low_sample=False,
            total_fix_sessions_by_bucket={"has_tests": 12, "has_typecheck": 0, "neither": 0},
            gaps_by_bucket={"has_tests": 1, "has_typecheck": 0, "neither": 0},
            gap_rate_by_bucket={"has_tests": 1 / 12, "has_typecheck": None, "neither": None},
            low_sample_by_bucket={"has_tests": False, "has_typecheck": True, "neither": True},
        )
        # Give velocity by_reason counts the new codes
        data.velocity_metrics.by_reason = {
            "matched_success": 1,
            "interrupt_mid_thought": 2,
            "context_rot": 3,
            "given_up": 4,
            "end_of_window": 1,
        }
        # And populate chains with one example per new reason
        data.chains = [
            ResolutionChain(
                initial_failure_ts=0, initial_command="pytest",
                claude_session_ids=["s1"], resolution_ts=100,
                resolution_command="", active_time_ms=240_000, wall_time_ms=200,
                project="auth", outcome="productive",
                closure_reason="interrupt_mid_thought",
                first_claude_prompt="fix the failing auth test",
            ),
            ResolutionChain(
                initial_failure_ts=0, initial_command="npm test",
                claude_session_ids=["s2"], resolution_ts=100,
                resolution_command="", active_time_ms=360_000, wall_time_ms=200,
                project="frontend", outcome="friction",
                closure_reason="context_rot",
                first_claude_prompt="understand why login breaks",
            ),
            ResolutionChain(
                initial_failure_ts=0, initial_command="cargo build",
                claude_session_ids=["s3"], resolution_ts=100,
                resolution_command="", active_time_ms=480_000, wall_time_ms=200,
                project="rustproj", outcome="productive",
                closure_reason="given_up",
                first_claude_prompt="refactor the service trait",
            ),
        ]
        return data

    def test_verification_gaps_section_renders(self):
        data = self._rich_data_with_new_signals()
        prompt = build_insights_prompt(data)
        assert "VERIFICATION GAPS" in prompt
        # Per-bucket headline includes the projects-with-tests count
        assert "projects with tests" in prompt
        assert "12 fix session(s)" in prompt
        assert "src/auth.py" in prompt

    def test_verification_gaps_empty_omitted_in_v3(self):
        """v3: when there are no fix sessions, the section is omitted from
        the prompt entirely. v2 emitted a 'No fix sessions' placeholder."""
        data = _fully_populated_data()  # default empty VerificationGapFindings
        prompt = build_insights_prompt(data)
        assert "VERIFICATION GAPS" not in prompt
        assert "No fix sessions" not in prompt

    def test_verification_gaps_low_sample(self):
        from ambient.detect.verification import VerificationGap, VerificationGapFindings
        data = _fully_populated_data()
        data.verification_gaps = VerificationGapFindings(
            gaps=[VerificationGap(
                session_id="s1", project="x", session_end_ts=0,
                session_cwd="/tmp", edited_files=["a.py"], bucket="has_tests")],
            total_fix_sessions=3,
            gap_rate=None,
            low_sample=True,
            total_fix_sessions_by_bucket={"has_tests": 3, "has_typecheck": 0, "neither": 0},
            gaps_by_bucket={"has_tests": 1, "has_typecheck": 0, "neither": 0},
            gap_rate_by_bucket={"has_tests": None, "has_typecheck": None, "neither": None},
            low_sample_by_bucket={"has_tests": True, "has_typecheck": True, "neither": True},
        )
        prompt = build_insights_prompt(data)
        assert "low sample" in prompt
        # No percentage should be published
        assert "%" not in prompt or prompt.count("%") == 0 or \
            "%)" not in [l for l in prompt.split("\n") if "low sample" in l][0]

    def test_abandonment_section_has_new_reasons(self):
        data = self._rich_data_with_new_signals()
        prompt = build_insights_prompt(data)
        assert "ABANDONMENT BY REASON" in prompt
        assert "interrupt_mid_thought" in prompt
        assert "context_rot" in prompt
        assert "given_up" in prompt
        # Example chain first_claude_prompt appears verbatim
        assert "fix the failing auth test" in prompt
        assert "understand why login breaks" in prompt

    def test_stuck_pattern_includes_opening_prompts(self):
        data = self._rich_data_with_new_signals()
        prompt = build_insights_prompt(data)
        assert "figure out why the auth test is broken" in prompt
        assert "fix this login flow" in prompt

    def test_vague_framing_label_not_emitted_into_prompt(self):
        """v3 drops the 'vague framing: X%' label from the LLM prompt because
        INSIGHTS_SYSTEM now tells the model not to grade developer prompts.
        The detector field remains for internal stuck-pattern grouping."""
        data = self._rich_data_with_new_signals()
        prompt = build_insights_prompt(data)
        assert "vague framing" not in prompt.lower()

    def test_tool_and_cluster_patterns_also_carry_prompts_in_prompt(self):
        data = self._rich_data_with_new_signals()
        prompt = build_insights_prompt(data)
        assert "debug this edit issue" in prompt
        assert "why is src/auth broken" in prompt


class TestRichTerminalSummary:
    def test_includes_top_prompt_and_sequence(self):
        summary = format_terminal_summary(_fully_populated_data())
        assert "Top repeated prompt" in summary
        assert '"commit this"' in summary
        assert "Top command sequence" in summary
        assert "pytest -x" in summary

    def test_includes_pending_count(self):
        summary = format_terminal_summary(_fully_populated_data())
        assert "Pending recs:" in summary
        assert "1" in summary

    def test_period_delta_surfaced_when_available(self):
        data = _fully_populated_data()
        data.comparison = PeriodComparison(
            velocity_delta_ms=-60_000,
            stuck_delta=-2,
            prior_date_range="2026-04-03 to 2026-04-09",
        )
        summary = format_terminal_summary(data)
        assert "vs prior" in summary

    def test_verification_gap_line_in_terminal(self):
        """v3 renders per-bucket. The 12 fix sessions all live in has_tests."""
        from ambient.detect.verification import VerificationGap, VerificationGapFindings
        data = _fully_populated_data()
        data.verification_gaps = VerificationGapFindings(
            gaps=[VerificationGap(
                session_id="s1", project="p", session_end_ts=0,
                session_cwd="/tmp", edited_files=["a.py"], bucket="has_tests")],
            total_fix_sessions=12,
            gap_rate=1 / 12,
            low_sample=False,
            total_fix_sessions_by_bucket={"has_tests": 12, "has_typecheck": 0, "neither": 0},
            gaps_by_bucket={"has_tests": 1, "has_typecheck": 0, "neither": 0},
            gap_rate_by_bucket={"has_tests": 1 / 12, "has_typecheck": None, "neither": None},
            low_sample_by_bucket={"has_tests": False, "has_typecheck": True, "neither": True},
        )
        summary = format_terminal_summary(data)
        assert "Verification gaps (tests)" in summary
        assert "12 fixes" in summary

    def test_terminal_summary_neither_bucket_does_not_show_percentage(self):
        """v3 fix for the v2 100% headline: an all-neither workload must not
        produce a '10/10 fixes (100%)' line in the terminal summary."""
        from ambient.detect.verification import VerificationGap, VerificationGapFindings
        data = _fully_populated_data()
        data.verification_gaps = VerificationGapFindings(
            gaps=[VerificationGap(
                session_id=f"s{i}", project="np", session_end_ts=0,
                session_cwd="/tmp", edited_files=["a.py"], bucket="neither")
                for i in range(10)],
            total_fix_sessions=10,
            gap_rate=1.0,  # legacy aggregate field; should not be displayed
            low_sample=False,
            total_fix_sessions_by_bucket={"has_tests": 0, "has_typecheck": 0, "neither": 10},
            gaps_by_bucket={"has_tests": 0, "has_typecheck": 0, "neither": 10},
            gap_rate_by_bucket={"has_tests": None, "has_typecheck": None, "neither": None},
            low_sample_by_bucket={"has_tests": True, "has_typecheck": True, "neither": False},
        )
        summary = format_terminal_summary(data)
        assert "100%" not in summary
        assert "10/10 fixes" not in summary
        # Should surface the neither sessions descriptively, not as a gap rate
        assert "non-verifiable projects" in summary
        assert "10" in summary

    def test_terminal_summary_per_bucket_rendering(self):
        """Per-bucket breakdown: has_tests + has_typecheck both render with rates."""
        from ambient.detect.verification import VerificationGap, VerificationGapFindings
        data = _fully_populated_data()
        data.verification_gaps = VerificationGapFindings(
            gaps=[],
            total_fix_sessions=22,
            gap_rate=None,
            low_sample=False,
            total_fix_sessions_by_bucket={"has_tests": 12, "has_typecheck": 10, "neither": 0},
            gaps_by_bucket={"has_tests": 3, "has_typecheck": 2, "neither": 0},
            gap_rate_by_bucket={"has_tests": 3 / 12, "has_typecheck": 2 / 10, "neither": None},
            low_sample_by_bucket={"has_tests": False, "has_typecheck": False, "neither": True},
        )
        summary = format_terminal_summary(data)
        assert "Verification gaps (tests): 3/12 fixes (25%)" in summary
        assert "Verification gaps (typecheck): 2/10 fixes (20%)" in summary

    def test_top_trigger_prompt_in_terminal(self):
        data = _fully_populated_data()
        data.stuck_patterns.patterns[0].opening_prompts = [
            "figure out why auth is broken",
        ]
        summary = format_terminal_summary(data)
        assert "Top trigger prompt:" in summary
        assert "figure out why auth is broken" in summary


class TestPromptBudgetTrimming:
    def test_over_budget_logs_warning_but_still_fires(self, tmp_path, caplog, monkeypatch):
        """v3 removed the proportional-shrink loop. When the prompt exceeds
        INSIGHTS_INPUT_BUDGET, the run logs a warning and fires the full
        prompt unchanged (confidence gates handle suppression upstream)."""
        import logging
        from ambient.present import insights as insights_mod

        config = _config(base_dir=tmp_path)
        data = _fully_populated_data()

        # Force the over-budget path by setting an absurdly small budget. The
        # populated fixture's prompt is several thousand chars, well above 1.
        monkeypatch.setattr(insights_mod, "INSIGHTS_INPUT_BUDGET", 1)

        captured = {"prompt_len": 0}

        def fake_call_api(config, system, prompt, model, max_tokens=3000, client=None):
            captured["prompt_len"] = len(prompt)
            return "report"

        with caplog.at_level(logging.WARNING, logger="ambient.present.insights"):
            with patch("ambient.present.api.call_api", side_effect=fake_call_api):
                result = insights_mod.generate_insights_report(data, config)

        assert result == "report"
        # No shrinking occurred — full prompt sent
        assert captured["prompt_len"] > 0
        # And the over-budget warning fired
        assert any(
            "INSIGHTS_PROMPT_OVER_BUDGET" in record.message
            for record in caplog.records
        )

    def test_under_budget_does_not_warn(self, tmp_path, caplog):
        """Sanity check: a normal-sized prompt does not trigger the warning."""
        import logging
        from ambient.present import insights as insights_mod

        config = _config(base_dir=tmp_path)
        data = _fully_populated_data()

        def fake_call_api(config, system, prompt, model, max_tokens=3000, client=None):
            return "report"

        with caplog.at_level(logging.WARNING, logger="ambient.present.insights"):
            with patch("ambient.present.api.call_api", side_effect=fake_call_api):
                insights_mod.generate_insights_report(data, config)

        assert not any(
            "INSIGHTS_PROMPT_OVER_BUDGET" in record.message
            for record in caplog.records
        )


class TestPendingRecommendations:
    def test_aggregate_lists_pending(self, tmp_path):
        config = _config(base_dir=tmp_path)
        rec_dir = config.recommendations_dir
        rec_dir.mkdir(parents=True, exist_ok=True)
        (rec_dir / "skill-commit-push.md").write_text(
            '---\ntype: skill\ntitle: "Commit and push"\nrationale: "typed 7 times"\n---\n\nbody\n'
        )
        (rec_dir / "alias-gp.md").write_text(
            '---\ntype: alias\ntitle: "gp alias"\nrationale: "sequence run 5x"\n---\n\nalias gp=git push\n'
        )
        data = aggregate_coaching_data(config, window_days=7, compare=False)
        assert len(data.pending_recommendations) == 2
        ids = {r["id"] for r in data.pending_recommendations}
        assert ids == {"skill-commit-push", "alias-gp"}
        types = {r["type"] for r in data.pending_recommendations}
        assert types == {"skill", "alias"}

    def test_aggregate_empty_when_no_recommendations_dir(self, tmp_path):
        config = _config(base_dir=tmp_path)
        data = aggregate_coaching_data(config, window_days=7, compare=False)
        assert data.pending_recommendations == []
