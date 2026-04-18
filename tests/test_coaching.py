"""Tests for coaching detectors: session outcome classification + stuck pattern grouping."""

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.coaching import (
    classify_session_outcome,
    classify_sessions,
    group_stuck_patterns,
)


def _make_session(
    prompt_count=10,
    error_count=0,
    tools=None,
    files=None,
    duration_ms=600_000,
    project="test-project",
    session_id="sess-1",
    event_type="claude_session",
    prompts=None,
):
    """Build a claude_session Event for testing."""
    return Event(
        ts_start=1000000,
        ts_end=1000000 + duration_ms,
        duration_ms=duration_ms,
        command="claude: test",
        exit_code=0,
        cwd="/home/user/test-project",
        tmux_pane=None,
        gap_ms=None,
        type=event_type,
        claude_session_id=session_id,
        claude_prompts=prompts if prompts is not None else ["test"] * prompt_count,
        claude_tools=tools,
        claude_files=files,
        claude_project=project,
        claude_prompt_count=prompt_count,
        claude_is_error_count=error_count,
    )


def _config(**overrides):
    return Config(**overrides)


# ── Unit 12: Session Outcome Classification ──


class TestClassifySessionOutcome:
    def test_productive_session(self):
        """10 prompts, 2 Write calls, 1 error → Productive."""
        event = _make_session(
            prompt_count=10,
            error_count=1,
            tools=[
                {"name": "Read", "files": ["a.py"]},
                {"name": "Write", "files": ["b.py"]},
                {"name": "Write", "files": ["c.py"]},
                {"name": "Bash", "files": []},
            ],
        )
        outcome = classify_session_outcome(event, _config())
        assert outcome.classification == "productive"
        assert outcome.thrash_score == 1 / 10

    def test_quick_session(self):
        """3 prompts, 1 tool call → Quick."""
        event = _make_session(
            prompt_count=3,
            error_count=0,
            tools=[{"name": "Read", "files": []}],
        )
        outcome = classify_session_outcome(event, _config())
        assert outcome.classification == "quick"

    def test_abandoned_session(self):
        """8 prompts, 0 Write/Edit, 3 errors, 10 min → Abandoned."""
        event = _make_session(
            prompt_count=8,
            error_count=3,
            tools=[
                {"name": "Bash", "files": []},
                {"name": "Read", "files": []},
                {"name": "Grep", "files": []},
            ],
            duration_ms=600_000,
        )
        outcome = classify_session_outcome(event, _config())
        assert outcome.classification == "abandoned"

    def test_friction_session(self):
        """6 prompts, 5 errors → Friction (thrash_score 0.83 > 0.5)."""
        event = _make_session(
            prompt_count=6,
            error_count=5,
            tools=[
                {"name": "Write", "files": ["a.py"]},
                {"name": "Bash", "files": []},
                {"name": "Edit", "files": ["b.py"]},
                {"name": "Bash", "files": []},
                {"name": "Bash", "files": []},
                {"name": "Bash", "files": []},
            ],
        )
        outcome = classify_session_outcome(event, _config())
        assert outcome.classification == "friction"
        assert outcome.thrash_score > 0.5

    def test_quick_precedence_over_abandoned(self):
        """2 prompts, 2 errors → Quick (precedence), not Abandoned."""
        event = _make_session(
            prompt_count=2,
            error_count=2,
            tools=[{"name": "Bash", "files": []}],
            duration_ms=600_000,
        )
        outcome = classify_session_outcome(event, _config())
        assert outcome.classification == "quick"

    def test_thrash_score_none_below_floor(self):
        """2 prompts, 2 errors → thrash_score is None (below floor of 3)."""
        event = _make_session(prompt_count=2, error_count=2)
        outcome = classify_session_outcome(event, _config())
        assert outcome.thrash_score is None

    def test_zero_prompts(self):
        """0 prompts → Quick, thrash_score is None."""
        event = _make_session(prompt_count=0, error_count=0, tools=[])
        outcome = classify_session_outcome(event, _config())
        assert outcome.classification == "quick"
        assert outcome.thrash_score is None

    def test_shell_command_skipped_by_classify_sessions(self):
        """Shell command events (type='command') are not classified."""
        shell = _make_session(event_type="command")
        session = _make_session(event_type="claude_session", prompt_count=10)
        findings = classify_sessions([shell, session], _config())
        assert len(findings.outcomes) == 1
        assert findings.outcomes[0].classification == "productive"

    def test_missing_error_count_treated_as_zero(self):
        """Session with claude_is_error_count=None → treated as 0 errors."""
        event = _make_session(prompt_count=10, error_count=0)
        event.claude_is_error_count = None
        outcome = classify_session_outcome(event, _config())
        assert outcome.error_count == 0
        assert outcome.classification == "productive"


class TestClassifySessions:
    def test_counts_by_classification(self):
        events = [
            _make_session(prompt_count=3, tools=[], session_id="s1"),  # quick
            _make_session(prompt_count=10, error_count=0, tools=[{"name": "Write", "files": []}] * 3, session_id="s2"),  # productive
            _make_session(prompt_count=10, error_count=0, tools=[{"name": "Write", "files": []}] * 3, session_id="s3"),  # productive
        ]
        findings = classify_sessions(events, _config())
        assert findings.count_by_classification == {"quick": 1, "productive": 2}

    def test_avg_thrash_score(self):
        events = [
            _make_session(prompt_count=6, error_count=3, tools=[{"name": "Write", "files": []}] * 4, session_id=f"s{i}")
            for i in range(5)
        ]
        findings = classify_sessions(events, _config())
        assert findings.avg_thrash_score is not None
        assert findings.low_sample is False
        assert abs(findings.avg_thrash_score - 0.5) < 0.01

    def test_avg_thrash_score_low_sample_gated(self):
        """4 scoring sessions → avg None, low_sample True."""
        events = [
            _make_session(prompt_count=6, error_count=3, tools=[{"name": "Write", "files": []}] * 4, session_id=f"s{i}")
            for i in range(4)
        ]
        findings = classify_sessions(events, _config())
        assert findings.avg_thrash_score is None
        assert findings.low_sample is True

    def test_no_sessions(self):
        findings = classify_sessions([], _config())
        assert findings.outcomes == []
        assert findings.avg_thrash_score is None
        assert findings.low_sample is False

    def test_zero_scoring_sessions_not_low_sample(self):
        """0 thrash-scoring sessions → avg None, low_sample False (nothing to aggregate)."""
        events = [
            _make_session(prompt_count=2, error_count=0, tools=[], session_id="s1"),
        ]
        findings = classify_sessions(events, _config())
        assert findings.avg_thrash_score is None
        assert findings.low_sample is False


# ── Unit 13: Stuck Pattern Grouping ──


class TestGroupStuckPatterns:
    def test_groups_friction_by_project(self):
        """3 Friction sessions on project 'auth' → one pattern with episode_count=3."""
        events = [
            _make_session(prompt_count=6, error_count=5, project="auth",
                          tools=[{"name": "Bash", "files": []}] * 4,
                          files=["src/auth.py"], session_id=f"s{i}")
            for i in range(3)
        ]
        findings = classify_sessions(events, _config())
        stuck = group_stuck_patterns(findings.outcomes, events, _config())
        assert stuck.total_stuck_sessions == 3
        assert len(stuck.patterns) == 1
        assert stuck.patterns[0].project == "auth"
        assert stuck.patterns[0].episode_count == 3
        assert "Bash" in stuck.patterns[0].failing_tools

    def test_separate_projects(self):
        """Friction sessions across 2 projects → separate patterns."""
        e1 = _make_session(prompt_count=6, error_count=5, project="auth",
                           tools=[{"name": "Bash", "files": []}] * 4, session_id="s1")
        e2 = _make_session(prompt_count=6, error_count=5, project="frontend",
                           tools=[{"name": "Bash", "files": []}] * 4, session_id="s2")
        findings = classify_sessions([e1, e2], _config())
        stuck = group_stuck_patterns(findings.outcomes, [e1, e2], _config())
        assert stuck.total_stuck_sessions == 2
        assert len(stuck.patterns) == 2
        projects = {p.project for p in stuck.patterns}
        assert projects == {"auth", "frontend"}

    def test_no_files_gets_unknown(self):
        """Friction session with no claude_files → 'unknown' file cluster."""
        event = _make_session(prompt_count=6, error_count=5, project="api",
                              tools=[{"name": "Bash", "files": []}] * 4,
                              files=None, session_id="s1")
        findings = classify_sessions([event], _config())
        stuck = group_stuck_patterns(findings.outcomes, [event], _config())
        assert stuck.patterns[0].file_cluster == ["unknown"]

    def test_no_stuck_sessions(self):
        """All productive sessions → empty findings."""
        event = _make_session(prompt_count=10, error_count=0,
                              tools=[{"name": "Write", "files": []}] * 4, session_id="s1")
        findings = classify_sessions([event], _config())
        stuck = group_stuck_patterns(findings.outcomes, [event], _config())
        assert stuck.total_stuck_sessions == 0
        assert stuck.patterns == []

    def test_abandoned_included_in_grouping(self):
        """Abandoned sessions are included alongside Friction in stuck grouping."""
        friction = _make_session(prompt_count=6, error_count=5, project="auth",
                                 tools=[{"name": "Bash", "files": []}] * 4, session_id="s1")
        abandoned = _make_session(prompt_count=8, error_count=3, project="auth",
                                  tools=[{"name": "Read", "files": []}, {"name": "Bash", "files": []}],
                                  duration_ms=600_000, session_id="s2")
        findings = classify_sessions([friction, abandoned], _config())
        stuck = group_stuck_patterns(findings.outcomes, [friction, abandoned], _config())
        assert stuck.total_stuck_sessions == 2
        assert stuck.patterns[0].episode_count == 2

    def test_sorted_by_episode_count(self):
        """Patterns sorted by episode_count descending."""
        events = []
        # 3 friction sessions on "auth"
        for i in range(3):
            events.append(_make_session(prompt_count=6, error_count=5, project="auth",
                                        tools=[{"name": "Bash", "files": []}] * 4, session_id=f"auth-{i}"))
        # 1 friction session on "frontend"
        events.append(_make_session(prompt_count=6, error_count=5, project="frontend",
                                    tools=[{"name": "Bash", "files": []}] * 4, session_id="fe-1"))

        findings = classify_sessions(events, _config())
        stuck = group_stuck_patterns(findings.outcomes, events, _config())
        assert stuck.patterns[0].project == "auth"
        assert stuck.patterns[0].episode_count == 3
        assert stuck.patterns[1].project == "frontend"
        assert stuck.patterns[1].episode_count == 1

    def test_pattern_low_sample_thrash_none(self):
        """StuckPattern with 2 scoring sessions → avg_thrash_score is None."""
        events = [
            _make_session(prompt_count=6, error_count=5, project="auth",
                          tools=[{"name": "Bash", "files": []}] * 4,
                          files=["src/auth.py"], session_id=f"s{i}")
            for i in range(2)
        ]
        findings = classify_sessions(events, _config())
        stuck = group_stuck_patterns(findings.outcomes, events, _config())
        assert len(stuck.patterns) == 1
        assert stuck.patterns[0].avg_thrash_score is None

    def test_pattern_enough_samples_thrash_float(self):
        """StuckPattern with 6 scoring sessions → avg_thrash_score is a float."""
        events = [
            _make_session(prompt_count=6, error_count=5, project="auth",
                          tools=[{"name": "Bash", "files": []}] * 4,
                          files=["src/auth.py"], session_id=f"s{i}")
            for i in range(6)
        ]
        findings = classify_sessions(events, _config())
        stuck = group_stuck_patterns(findings.outcomes, events, _config())
        assert len(stuck.patterns) == 1
        assert stuck.patterns[0].avg_thrash_score is not None
        assert isinstance(stuck.patterns[0].avg_thrash_score, float)


class TestToolLevelStuckPatterns:
    def test_tool_grouping_across_projects(self):
        """Edit failing across 2 projects → one tool_level_pattern with episode_count=4."""
        events = []
        for proj, i in [("auth", 0), ("auth", 1), ("frontend", 0), ("frontend", 1)]:
            events.append(_make_session(
                prompt_count=6, error_count=5, project=proj,
                tools=[{"name": "Edit", "files": ["x.py"]}, {"name": "Bash", "files": []}],
                files=["x.py"], session_id=f"{proj}-s{i}",
            ))
        findings = classify_sessions(events, _config())
        stuck = group_stuck_patterns(findings.outcomes, events, _config())
        edit_patterns = [t for t in stuck.tool_level_patterns if t.tool_name == "Edit"]
        assert len(edit_patterns) == 1
        assert edit_patterns[0].episode_count == 4
        assert set(edit_patterns[0].projects) == {"auth", "frontend"}

    def test_single_session_tool_excluded(self):
        """Tool appearing in only 1 stuck session is skipped (covered by project pattern)."""
        events = [
            _make_session(prompt_count=6, error_count=5, project="auth",
                          tools=[{"name": "Write", "files": []}, {"name": "Bash", "files": []}] * 2,
                          session_id="s1"),
        ]
        findings = classify_sessions(events, _config())
        stuck = group_stuck_patterns(findings.outcomes, events, _config())
        assert stuck.tool_level_patterns == []

    def test_tool_low_sample_avg_thrash_none(self):
        """Tool pattern with <thrash_aggregate_min_n scoring sessions → avg_thrash_score is None."""
        events = [
            _make_session(prompt_count=6, error_count=5, project="auth",
                          tools=[{"name": "Bash", "files": []}] * 2,
                          session_id=f"s{i}")
            for i in range(3)  # 3 < default aggregate min 5
        ]
        findings = classify_sessions(events, _config())
        stuck = group_stuck_patterns(findings.outcomes, events, _config())
        bash_patterns = [t for t in stuck.tool_level_patterns if t.tool_name == "Bash"]
        assert bash_patterns[0].avg_thrash_score is None

    def test_tool_sorted_by_episode_count(self):
        """Tool patterns sorted descending by episode_count."""
        events = []
        # 3 sessions using Edit + Bash, 2 sessions using only Read
        for i in range(3):
            events.append(_make_session(
                prompt_count=6, error_count=5, project="p1",
                tools=[{"name": "Edit", "files": []}, {"name": "Bash", "files": []}] * 2,
                session_id=f"eb-{i}",
            ))
        for i in range(2):
            events.append(_make_session(
                prompt_count=6, error_count=5, project="p2",
                tools=[{"name": "Read", "files": []}, {"name": "Edit", "files": []}] * 2,
                session_id=f"r-{i}",
            ))
        findings = classify_sessions(events, _config())
        stuck = group_stuck_patterns(findings.outcomes, events, _config())
        tool_counts = [(t.tool_name, t.episode_count) for t in stuck.tool_level_patterns]
        # Edit appears in all 5 (3 eb + 2 r), Bash in 3, Read in 2
        counts_by_name = dict(tool_counts)
        assert counts_by_name["Edit"] == 5
        assert counts_by_name["Bash"] == 3
        # Should be sorted desc
        assert [c for _, c in tool_counts] == sorted([c for _, c in tool_counts], reverse=True)


class TestFileClusterStuckPatterns:
    def test_shared_directory_prefix_detected(self):
        """3 sessions touching agents/*.md → one cluster with path_fragment='agents/'."""
        events = []
        for i, fname in enumerate(["pm.md", "reviewer.md", "planner.md"]):
            events.append(_make_session(
                prompt_count=6, error_count=5, project="scheduler",
                tools=[{"name": "Edit", "files": []}, {"name": "Bash", "files": []}] * 2,
                files=[f"agents/{fname}", f"agents/helpers/shared.md"],
                session_id=f"s{i}",
            ))
        findings = classify_sessions(events, _config())
        stuck = group_stuck_patterns(findings.outcomes, events, _config())
        agents_clusters = [c for c in stuck.file_cluster_patterns if c.path_fragment == "agents/"]
        assert len(agents_clusters) == 1
        assert agents_clusters[0].episode_count == 3
        assert "scheduler" in agents_clusters[0].projects

    def test_singleton_cluster_excluded(self):
        """Single session with unique file prefix is excluded from file_cluster_patterns."""
        events = [
            _make_session(prompt_count=6, error_count=5, project="p",
                          tools=[{"name": "Edit", "files": []}] * 2,
                          files=["lonely/file.py"], session_id="s1"),
        ]
        findings = classify_sessions(events, _config())
        stuck = group_stuck_patterns(findings.outcomes, events, _config())
        assert stuck.file_cluster_patterns == []

    def test_no_files_skipped_not_unknown(self):
        """Stuck session with empty files list does not produce a file_cluster_pattern."""
        events = [
            _make_session(prompt_count=6, error_count=5, project="p",
                          tools=[{"name": "Edit", "files": []}] * 2,
                          files=None, session_id=f"s{i}")
            for i in range(3)
        ]
        findings = classify_sessions(events, _config())
        stuck = group_stuck_patterns(findings.outcomes, events, _config())
        # No file-cluster pattern (no file signal)
        assert stuck.file_cluster_patterns == []
        assert not any(c.path_fragment == "unknown" for c in stuck.file_cluster_patterns)

    def test_project_level_patterns_unchanged(self):
        """Existing patterns field is not mutated by the new groupings."""
        events = [
            _make_session(prompt_count=6, error_count=5, project="auth",
                          tools=[{"name": "Bash", "files": []}] * 4,
                          files=["src/auth.py"], session_id=f"s{i}")
            for i in range(3)
        ]
        findings = classify_sessions(events, _config())
        stuck = group_stuck_patterns(findings.outcomes, events, _config())
        # Project-level still works the same as before
        assert len(stuck.patterns) == 1
        assert stuck.patterns[0].project == "auth"
        assert stuck.patterns[0].episode_count == 3


class TestOpeningPromptsAndVagueFraming:
    """Stuck patterns carry opening prompts of their stuck sessions plus a
    vague-framing diagnostic. Hedged under low sample."""

    def test_opening_prompt_captured(self):
        event = _make_session(
            prompt_count=10, error_count=0,
            tools=[{"name": "Write", "files": []}] * 3,
            prompts=["implement the login refactor"] + ["test"] * 9,
        )
        outcome = classify_session_outcome(event, _config())
        assert outcome.opening_prompt == "implement the login refactor"

    def test_opening_prompt_truncated(self):
        from ambient.detect.coaching import OPENING_PROMPT_MAX_LENGTH
        event = _make_session(
            prompt_count=10,
            prompts=["x" * 500] + ["test"] * 9,
        )
        outcome = classify_session_outcome(event, _config())
        assert len(outcome.opening_prompt) == OPENING_PROMPT_MAX_LENGTH

    def test_stuck_pattern_carries_opening_prompts(self):
        """Stuck pattern's opening_prompts list matches its stuck sessions."""
        events = []
        session_prompts = [
            "fix this auth bug",
            "debug the login issue",
            "why is the token expired",
        ]
        for i, p in enumerate(session_prompts):
            events.append(_make_session(
                prompt_count=6, error_count=5, project="auth",
                tools=[{"name": "Bash", "files": []}] * 4,
                session_id=f"s{i}",
                prompts=[p] + ["follow up"] * 5,
            ))
        findings = classify_sessions(events, _config())
        stuck = group_stuck_patterns(findings.outcomes, events, _config())
        assert stuck.patterns[0].opening_prompts == session_prompts

    def test_vague_framing_fraction_emitted_above_floor(self):
        """>=thrash_aggregate_min_n stuck sessions → fraction emits."""
        events = []
        # 5 sessions, 3 vague, 2 specific
        vague = ["fix this bug", "debug it", "figure out why it fails"]
        specific = ["rename User to Account", "add a logout endpoint"]
        for i, p in enumerate(vague + specific):
            events.append(_make_session(
                prompt_count=6, error_count=5, project="auth",
                tools=[{"name": "Bash", "files": []}] * 4,
                session_id=f"s{i}",
                prompts=[p] + ["fu"] * 5,
            ))
        findings = classify_sessions(events, _config())
        stuck = group_stuck_patterns(findings.outcomes, events, _config())
        pattern = stuck.patterns[0]
        assert pattern.vague_framing_fraction is not None
        assert abs(pattern.vague_framing_fraction - 3 / 5) < 0.001

    def test_vague_framing_fraction_none_under_low_sample(self):
        """<thrash_aggregate_min_n stuck sessions → fraction is None."""
        events = [
            _make_session(
                prompt_count=6, error_count=5, project="auth",
                tools=[{"name": "Bash", "files": []}] * 4,
                session_id=f"s{i}",
                prompts=["figure out why it crashes"] + ["fu"] * 5,
            )
            for i in range(3)
        ]
        findings = classify_sessions(events, _config())
        stuck = group_stuck_patterns(findings.outcomes, events, _config())
        # Only 3 stuck < 5 min_n → None
        assert stuck.patterns[0].vague_framing_fraction is None
        # But opening_prompts are still populated regardless
        assert len(stuck.patterns[0].opening_prompts) == 3

    def test_vague_framing_fraction_zero_when_all_specific(self):
        """If every prompt is specific, fraction is 0.0, not None."""
        events = []
        specific = [f"rename variable_{i} to new_name_{i}" for i in range(5)]
        for i, p in enumerate(specific):
            events.append(_make_session(
                prompt_count=6, error_count=5, project="auth",
                tools=[{"name": "Bash", "files": []}] * 4,
                session_id=f"s{i}",
                prompts=[p] + ["fu"] * 5,
            ))
        findings = classify_sessions(events, _config())
        stuck = group_stuck_patterns(findings.outcomes, events, _config())
        assert stuck.patterns[0].vague_framing_fraction == 0.0

    def test_tool_and_cluster_patterns_also_carry_prompts(self):
        """ToolStuckPattern and FileClusterStuckPattern carry opening_prompts too."""
        events = [
            _make_session(
                prompt_count=6, error_count=5, project=proj,
                tools=[{"name": "Edit", "files": []}, {"name": "Bash", "files": []}],
                files=["src/auth.py"],
                session_id=f"{proj}-s{i}",
                prompts=[f"fix the {proj} bug {i}"] + ["fu"] * 5,
            )
            for proj in ("auth", "frontend")
            for i in range(2)
        ]
        findings = classify_sessions(events, _config())
        stuck = group_stuck_patterns(findings.outcomes, events, _config())
        edit_patterns = [t for t in stuck.tool_level_patterns if t.tool_name == "Edit"]
        assert edit_patterns
        assert len(edit_patterns[0].opening_prompts) == 4
        # File cluster: all sessions touch src/auth.py
        clusters = stuck.file_cluster_patterns
        assert clusters
        assert len(clusters[0].opening_prompts) == 4

    def test_custom_vague_framing_patterns(self):
        """Config-driven patterns: adding 'TODO:' matches it."""
        config = _config(coaching_vague_framing_patterns=[r"^TODO:"])
        events = [
            _make_session(
                prompt_count=6, error_count=5, project="auth",
                tools=[{"name": "Bash", "files": []}] * 4,
                session_id=f"s{i}",
                prompts=["TODO: something" if i < 3 else "specific task"] + ["fu"] * 5,
            )
            for i in range(5)
        ]
        findings = classify_sessions(events, config)
        stuck = group_stuck_patterns(findings.outcomes, events, config)
        assert abs(stuck.patterns[0].vague_framing_fraction - 3 / 5) < 0.001
