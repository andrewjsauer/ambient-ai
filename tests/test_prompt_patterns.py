import pytest

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.prompt_patterns import (
    PromptPatternFindings,
    _is_noise,
    _normalize,
    detect_prompt_patterns,
)


def _make_claude_event(
    prompts: list[str],
    session_id: str = "sess-1",
    project: str | None = "/home/user/myproject",
    ts_start: int = 1000,
    ts_end: int | None = None,
) -> Event:
    return Event(
        ts_start=ts_start,
        ts_end=ts_end if ts_end is not None else ts_start + 1000,
        duration_ms=1000,
        command="claude",
        exit_code=0,
        cwd="/tmp",
        tmux_pane=None,
        gap_ms=None,
        type="claude_session",
        claude_session_id=session_id,
        claude_prompts=prompts,
        claude_project=project,
    )


@pytest.fixture
def config():
    return Config(prompt_pattern_min_frequency=3)


# --- Normalization tests ---


def test_normalize_lowercase():
    assert _normalize("Fix The Bug") == "fix the bug"


def test_normalize_strips_file_paths():
    result = _normalize("fix the test in /src/foo.py")
    assert "/src/foo.py" not in result
    assert "fix the test in" in result


def test_normalize_strips_slash_command():
    assert _normalize("/commit push changes") == "push changes"
    assert _normalize("/review check the code") == "check the code"


def test_normalize_collapses_whitespace():
    assert _normalize("fix   the    bug") == "fix the bug"


def test_normalize_truncates_to_200():
    long_prompt = "a " * 200
    assert len(_normalize(long_prompt)) <= 200


# --- Noise filter tests ---


def test_noise_short_prompts():
    assert _is_noise("y")
    assert _is_noise("ok")
    assert _is_noise("")


def test_noise_known_patterns():
    assert _is_noise("clear")
    assert _is_noise("yes")
    assert _is_noise("exit")


def test_not_noise_real_prompt():
    assert not _is_noise("fix the failing test")
    assert not _is_noise("commit and push")


# --- Happy path: repeated prompt across sessions ---


def test_repeated_prompt_across_sessions(config):
    events = [
        _make_claude_event(["commit and push"], session_id=f"sess-{i}")
        for i in range(5)
    ]
    result = detect_prompt_patterns(events, config)

    assert result.total_prompts == 5
    single_patterns = [p for p in result.patterns if " -> " not in p.normalized_prompt]
    assert any(p.normalized_prompt == "commit and push" and p.count == 5 for p in single_patterns)


# --- Happy path: file path normalization merges prompts ---


def test_file_path_normalization_merges(config):
    events = [
        _make_claude_event(["fix the test in /src/foo.py"], session_id="s1"),
        _make_claude_event(["fix the test in /src/bar.py"], session_id="s2"),
        _make_claude_event(["fix the test in /src/baz.py"], session_id="s3"),
    ]
    result = detect_prompt_patterns(events, config)

    single_patterns = [p for p in result.patterns if " -> " not in p.normalized_prompt]
    assert any(p.normalized_prompt == "fix the test in" and p.count == 3 for p in single_patterns)


# --- Edge case: slash commands strip prefix ---


def test_slash_commands_strip_prefix(config):
    events = [
        _make_claude_event(["/commit push to main"], session_id=f"s{i}")
        for i in range(4)
    ]
    result = detect_prompt_patterns(events, config)

    single_patterns = [p for p in result.patterns if " -> " not in p.normalized_prompt]
    assert any(p.normalized_prompt == "push to main" and p.count == 4 for p in single_patterns)


# --- Edge case: short/noise prompts filtered ---


def test_noise_prompts_filtered(config):
    events = [
        _make_claude_event(["yes", "ok", "y", "clear"], session_id=f"s{i}")
        for i in range(5)
    ]
    result = detect_prompt_patterns(events, config)

    assert result.total_prompts == 0
    assert result.patterns == []


# --- Edge case: below min_frequency not reported ---


def test_below_min_frequency_not_reported(config):
    events = [
        _make_claude_event(["unique prompt alpha"], session_id="s1"),
        _make_claude_event(["unique prompt beta"], session_id="s2"),
    ]
    result = detect_prompt_patterns(events, config)

    assert result.total_prompts == 2
    assert result.patterns == []


# --- Edge case: empty prompt list ---


def test_empty_events(config):
    result = detect_prompt_patterns([], config)

    assert result.total_prompts == 0
    assert result.patterns == []


def test_no_claude_sessions(config):
    events = [
        Event(
            ts_start=1000, ts_end=2000, duration_ms=1000,
            command="ls", exit_code=0, cwd="/tmp", tmux_pane=None, gap_ms=None,
            type="command",
        )
    ]
    result = detect_prompt_patterns(events, config)

    assert result.total_prompts == 0
    assert result.patterns == []


# --- Edge case: N-gram within single session ---


def test_ngram_within_single_session(config):
    # Same sequence of 2 prompts repeated 3 times in one session
    prompts = ["run tests", "fix errors"] * 3
    events = [_make_claude_event(prompts, session_id="s1")]
    result = detect_prompt_patterns(events, config)

    ngram_patterns = [p for p in result.patterns if " -> " in p.normalized_prompt]
    assert any(
        p.normalized_prompt == "run tests -> fix errors" and p.count >= 3
        for p in ngram_patterns
    )


# --- Within-session n-grams never emit cross-session adjacency ---


def test_no_within_session_ngrams_across_sessions(config):
    """Within-session scope must not contain n-grams spanning multiple sessions."""
    events = [
        _make_claude_event(["run tests"], session_id="s1"),
        _make_claude_event(["fix errors"], session_id="s2"),
        _make_claude_event(["run tests"], session_id="s3"),
        _make_claude_event(["fix errors"], session_id="s4"),
        _make_claude_event(["run tests"], session_id="s5"),
        _make_claude_event(["fix errors"], session_id="s6"),
    ]
    result = detect_prompt_patterns(events, config)

    within = [p for p in result.patterns if p.scope == "within_session"]
    assert not any(
        "run tests -> fix errors" in p.normalized_prompt for p in within
    )


# --- Projects tracked ---


def test_projects_tracked(config):
    events = [
        _make_claude_event(["commit and push"], session_id="s1", project="proj-a"),
        _make_claude_event(["commit and push"], session_id="s2", project="proj-b"),
        _make_claude_event(["commit and push"], session_id="s3", project="proj-a"),
    ]
    result = detect_prompt_patterns(events, config)

    pattern = next(p for p in result.patterns if p.normalized_prompt == "commit and push")
    assert sorted(pattern.projects) == ["proj-a", "proj-b"]


# --- Patterns sorted by count descending ---


def test_patterns_sorted_by_count(config):
    events = [
        _make_claude_event(["commit and push"], session_id=f"s{i}")
        for i in range(5)
    ] + [
        _make_claude_event(["run the tests"], session_id=f"t{i}")
        for i in range(3)
    ]
    result = detect_prompt_patterns(events, config)

    single_patterns = [p for p in result.patterns if " -> " not in p.normalized_prompt]
    assert len(single_patterns) == 2
    assert single_patterns[0].count >= single_patterns[1].count


# --- Config fields respected ---


def test_custom_min_frequency():
    config = Config(prompt_pattern_min_frequency=5)
    events = [
        _make_claude_event(["commit and push"], session_id=f"s{i}")
        for i in range(4)
    ]
    result = detect_prompt_patterns(events, config)

    # 4 occurrences < min_frequency of 5
    assert result.patterns == []


def test_events_with_none_prompts(config):
    event = Event(
        ts_start=1000, ts_end=2000, duration_ms=1000,
        command="claude", exit_code=0, cwd="/tmp", tmux_pane=None, gap_ms=None,
        type="claude_session",
        claude_session_id="s1",
        claude_prompts=None,
    )
    result = detect_prompt_patterns([event], config)

    assert result.total_prompts == 0
    assert result.patterns == []


# --- Cross-session n-grams ---


class TestCrossSessionNgrams:
    def test_cross_session_ngram_within_time_window(self, config):
        """Same prompt sequence repeated across N sessions within 24h → cross-session pattern emitted."""
        events = []
        for i in range(4):
            # Two adjacent sessions 1h apart, each with one prompt
            events.append(_make_claude_event(
                ["plan the feature"],
                session_id=f"plan-{i}",
                ts_start=1_000_000 + i * 2 * 3_600_000,  # 2h apart
            ))
            events.append(_make_claude_event(
                ["implement the feature"],
                session_id=f"impl-{i}",
                ts_start=1_000_000 + i * 2 * 3_600_000 + 3_600_000,
            ))
        result = detect_prompt_patterns(events, config)
        cross = [p for p in result.patterns if p.scope == "cross_session"]
        assert any(
            p.normalized_prompt == "plan the feature -> implement the feature" and p.count >= 3
            for p in cross
        )

    def test_max_gap_breaks_cross_session_ngram(self):
        """Sessions >24h apart do not produce cross-session n-grams."""
        config = Config(
            prompt_pattern_min_frequency=2,
            prompt_pattern_cross_session_max_gap_ms=3_600_000,  # 1h
        )
        # Two sessions with 2h gap > 1h max → sentinel breaks the ngram
        events = [
            _make_claude_event(["a prompt"], session_id="s1", ts_start=1_000_000, ts_end=1_010_000),
            _make_claude_event(["b prompt"], session_id="s2",
                               ts_start=1_010_000 + 2 * 3_600_000,
                               ts_end=1_020_000 + 2 * 3_600_000),
            _make_claude_event(["a prompt"], session_id="s3", ts_start=50_000_000, ts_end=50_010_000),
            _make_claude_event(["b prompt"], session_id="s4",
                               ts_start=50_010_000 + 2 * 3_600_000,
                               ts_end=50_020_000 + 2 * 3_600_000),
        ]
        result = detect_prompt_patterns(events, config)
        cross = [p for p in result.patterns if p.scope == "cross_session"]
        assert not any(
            "a prompt -> b prompt" in p.normalized_prompt for p in cross
        )

    def test_single_session_repetition_not_counted_as_cross(self, config):
        """When an n-gram is entirely within one session, it does not appear in cross-session scope."""
        prompts = ["step a", "step b"] * 3
        events = [_make_claude_event(prompts, session_id="s1", ts_start=1_000_000)]
        result = detect_prompt_patterns(events, config)
        cross = [p for p in result.patterns if p.scope == "cross_session"]
        assert not any(
            "step a -> step b" in p.normalized_prompt for p in cross
        )
        # But within-session pattern must be present
        within = [p for p in result.patterns if p.scope == "within_session"]
        assert any(
            p.normalized_prompt == "step a -> step b" and p.count >= 3
            for p in within
        )

    def test_cross_session_requires_project(self, config):
        """Sessions without a project are excluded from cross-session pass."""
        events = [
            _make_claude_event(["plan x"], session_id=f"s{i}", project=None,
                               ts_start=1_000_000 + i * 3_600_000)
            for i in range(3)
        ] + [
            _make_claude_event(["do x"], session_id=f"d{i}", project=None,
                               ts_start=1_000_000 + i * 3_600_000 + 1_800_000)
            for i in range(3)
        ]
        result = detect_prompt_patterns(events, config)
        cross = [p for p in result.patterns if p.scope == "cross_session"]
        assert cross == []

    def test_cross_session_scoped_per_project(self, config):
        """Cross-session n-grams do not bridge across projects."""
        events = []
        # Project A: plan -> implement repeated
        for i in range(3):
            events.append(_make_claude_event(
                ["plan"], session_id=f"a-plan-{i}", project="proj-a",
                ts_start=1_000_000 + i * 7_200_000,
            ))
            events.append(_make_claude_event(
                ["implement"], session_id=f"a-impl-{i}", project="proj-a",
                ts_start=1_000_000 + i * 7_200_000 + 3_600_000,
            ))
        # Project B: interleaved but different project — should not link with A
        events.append(_make_claude_event(
            ["implement"], session_id="b-impl-1", project="proj-b",
            ts_start=1_000_000 + 1_800_000,
        ))

        result = detect_prompt_patterns(events, config)
        cross = [p for p in result.patterns if p.scope == "cross_session"]
        # The "plan -> implement" pattern should be attributed only to proj-a
        plan_impl = [p for p in cross if p.normalized_prompt == "plan -> implement"]
        assert plan_impl
        assert plan_impl[0].projects == ["proj-a"]

    def test_within_and_cross_coexist_on_same_gram(self, config):
        """A sequence that repeats both within and across sessions emits both scopes."""
        # Session 1 has "plan it -> ship it" 3 times within itself
        s1 = _make_claude_event(["plan it", "ship it"] * 3, session_id="s1", ts_start=1_000_000)
        # Subsequent sessions contribute cross-session occurrences
        s2 = _make_claude_event(["plan it"], session_id="s2", ts_start=2_000_000)
        s3 = _make_claude_event(["ship it"], session_id="s3", ts_start=3_000_000)
        s4 = _make_claude_event(["plan it"], session_id="s4", ts_start=4_000_000)
        s5 = _make_claude_event(["ship it"], session_id="s5", ts_start=5_000_000)
        s6 = _make_claude_event(["plan it"], session_id="s6", ts_start=6_000_000)
        s7 = _make_claude_event(["ship it"], session_id="s7", ts_start=7_000_000)
        result = detect_prompt_patterns([s1, s2, s3, s4, s5, s6, s7], config)

        within = [p for p in result.patterns
                  if p.scope == "within_session" and p.normalized_prompt == "plan it -> ship it"]
        cross = [p for p in result.patterns
                 if p.scope == "cross_session" and p.normalized_prompt == "plan it -> ship it"]
        assert within, "expected within-session plan it -> ship it pattern"
        assert cross, "expected cross-session plan it -> ship it pattern"
