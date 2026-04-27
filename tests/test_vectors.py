"""Tests for the vector aggregation detector (v4 Phase 3)."""

import pytest

from ambient.detect.vectors import (
    Vector,
    VectorCategory,
    VectorFindings,
    StopReason,
    StopEvent,
    classify_vector,
)


# ---------- Unit 1: data model + classifier ----------

class TestStopEvent:
    def test_priority_ordering(self):
        # Tie-break: exit > pause > focus_change > enter > end_of_window
        ts = 1_000_000
        exit_e = StopEvent(ts_ms=ts, reason="exit")
        pause_e = StopEvent(ts_ms=ts, reason="pause")
        focus_e = StopEvent(ts_ms=ts, reason="focus_change")
        enter_e = StopEvent(ts_ms=ts, reason="enter")
        eow_e = StopEvent(ts_ms=ts, reason="end_of_window")
        priorities = [exit_e.priority, pause_e.priority, focus_e.priority,
                      enter_e.priority, eow_e.priority]
        # Strictly decreasing
        assert priorities == sorted(priorities, reverse=True)
        assert exit_e.priority > pause_e.priority > focus_e.priority > enter_e.priority


class TestClassifyVectorSlash:
    def test_review_slash_command(self):
        assert classify_vector(
            "/compound-engineering:ce-review",
            "enter",
            slash_command="/compound-engineering:ce-review",
        ) == "review"

    def test_planning_slash_command(self):
        assert classify_vector(
            "/compound-engineering:ce-plan",
            "enter",
            slash_command="/compound-engineering:ce-plan",
        ) == "planning"

    def test_execution_slash_command_via_taxonomy(self):
        assert classify_vector(
            "/ship",
            "enter",
            slash_command="/ship",
        ) == "execution"

    def test_meta_slash_command(self):
        assert classify_vector("/clear", "enter", slash_command="/clear") == "meta"

    def test_unknown_slash_demotes_to_freeform(self):
        # The slash taxonomy returns "other" for unknown commands; the vector
        # classifier demotes "other" to "freeform" so the report's
        # classification mix doesn't carry an unhelpful bucket.
        assert classify_vector(
            "/totally-made-up",
            "enter",
            slash_command="/totally-made-up",
        ) == "freeform"

    def test_user_override_via_overrides_arg(self):
        assert classify_vector(
            "/my-cmd",
            "enter",
            slash_command="/my-cmd",
            overrides={"/my-cmd": "review"},
        ) == "review"


class TestClassifyVectorPause:
    def test_pause_with_empty_text_is_thinking(self):
        assert classify_vector("", "pause") == "thinking"

    def test_pause_with_short_text_is_thinking(self):
        assert classify_vector("ok", "pause") == "thinking"

    def test_pause_with_long_text_falls_through_to_keyword(self):
        # 20+ chars of execution-keyword text → execution, not thinking.
        long_exec = "npm test --watch --coverage"
        assert classify_vector(long_exec, "pause") == "execution"


class TestClassifyVectorKeyword:
    @pytest.mark.parametrize("cmd,expected", [
        ("npm test", "execution"),
        ("pytest -x", "execution"),
        ("make build", "execution"),
        ("git status", "execution"),
        ("gh pr create", "execution"),
        ("docker compose up", "execution"),
        ("cargo test", "execution"),
        ("python script.py", "execution"),
    ])
    def test_execution_prefixes(self, cmd, expected):
        assert classify_vector(cmd, "enter") == expected

    def test_case_insensitive_keyword_match(self):
        # NPM (uppercase) still matches the keyword set.
        assert classify_vector("NPM TEST", "enter") == "execution"

    def test_non_execution_text_is_freeform(self):
        assert classify_vector("how does this work", "enter") == "freeform"

    def test_empty_text_is_freeform(self):
        assert classify_vector("", "enter") == "freeform"

    def test_focus_change_with_empty_text_is_freeform(self):
        # focus_change alone doesn't pin classification.
        assert classify_vector("", "focus_change") == "freeform"

    def test_end_of_window_with_keyword_still_classifies(self):
        # end_of_window doesn't change classification logic; the last text wins.
        assert classify_vector("npm test", "end_of_window") == "execution"


class TestVectorDataclass:
    def test_basic_construction(self):
        v = Vector(
            ts_start=1_000_000,
            ts_end=1_060_000,
            duration_ms=60_000,
            stop_reason="enter",
            last_command_or_prompt="npm test",
            project="ambient-ai",
        )
        assert v.duration_ms == 60_000
        assert v.classification == "freeform"  # default
        assert v.app_focus is None
        assert v.tmux_pane_focus is None
        assert v.pause_duration_ms is None

    def test_with_pause_duration(self):
        v = Vector(
            ts_start=1_000_000,
            ts_end=1_180_000,
            duration_ms=180_000,
            stop_reason="pause",
            last_command_or_prompt="",
            project="ambient-ai",
            pause_duration_ms=180_000,
            classification="thinking",
        )
        assert v.pause_duration_ms == 180_000
        assert v.classification == "thinking"


class TestVectorFindingsShape:
    def test_default_is_empty(self):
        f = VectorFindings()
        assert f.vectors == []
        assert f.count_by_stop_reason == {}
        assert f.total_duration_by_stop_reason == {}
        assert f.count_by_project == {}
        assert f.count_by_classification == {}


# ---------- Unit 2: detect_vectors ----------

from datetime import datetime, timezone
from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.focus_events import FocusEvent
from ambient.detect.pauses import PauseClassification, PauseFindings
from ambient.detect.vectors import detect_vectors


def _ms(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)


def _command_event(ts_ms: int, duration_ms: int, command: str, cwd: str = "/repo/proj") -> Event:
    return Event(
        ts_start=ts_ms, ts_end=ts_ms + duration_ms, duration_ms=duration_ms,
        command=command, exit_code=0, cwd=cwd,
        tmux_pane=None, gap_ms=None, type="command",
    )


def _claude_session_event(
    ts_ms: int, duration_ms: int, prompts: list[str], project_path: str = "/repo/proj",
) -> Event:
    return Event(
        ts_start=ts_ms, ts_end=ts_ms + duration_ms, duration_ms=duration_ms,
        command="claude: ...", exit_code=0, cwd=project_path,
        tmux_pane=None, gap_ms=None, type="claude_session",
        claude_prompts=prompts, claude_project=project_path,
    )


def _focus(ts_ms: int, bundle_id: str = "com.apple.Terminal", source: str = "nsworkspace") -> FocusEvent:
    return FocusEvent(
        ts=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
        source=source, event="app_activated",
        bundle_id=bundle_id, app_name=bundle_id.split(".")[-1], pid=1,
    )


def _pause(end_ts_ms: int, gap_ms: int, label: str = "evaluating", preceding: str = "") -> PauseClassification:
    """Build a PauseClassification matching pauses.classify() semantics.

    `ts_start` in PauseClassification is set to the NEXT event's ts_start
    (see pauses.py:179) — i.e. the gap/pause-end timestamp. The detector
    uses this directly as the stop-event timestamp; the helper takes
    end_ts_ms to make the test intent explicit.
    """
    return PauseClassification(
        gap_ms=gap_ms, label=label,
        probabilities={label: 1.0},
        preceding_command=preceding, following_command="",
        ts_start=end_ts_ms,
    )


def _pause_findings(*classifications) -> PauseFindings:
    return PauseFindings(available=True, classifications=list(classifications))


def _config(tmp_path) -> Config:
    cfg = Config(base_dir=tmp_path / ".ambient")
    cfg.ensure_dirs()
    return cfg


WINDOW_START = _ms("2026-04-27T10:00:00Z")
WINDOW_END = _ms("2026-04-27T20:00:00Z")


class TestDetectVectorsHappyPath:
    def test_three_command_events_produce_three_vectors(self, tmp_path):
        cfg = _config(tmp_path)
        events = [
            _command_event(WINDOW_START + 60_000, 1_000, "ls"),
            _command_event(WINDOW_START + 600_000, 1_000, "git status"),
            _command_event(WINDOW_START + 1_800_000, 1_000, "npm test"),
        ]
        result = detect_vectors(events, [], None, WINDOW_START, WINDOW_END, cfg)
        # 3 enter stops + 1 end_of_window = 4 stops → 4 vectors (last is the
        # tail from the third command's ts_end up to window_end).
        assert len(result.vectors) == 4
        reasons = [v.stop_reason for v in result.vectors]
        assert reasons[:3] == ["enter", "enter", "enter"]
        assert reasons[-1] == "end_of_window"

    def test_pause_terminated_vector_carries_pause_duration(self, tmp_path):
        cfg = _config(tmp_path)
        events = [_command_event(WINDOW_START + 60_000, 1_000, "ls", cwd="/repo/proj")]
        # Pause classified "stuck" at WINDOW_START + 600_000, gap 120_000ms.
        pauses = _pause_findings(
            _pause(WINDOW_START + 600_000, 120_000, label="stuck", preceding="ls"),
        )
        result = detect_vectors(events, [], pauses, WINDOW_START, WINDOW_END, cfg)
        # Vector terminated by the pause has its pause_duration_ms set.
        pause_vectors = [v for v in result.vectors if v.stop_reason == "pause"]
        assert len(pause_vectors) == 1
        assert pause_vectors[0].pause_duration_ms == 120_000

    def test_classification_uses_slash_taxonomy_when_marker_present(self, tmp_path):
        cfg = _config(tmp_path)
        body = (
            "<command-message>compound-engineering:ce-review</command-message>\n"
            "<command-name>/compound-engineering:ce-review</command-name>\n"
            "<command-args></command-args>"
        )
        events = [
            _claude_session_event(WINDOW_START + 60_000, 1_000, prompts=[body]),
        ]
        result = detect_vectors(events, [], None, WINDOW_START, WINDOW_END, cfg)
        review_vectors = [v for v in result.vectors if v.classification == "review"]
        assert len(review_vectors) == 1


class TestDetectVectorsEdgeCases:
    def test_empty_inputs(self, tmp_path):
        cfg = _config(tmp_path)
        result = detect_vectors([], [], None, WINDOW_START, WINDOW_END, cfg)
        # Only end_of_window stop fires; vectors_list contains the single
        # window-spanning vector.
        assert len(result.vectors) == 1
        assert result.vectors[0].stop_reason == "end_of_window"
        assert result.vectors[0].duration_ms == WINDOW_END - WINDOW_START

    def test_window_end_le_start_returns_empty(self, tmp_path):
        cfg = _config(tmp_path)
        # Inverted window
        result = detect_vectors([], [], None, WINDOW_END, WINDOW_START, cfg)
        assert result.vectors == []

    def test_pause_below_threshold_does_not_emit_stop(self, tmp_path):
        cfg = _config(tmp_path)
        # Pause label "routine" is below default threshold "evaluating"
        pauses = _pause_findings(
            _pause(WINDOW_START + 600_000, 60_000, label="routine"),
        )
        result = detect_vectors([], [], pauses, WINDOW_START, WINDOW_END, cfg)
        assert "pause" not in result.count_by_stop_reason

    def test_focus_change_debounce_collapses_rapid_events(self, tmp_path):
        cfg = _config(tmp_path)
        cfg.vector_focus_debounce_ms = 2000  # 2s
        # 5 focus events within 1.5s — should collapse to 1 stop.
        focus = [
            _focus(WINDOW_START + 60_000),
            _focus(WINDOW_START + 60_300, "com.apple.Safari"),
            _focus(WINDOW_START + 60_600, "com.apple.Terminal"),
            _focus(WINDOW_START + 60_900, "com.apple.Safari"),
            _focus(WINDOW_START + 61_200, "com.apple.Terminal"),
        ]
        result = detect_vectors([], focus, None, WINDOW_START, WINDOW_END, cfg)
        focus_count = result.count_by_stop_reason.get("focus_change", 0)
        assert focus_count == 1, f"expected 1 debounced focus stop, got {focus_count}"

    def test_stop_priority_tiebreak_at_same_ms(self, tmp_path):
        cfg = _config(tmp_path)
        # Both a command (enter) and a pause (pause) at the same ts_end.
        # Pause should win (higher priority).
        ts = WINDOW_START + 600_000
        events = [_command_event(ts - 1_000, 1_000, "ls")]  # ts_end = ts
        # Pause-end ts == ts (collision with the enter stop at ts).
        pauses = _pause_findings(_pause(ts, 30_000, label="stuck"))
        result = detect_vectors(events, [], pauses, WINDOW_START, WINDOW_END, cfg)
        # The vector ending at `ts` should be a pause vector, not enter.
        v_at_ts = next((v for v in result.vectors if v.ts_end == ts), None)
        assert v_at_ts is not None
        assert v_at_ts.stop_reason == "pause"

    def test_end_of_window_vector_closes_at_window_end(self, tmp_path):
        cfg = _config(tmp_path)
        events = [_command_event(WINDOW_START + 60_000, 1_000, "ls")]
        result = detect_vectors(events, [], None, WINDOW_START, WINDOW_END, cfg)
        last = result.vectors[-1]
        assert last.stop_reason == "end_of_window"
        assert last.ts_end == WINDOW_END

    def test_deterministic_repeat_runs(self, tmp_path):
        cfg = _config(tmp_path)
        events = [
            _command_event(WINDOW_START + 60_000, 1_000, "ls"),
            _command_event(WINDOW_START + 600_000, 1_000, "git status"),
        ]
        focus = [_focus(WINDOW_START + 300_000)]
        pauses = _pause_findings(
            _pause(WINDOW_START + 1_200_000, 60_000, label="evaluating"),
        )
        r1 = detect_vectors(events, focus, pauses, WINDOW_START, WINDOW_END, cfg)
        r2 = detect_vectors(events, focus, pauses, WINDOW_START, WINDOW_END, cfg)
        # Same inputs → identical vector list (no clock-dependent ordering).
        assert [v.ts_end for v in r1.vectors] == [v.ts_end for v in r2.vectors]
        assert [v.stop_reason for v in r1.vectors] == [v.stop_reason for v in r2.vectors]


class TestDetectVectorsAggregates:
    def test_count_by_stop_reason_matches_per_vector_breakdown(self, tmp_path):
        cfg = _config(tmp_path)
        events = [
            _command_event(WINDOW_START + 60_000, 1_000, "ls"),
            _command_event(WINDOW_START + 600_000, 1_000, "npm test"),
        ]
        result = detect_vectors(events, [], None, WINDOW_START, WINDOW_END, cfg)
        # Reconstruct counts from per-vector list.
        from collections import Counter
        per_vector = Counter(v.stop_reason for v in result.vectors)
        assert dict(result.count_by_stop_reason) == dict(per_vector)

    def test_total_duration_matches_window_span(self, tmp_path):
        cfg = _config(tmp_path)
        events = [
            _command_event(WINDOW_START + 60_000, 1_000, "ls"),
            _command_event(WINDOW_START + 600_000, 1_000, "npm test"),
        ]
        result = detect_vectors(events, [], None, WINDOW_START, WINDOW_END, cfg)
        total = sum(v.duration_ms for v in result.vectors)
        # All vectors together cover [window_start, window_end] exactly.
        assert total == WINDOW_END - WINDOW_START


# ---------- Unit 3: aggregation surfaces ----------

from ambient.detect.vectors import (
    stop_reason_summary,
    top_vectors_per_project,
    vectors_by_day,
)


def _v(ts_start: int, duration_ms: int, project: str = "p", stop: str = "enter") -> Vector:
    return Vector(
        ts_start=ts_start,
        ts_end=ts_start + duration_ms,
        duration_ms=duration_ms,
        stop_reason=stop,  # type: ignore[arg-type]
        last_command_or_prompt="",
        project=project,
    )


class TestTopVectorsPerProject:
    def test_returns_top_n_by_duration(self):
        f = VectorFindings(vectors=[
            _v(0, 100, "a"),
            _v(0, 500, "a"),
            _v(0, 200, "a"),
            _v(0, 1000, "b"),
        ])
        result = top_vectors_per_project(f, n=2)
        assert [v.duration_ms for v in result["a"]] == [500, 200]
        assert [v.duration_ms for v in result["b"]] == [1000]

    def test_n_zero_returns_empty(self):
        f = VectorFindings(vectors=[_v(0, 100, "a")])
        assert top_vectors_per_project(f, n=0) == {}

    def test_empty_findings_returns_empty(self):
        assert top_vectors_per_project(VectorFindings(), n=3) == {}


class TestVectorsByDay:
    def test_buckets_by_local_date(self):
        ts1 = _ms("2026-04-27T10:00:00Z")
        ts2 = _ms("2026-04-28T10:00:00Z")
        f = VectorFindings(vectors=[_v(ts1, 1000), _v(ts2, 1000), _v(ts2, 2000)])
        result = vectors_by_day(f)
        assert len(result) == 2
        # Two vectors on the second day.
        days = sorted(result.keys())
        assert len(result[days[1]]) == 2


class TestPhase3ReviewRegressions:
    """Regression tests for findings from the Phase 3 review (commit 9fba7a5)."""

    def test_pause_stop_uses_ts_start_directly_not_added_to_gap(self, tmp_path):
        # P0: PauseClassification.ts_start IS the pause-end (matches the
        # next event's ts_start; see pauses.py:179). Earlier code added
        # gap_ms again, double-counting. Verify the stop fires at ts_start.
        cfg = _config(tmp_path)
        pause_end_ts = WINDOW_START + 600_000
        pauses = _pause_findings(_pause(pause_end_ts, 60_000, label="stuck"))
        result = detect_vectors([], [], pauses, WINDOW_START, WINDOW_END, cfg)
        pause_vectors = [v for v in result.vectors if v.stop_reason == "pause"]
        assert len(pause_vectors) == 1
        # The vector ending at the pause must end at pause_end_ts, NOT at
        # pause_end_ts + 60_000 (which would be past the actual pause end).
        assert pause_vectors[0].ts_end == pause_end_ts

    def test_unknown_pause_label_logs_warning(self, tmp_path, caplog):
        # C4: unrecognized labels are silently dropped. Now they log once
        # per detector run so the user knows pauses dropped due to label drift.
        cfg = _config(tmp_path)
        pauses = _pause_findings(
            _pause(WINDOW_START + 600_000, 60_000, label="long-tail-novel-label"),
        )
        with caplog.at_level("WARNING"):
            detect_vectors([], [], pauses, WINDOW_START, WINDOW_END, cfg)
        assert any("long-tail-novel-label" in rec.message for rec in caplog.records)

    def test_last_project_carries_to_pause_terminated_empty_vector(self, tmp_path):
        # C3: a pause-terminated vector with no events_in_vector previously
        # collapsed to project='unknown'. Now it carries forward the prior
        # vector's project.
        cfg = _config(tmp_path)
        events = [_command_event(WINDOW_START + 60_000, 1_000, "ls", cwd="/repo/proj-a")]
        # Pause ends well after the command's ts_end → no events fall inside
        # the pause vector.
        pauses = _pause_findings(
            _pause(WINDOW_START + 600_000, 300_000, label="stuck", preceding="ls"),
        )
        result = detect_vectors(events, [], pauses, WINDOW_START, WINDOW_END, cfg)
        pause_v = next(v for v in result.vectors if v.stop_reason == "pause")
        assert pause_v.project == "proj-a"  # not "unknown"

    def test_focus_events_sorted_before_debounce(self, tmp_path):
        # C5: out-of-order focus events were dropped erroneously. Sort then
        # debounce so a legitimate later-arriving-but-earlier-ts event still
        # fires its stop.
        cfg = _config(tmp_path)
        cfg.vector_focus_debounce_ms = 1000
        # Two events 5 seconds apart, but presented in REVERSE chronological order.
        focus = [
            _focus(WINDOW_START + 70_000, "com.apple.Terminal"),
            _focus(WINDOW_START + 60_000, "com.apple.Terminal"),  # earlier ts, listed second
        ]
        result = detect_vectors([], focus, None, WINDOW_START, WINDOW_END, cfg)
        # Both events fire (5s apart > 1s debounce), regardless of input order.
        focus_count = result.count_by_stop_reason.get("focus_change", 0)
        assert focus_count == 2

    def test_vector_category_no_other_member(self):
        # K3: VectorCategory previously had "other" but classify_vector
        # demoted "other" to "freeform", leaving "other" unreachable. Removed.
        from typing import get_args
        from ambient.detect.vectors import VectorCategory
        assert "other" not in get_args(VectorCategory)


class TestPhase3InsightsSystemNumbering:
    def test_section_numbers_are_unique_and_monotonic(self):
        # C2 / M-02 / K9: Phase 3 introduced VECTORS as section 4 but Top
        # Finding was already 4. Numbering must be unique and monotonic.
        from ambient.present.insights import INSIGHTS_SYSTEM
        import re

        section_lines = re.findall(r"^(\d+)\. \*\*", INSIGHTS_SYSTEM, flags=re.MULTILINE)
        nums = [int(n) for n in section_lines]
        assert nums == sorted(set(nums)), f"section numbers not unique/monotonic: {nums}"
        assert nums == list(range(1, len(nums) + 1)), f"section numbers not contiguous from 1: {nums}"


class TestStopReasonSummary:
    def test_sorts_by_total_duration_desc(self):
        f = VectorFindings(
            vectors=[],
            count_by_stop_reason={"enter": 10, "pause": 2},
            total_duration_by_stop_reason={"enter": 600_000, "pause": 1_800_000},
        )
        result = stop_reason_summary(f)
        # pause has 1.8M ms total > enter's 600k → pause comes first.
        assert result[0][0] == "pause"
        assert result[0][1] == 2
        assert result[1][0] == "enter"


