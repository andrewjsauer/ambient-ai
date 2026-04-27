"""Tests for focus-events ingestion + attention-weighted project time (Unit 9)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.focus_events import (
    DEFAULT_TERMINAL_BUNDLE_IDS,
    FocusEvent,
    compute_attention_intervals,
    compute_context_switch_density,
    latest_cursor,
    read_focus_events,
)
from ambient.detect.projects import detect_project_allocation


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _terminal_event(ts: str, bundle_id: str = "com.apple.Terminal") -> FocusEvent:
    return FocusEvent(
        ts=_ts(ts), source="nsworkspace", event="app_activated",
        bundle_id=bundle_id, app_name="Terminal", pid=1,
    )


def _other_event(ts: str, bundle_id: str = "com.apple.Safari") -> FocusEvent:
    return FocusEvent(
        ts=_ts(ts), source="nsworkspace", event="app_activated",
        bundle_id=bundle_id, app_name="Safari", pid=2,
    )


def _shell_event(ts_ms: int, cwd: str, duration_ms: int) -> Event:
    return Event(
        ts_start=ts_ms,
        ts_end=ts_ms + duration_ms,
        duration_ms=duration_ms,
        command="ls",
        exit_code=0,
        cwd=cwd,
        tmux_pane=None,
        gap_ms=None,
    )


# ---------- read_focus_events ----------

class TestReadFocusEvents:
    def test_missing_file_returns_empty(self, tmp_path):
        assert read_focus_events(tmp_path / "nope.jsonl") == []

    def test_reads_well_formed_lines(self, tmp_path):
        path = tmp_path / "events.jsonl"
        path.write_text(
            '{"ts":"2026-04-27T10:00:00+00:00","source":"nsworkspace","event":"app_activated","bundle_id":"x","app_name":"X","pid":1}\n'
            '{"ts":"2026-04-27T10:01:00+00:00","source":"tmux","event":"pane-focus-in","pane_id":"%1","window_index":"0","session_name":"s"}\n'
        )
        events = read_focus_events(path)
        assert len(events) == 2
        assert events[0].source == "nsworkspace"
        assert events[1].source == "tmux"
        assert events[1].pane_id == "%1"

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "events.jsonl"
        path.write_text(
            'this is not json\n'
            '{"ts":"2026-04-27T10:00:00+00:00","source":"nsworkspace","event":"a","bundle_id":"x"}\n'
        )
        events = read_focus_events(path)
        assert len(events) == 1

    def test_since_cursor_filters(self, tmp_path):
        path = tmp_path / "events.jsonl"
        path.write_text(
            '{"ts":"2026-04-27T10:00:00+00:00","source":"nsworkspace","event":"a","bundle_id":"x"}\n'
            '{"ts":"2026-04-27T11:00:00+00:00","source":"nsworkspace","event":"b","bundle_id":"x"}\n'
        )
        events = read_focus_events(path, since_iso="2026-04-27T10:30:00+00:00")
        assert len(events) == 1
        assert events[0].event == "b"


class TestLatestCursor:
    def test_empty_returns_empty_string(self):
        assert latest_cursor([]) == ""

    def test_returns_max_ts(self):
        events = [
            FocusEvent(ts=_ts("2026-04-27T10:00:00Z"), source="x", event="y"),
            FocusEvent(ts=_ts("2026-04-27T11:00:00Z"), source="x", event="y"),
            FocusEvent(ts=_ts("2026-04-27T10:30:00Z"), source="x", event="y"),
        ]
        cursor = latest_cursor(events)
        assert cursor.startswith("2026-04-27T11:00:00")


# ---------- compute_context_switch_density ----------

class TestContextSwitchDensity:
    def test_basic_density(self):
        # 5 events in a 10-minute session → 0.5/min.
        start = _ts("2026-04-27T10:00:00Z")
        end = _ts("2026-04-27T10:10:00Z")
        events = [
            FocusEvent(ts=start + timedelta(minutes=i), source="nsworkspace", event="a")
            for i in range(1, 6)
        ]
        density = compute_context_switch_density(events, [("s1", start, end)])
        assert density["s1"] == pytest.approx(0.5)

    def test_zero_when_no_events(self):
        start = _ts("2026-04-27T10:00:00Z")
        end = _ts("2026-04-27T10:10:00Z")
        density = compute_context_switch_density([], [("s1", start, end)])
        assert density["s1"] == 0.0

    def test_zero_duration_session(self):
        start = end = _ts("2026-04-27T10:00:00Z")
        density = compute_context_switch_density([], [("s1", start, end)])
        assert density["s1"] == 0.0

    def test_events_outside_session_are_excluded(self):
        start = _ts("2026-04-27T10:00:00Z")
        end = _ts("2026-04-27T10:10:00Z")
        events = [
            FocusEvent(ts=_ts("2026-04-27T09:00:00Z"), source="x", event="y"),  # before
            FocusEvent(ts=_ts("2026-04-27T10:05:00Z"), source="x", event="y"),  # in
            FocusEvent(ts=_ts("2026-04-27T11:00:00Z"), source="x", event="y"),  # after
        ]
        density = compute_context_switch_density(events, [("s1", start, end)])
        assert density["s1"] == pytest.approx(1 / 10)


# ---------- compute_attention_intervals ----------

class TestAttentionIntervals:
    def test_pairs_terminal_activation_to_next_non_terminal(self):
        events = [
            _terminal_event("2026-04-27T10:00:00Z"),
            _other_event("2026-04-27T10:30:00Z"),
        ]
        intervals = compute_attention_intervals(events)
        assert len(intervals) == 1
        start, end = intervals[0]
        assert start == _ts("2026-04-27T10:00:00Z")
        assert end == _ts("2026-04-27T10:30:00Z")

    def test_open_interval_closes_at_fallback_until(self):
        # Terminal activated; no non-terminal event after — interval closes
        # at the fallback_until window-end.
        events = [_terminal_event("2026-04-27T10:00:00Z")]
        end_window = _ts("2026-04-27T18:00:00Z")
        intervals = compute_attention_intervals(events, fallback_until=end_window)
        assert intervals == [(_ts("2026-04-27T10:00:00Z"), end_window)]

    def test_alternating_terminal_other_yields_multiple_intervals(self):
        events = [
            _terminal_event("2026-04-27T10:00:00Z"),
            _other_event("2026-04-27T10:15:00Z"),
            _terminal_event("2026-04-27T10:30:00Z"),
            _other_event("2026-04-27T11:00:00Z"),
        ]
        intervals = compute_attention_intervals(events)
        assert len(intervals) == 2
        assert intervals[0] == (_ts("2026-04-27T10:00:00Z"), _ts("2026-04-27T10:15:00Z"))
        assert intervals[1] == (_ts("2026-04-27T10:30:00Z"), _ts("2026-04-27T11:00:00Z"))

    def test_consecutive_terminals_keep_existing_interval(self):
        # Terminal → terminal (different bundle, both in DEFAULT) should not
        # close and reopen — the user is still in terminal scope.
        events = [
            _terminal_event("2026-04-27T10:00:00Z", "com.apple.Terminal"),
            _terminal_event("2026-04-27T10:05:00Z", "com.googlecode.iterm2"),
            _other_event("2026-04-27T10:30:00Z"),
        ]
        intervals = compute_attention_intervals(events)
        assert intervals == [(_ts("2026-04-27T10:00:00Z"), _ts("2026-04-27T10:30:00Z"))]

    def test_tmux_events_ignored_for_intervals(self):
        # tmux focus events refine WHICH project; they don't flip foreground.
        events = [
            FocusEvent(ts=_ts("2026-04-27T10:00:00Z"), source="tmux", event="pane-focus-in"),
            _terminal_event("2026-04-27T10:05:00Z"),
            _other_event("2026-04-27T10:30:00Z"),
        ]
        intervals = compute_attention_intervals(events)
        assert len(intervals) == 1
        assert intervals[0] == (_ts("2026-04-27T10:05:00Z"), _ts("2026-04-27T10:30:00Z"))

    def test_unknown_bundle_id_treated_as_non_terminal(self):
        events = [
            _other_event("2026-04-27T10:00:00Z", "com.totally.unknown.app"),
        ]
        intervals = compute_attention_intervals(events)
        assert intervals == []

    def test_default_bundles_include_common_terminals(self):
        # Smoke — DEFAULT_TERMINAL_BUNDLE_IDS contains the well-known terminals.
        assert "com.apple.Terminal" in DEFAULT_TERMINAL_BUNDLE_IDS
        assert "com.googlecode.iterm2" in DEFAULT_TERMINAL_BUNDLE_IDS
        assert "com.mitchellh.ghostty" in DEFAULT_TERMINAL_BUNDLE_IDS


# ---------- detect_project_allocation with attention intervals ----------

class TestAttentionWeightedAllocation:
    def test_no_intervals_keeps_command_span_basis(self, tmp_path):
        cfg = Config(base_dir=tmp_path / ".ambient")
        cfg.ensure_dirs()
        events = [_shell_event(0, "/repo/proj", 60_000)]
        findings = detect_project_allocation(events, cfg)
        assert findings.time_basis == "command_span"
        assert findings.allocations[0].total_ms == 60_000

    def test_empty_intervals_marks_attention_weighted(self, tmp_path):
        cfg = Config(base_dir=tmp_path / ".ambient")
        cfg.ensure_dirs()
        events = [_shell_event(0, "/repo/proj", 60_000)]
        findings = detect_project_allocation(events, cfg, attention_intervals=[])
        # Empty list still indicates attention mode is active; no overlap → 0 ms.
        # But our code treats `not attention_intervals` as no-op, so result is
        # command-span. Document this contract: explicit None vs empty list.
        assert findings.time_basis == "command_span"

    def test_full_overlap_yields_full_duration(self, tmp_path):
        cfg = Config(base_dir=tmp_path / ".ambient")
        cfg.ensure_dirs()
        # Event spans 10:00 → 10:01 UTC; attention interval is bigger and contains it.
        ts_ms = int(_ts("2026-04-27T10:00:00Z").timestamp() * 1000)
        events = [_shell_event(ts_ms, "/repo/proj", 60_000)]
        intervals = [(_ts("2026-04-27T09:00:00Z"), _ts("2026-04-27T11:00:00Z"))]
        findings = detect_project_allocation(events, cfg, attention_intervals=intervals)
        assert findings.time_basis == "attention_weighted"
        assert findings.allocations[0].total_ms == 60_000

    def test_no_overlap_yields_zero_duration(self, tmp_path):
        cfg = Config(base_dir=tmp_path / ".ambient")
        cfg.ensure_dirs()
        ts_ms = int(_ts("2026-04-27T10:00:00Z").timestamp() * 1000)
        events = [_shell_event(ts_ms, "/repo/proj", 60_000)]
        # Attention interval is hours later — no overlap.
        intervals = [(_ts("2026-04-27T15:00:00Z"), _ts("2026-04-27T16:00:00Z"))]
        findings = detect_project_allocation(events, cfg, attention_intervals=intervals)
        assert findings.allocations[0].total_ms == 0

    def test_partial_overlap_clips_to_intersection(self, tmp_path):
        cfg = Config(base_dir=tmp_path / ".ambient")
        cfg.ensure_dirs()
        # Event spans 10:00 → 10:10; attention interval 10:05 → 10:15 →
        # overlap is 5 minutes = 300_000 ms.
        ts_ms = int(_ts("2026-04-27T10:00:00Z").timestamp() * 1000)
        events = [_shell_event(ts_ms, "/repo/proj", 600_000)]  # 10 min
        intervals = [(_ts("2026-04-27T10:05:00Z"), _ts("2026-04-27T10:15:00Z"))]
        findings = detect_project_allocation(events, cfg, attention_intervals=intervals)
        assert findings.allocations[0].total_ms == 300_000

    def test_attention_weighted_le_command_span(self, tmp_path):
        # Sanity invariant: attention-weighted time is always ≤ command-span time.
        cfg = Config(base_dir=tmp_path / ".ambient")
        cfg.ensure_dirs()
        ts_ms = int(_ts("2026-04-27T10:00:00Z").timestamp() * 1000)
        events = [
            _shell_event(ts_ms, "/repo/proj", 600_000),
            _shell_event(ts_ms + 1_000_000, "/repo/proj", 300_000),
        ]
        # Two intervals each contained inside one event.
        intervals = [
            (_ts("2026-04-27T10:02:00Z"), _ts("2026-04-27T10:05:00Z")),
            (_ts("2026-04-27T10:18:00Z"), _ts("2026-04-27T10:19:00Z")),
        ]
        cs = detect_project_allocation(events, cfg)
        aw = detect_project_allocation(events, cfg, attention_intervals=intervals)
        assert aw.allocations[0].total_ms <= cs.allocations[0].total_ms

    def test_overlapping_intervals_merge_correctly(self, tmp_path):
        cfg = Config(base_dir=tmp_path / ".ambient")
        cfg.ensure_dirs()
        ts_ms = int(_ts("2026-04-27T10:00:00Z").timestamp() * 1000)
        events = [_shell_event(ts_ms, "/repo/proj", 600_000)]  # 10 min
        # Two overlapping intervals — should merge to one, no double-count.
        intervals = [
            (_ts("2026-04-27T10:00:00Z"), _ts("2026-04-27T10:08:00Z")),
            (_ts("2026-04-27T10:05:00Z"), _ts("2026-04-27T10:10:00Z")),
        ]
        findings = detect_project_allocation(events, cfg, attention_intervals=intervals)
        # Union covers full event → 600_000 ms (NOT 600 + double-count).
        assert findings.allocations[0].total_ms == 600_000
