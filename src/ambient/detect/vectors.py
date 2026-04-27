"""Vector aggregation detector (v4 Phase 3).

Re-shapes existing signals (shell commands, Claude prompts, pauses, focus
events) into the stop-point/event-vector model from
docs/brainstorms/2026-04-26-stop-point-event-taxonomy.md.

A *vector* is a stretch of activity terminated by a stop event:
    - "enter"        — a command was submitted or a Claude prompt arrived
    - "pause"        — a pause classifier flagged the gap as evaluating/stuck
    - "focus_change" — NSWorkspace or tmux focus shifted
    - "exit"         — a session ended (reserved for explicit termination)
    - "end_of_window" — synthesized at window_end so the last vector closes

Each vector carries: when it started, when it ended, why it ended, the last
submitted text, the project, app/pane focus, and a heuristic classification.

Read-only contract: never modifies inputs. Failures degrade to an empty
VectorFindings via the orchestrator's _safe_run wrapper.

This module ships in two waves: Unit 1 lands the data model + classifier
(this file's first version); Unit 2 adds detect_vectors; Unit 3 adds the
aggregation helpers.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Iterable, Literal

from ambient.detect.slash_taxonomy import (
    classify_slash_command,
    extract_slash_command,
)

if TYPE_CHECKING:
    from ambient.capture.reader import Event
    from ambient.config import Config
    from ambient.detect.focus_events import FocusEvent
    from ambient.detect.pauses import PauseFindings

logger = logging.getLogger(__name__)

StopReason = Literal["enter", "pause", "focus_change", "exit", "end_of_window"]
VectorCategory = Literal[
    "planning", "execution", "review", "design",
    "thinking", "freeform", "meta",
]

# Tie-break order when multiple stop events fire at the same ms. The "harder"
# stop reason wins (matches the user's mental model — exit > pause > focus > enter).
_STOP_PRIORITY: dict[StopReason, int] = {
    "exit": 4,
    "pause": 3,
    "focus_change": 2,
    "enter": 1,
    "end_of_window": 0,  # only emitted once at window_end; never collides
}


# Execution-keyword set: a vector ending in a command or prompt that starts
# with one of these tokens classifies as "execution" when there's no slash
# marker. Conservative; tune against real shell history.
_EXECUTION_PREFIXES: frozenset[str] = frozenset({
    "make", "npm", "pnpm", "yarn", "cargo", "go", "pytest", "python", "python3",
    "bin/rails", "rails", "bundle", "docker", "gh", "git", "tsc", "eslint",
    "ruff", "mypy", "rspec", "jest", "vitest",
})

# When stop_reason == "pause" and last_text is short or empty, the vector
# is just thinking time. Threshold for "short": 20 chars (intentionally low).
_THINKING_TEXT_MAX_CHARS = 20


@dataclass(frozen=True)
class StopEvent:
    """A boundary between two vectors.

    `ts_ms` is the epoch-ms timestamp at which the boundary fires.
    `reason` is the stop category. `text` is the last submitted command or
    prompt text (empty for focus_change / end_of_window).
    `pause_duration_ms` is populated only when reason == "pause".
    `project`, `app_focus`, `tmux_pane_focus` carry the *outgoing* context
    that the closing vector should record.
    """

    ts_ms: int
    reason: StopReason
    text: str = ""
    project: str = ""
    app_focus: str | None = None
    tmux_pane_focus: str | None = None
    pause_duration_ms: int | None = None

    @property
    def priority(self) -> int:
        return _STOP_PRIORITY.get(self.reason, 0)


@dataclass
class Vector:
    """A stretch of activity terminated by a stop event."""

    ts_start: int
    ts_end: int
    duration_ms: int
    stop_reason: StopReason
    last_command_or_prompt: str
    project: str
    app_focus: str | None = None
    tmux_pane_focus: str | None = None
    pause_duration_ms: int | None = None
    classification: VectorCategory = "freeform"


@dataclass
class VectorFindings:
    """Result of detect_vectors. All counters mirror `vectors` for cheap rendering."""

    vectors: list[Vector] = field(default_factory=list)
    count_by_stop_reason: dict[StopReason, int] = field(default_factory=dict)
    total_duration_by_stop_reason: dict[StopReason, int] = field(default_factory=dict)
    count_by_project: dict[str, int] = field(default_factory=dict)
    count_by_classification: dict[VectorCategory, int] = field(default_factory=dict)
    window_start_iso: str = ""
    window_end_iso: str = ""


def classify_vector(
    last_text: str,
    stop_reason: StopReason,
    slash_command: str | None = None,
    overrides: dict[str, str] | None = None,
) -> VectorCategory:
    """Heuristic classifier mapping a vector's terminating context to a category.

    Order of precedence:
    1. Slash command (defer to slash_taxonomy; "other" demotes to "freeform"
       to avoid leaking the "other" bucket into vector classification).
    2. stop_reason == "pause" with empty/short last_text → "thinking" (the
       vector was a long pause with nothing meaningful entered, classic
       "evaluating" signal from the pause GMM).
    3. last_text starts with an execution-keyword token → "execution".
    4. Fallback → "freeform".
    """
    cmd = (slash_command or "").strip()
    if cmd:
        cat = classify_slash_command(cmd, overrides=overrides)
        if cat == "other":
            return "freeform"
        return cat  # type: ignore[return-value]

    text = (last_text or "").strip()
    if stop_reason == "pause" and len(text) <= _THINKING_TEXT_MAX_CHARS:
        return "thinking"

    if text:
        first_token = text.split(None, 1)[0].lower()
        if first_token in _EXECUTION_PREFIXES:
            return "execution"

    return "freeform"


# --------------------------------------------------------------------------
# Unit 2: stop-event enumeration + vector detection
# --------------------------------------------------------------------------


# Pause-label severity ordering. The detector emits a pause stop event when
# the GMM classifier's label is at or above the configured min_label.
_PAUSE_LABEL_SEVERITY: dict[str, int] = {
    "routine": 0,
    "evaluating": 1,
    "stuck": 2,
}


def _pause_qualifies(label: str, min_label: str) -> bool:
    """Return True if a pause classification is severe enough to emit a stop."""
    label_sev = _PAUSE_LABEL_SEVERITY.get(label, -1)
    threshold = _PAUSE_LABEL_SEVERITY.get(min_label, _PAUSE_LABEL_SEVERITY["evaluating"])
    return label_sev >= threshold


def _project_from_event(event) -> str:
    """Project derivation. Defers to projects._derive_project to avoid the
    third copy of identical logic the maintainability review flagged.
    """
    from ambient.detect.projects import _derive_project
    return _derive_project(event)


def _enumerate_stops(
    events,
    focus_events,
    pauses,
    window_start_ms: int,
    window_end_ms: int,
    config,
) -> list[StopEvent]:
    """Collect every stop event from every source, sort, debounce, deduplicate.

    Tie-breaking: when multiple stops share a timestamp, the highest-priority
    reason (per _STOP_PRIORITY) wins; the others are discarded for that ms.
    """
    stops: list[StopEvent] = []

    # 1. Shell command + claude_session events → "enter" stop at ts_end.
    # Plus: events that follow a session boundary (gap_ms above the configured
    # threshold or session_boundary=True) emit an "exit" stop at the start of
    # the gap. Without this, vectors silently span overnight idle gaps because
    # pauses.classify() deliberately skips session-boundary gaps and never
    # emits a PauseClassification for them. Real-data validation surfaced
    # 18-hour 'enter' vectors ending in `tmux attach` because of this.
    session_boundary_ms = getattr(config, "session_boundary_ms", 600_000)
    for ev in events or []:
        # Session-boundary "exit" stop: emitted at the END of the prior
        # activity (i.e. ev.ts_start - ev.gap_ms ≈ previous event's ts_end).
        gap = getattr(ev, "gap_ms", None) or 0
        is_boundary = bool(getattr(ev, "session_boundary", False)) or gap > session_boundary_ms
        if is_boundary and gap > 0:
            exit_ts = ev.ts_start - gap
            if window_start_ms <= exit_ts <= window_end_ms:
                stops.append(StopEvent(
                    ts_ms=exit_ts,
                    reason="exit",
                    text="",
                    project=_project_from_event(ev),
                ))

        ts_end = ev.ts_start + max(ev.duration_ms, 0)
        if ts_end < window_start_ms or ts_end > window_end_ms:
            continue
        if ev.type == "command":
            stops.append(StopEvent(
                ts_ms=ts_end,
                reason="enter",
                text=ev.command or "",
                project=_project_from_event(ev),
            ))
        elif ev.type == "claude_session":
            prompts = getattr(ev, "claude_prompts", None) or []
            text = prompts[0] if prompts else (ev.command or "")
            stops.append(StopEvent(
                ts_ms=ts_end,
                reason="enter",
                text=text,
                project=_project_from_event(ev),
            ))

    # 2. Pause classifications → "pause" stop at the gap's end ts.
    # PauseClassification.ts_start is the *next* event's ts_start (set in
    # pauses.classify()), which equals the pause-end. Do NOT add gap_ms — that
    # would double-count the gap and place pause stops past the next command.
    if pauses is not None:
        classifications = getattr(pauses, "classifications", None) or []
        min_label = getattr(config, "vector_pause_min_label", "evaluating")
        unrecognized_labels: set[str] = set()
        for pc in classifications:
            if pc.label not in _PAUSE_LABEL_SEVERITY:
                unrecognized_labels.add(pc.label)
                continue
            if not _pause_qualifies(pc.label, min_label):
                continue
            ts_end = pc.ts_start or 0
            if ts_end < window_start_ms or ts_end > window_end_ms:
                continue
            stops.append(StopEvent(
                ts_ms=ts_end,
                reason="pause",
                text=pc.preceding_command or "",
                project="",  # pause carries no project; resolved during aggregation
                pause_duration_ms=pc.gap_ms,
            ))
        if unrecognized_labels:
            logger.warning(
                "vectors: %d pause label(s) not in known severity table %s; "
                "pause stops dropped silently. Add to _PAUSE_LABEL_SEVERITY if intended.",
                len(unrecognized_labels), sorted(unrecognized_labels),
            )

    # 3. Focus events → "focus_change" stops, debounced.
    # Sort by ts BEFORE debouncing so out-of-order input (concurrent NSWorkspace
    # + tmux writers may interleave) doesn't drop a legitimate event whose ts
    # is earlier than the previously-seen one.
    debounce_ms = max(0, getattr(config, "vector_focus_debounce_ms", 2000))
    sorted_focus = sorted(focus_events or [], key=lambda fe: fe.ts)
    last_focus_ts: int | None = None
    for fe in sorted_focus:
        ts_ms = int(fe.ts.timestamp() * 1000)
        if ts_ms < window_start_ms or ts_ms > window_end_ms:
            continue
        if last_focus_ts is not None and (ts_ms - last_focus_ts) < debounce_ms:
            continue
        last_focus_ts = ts_ms
        app_focus = fe.bundle_id or fe.app_name
        tmux_pane_focus = fe.pane_id
        stops.append(StopEvent(
            ts_ms=ts_ms,
            reason="focus_change",
            text="",
            project="",
            app_focus=app_focus,
            tmux_pane_focus=tmux_pane_focus,
        ))

    # 4. Synthesize end_of_window so the last open vector closes cleanly.
    stops.append(StopEvent(ts_ms=window_end_ms, reason="end_of_window"))

    # Sort by ts; on ties, keep the highest-priority reason and discard the rest.
    stops.sort(key=lambda s: (s.ts_ms, -s.priority))
    deduped: list[StopEvent] = []
    seen_ts: set[int] = set()
    for s in stops:
        if s.ts_ms in seen_ts:
            continue
        seen_ts.add(s.ts_ms)
        deduped.append(s)
    return deduped


def _project_for_window(events_in_window: list, fallback: str) -> str:
    """Pick the most-frequent project among events that fall inside a vector."""
    if not events_in_window:
        return fallback or "unknown"
    counts: Counter[str] = Counter(_project_from_event(e) for e in events_in_window)
    return counts.most_common(1)[0][0]


def detect_vectors(
    events: list["Event"] | None,
    focus_events: list["FocusEvent"] | None,
    pauses: "PauseFindings | None",
    window_start_ms: int,
    window_end_ms: int,
    config: "Config",
) -> VectorFindings:
    """Build VectorFindings for the window.

    Args:
        events: list[Event] from read_events; covers shell commands + claude sessions.
        focus_events: list[FocusEvent] from read_focus_events; empty when capture is off.
        pauses: PauseFindings (or None) from detect_pauses; classifications drive
            "pause" stop events when their label is at or above the config threshold.
        window_start_ms, window_end_ms: epoch-ms window bounds.
        config: source of vector_pause_min_label, vector_focus_debounce_ms,
            slash_taxonomy_overrides.
    """
    if window_end_ms <= window_start_ms:
        return VectorFindings(
            window_start_iso=_ms_to_iso(window_start_ms),
            window_end_iso=_ms_to_iso(window_end_ms),
        )

    stops = _enumerate_stops(
        events, focus_events, pauses, window_start_ms, window_end_ms, config,
    )
    if not stops:
        return VectorFindings(
            window_start_iso=_ms_to_iso(window_start_ms),
            window_end_iso=_ms_to_iso(window_end_ms),
        )

    overrides = getattr(config, "slash_taxonomy_overrides", None) or None
    sorted_events = sorted(events or [], key=lambda e: e.ts_start)

    vectors: list[Vector] = []
    cursor_ms = window_start_ms
    last_app_focus: str | None = None
    last_tmux_focus: str | None = None
    last_project: str = "unknown"  # carries forward when a vector has no events

    for i, stop in enumerate(stops):
        if stop.ts_ms < cursor_ms:
            # Stop predates cursor (shouldn't happen with the global sort, but
            # defend against future changes). Skip without rewinding.
            continue
        if stop.ts_ms == cursor_ms:
            # Empty vector — no time elapsed. Update focus state so it carries
            # to the next vector, but do not emit a zero-duration record.
            if stop.reason == "focus_change":
                last_app_focus = stop.app_focus
                last_tmux_focus = stop.tmux_pane_focus
            continue

        events_in_vector = [
            e for e in sorted_events
            if cursor_ms <= e.ts_start < stop.ts_ms
        ]
        # Project resolution: stop.project (set for command/session/exit stops)
        # wins; otherwise the most-frequent project among events_in_vector;
        # otherwise carry forward last_project so empty pause-terminated
        # vectors don't collapse to "unknown".
        if stop.project:
            project = stop.project
        elif events_in_vector:
            project = _project_for_window(events_in_vector, fallback=last_project)
        else:
            project = last_project
        last_project = project

        last_text = stop.text
        slash_cmd = extract_slash_command(last_text) if last_text else None
        cls = classify_vector(
            last_text=last_text,
            stop_reason=stop.reason,
            slash_command=slash_cmd,
            overrides=overrides,
        )
        vectors.append(Vector(
            ts_start=cursor_ms,
            ts_end=stop.ts_ms,
            duration_ms=stop.ts_ms - cursor_ms,
            stop_reason=stop.reason,
            last_command_or_prompt=last_text,
            project=project,
            app_focus=last_app_focus,
            tmux_pane_focus=last_tmux_focus,
            pause_duration_ms=stop.pause_duration_ms,
            classification=cls,
        ))

        # After an exit stop, skip the idle gap: advance the cursor past every
        # adjacent exit stop to the next non-exit stop's ts. Without this, the
        # idle period (overnight, weekend) would render as a ghost vector
        # ending in the next 'enter' — not a real activity stretch.
        if stop.reason == "exit":
            next_active_ts = stop.ts_ms
            for later in stops[i + 1:]:
                if later.reason != "exit" and later.ts_ms > stop.ts_ms:
                    next_active_ts = later.ts_ms
                    break
            cursor_ms = next_active_ts
        else:
            cursor_ms = stop.ts_ms

        if stop.reason == "focus_change":
            last_app_focus = stop.app_focus
            last_tmux_focus = stop.tmux_pane_focus

    return _build_findings(vectors, window_start_ms, window_end_ms)


def _build_findings(
    vectors: list[Vector],
    window_start_ms: int,
    window_end_ms: int,
) -> VectorFindings:
    count_by_stop_reason: dict[str, int] = {}
    total_duration_by_stop_reason: dict[str, int] = {}
    count_by_project: dict[str, int] = {}
    count_by_classification: dict[str, int] = {}
    for v in vectors:
        count_by_stop_reason[v.stop_reason] = count_by_stop_reason.get(v.stop_reason, 0) + 1
        total_duration_by_stop_reason[v.stop_reason] = (
            total_duration_by_stop_reason.get(v.stop_reason, 0) + v.duration_ms
        )
        count_by_project[v.project] = count_by_project.get(v.project, 0) + 1
        count_by_classification[v.classification] = (
            count_by_classification.get(v.classification, 0) + 1
        )
    return VectorFindings(
        vectors=vectors,
        count_by_stop_reason=count_by_stop_reason,
        total_duration_by_stop_reason=total_duration_by_stop_reason,
        count_by_project=count_by_project,
        count_by_classification=count_by_classification,
        window_start_iso=_ms_to_iso(window_start_ms),
        window_end_iso=_ms_to_iso(window_end_ms),
    )


def _ms_to_iso(ms: int) -> str:
    if ms <= 0:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Unit 3: aggregation surfaces (per-project, per-day, per-stop-reason)
# --------------------------------------------------------------------------


def top_vectors_per_project(
    findings: VectorFindings, n: int,
) -> dict[str, list[Vector]]:
    """For each project, return its top-n longest vectors sorted by duration desc."""
    if n <= 0 or not findings.vectors:
        return {}
    by_project: dict[str, list[Vector]] = {}
    for v in findings.vectors:
        by_project.setdefault(v.project, []).append(v)
    return {
        proj: sorted(vs, key=lambda v: v.duration_ms, reverse=True)[:n]
        for proj, vs in by_project.items()
    }


def vectors_by_day(findings: VectorFindings) -> dict:
    """Bucket vectors by their start-day (local time).

    Mirrors insights._format_by_day_summary's date-bucketing convention so the
    by-day renderer can show vector activity alongside project time.
    """
    from collections import defaultdict
    out: dict = defaultdict(list)
    for v in findings.vectors:
        d = datetime.fromtimestamp(v.ts_start / 1000).date()
        out[d].append(v)
    return dict(out)


def stop_reason_summary(
    findings: VectorFindings,
) -> list[tuple[StopReason, int, int]]:
    """Return [(reason, count, total_duration_ms), ...] sorted by total_duration desc."""
    rows: list[tuple[StopReason, int, int]] = []
    for reason, count in findings.count_by_stop_reason.items():
        total = findings.total_duration_by_stop_reason.get(reason, 0)
        rows.append((reason, count, total))  # type: ignore[arg-type]
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows


