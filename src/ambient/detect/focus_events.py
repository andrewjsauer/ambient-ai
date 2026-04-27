"""Focus-events ingestion + derived metrics (Phase 2 Unit 9).

Reads ~/.ambient/focus-events.jsonl with a cursor and produces two outputs:

1. `compute_context_switch_density(focus_events, sessions)` — per-session
   focus-events-per-minute metric. Surfaces in coaching as a side dimension on
   _section_session_outcomes ("4.2/min friction vs 1.1/min productive").

2. `compute_attention_intervals(focus_events, terminal_bundle_ids)` — list of
   `(start_ts, end_ts)` windows during which the user's terminal/Claude was
   the foreground app. Fed into `projects.detect_project_allocation` so the
   project ledger's per-project hours flip from gap-based "command-span time"
   to attention-weighted "active time" — the metric the developer actually
   wants to see.

Read-only contract: never modifies the focus-events file. Failures degrade to
empty results so the rest of the insights pipeline keeps working when the
focus listener is off or the file is missing.

Privacy contract: this detector is read-only over data already filtered by
Units 7 + 8 to the privacy-bounded payload. It does not introduce a new PII
surface and cites docs/PRIVACY.md clauses 6 and 7 by reference.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Default terminal/Claude bundle IDs treated as "in-focus = working in ambient
# scope". User-tunable via Config.terminal_bundle_ids. Conservative default —
# add new bundle IDs here as users report them. Names below are the common
# macOS terminals + Claude Code desktop; an unknown bundle is simply not
# counted as terminal time, which biases "active time" downward (safer than
# overcounting).
DEFAULT_TERMINAL_BUNDLE_IDS: frozenset[str] = frozenset({
    "com.apple.Terminal",
    "com.googlecode.iterm2",
    "net.kovidgoyal.kitty",
    "com.mitchellh.ghostty",
    "dev.warp.Warp-Stable",
    "io.alacritty",
    "com.tabby.Tabby",
    "com.anthropic.claudefordesktop",  # Claude Code desktop, if present
    "ai.claude.desktop",
})


@dataclass(frozen=True)
class FocusEvent:
    ts: datetime
    source: str
    event: str
    bundle_id: str | None = None
    app_name: str | None = None
    pid: int | None = None
    pane_id: str | None = None
    window_index: str | None = None
    session_name: str | None = None


def read_focus_events(
    path: Path,
    *,
    since_iso: str | None = None,
) -> list[FocusEvent]:
    """Read focus-events.jsonl, returning events with ts > since_iso.

    Returns [] when the file is missing, empty, or unreadable. Malformed lines
    are skipped with a debug log.

    The since_iso bound is normalized: naive datetimes are treated as UTC so
    the comparison against UTC-aware event timestamps succeeds. Without this
    normalization, the comparison raises TypeError, gets swallowed by the
    caller's _safe_run, and the entire focus-events pipeline silently no-ops.
    """
    if not path.exists():
        return []
    since: datetime | None = None
    if since_iso:
        try:
            parsed = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            since = parsed

    events: list[FocusEvent] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line_num, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("focus_events: skip malformed line %d", line_num)
                    continue
                ts = _parse_ts(obj.get("ts"))
                if ts is None:
                    continue
                if since is not None and ts <= since:
                    continue
                events.append(FocusEvent(
                    ts=ts,
                    source=obj.get("source", "unknown"),
                    event=obj.get("event", "unknown"),
                    bundle_id=obj.get("bundle_id"),
                    app_name=obj.get("app_name"),
                    pid=obj.get("pid"),
                    pane_id=obj.get("pane_id"),
                    window_index=obj.get("window_index"),
                    session_name=obj.get("session_name"),
                ))
    except OSError as e:
        logger.warning("focus_events: cannot read %s: %s", path, e)
        return []
    return events


def latest_cursor(events: list[FocusEvent]) -> str:
    """Return the latest event ts as ISO-8601 for cursor advancement.

    Empty input returns the empty string so the caller can leave the cursor
    untouched on no-op ticks.

    Clock-skew defense: if the latest event ts is more than one hour in the
    future (clock skew, NTP step, manual file edit), clamp the cursor to
    `now`. Without this, a single bad timestamp would latch the cursor to
    year 9999 and silently filter every subsequent focus event forever —
    recovery would require hand-editing state.json.
    """
    if not events:
        return ""
    max_ts = max(e.ts for e in events)
    horizon = datetime.now(timezone.utc) + timedelta(hours=1)
    if max_ts > horizon:
        logger.warning(
            "focus_events: latest event ts %s is far in the future; "
            "clamping cursor to now to avoid poisoning",
            max_ts.isoformat(),
        )
        return datetime.now(timezone.utc).isoformat()
    return max_ts.isoformat()


def compute_context_switch_density(
    events: list[FocusEvent],
    session_intervals: Iterable[tuple[str, datetime, datetime]],
) -> dict[str, float]:
    """Per-session focus-events-per-minute density.

    Args:
        events: focus events for the analysis window.
        session_intervals: iterable of (session_id, start, end) tuples. The
            caller derives this from the existing event stream / coaching
            outcomes.

    Output: dict[session_id, switches_per_minute]. Sessions with no focus
    events get 0.0 (stable, comparable across sessions). Zero-length sessions
    get 0.0 to avoid division by zero.
    """
    out: dict[str, float] = {}
    for session_id, start, end in session_intervals:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        duration_min = max((end - start).total_seconds() / 60.0, 0.0)
        if duration_min <= 0:
            out[session_id] = 0.0
            continue
        n = sum(1 for e in events if start <= e.ts <= end)
        out[session_id] = n / duration_min
    return out


def compute_attention_intervals(
    events: list[FocusEvent],
    terminal_bundle_ids: Iterable[str] | None = None,
    *,
    fallback_until: datetime | None = None,
) -> list[tuple[datetime, datetime]]:
    """Return time intervals during which a terminal/Claude bundle was foreground.

    Walk events in ts order. When a terminal app is activated, open an interval
    starting at that ts. When a non-terminal app is activated, close the open
    interval at that ts. The final open interval is closed at `fallback_until`
    if provided; otherwise it is closed at the timestamp of the last event so
    we never report an unbounded interval.

    Source filter: only `nsworkspace` events with bundle_id checks contribute.
    tmux events do not flip foreground app state — they only refine WHICH
    project the user was in once the terminal already had focus.
    """
    bundle_set = frozenset(terminal_bundle_ids) if terminal_bundle_ids else DEFAULT_TERMINAL_BUNDLE_IDS

    # Normalize fallback_until to UTC-aware so comparisons against event
    # timestamps (always UTC-aware) don't raise TypeError.
    if fallback_until is not None and fallback_until.tzinfo is None:
        fallback_until = fallback_until.replace(tzinfo=timezone.utc)

    intervals: list[tuple[datetime, datetime]] = []
    open_start: datetime | None = None

    nsw_events = sorted(
        (e for e in events if e.source == "nsworkspace"),
        key=lambda e: e.ts,
    )

    for e in nsw_events:
        # Skip events with no bundle_id when deciding terminal-status. An
        # nsworkspace activation without a bundle_id is too ambiguous to
        # treat as "left the terminal" (would falsely close intervals on
        # unsigned helper apps with bundle_id=None).
        if e.bundle_id is None:
            continue
        is_terminal = e.bundle_id in bundle_set
        if is_terminal and open_start is None:
            open_start = e.ts
        elif not is_terminal and open_start is not None:
            if e.ts > open_start:
                intervals.append((open_start, e.ts))
            open_start = None

    if open_start is not None:
        end_ts = fallback_until if fallback_until is not None else (
            nsw_events[-1].ts if nsw_events else open_start
        )
        if end_ts > open_start:
            intervals.append((open_start, end_ts))

    return intervals


def _parse_ts(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
