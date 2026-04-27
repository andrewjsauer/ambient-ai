"""Per-project time allocation detector."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Literal

from ambient.capture.reader import Event
from ambient.config import Config

TimeBasis = Literal["command_span", "attention_weighted"]


@dataclass
class ProjectAllocation:
    project: str
    total_ms: int
    session_count: int
    event_count: int


@dataclass
class ProjectFindings:
    allocations: list[ProjectAllocation] = field(default_factory=list)
    context_switches: int = 0
    primary_project: str = ""
    # v4 Phase 2 Unit 9: "command_span" sums event durations as-is (gap-based);
    # "attention_weighted" clips each event's contribution to the intersection
    # with focus-event attention intervals. Renderers use this label so users
    # know which time math produced the number they're looking at.
    time_basis: TimeBasis = "command_span"


def _derive_project(event: Event) -> str:
    """Derive a project name from an event."""
    if event.type == "claude_session" and event.claude_project:
        # Use the last meaningful path component
        path = PurePosixPath(event.claude_project)
        return path.name or "unknown"

    if event.cwd:
        path = PurePosixPath(event.cwd)
        # Skip home directory and common non-project paths
        name = path.name
        if name in ("", "~", "/", "tmp"):
            return "unknown"
        return name

    return "unknown"


def detect_project_allocation(
    events: list[Event],
    config: Config,
    *,
    attention_intervals: list[tuple[datetime, datetime]] | None = None,
) -> ProjectFindings:
    """Compute per-project time allocation from events.

    When `attention_intervals` is provided, each event's contribution to its
    project is clipped to the intersection of (event time span) and (the
    union of attention intervals). The resulting `time_basis` is set to
    "attention_weighted" so renderers can label the output correctly.

    When None (or empty), behavior is unchanged: total_ms is the gap-based
    sum of event.duration_ms, and time_basis stays "command_span".
    """
    if not events:
        basis: TimeBasis = (
            "attention_weighted" if attention_intervals is not None else "command_span"
        )
        return ProjectFindings(time_basis=basis)

    use_attention = bool(attention_intervals)
    intervals_ms: list[tuple[int, int]] = []
    if use_attention:
        intervals_ms = _intervals_to_epoch_ms(attention_intervals or [])

    project_data: dict[str, dict] = {}
    prev_project = None
    context_switches = 0

    for event in events:
        project = _derive_project(event)

        if project not in project_data:
            project_data[project] = {"total_ms": 0, "session_count": 0, "event_count": 0}

        if use_attention:
            event_start = event.ts_start
            event_end = event.ts_start + max(event.duration_ms, 0)
            contribution = _overlap_ms((event_start, event_end), intervals_ms)
        else:
            contribution = event.duration_ms

        project_data[project]["total_ms"] += contribution
        project_data[project]["event_count"] += 1
        if event.type == "claude_session":
            project_data[project]["session_count"] += 1

        if prev_project is not None and project != prev_project:
            context_switches += 1
        prev_project = project

    allocations = sorted(
        [
            ProjectAllocation(
                project=name,
                total_ms=data["total_ms"],
                session_count=data["session_count"],
                event_count=data["event_count"],
            )
            for name, data in project_data.items()
        ],
        key=lambda a: a.total_ms,
        reverse=True,
    )

    primary = allocations[0].project if allocations else ""

    return ProjectFindings(
        allocations=allocations,
        context_switches=context_switches,
        primary_project=primary,
        time_basis="attention_weighted" if use_attention else "command_span",
    )


def _intervals_to_epoch_ms(
    intervals: list[tuple[datetime, datetime]],
) -> list[tuple[int, int]]:
    """Convert datetime intervals to (epoch_ms_start, epoch_ms_end), sorted/merged."""
    out: list[tuple[int, int]] = []
    for start, end in intervals:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        if end <= start:
            continue
        out.append((int(start.timestamp() * 1000), int(end.timestamp() * 1000)))
    out.sort()
    # Merge overlapping intervals so overlap math is straightforward.
    merged: list[tuple[int, int]] = []
    for s, e in out:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _overlap_ms(span: tuple[int, int], intervals_ms: list[tuple[int, int]]) -> int:
    """Sum of overlap (in ms) between `span` and each interval in `intervals_ms`."""
    total = 0
    s, e = span
    if e <= s:
        return 0
    for i_s, i_e in intervals_ms:
        if i_e <= s:
            continue
        if i_s >= e:
            break  # intervals sorted; remainder is to the right of span
        total += min(e, i_e) - max(s, i_s)
    return total
