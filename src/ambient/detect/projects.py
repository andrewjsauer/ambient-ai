"""Per-project time allocation detector."""

from dataclasses import dataclass, field
from pathlib import PurePosixPath

from ambient.capture.reader import Event
from ambient.config import Config


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


def detect_project_allocation(events: list[Event], config: Config) -> ProjectFindings:
    """Compute per-project time allocation from events."""
    if not events:
        return ProjectFindings()

    # Group events by project
    project_data: dict[str, dict] = {}
    prev_project = None
    context_switches = 0

    for event in events:
        project = _derive_project(event)

        if project not in project_data:
            project_data[project] = {"total_ms": 0, "session_count": 0, "event_count": 0}

        project_data[project]["total_ms"] += event.duration_ms
        project_data[project]["event_count"] += 1
        if event.type == "claude_session":
            project_data[project]["session_count"] += 1

        if prev_project is not None and project != prev_project:
            context_switches += 1
        prev_project = project

    # Build sorted allocations (by time descending)
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
    )
