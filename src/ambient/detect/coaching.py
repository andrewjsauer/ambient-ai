"""Coaching detectors: session outcome classification, thrash scoring, stuck pattern grouping."""

from dataclasses import dataclass, field

from ambient.capture.reader import Event
from ambient.config import Config


@dataclass
class SessionOutcome:
    session_id: str
    classification: str  # "productive", "friction", "quick", "abandoned"
    thrash_score: float | None
    project: str
    duration_ms: int
    prompt_count: int
    error_count: int
    tools: list[dict]
    files: list[str]


@dataclass
class CoachingFindings:
    outcomes: list[SessionOutcome] = field(default_factory=list)
    count_by_classification: dict[str, int] = field(default_factory=dict)
    avg_thrash_score: float | None = None
    low_sample: bool = False


@dataclass
class StuckPattern:
    project: str
    file_cluster: list[str]
    failing_tools: list[str]
    episode_count: int
    avg_thrash_score: float | None
    total_duration_ms: int
    session_ids: list[str]


@dataclass
class ToolStuckPattern:
    tool_name: str
    episode_count: int  # stuck sessions where this tool appeared
    projects: list[str]
    avg_thrash_score: float | None
    total_duration_ms: int


@dataclass
class FileClusterStuckPattern:
    path_fragment: str
    episode_count: int
    projects: list[str]
    failing_tools: list[str]
    total_duration_ms: int


@dataclass
class StuckPatternFindings:
    patterns: list[StuckPattern] = field(default_factory=list)
    total_stuck_sessions: int = 0
    tool_level_patterns: list[ToolStuckPattern] = field(default_factory=list)
    file_cluster_patterns: list[FileClusterStuckPattern] = field(default_factory=list)


def _compute_thrash_score(error_count: int, prompt_count: int, min_prompts: int) -> float | None:
    if prompt_count < min_prompts:
        return None
    return error_count / prompt_count


def _count_tool_calls(tools: list[dict] | None) -> int:
    return len(tools) if tools else 0


def _has_write_edit(tools: list[dict] | None) -> bool:
    if not tools:
        return False
    return any(t.get("name") in ("Write", "Edit") for t in tools)


def _extract_tool_names(tools: list[dict] | None) -> list[str]:
    if not tools:
        return []
    return list({t.get("name", "unknown") for t in tools})


def _dominant_path_prefix(files: list[str]) -> str:
    """Per-session dominant path prefix for file-cluster grouping.

    Strategy:
    - Longest common directory prefix of the session's files if >=2 chars.
    - Otherwise the basename of the first file (so single-file sessions still group).
    - Empty string if there are no files.
    """
    if not files:
        return ""
    if len(files) == 1:
        name = files[0].rsplit("/", 1)[-1]
        return name if name else files[0]

    # Longest common char prefix, then trim to the last directory boundary
    prefix = files[0]
    for f in files[1:]:
        while not f.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                break
        if not prefix:
            break

    if "/" in prefix:
        prefix = prefix.rsplit("/", 1)[0] + "/"
    if len(prefix) >= 2:
        return prefix

    # Fall back to basename of first file
    first = files[0]
    return first.rsplit("/", 1)[-1] or first


def classify_session_outcome(event: Event, config: Config) -> SessionOutcome:
    prompt_count = event.claude_prompt_count or 0
    error_count = event.claude_is_error_count or 0
    tools = event.claude_tools or []
    files = event.claude_files or []
    project = event.claude_project or event.cwd or "unknown"
    duration_ms = event.duration_ms
    session_id = event.claude_session_id or ""

    thrash_score = _compute_thrash_score(error_count, prompt_count, config.thrash_min_prompts)

    # Strict precedence ordering
    tool_count = _count_tool_calls(tools)

    if prompt_count < 5 and tool_count < 3:
        classification = "quick"
    elif (prompt_count > 1
          and not _has_write_edit(tools)
          and error_count > 0
          and duration_ms > 300_000):
        classification = "abandoned"
    elif thrash_score is not None and thrash_score > config.thrash_score_threshold:
        classification = "friction"
    else:
        classification = "productive"

    return SessionOutcome(
        session_id=session_id,
        classification=classification,
        thrash_score=thrash_score,
        project=project,
        duration_ms=duration_ms,
        prompt_count=prompt_count,
        error_count=error_count,
        tools=tools,
        files=files,
    )


def classify_sessions(events: list[Event], config: Config) -> CoachingFindings:
    outcomes = []
    for event in events:
        if event.type != "claude_session":
            continue
        outcomes.append(classify_session_outcome(event, config))

    count_by = {}
    for o in outcomes:
        count_by[o.classification] = count_by.get(o.classification, 0) + 1

    scores = [o.thrash_score for o in outcomes if o.thrash_score is not None]
    if len(scores) >= config.thrash_aggregate_min_n:
        avg_score = sum(scores) / len(scores)
        low_sample = False
    else:
        avg_score = None
        low_sample = 0 < len(scores) < config.thrash_aggregate_min_n

    return CoachingFindings(
        outcomes=outcomes,
        count_by_classification=count_by,
        avg_thrash_score=avg_score,
        low_sample=low_sample,
    )


def group_stuck_patterns(
    outcomes: list[SessionOutcome],
    events: list[Event],
    config: Config,
) -> StuckPatternFindings:
    # Filter to Friction and Abandoned outcomes
    stuck = [o for o in outcomes if o.classification in ("friction", "abandoned")]

    if not stuck:
        return StuckPatternFindings()

    # Group by project
    by_project: dict[str, list[SessionOutcome]] = {}
    for o in stuck:
        by_project.setdefault(o.project, []).append(o)

    patterns = []
    for project, project_outcomes in by_project.items():
        # Collect all files and tool names across stuck sessions in this project
        all_files: list[str] = []
        all_tool_names: list[str] = []
        session_ids: list[str] = []
        total_duration = 0
        thrash_scores: list[float] = []

        for o in project_outcomes:
            all_files.extend(o.files)
            all_tool_names.extend(_extract_tool_names(o.tools))
            session_ids.append(o.session_id)
            total_duration += o.duration_ms
            if o.thrash_score is not None:
                thrash_scores.append(o.thrash_score)

        # Deduplicate files and tool names
        unique_files = sorted(set(all_files)) if all_files else ["unknown"]
        unique_tools = sorted(set(all_tool_names)) if all_tool_names else ["unknown"]

        if len(thrash_scores) >= config.thrash_aggregate_min_n:
            avg_thrash = sum(thrash_scores) / len(thrash_scores)
        else:
            avg_thrash = None

        patterns.append(StuckPattern(
            project=project,
            file_cluster=unique_files,
            failing_tools=unique_tools,
            episode_count=len(project_outcomes),
            avg_thrash_score=avg_thrash,
            total_duration_ms=total_duration,
            session_ids=session_ids,
        ))

    # Sort by episode_count descending
    patterns.sort(key=lambda p: p.episode_count, reverse=True)

    # --- Tool-level grouping: one pattern per failing tool across stuck sessions ---
    tool_level = _group_by_tool(stuck, config)

    # --- File-cluster grouping: one pattern per dominant-prefix bucket ---
    file_clusters = _group_by_file_cluster(stuck, config)

    return StuckPatternFindings(
        patterns=patterns,
        total_stuck_sessions=len(stuck),
        tool_level_patterns=tool_level,
        file_cluster_patterns=file_clusters,
    )


def _group_by_tool(
    stuck: list[SessionOutcome], config: Config
) -> list[ToolStuckPattern]:
    by_tool: dict[str, list[SessionOutcome]] = {}
    for o in stuck:
        for tool_name in _extract_tool_names(o.tools):
            by_tool.setdefault(tool_name, []).append(o)

    results: list[ToolStuckPattern] = []
    for tool_name, tool_outcomes in by_tool.items():
        if len(tool_outcomes) < 2:
            continue  # single-session tool patterns are redundant with project-level
        thrash_scores = [o.thrash_score for o in tool_outcomes if o.thrash_score is not None]
        if len(thrash_scores) >= config.thrash_aggregate_min_n:
            avg_thrash = sum(thrash_scores) / len(thrash_scores)
        else:
            avg_thrash = None
        results.append(ToolStuckPattern(
            tool_name=tool_name,
            episode_count=len(tool_outcomes),
            projects=sorted({o.project for o in tool_outcomes}),
            avg_thrash_score=avg_thrash,
            total_duration_ms=sum(o.duration_ms for o in tool_outcomes),
        ))

    results.sort(key=lambda p: p.episode_count, reverse=True)
    return results


def _group_by_file_cluster(
    stuck: list[SessionOutcome], config: Config
) -> list[FileClusterStuckPattern]:
    by_prefix: dict[str, list[SessionOutcome]] = {}
    for o in stuck:
        prefix = _dominant_path_prefix(o.files or [])
        if not prefix:
            continue  # skip sessions with no file signal
        by_prefix.setdefault(prefix, []).append(o)

    results: list[FileClusterStuckPattern] = []
    for prefix, cluster_outcomes in by_prefix.items():
        if len(cluster_outcomes) < 2:
            continue  # singletons already covered by project grouping
        tools: set[str] = set()
        for o in cluster_outcomes:
            tools.update(_extract_tool_names(o.tools))
        results.append(FileClusterStuckPattern(
            path_fragment=prefix,
            episode_count=len(cluster_outcomes),
            projects=sorted({o.project for o in cluster_outcomes}),
            failing_tools=sorted(tools),
            total_duration_ms=sum(o.duration_ms for o in cluster_outcomes),
        ))

    results.sort(key=lambda p: p.episode_count, reverse=True)
    return results
