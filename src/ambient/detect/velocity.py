"""Resolution velocity tracker: detect fail→Claude→success chains, measure active time."""

import statistics
from dataclasses import dataclass, field

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.correlator import BENIGN_NONZERO_COMMANDS, _base_command


FIRST_PROMPT_MAX_LENGTH = 120


@dataclass
class ResolutionChain:
    initial_failure_ts: int
    initial_command: str
    claude_session_ids: list[str]
    resolution_ts: int
    resolution_command: str
    active_time_ms: int
    wall_time_ms: int
    project: str
    outcome: str  # worst session outcome in chain
    closure_reason: str  # "matched_success" | "idle_break" | "end_of_window"
    first_claude_prompt: str = ""  # truncated first prompt of the first claude session in the chain

    @property
    def resolved(self) -> bool:
        # Back-compat: prior API exposed `resolved: bool`. True iff the chain
        # was closed by a matching-command success (not idle break / end of window).
        return self.closure_reason == "matched_success"


@dataclass
class VelocityMetrics:
    avg_ms: int = 0
    median_ms: int = 0
    p90_ms: int = 0
    total_chains: int = 0
    resolved_count: int = 0
    by_project: dict = field(default_factory=dict)
    by_reason: dict[str, int] = field(default_factory=dict)


def _is_failed_command(event: Event) -> bool:
    if event.type != "command":
        return False
    if event.exit_code == 0:
        return False
    return _base_command(event.command) not in BENIGN_NONZERO_COMMANDS


def _derive_project(event: Event) -> str:
    if event.type == "claude_session" and event.claude_project:
        return event.claude_project
    if event.cwd:
        return event.cwd.rstrip("/").rsplit("/", 1)[-1]
    return "unknown"


# Outcome severity for "worst wins" logic
_OUTCOME_SEVERITY = {"productive": 0, "quick": 1, "abandoned": 2, "friction": 3}


def detect_resolution_chains(
    events: list[Event],
    config: Config,
    session_outcomes: dict[str, str] | None = None,
) -> list[ResolutionChain]:
    """Walk events chronologically per project, stitching fail→Claude→success chains.

    Args:
        events: All events for the time window.
        config: Config with velocity_idle_break_ms.
        session_outcomes: Optional map of session_id -> classification from coaching detector.
    """
    if not events:
        return []

    outcomes = session_outcomes or {}
    idle_break = config.velocity_idle_break_ms

    # Group events by project, sorted by ts_start
    by_project: dict[str, list[Event]] = {}
    for e in events:
        proj = _derive_project(e)
        by_project.setdefault(proj, []).append(e)

    chains: list[ResolutionChain] = []

    for project, proj_events in by_project.items():
        proj_events.sort(key=lambda e: e.ts_start)

        # State for current open chain
        chain_start_ts: int | None = None
        chain_command: str = ""
        chain_base_cmd: str = ""
        chain_session_ids: list[str] = []
        chain_active_ms: int = 0
        chain_worst_outcome: str = "productive"
        chain_has_claude: bool = False
        chain_first_prompt: str = ""
        last_event_end: int = 0

        for event in proj_events:
            # Check idle break
            if chain_start_ts is not None and last_event_end > 0:
                gap = event.ts_start - last_event_end
                if gap > idle_break:
                    # Break the chain — unresolved
                    chains.append(ResolutionChain(
                        initial_failure_ts=chain_start_ts,
                        initial_command=chain_command,
                        claude_session_ids=chain_session_ids,
                        resolution_ts=last_event_end,
                        resolution_command="",
                        active_time_ms=chain_active_ms,
                        wall_time_ms=last_event_end - chain_start_ts,
                        project=project,
                        outcome=chain_worst_outcome,
                        closure_reason="idle_break",
                        first_claude_prompt=chain_first_prompt,
                    ))
                    chain_start_ts = None

            if chain_start_ts is None:
                # Looking for a failed command to start a chain
                if _is_failed_command(event):
                    chain_start_ts = event.ts_start
                    chain_command = event.command
                    chain_base_cmd = _base_command(event.command)
                    chain_session_ids = []
                    chain_active_ms = event.duration_ms
                    chain_worst_outcome = "productive"
                    chain_has_claude = False
                    chain_first_prompt = ""
                    last_event_end = event.ts_end
                continue

            # Chain is open — process event
            if event.type == "claude_session":
                # Capture the first prompt of the first claude session in the chain
                if not chain_has_claude and event.claude_prompts:
                    chain_first_prompt = event.claude_prompts[0][:FIRST_PROMPT_MAX_LENGTH]
                chain_has_claude = True
                sid = event.claude_session_id or ""
                chain_session_ids.append(sid)
                chain_active_ms += event.duration_ms
                # Track worst outcome
                ev_outcome = outcomes.get(sid, "productive")
                if _OUTCOME_SEVERITY.get(ev_outcome, 0) > _OUTCOME_SEVERITY.get(chain_worst_outcome, 0):
                    chain_worst_outcome = ev_outcome
                last_event_end = event.ts_end

            elif event.type == "command":
                chain_active_ms += event.duration_ms
                last_event_end = event.ts_end

                # Check for resolution: success + matching base command + had Claude involvement
                if (event.exit_code == 0
                        and chain_has_claude
                        and _base_command(event.command) == chain_base_cmd):
                    chains.append(ResolutionChain(
                        initial_failure_ts=chain_start_ts,
                        initial_command=chain_command,
                        claude_session_ids=chain_session_ids,
                        resolution_ts=event.ts_end,
                        resolution_command=event.command,
                        active_time_ms=chain_active_ms,
                        wall_time_ms=event.ts_end - chain_start_ts,
                        project=project,
                        outcome=chain_worst_outcome,
                        closure_reason="matched_success",
                        first_claude_prompt=chain_first_prompt,
                    ))
                    chain_start_ts = None

        # Close any open chain as unresolved at end of events
        if chain_start_ts is not None:
            chains.append(ResolutionChain(
                initial_failure_ts=chain_start_ts,
                initial_command=chain_command,
                claude_session_ids=chain_session_ids,
                resolution_ts=last_event_end,
                resolution_command="",
                active_time_ms=chain_active_ms,
                wall_time_ms=last_event_end - chain_start_ts,
                project=project,
                outcome=chain_worst_outcome,
                closure_reason="end_of_window",
                first_claude_prompt=chain_first_prompt,
            ))

    return chains


def compute_velocity_metrics(
    chains: list[ResolutionChain],
    min_chains: int = 5,
) -> VelocityMetrics:
    """Compute velocity metrics from resolution chains.

    Args:
        min_chains: Minimum resolved chains for meaningful metrics.
            When below this threshold, metrics are still computed but
            consumers should treat them as low-confidence.
    """
    resolved = [c for c in chains if c.resolved]
    active_times = [c.active_time_ms for c in resolved]

    by_reason: dict[str, int] = {}
    for c in chains:
        by_reason[c.closure_reason] = by_reason.get(c.closure_reason, 0) + 1

    metrics = VelocityMetrics(
        total_chains=len(chains),
        resolved_count=len(resolved),
        by_reason=by_reason,
    )

    if active_times:
        metrics.avg_ms = int(statistics.mean(active_times))
        metrics.median_ms = int(statistics.median(active_times))
        if len(active_times) >= 2:
            sorted_times = sorted(active_times)
            p90_idx = int(len(sorted_times) * 0.9)
            metrics.p90_ms = sorted_times[min(p90_idx, len(sorted_times) - 1)]
        else:
            metrics.p90_ms = active_times[0]

    # Per-project breakdown
    by_project: dict[str, list[int]] = {}
    for c in resolved:
        by_project.setdefault(c.project, []).append(c.active_time_ms)

    for proj, times in by_project.items():
        proj_metrics = VelocityMetrics(
            avg_ms=int(statistics.mean(times)),
            median_ms=int(statistics.median(times)),
            total_chains=sum(1 for c in chains if c.project == proj),
            resolved_count=len(times),
        )
        if len(times) >= 2:
            sorted_t = sorted(times)
            p90_idx = int(len(sorted_t) * 0.9)
            proj_metrics.p90_ms = sorted_t[min(p90_idx, len(sorted_t) - 1)]
        else:
            proj_metrics.p90_ms = times[0]
        metrics.by_project[proj] = proj_metrics

    return metrics
