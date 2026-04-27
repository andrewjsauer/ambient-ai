"""Insights aggregate: run every algorithmic detector, hand the result to an LLM.

CoachingData is the single top-level aggregate passed to the insights prompt.
Despite the name, it carries more than coaching — it is the full detector
snapshot over a window (prompt patterns, compression sequences, shell↔Claude
correlations, outcomes, velocity, stuck patterns). Name retained for import
stability; prefer importing this as the insights aggregate.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ambient.capture.reader import read_events
from ambient.config import Config
from ambient.detect._claude_session_walk import PromptRecord, walk_prompts
from ambient.detect.coaching import (
    CoachingFindings,
    StuckPatternFindings,
    classify_sessions,
    group_stuck_patterns,
)
from ambient.detect.command_mix import CommandMixFindings, detect_command_mix
from ambient.detect.compression import CompressionFindings, detect_compression
from ambient.detect.correlator import CorrelationFindings, correlate_signals
from ambient.detect.focus_events import (
    compute_attention_intervals,
    compute_context_switch_density,
    read_focus_events,
)
from ambient.detect.freeform_fraction import (
    FreeformFraction,
    detect_freeform_fraction,
)
from ambient.detect.vectors import (
    VectorFindings,
    detect_vectors,
    stop_reason_summary,
    top_vectors_per_project,
)
from ambient.detect.git_activity import (
    GitActivityFindings,
    detect_git_activity,
)
from ambient.detect.project_capabilities import clear_capability_cache
from ambient.detect.project_ledger import ProjectLedger, detect_project_ledger
from ambient.detect.prompt_patterns import (
    PromptPatternFindings,
    detect_prompt_patterns,
)
from ambient.detect.velocity import (
    VelocityMetrics,
    ResolutionChain,
    compute_velocity_metrics,
    detect_resolution_chains,
)
from ambient.detect.verification import (
    VerificationGapFindings,
    detect_verification_gaps,
)


logger = logging.getLogger(__name__)

INSIGHTS_INPUT_BUDGET = 30_000

# Per-section example caps. Sections suppress themselves when their evidence
# list is empty (see Unit 8); over-budget runs log a warning rather than
# shrink these caps. Phase 4's baseline anomaly gate will tighten further.
_DEFAULT_CAPS = {
    "prompts_within": 10,
    "prompts_cross": 5,
    "sequences": 5,
    "correlation_examples": 3,
    "resolution_chains_resolved": 5,
    "resolution_chains_unresolved": 5,
    "stuck_projects": 5,
    "stuck_tools": 5,
    "stuck_clusters": 5,
    "stuck_opening_prompts": 3,  # per-pattern cap for verbatim prompt quotes
    "verification_gaps": 8,
    "abandonment_examples": 3,  # per new closure_reason
    "pending_recs": 10,
    "git_commits_per_project": 8,  # cap commits cited per project in GIT ACTIVITY
    "vectors_per_project": 3,  # v4 Phase 3: longest-N vectors per project in VECTORS
}

INSIGHTS_SYSTEM = """You are a senior engineer reviewing a developer's week of work. The data is rich (real prompts, real commands, real file paths, real chains with failure-and-resolution context, real verification gaps bucketed by project capability, and a per-project git-activity ledger of what actually shipped). Your job is to produce a tight engineering review grounded in those specifics — not behavioral coaching, not generalities, not padding.

Sections to produce, in this order. Only emit a section when the input data contains a corresponding section. If the input has no data for a topic, omit that section entirely from your output. Do not write "no data" or "insufficient sample" filler.

1. **What Shipped** — when the GIT ACTIVITY section is present in the input, open the report by naming what actually shipped this window. Cite specific commit subjects and per-project commit counts. This is the denominator against which every later finding should be framed. When GIT ACTIVITY is absent (no commits in any project this window), omit this section like any other — do not write "no commits" filler.
2. **What You Worked On** — when the PROJECT LEDGER section is present, name the top projects by active time. For each, cite the active time, top files touched, and the one-line summary verbatim. Quote the user's distinctive wording from the recent prompts where it appears in the summary. Do not invent specifics not present in the input. When PROJECT LEDGER is absent, omit this section.
3. **Command Mix** — when the COMMAND MIX section is present, report the planning/execution/review ratios per project. Frame as signal, not prescription: a high freeform fraction means most work bypasses structured commands, which is a fact about how the developer works, not a problem to fix. When COMMAND MIX is absent, omit this section.
4. **Vectors** — when the VECTORS section is present, name the longest 1-3 vectors by duration and the dominant stop reason. Describe what the data says (e.g. "longest vector ended in /clear, suggesting context-rot"); do NOT invent hypotheses about vector content the data does not support. When VECTORS is absent, omit this section.
5. **Top Finding** — the single most-actionable engineering observation this window, in one paragraph. Cite at least one verbatim quote from the data (a prompt, command, file path, or commit). Frame it against shipped work when possible.
6. **Recurring Patterns** — the most-repeated prompts and command sequences. Quote the normalized pattern text verbatim. Note which projects they appear in.
7. **Stuck Episode Analysis** — patterns in the stuck sessions, broken out by project, by failing tool, and by file cluster. Quote tool names and file paths from the data. When the data includes opening prompts for stuck sessions, quote at least one verbatim.
8. **Verification Gaps** — when the data includes a verification-gap section, report the per-bucket rates ("projects with tests", "projects with typecheck/build only", "projects with no detected verification target"). The "neither" bucket is descriptive, not a gap — projects with no test or typecheck target cannot be verified by definition; do not frame those sessions as the developer skipping tests. When a bucket is low-sample, hedge or omit. Cite one specific example gap (session id, file edited, project).
9. **Resolution Velocity** — how quickly problems close. Cite avg/median/p90 and at least one example resolved chain. When abandoned chains have specific reason codes (interrupt_mid_thought, context_rot, given_up, end_of_window), narrate WHY chains aren't closing, not just that they aren't.
10. **Trend vs. Prior Week** — only when the period-comparison section is present in the input. State the deltas in plain numbers. Never invent a trend.
11. **Surprise of the Week** — exactly one non-obvious cross-signal observation the algorithmic detectors did NOT highlight directly. Must cross-reference two data sources (e.g., a prompt pattern AND a stuck file cluster, a command sequence AND a correlation pattern, or a commit AND a verification gap). Cite the specific data. If no real surprise exists, omit this section entirely. Do not invent a surprise to fill space.
12. **Diagnostic Questions** — conclude with exactly three questions a senior engineer would ask the developer after reading this report. Each question must reference a specific data point above (count, project, file, command, chain id, or commit) and be answerable in one sentence. Do not issue directives. Do not prescribe behavior changes. The goal is to force the developer's own synthesis, not to tell them what to do.

Pending recommendations, when present in the input, are surfaced in an appendix you do not need to write — the report renderer handles that separately.

Rules:
- Every claim about a pattern must be grounded in a verbatim quote from the data — a prompt, command, file path, or tool name. No generic phrasing like "you often type similar prompts" without a specific quoted example.
- When a metric is below sample threshold (marked "low sample" in the data), hedge explicitly or omit the claim. Do not report averages or trends on insufficient samples.
- Prefer concrete counts and durations over vague language. "You ran `pytest -x && git add` 14 times" beats "you often run this sequence".
- A section absent from the input means the corresponding finding has no support. Omit, do not acknowledge.
- A 100% rate on any metric is suspect — call it out and ask whether the detector is correct, rather than presenting it as a confident finding.
- Do not grade the developer's continuation prompts ("go for it", "yes lets fix these") as risk signals. These are normal acceptance responses; the agent's planning posture before editing is the relevant signal, not the developer's reply after.
- Length: aim for 600-900 words total. Density over breadth. Shorter when there is less to say.

Vocabulary — prefer these industry-standard terms when naming patterns, so language stays consistent across weeks:
- **prompt debt**: near-duplicate prompts accumulating across sessions — asking variants of the same question repeatedly.
- **verification gap**: a fix that shipped without a subsequent verifying command (test or typecheck/build, depending on project capability) proving it worked.
- **context rot**: a Claude session dominated by Read/Grep/ToolSearch calls with no Edit/Write — the agent hunted for context it couldn't find.
- **cognitive debt**: loss of comprehension from fast AI-generated output — code that ships without the developer understanding the full system.
- **interrupt mid-thought**: a session that ended with the agent blocked on a user confirmation (AskUserQuestion) rather than completing its work.
"""


@dataclass
class PeriodComparison:
    """Week-over-week deltas between current and prior windows.

    All deltas are `current - prior`: negative velocity_delta_ms means faster;
    negative stuck_delta means fewer stuck episodes. `insufficient_data_reason`
    is set when the gates are not met; in that case the deltas are None.
    """

    velocity_delta_ms: int | None = None
    stuck_delta: int | None = None
    thrash_delta: float | None = None
    new_patterns: list[str] = field(default_factory=list)  # top normalized prompts current∖prior
    dropped_patterns: list[str] = field(default_factory=list)  # top normalized prompts prior∖current
    insufficient_data_reason: str | None = None
    prior_date_range: str = ""


@dataclass
class CoachingData:
    """Top-level insights aggregate.

    Carries every detector output over the window so the LLM prompt can cite
    concrete examples. Name retained for import stability.
    """

    coaching_findings: CoachingFindings
    stuck_patterns: StuckPatternFindings
    velocity_metrics: VelocityMetrics
    chains: list[ResolutionChain]
    window_days: int
    date_range: str
    # Extended detector outputs — default-factory so old callers still work.
    prompt_patterns: PromptPatternFindings = field(
        default_factory=lambda: PromptPatternFindings(patterns=[], total_prompts=0)
    )
    compression: CompressionFindings = field(
        default_factory=lambda: CompressionFindings(sequences=[], compression_ratio=1.0)
    )
    correlations: CorrelationFindings = field(default_factory=CorrelationFindings)
    comparison: PeriodComparison | None = None
    pending_recommendations: list[dict] = field(default_factory=list)
    verification_gaps: VerificationGapFindings = field(
        default_factory=VerificationGapFindings
    )
    # v3 Unit 3: per-project shipped-work summary from `git log`. Surfaced as
    # the FIRST section of the prompt so every later finding has a denominator.
    git_activity: GitActivityFindings = field(default_factory=GitActivityFindings)
    # v4 Phase 1 Unit 4: per-project ledger of what the user worked on
    # (active time, top files, LLM-summarized one-liner). Optional — None when
    # the detector was skipped (e.g. test fixtures or aggregation failures).
    project_ledger: ProjectLedger | None = None
    # v4 Phase 1 Unit 2: per-project intent mix (planning/execution/review/...)
    command_mix: CommandMixFindings | None = None
    # v4 Phase 1 Unit 3: % of prompts with no slash command, plus prior delta.
    freeform_fraction: FreeformFraction | None = None
    # v4 Phase 1 Unit 5: render mode flag. When True, format_terminal_summary
    # emits a daily timeline view of project_ledger + command_mix instead of
    # the aggregate. The LLM prompt is unchanged regardless.
    by_day: bool = False
    # Raw events captured during aggregation; used by the by-day renderer to
    # bucket time per local day without re-reading the event log. None when
    # not yet populated (e.g. test fixtures); the by-day renderer treats None
    # the same as an empty list.
    events: list | None = None
    # v4 Phase 2 Unit 9: per-session focus-event density (switches per minute).
    # Maps Claude session_id → density. None when focus capture is off; empty
    # dict when capture is on but no events fell inside any session window.
    focus_density: dict[str, float] | None = None
    # v4 Phase 3: vector aggregation findings — stretches of activity terminated
    # by stop events (Enter, pause, focus_change, exit, end_of_window). None
    # when the detector was skipped or failed; renderer omits the section.
    vectors: VectorFindings | None = None


def _safe_run(fn, *args, default, label, **kwargs):
    """Run a detector, logging and substituting `default` on failure.

    Insights must never crash because one detector blew up on odd data.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("INSIGHTS_DETECTOR_FAILED detector=%s error=%s", label, exc)
        return default


_PATTERN_CHURN_TOP_N = 5
_COMPARISON_MIN_STUCK = 3


def _aggregate_window(
    config: Config,
    start: datetime,
    end: datetime,
    window_days: int,
    *,
    skip_phase1: bool = False,
    prior_prompts: list[PromptRecord] | None = None,
) -> CoachingData:
    """Run every detector over a date window and assemble a CoachingData.

    Does not populate `.comparison`; the caller fills that in if applicable.

    `skip_phase1=True` skips command_mix / freeform_fraction / project_ledger
    (used for the prior window — `compute_period_comparison` never reads them,
    so running Haiku summaries on a discarded ledger is wasted spend).

    `prior_prompts` (when provided alongside `skip_phase1=False`) is the
    materialized prompt list for the prior window so freeform_fraction can
    compute its delta without re-walking.
    """
    date_range = f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
    events = read_events(config, start=start, end=end)

    findings = classify_sessions(events, config)
    stuck = group_stuck_patterns(findings.outcomes, events, config)

    outcome_map = {o.session_id: o.classification for o in findings.outcomes}
    chains = detect_resolution_chains(events, config, session_outcomes=outcome_map)
    velocity = compute_velocity_metrics(chains, min_chains=config.velocity_min_chains)

    prompt_patterns = _safe_run(
        detect_prompt_patterns, events, config,
        default=PromptPatternFindings(patterns=[], total_prompts=0),
        label="prompt_patterns",
    )
    compression = _safe_run(
        detect_compression, events, config,
        default=CompressionFindings(sequences=[], compression_ratio=1.0),
        label="compression",
    )
    correlations = _safe_run(
        correlate_signals, events,
        default=CorrelationFindings(),
        label="correlator",
    )
    verification_gaps = _safe_run(
        detect_verification_gaps, events, config,
        default=VerificationGapFindings(),
        label="verification_gaps",
    )
    git_activity = _safe_run(
        detect_git_activity, events, start, end, config,
        default=GitActivityFindings(),
        label="git_activity",
    )

    # v4 Phase 1 detectors: walk ~/.claude/projects/*.jsonl ONCE for this window
    # and share the result across command_mix, freeform_fraction, and
    # project_ledger. Skipped entirely on the prior window — comparison consumes
    # only velocity/stuck/patterns from the prior call.
    if skip_phase1:
        command_mix = None
        freeform_fraction = None
        project_ledger = None
        vector_findings = None
    else:
        prompts = _safe_run(
            _walk_prompts_for_window, config.claude_projects_dir, start, end,
            default=[],
            label="phase1_walk",
        )
        # v4 Phase 2 Unit 9: read focus events for the window. When focus
        # capture is on, attention_intervals flips project_ledger time math
        # from "command_span" to "attention_weighted" and feeds context-
        # switch density into the coaching layer. When focus capture is off
        # (default), focus_events is empty and attention_intervals is None,
        # so behavior is identical to Phase 1.
        #
        # `start` from `aggregate_coaching_data` is naive local time;
        # `read_focus_events` normalizes naive bounds to UTC for the
        # comparison against UTC-aware event timestamps.
        focus_events = _safe_run(
            read_focus_events, config.focus_events_path,
            since_iso=start.isoformat(),
            default=[],
            label="focus_events_read",
        )
        attention_intervals = (
            compute_attention_intervals(focus_events, fallback_until=end)
            if focus_events else None
        )
        focus_density = _compute_focus_density(focus_events, events) if focus_events else None

        command_mix = _safe_run(
            detect_command_mix, config.claude_projects_dir, start, end, config,
            prompts=prompts,
            default=None,
            label="command_mix",
        )
        freeform_fraction = _safe_run(
            detect_freeform_fraction, config.claude_projects_dir, start, end, config,
            prompts=prompts, prior_prompts=prior_prompts,
            default=None,
            label="freeform_fraction",
        )
        project_ledger = _safe_run(
            detect_project_ledger, events, config.claude_projects_dir, start, end, config,
            prompts=prompts, attention_intervals=attention_intervals,
            default=None,
            label="project_ledger",
        )

        # v4 Phase 3: vector aggregation. Re-shapes existing signals into the
        # stop-point model. No new capture; reads events + focus_events + (if
        # available) the GMM pause classifier. Detector failures degrade to None.
        pauses = _safe_run(
            _detect_pauses_safe, events, config,
            default=None,
            label="vectors_pauses",
        )
        window_start_ms = int(start.timestamp() * 1000)
        window_end_ms = int(end.timestamp() * 1000)
        vector_findings = _safe_run(
            detect_vectors, events, focus_events, pauses,
            window_start_ms, window_end_ms, config,
            default=None,
            label="vectors",
        )

    return CoachingData(
        coaching_findings=findings,
        stuck_patterns=stuck,
        velocity_metrics=velocity,
        chains=chains,
        window_days=window_days,
        date_range=date_range,
        prompt_patterns=prompt_patterns,
        compression=compression,
        correlations=correlations,
        verification_gaps=verification_gaps,
        git_activity=git_activity,
        command_mix=command_mix,
        freeform_fraction=freeform_fraction,
        project_ledger=project_ledger,
        events=events,
        focus_density=focus_density if not skip_phase1 else None,
        vectors=vector_findings if not skip_phase1 else None,
    )


def _walk_prompts_for_window(
    claude_projects_dir, start: datetime, end: datetime
) -> list[PromptRecord]:
    """Materialize a single prompt walk for the window. Errors propagate to _safe_run."""
    return list(walk_prompts(claude_projects_dir, start, end))


def _detect_pauses_safe(events, config):
    """Run pause classification for vectors; returns None when unavailable.

    The pause classifier is calibrated lazily; its findings.available flag tells
    us whether the model has been fit. We treat any unavailability as 'no pause
    stops' rather than crashing the vector detector.
    """
    from ambient.detect.pauses import classify
    findings = classify(events, config)
    return findings if findings.available else None


def _compute_focus_density(focus_events, events) -> dict[str, float]:
    """Build session-id → switches-per-minute density from focus events + events.

    Session intervals are derived from claude_session events: each event's
    `claude_session_id`, ts_start, and duration_ms become a (session_id, start, end)
    tuple. Shell events have no session concept and are excluded.
    """
    from datetime import timezone as _tz
    intervals = []
    for ev in events or []:
        if ev.type != "claude_session":
            continue
        session_id = getattr(ev, "claude_session_id", None)
        if not session_id:
            continue
        start_dt = datetime.fromtimestamp(ev.ts_start / 1000, tz=_tz.utc)
        end_dt = datetime.fromtimestamp(
            (ev.ts_start + max(ev.duration_ms, 0)) / 1000, tz=_tz.utc,
        )
        intervals.append((session_id, start_dt, end_dt))
    return compute_context_switch_density(focus_events, intervals)


def compute_period_comparison(
    current: CoachingData, prior: CoachingData, config: Config
) -> PeriodComparison:
    """Diff two windows. Gated by velocity_min_chains + stuck floor on both sides."""
    cur_velocity = current.velocity_metrics
    pri_velocity = prior.velocity_metrics
    cur_stuck = current.stuck_patterns.total_stuck_sessions
    pri_stuck = prior.stuck_patterns.total_stuck_sessions

    if (
        cur_velocity.resolved_count < config.velocity_min_chains
        or pri_velocity.resolved_count < config.velocity_min_chains
    ):
        return PeriodComparison(
            insufficient_data_reason=(
                f"Each window needs at least {config.velocity_min_chains} resolved chains "
                f"(current: {cur_velocity.resolved_count}, prior: {pri_velocity.resolved_count})."
            ),
            prior_date_range=prior.date_range,
        )
    if cur_stuck < _COMPARISON_MIN_STUCK or pri_stuck < _COMPARISON_MIN_STUCK:
        return PeriodComparison(
            insufficient_data_reason=(
                f"Each window needs at least {_COMPARISON_MIN_STUCK} stuck sessions "
                f"(current: {cur_stuck}, prior: {pri_stuck})."
            ),
            prior_date_range=prior.date_range,
        )

    # Top-N normalized prompts for pattern churn (within_session scope only;
    # cross-session already captures multi-session repetition)
    def _top_prompts(data: CoachingData) -> set[str]:
        ranked = sorted(
            (p for p in data.prompt_patterns.patterns if p.scope == "within_session"),
            key=lambda p: p.count,
            reverse=True,
        )
        return {p.normalized_prompt for p in ranked[:_PATTERN_CHURN_TOP_N]}

    current_top = _top_prompts(current)
    prior_top = _top_prompts(prior)

    thrash_delta: float | None = None
    cur_thrash = current.coaching_findings.avg_thrash_score
    pri_thrash = prior.coaching_findings.avg_thrash_score
    if cur_thrash is not None and pri_thrash is not None:
        thrash_delta = cur_thrash - pri_thrash

    return PeriodComparison(
        velocity_delta_ms=cur_velocity.avg_ms - pri_velocity.avg_ms,
        stuck_delta=cur_stuck - pri_stuck,
        thrash_delta=thrash_delta,
        new_patterns=sorted(current_top - prior_top)[:_PATTERN_CHURN_TOP_N],
        dropped_patterns=sorted(prior_top - current_top)[:_PATTERN_CHURN_TOP_N],
        prior_date_range=prior.date_range,
    )


def aggregate_coaching_data(
    config: Config, window_days: int = 7, compare: bool = True
) -> CoachingData:
    # Reset the per-run project-capability cache so each insights run sees a
    # fresh probe (capabilities can change between runs as projects evolve).
    clear_capability_cache()

    end = datetime.now()
    start = end - timedelta(days=window_days)

    if compare:
        # Prior window: equal length, ending where current window starts.
        prior_end = start
        prior_start = prior_end - timedelta(days=window_days)
        # Materialize the prior prompt walk once; share with current's
        # freeform_fraction so its delta computation doesn't re-walk.
        prior_prompts = _safe_run(
            _walk_prompts_for_window, config.claude_projects_dir, prior_start, prior_end,
            default=[],
            label="phase1_walk_prior",
        )
        # Phase 1 detectors are skipped on the prior aggregation — comparison
        # never reads them, so running per-project Haiku summaries on a
        # throwaway ledger would just waste spend.
        prior = _aggregate_window(
            config, prior_start, prior_end, window_days, skip_phase1=True
        )
        current = _aggregate_window(
            config, start, end, window_days, prior_prompts=prior_prompts
        )
        current.comparison = compute_period_comparison(current, prior, config)
    else:
        current = _aggregate_window(config, start, end, window_days)

    # Pending recommendations — staged by the daemon, surfaced here for
    # visibility in the insights report. Deferred import to avoid a cycle
    # if recommender ever imports insights.
    from ambient.present.recommender import list_pending_recommendations

    current.pending_recommendations = _safe_run(
        list_pending_recommendations, config,
        default=[],
        label="pending_recommendations",
    )

    return current


def _truncate(text: str, max_len: int = 140) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _section_recurring_prompts(data: CoachingData, caps: dict) -> list[str]:
    patterns = data.prompt_patterns.patterns
    within = sorted(
        (p for p in patterns if p.scope == "within_session"),
        key=lambda p: p.count, reverse=True,
    )[: caps["prompts_within"]]
    cross = sorted(
        (p for p in patterns if p.scope == "cross_session"),
        key=lambda p: p.count, reverse=True,
    )[: caps["prompts_cross"]]

    # Dedupe: drop within-session patterns whose normalized text also appears cross-session
    cross_norms = {p.normalized_prompt for p in cross}
    within = [p for p in within if p.normalized_prompt not in cross_norms]

    if not within and not cross:
        return []
    lines = [f"\nRECURRING PROMPTS ({data.prompt_patterns.total_prompts} prompts analyzed):"]

    if within:
        lines.append("  Within-session (same session, repeated):")
        for p in within:
            proj = f" [{', '.join(p.projects)}]" if p.projects else ""
            example = _truncate(p.raw_examples[0]) if p.raw_examples else p.normalized_prompt
            lines.append(f"    x{p.count} \"{_truncate(p.normalized_prompt)}\"{proj}")
            if example and example != p.normalized_prompt:
                lines.append(f"      example: \"{example}\"")
    if cross:
        lines.append("  Cross-session (same project, different sessions):")
        for p in cross:
            proj = f" [{', '.join(p.projects)}]" if p.projects else ""
            lines.append(f"    x{p.count} \"{_truncate(p.normalized_prompt)}\"{proj}")
    return lines


def _section_command_sequences(data: CoachingData, caps: dict) -> list[str]:
    sequences = data.compression.sequences[: caps["sequences"]]
    if not sequences:
        return []
    lines = ["\nRECURRING COMMAND SEQUENCES:"]
    for s in sequences:
        seq_text = " -> ".join(_truncate(c, 60) for c in s.sequence)
        total_min = s.total_time_ms / 60000
        lines.append(
            f"    x{s.count} {seq_text}  (total {total_min:.1f} min, gain {s.compression_gain})"
        )
    return lines


def _section_correlations(data: CoachingData, caps: dict) -> list[str]:
    if not data.correlations.patterns:
        return []
    lines = ["\nSHELL ↔ CLAUDE CORRELATIONS:"]
    for p in data.correlations.patterns:
        lines.append(f"  {p.pattern_type}: {p.count} occurrences")
        for ex in p.examples[: caps["correlation_examples"]]:
            cmd = _truncate(str(ex.get("command", "")), 100)
            gap = ex.get("gap_ms", 0) or 0
            gap_s = gap / 1000
            lines.append(f"    - \"{cmd}\" (gap {gap_s:.0f}s)")
    return lines


def _section_resolution_chains(data: CoachingData, caps: dict) -> list[str]:
    resolved = [c for c in data.chains if c.closure_reason == "matched_success"]
    unresolved = [c for c in data.chains if c.closure_reason != "matched_success"]
    resolved.sort(key=lambda c: c.active_time_ms)  # fastest first
    unresolved.sort(key=lambda c: c.active_time_ms, reverse=True)  # longest first
    resolved = resolved[: caps["resolution_chains_resolved"]]
    unresolved = unresolved[: caps["resolution_chains_unresolved"]]

    if not data.chains:
        return []
    lines = ["\nTOP RESOLUTION CHAINS:"]
    if resolved:
        lines.append("  Resolved (fastest first):")
        for c in resolved:
            prompt = f" prompt: \"{_truncate(c.first_claude_prompt, 80)}\"" if c.first_claude_prompt else ""
            lines.append(
                f"    [{c.project}] \"{_truncate(c.initial_command, 60)}\" "
                f"-> \"{_truncate(c.resolution_command, 60)}\" "
                f"({c.active_time_ms / 60000:.1f} min active){prompt}"
            )
    if unresolved:
        lines.append("  Unresolved (longest first):")
        for c in unresolved:
            prompt = f" prompt: \"{_truncate(c.first_claude_prompt, 80)}\"" if c.first_claude_prompt else ""
            lines.append(
                f"    [{c.project}] \"{_truncate(c.initial_command, 60)}\" "
                f"closed by {c.closure_reason} "
                f"({c.active_time_ms / 60000:.1f} min active){prompt}"
            )
    return lines


def _section_git_activity(data: CoachingData, caps: dict) -> list[str]:
    ga = data.git_activity
    if ga.total_commits == 0:
        return []
    lines = [
        f"\nGIT ACTIVITY (what shipped this window — {ga.total_commits} "
        f"commits across {len(ga.by_project)} project(s), "
        f"{ga.total_lines_changed} lines changed):"
    ]
    cap = caps["git_commits_per_project"]
    # Order projects by total commits descending so the most-active project
    # leads. Within a project, commits are already newest-first.
    ordered = sorted(
        ga.by_project.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )
    for project, commits in ordered:
        proj_lines = sum(c.insertions + c.deletions for c in commits)
        lines.append(
            f"  [{project}] {len(commits)} commit(s), {proj_lines} lines changed:"
        )
        for c in commits[:cap]:
            lines.append(
                f"    {c.sha[:8]} {_truncate(c.subject, 80)} "
                f"(+{c.insertions}/-{c.deletions}, {c.files_changed} file(s))"
            )
        if len(commits) > cap:
            lines.append(f"    ... and {len(commits) - cap} more commit(s)")
    return lines


def _section_project_ledger(data: CoachingData, caps: dict) -> list[str]:
    """v4 Phase 1 Unit 5: render the per-project ledger (active time + summary)."""
    pl = data.project_ledger
    if pl is None or not pl.entries:
        return []
    time_label = "active time" if pl.time_basis == "attention_weighted" else "command-span time"
    lines = [f"\nPROJECT LEDGER (what you worked on; {time_label}):"]
    for entry in pl.entries:
        minutes = entry.active_ms // 60_000
        if minutes >= 60:
            time_str = f"{minutes / 60:.1f}h"
        else:
            time_str = f"{minutes}min"
        sessions = f"{entry.session_count} session(s)"
        files_str = ""
        if entry.top_files:
            files_str = f", top files: {', '.join(entry.top_files[:5])}"
        lines.append(f"  [{entry.project}] {time_str} · {sessions}{files_str}")
        if entry.summary:
            lines.append(f"    summary: {entry.summary}")
    return lines


def _section_command_mix(data: CoachingData, caps: dict) -> list[str]:
    """v4 Phase 1 Unit 5: render planning/execution/review ratios."""
    cm = data.command_mix
    if cm is None or cm.overall.total == 0:
        return []
    lines = ["\nCOMMAND MIX (intent ratios; signal not prescription):"]
    overall_ratios = cm.overall.ratios
    lines.append(
        "  Overall: "
        f"plan {overall_ratios['planning'] * 100:.0f}% / "
        f"exec {overall_ratios['execution'] * 100:.0f}% / "
        f"review {overall_ratios['review'] * 100:.0f}% / "
        f"freeform {overall_ratios['freeform'] * 100:.0f}% "
        f"(of {cm.overall.total} prompts)"
    )
    for project, mix in sorted(cm.per_project.items(), key=lambda kv: kv[1].total, reverse=True):
        ratios = mix.ratios
        lines.append(
            f"  [{project}] "
            f"plan {ratios['planning'] * 100:.0f}% / "
            f"exec {ratios['execution'] * 100:.0f}% / "
            f"review {ratios['review'] * 100:.0f}% / "
            f"freeform {ratios['freeform'] * 100:.0f}% "
            f"(of {mix.total} prompts)"
        )
    return lines


def _section_vectors(data: CoachingData, caps: dict) -> list[str]:
    """v4 Phase 3: render the longest vectors per project + stop-reason mix.

    Empty when vectors is None or no vectors. Per Phase 3 → Phase 4 gate, the
    deeper LLM-narrative description in INSIGHTS_SYSTEM is intentionally
    minimal in this first cut — the user must judge the section useful on a
    real weekly run before the wider framing lands.
    """
    vf = data.vectors
    if vf is None or not vf.vectors:
        return []
    n_per_project = caps.get("vectors_per_project", 3)

    lines = [
        f"\nVECTORS ({len(vf.vectors)} stretches of activity terminated by stop events):"
    ]

    # Stop-reason mix line.
    summary = stop_reason_summary(vf)
    total_count = sum(c for _, c, _ in summary)
    total_dur = sum(d for _, _, d in summary)
    if total_count > 0 and total_dur > 0:
        mix_chunks = []
        for reason, count, dur in summary:
            pct = dur / total_dur * 100 if total_dur else 0
            mix_chunks.append(f"{reason} {pct:.0f}% ({count} vectors, {dur // 60_000}min)")
        lines.append("  Stop-reason mix: " + " · ".join(mix_chunks))

    # Per-project top-N longest vectors.
    by_project = top_vectors_per_project(vf, n=n_per_project)
    # Sort projects by their longest vector's duration desc.
    projects_ordered = sorted(
        by_project.items(),
        key=lambda kv: kv[1][0].duration_ms if kv[1] else 0,
        reverse=True,
    )
    for project, vectors in projects_ordered:
        if not vectors:
            continue
        chunks = []
        for v in vectors:
            mins = v.duration_ms // 60_000
            time_str = f"{v.duration_ms / 60_000:.1f}min"
            text = _truncate(v.last_command_or_prompt, 50)
            chunks.append(f"{time_str} ({v.stop_reason}) {text}")
        lines.append(f"  [{project}] longest {len(vectors)}: " + " · ".join(chunks))
    return lines


def _section_freeform_fraction(data: CoachingData, caps: dict) -> list[str]:
    """v4 Phase 1 Unit 5: render % of prompts with no slash command + delta."""
    ff = data.freeform_fraction
    if ff is None or ff.total_prompts == 0:
        return []
    pct = ff.overall_pct * 100
    delta_str = ""
    if ff.delta_pct is not None:
        sign = "+" if ff.delta_pct >= 0 else ""
        delta_str = f" ({sign}{ff.delta_pct * 100:.1f}pp vs prior)"
    lines = [
        f"\nFREEFORM FRACTION ({pct:.1f}% of {ff.total_prompts} prompts had no slash command{delta_str}):"
    ]
    if ff.per_project:
        for project, project_pct in sorted(ff.per_project.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"  [{project}] {project_pct * 100:.0f}%")
    return lines


def _section_session_outcomes(data: CoachingData) -> list[str]:
    counts = data.coaching_findings.count_by_classification
    total = sum(counts.values())
    if total == 0:
        return []
    lines = [f"\nSESSION OUTCOMES ({total} sessions):"]
    for cls in ("productive", "friction", "quick", "abandoned"):
        n = counts.get(cls, 0)
        pct = n / total * 100 if total else 0
        lines.append(f"  {cls}: {n} ({pct:.0f}%)")
    avg_thrash = data.coaching_findings.avg_thrash_score
    if avg_thrash is not None:
        lines.append(f"  Average thrash score: {avg_thrash:.2f}")
    elif data.coaching_findings.low_sample:
        lines.append("  Average thrash score: insufficient sample")

    # v4 Phase 2 Unit 9: context-switch density per outcome class. Only emit
    # when focus capture is on AND at least one session had focus events.
    if data.focus_density:
        density_by_class: dict[str, list[float]] = {}
        for outcome in data.coaching_findings.outcomes:
            d = data.focus_density.get(outcome.session_id)
            if d is None:
                continue
            density_by_class.setdefault(outcome.classification, []).append(d)
        if density_by_class:
            lines.append("  Context-switch density (focus events/min):")
            for cls in ("productive", "friction", "quick", "abandoned"):
                xs = density_by_class.get(cls, [])
                if xs:
                    avg = sum(xs) / len(xs)
                    lines.append(f"    {cls}: {avg:.1f}/min (n={len(xs)})")
    return lines


def _format_opening_prompts(prompts: list[str], cap: int) -> list[str]:
    """Render up to `cap` verbatim opening prompts as indented quoted lines."""
    if not prompts:
        return []
    return [f"    opening: \"{_truncate(p, 120)}\"" for p in prompts[:cap]]


def _section_stuck_by_project(data: CoachingData, caps: dict) -> list[str]:
    sp = data.stuck_patterns
    if not sp.patterns:
        return []
    lines = [f"\nSTUCK PATTERNS — BY PROJECT ({sp.total_stuck_sessions} stuck sessions):"]
    for p in sp.patterns[: caps["stuck_projects"]]:
        thrash_str = (
            f"avg thrash {p.avg_thrash_score:.2f}"
            if p.avg_thrash_score is not None
            else "thrash N/A (low sample)"
        )
        lines.append(
            f"  {p.project}: {p.episode_count} episodes, {thrash_str}, "
            f"tools: {', '.join(p.failing_tools)}, "
            f"total time: {p.total_duration_ms / 60000:.1f}min"
        )
        if p.file_cluster and p.file_cluster != ["unknown"]:
            lines.append(f"    files: {', '.join(p.file_cluster[:5])}")
        lines.extend(_format_opening_prompts(p.opening_prompts, caps["stuck_opening_prompts"]))
    return lines


def _section_stuck_by_tool(data: CoachingData, caps: dict) -> list[str]:
    tools = data.stuck_patterns.tool_level_patterns[: caps["stuck_tools"]]
    if not tools:
        return []
    lines = ["\nSTUCK PATTERNS — BY FAILING TOOL:"]
    for t in tools:
        thrash_str = (
            f"avg thrash {t.avg_thrash_score:.2f}"
            if t.avg_thrash_score is not None
            else "thrash N/A (low sample)"
        )
        lines.append(
            f"  {t.tool_name}: {t.episode_count} stuck sessions across "
            f"{len(t.projects)} project(s) [{', '.join(t.projects)}], "
            f"{thrash_str}, total time: {t.total_duration_ms / 60000:.1f}min"
        )
        lines.extend(_format_opening_prompts(t.opening_prompts, caps["stuck_opening_prompts"]))
    return lines


def _section_stuck_by_cluster(data: CoachingData, caps: dict) -> list[str]:
    clusters = data.stuck_patterns.file_cluster_patterns[: caps["stuck_clusters"]]
    if not clusters:
        return []
    lines = ["\nSTUCK PATTERNS — BY FILE CLUSTER:"]
    for c in clusters:
        lines.append(
            f"  {c.path_fragment}: {c.episode_count} stuck sessions "
            f"[{', '.join(c.projects)}], tools: {', '.join(c.failing_tools)}, "
            f"total time: {c.total_duration_ms / 60000:.1f}min"
        )
        lines.extend(_format_opening_prompts(c.opening_prompts, caps["stuck_opening_prompts"]))
    return lines


_BUCKET_LABELS = {
    "has_tests": "projects with tests",
    "has_typecheck": "projects with typecheck/build only",
    "neither": "projects with no detected verification target",
}


def _section_verification_gaps(data: CoachingData, caps: dict) -> list[str]:
    vg = data.verification_gaps
    if vg.total_fix_sessions == 0:
        return []
    # Suppress the section header entirely when only `neither`-bucket sessions
    # exist — those projects have no verification capability so framing them as
    # "verification gaps" is a category error. The terminal summary surfaces
    # the count under "Fixes in non-verifiable projects" instead.
    verifiable = (
        vg.total_fix_sessions_by_bucket.get("has_tests", 0)
        + vg.total_fix_sessions_by_bucket.get("has_typecheck", 0)
    )
    if verifiable == 0:
        return []
    lines = [
        "\nVERIFICATION GAPS (fixes not followed by a verifying command,"
        " bucketed by project capability):"
    ]

    # Per-bucket headlines so readers see which projects can actually be
    # verified at all, separately from which ones skipped verification.
    for bucket in ("has_tests", "has_typecheck", "neither"):
        total = vg.total_fix_sessions_by_bucket.get(bucket, 0)
        if total == 0:
            continue
        bucket_gaps = vg.gaps_by_bucket.get(bucket, 0)
        label = _BUCKET_LABELS[bucket]
        if bucket == "neither":
            lines.append(
                f"  {label}: {total} fix session(s) — no verification possible"
                " (no test or typecheck target detected)"
            )
            continue
        if vg.low_sample_by_bucket.get(bucket, False):
            lines.append(
                f"  {label}: {bucket_gaps} gap(s) of {total} fix session(s)"
                " (low sample — no rate published)"
            )
        else:
            rate = vg.gap_rate_by_bucket.get(bucket)
            pct = (rate or 0) * 100
            lines.append(
                f"  {label}: {bucket_gaps} of {total} fix session(s) shipped"
                f" without a verifying command ({pct:.0f}%)"
            )

    # Example gaps, capped. Annotate each with its bucket so the reader can
    # tell test-skipped from no-target-exists at a glance.
    for g in vg.gaps[: caps["verification_gaps"]]:
        files_preview = ", ".join(g.edited_files[:3]) if g.edited_files else "(no files)"
        bucket_note = (
            "" if g.bucket == "has_tests"
            else f" [{g.bucket}]"
        )
        lines.append(
            f"    [{g.project}]{bucket_note} session {g.session_id[:12]}…"
            f" edited: {files_preview}"
        )
    return lines


def _section_abandonment_reasons(data: CoachingData, caps: dict) -> list[str]:
    v = data.velocity_metrics
    by_reason = v.by_reason or {}
    if not by_reason:
        return []
    lines = ["\nABANDONMENT BY REASON:"]
    # Order: specific idle codes first, then end_of_window, then matched_success
    order = (
        "matched_success",
        "interrupt_mid_thought",
        "context_rot",
        "given_up",
        "end_of_window",
    )
    for key in order:
        count = by_reason.get(key, 0)
        if count:
            lines.append(f"  {key}: {count}")
    # Example chains per new idle reason
    for key in ("interrupt_mid_thought", "context_rot", "given_up"):
        examples = [
            c for c in data.chains if c.closure_reason == key
        ][: caps["abandonment_examples"]]
        if examples:
            lines.append(f"  Examples — {key}:")
            for c in examples:
                prompt = (
                    f" prompt: \"{_truncate(c.first_claude_prompt, 80)}\""
                    if c.first_claude_prompt else ""
                )
                lines.append(
                    f"    [{c.project}] \"{_truncate(c.initial_command, 60)}\" "
                    f"({c.active_time_ms / 60000:.1f} min active){prompt}"
                )
    return lines


def _section_velocity(data: CoachingData) -> list[str]:
    v = data.velocity_metrics
    if v.total_chains == 0:
        return []
    lines = [f"\nRESOLUTION VELOCITY ({v.total_chains} chains, {v.resolved_count} resolved):"]
    if v.by_reason:
        matched = v.by_reason.get("matched_success", 0)
        idle = v.by_reason.get("idle_break", 0)
        eow = v.by_reason.get("end_of_window", 0)
        lines.append(
            f"  Closure reasons: matched-success={matched}, "
            f"idle-break={idle}, end-of-window={eow}"
        )
    if v.resolved_count > 0:
        lines.append(f"  Average active time: {v.avg_ms / 1000:.0f}s ({v.avg_ms / 60000:.1f}min)")
        lines.append(f"  Median: {v.median_ms / 1000:.0f}s")
        lines.append(f"  p90: {v.p90_ms / 1000:.0f}s")
        if v.by_project:
            lines.append("  By project:")
            for proj, pm in v.by_project.items():
                lines.append(f"    {proj}: avg {pm.avg_ms / 1000:.0f}s, {pm.resolved_count} resolved")
    else:
        lines.append("  No resolved chains in this window.")
    return lines


def _section_period_comparison(data: CoachingData) -> list[str]:
    c = data.comparison
    # Suppress entirely when there's nothing real to compare. The narrative
    # layer is instructed to stay silent on trends rather than pad with
    # "insufficient data" prose.
    if c is None or c.insufficient_data_reason:
        return []
    lines = ["\nPERIOD COMPARISON (current vs. prior equal-length window):"]
    lines.append(f"  Prior window: {c.prior_date_range}")
    if c.velocity_delta_ms is not None:
        direction = "faster" if c.velocity_delta_ms < 0 else "slower"
        lines.append(
            f"  Velocity delta: {c.velocity_delta_ms / 60000:+.1f} min "
            f"({direction} vs prior)"
        )
    if c.stuck_delta is not None:
        lines.append(f"  Stuck-session delta: {c.stuck_delta:+d}")
    if c.thrash_delta is not None:
        lines.append(f"  Thrash-score delta: {c.thrash_delta:+.2f}")
    if c.new_patterns:
        lines.append(f"  New top patterns this week: {', '.join(repr(p) for p in c.new_patterns)}")
    if c.dropped_patterns:
        lines.append(f"  Dropped top patterns: {', '.join(repr(p) for p in c.dropped_patterns)}")
    return lines


def _section_pending_recommendations(data: CoachingData, caps: dict) -> list[str]:
    recs = data.pending_recommendations[: caps["pending_recs"]]
    if not recs:
        return []
    lines = ["\nPENDING RECOMMENDATIONS (staged in ~/.ambient/recommendations/):"]
    for r in recs:
        lines.append(f"  [{r.get('type', 'unknown')}] {r.get('id')} — {r.get('title', '')}")
    return lines


def build_insights_prompt(data: CoachingData, caps: dict | None = None) -> str:
    caps = caps or _DEFAULT_CAPS
    sections: list[str] = [
        f"COACHING DATA — {data.date_range} ({data.window_days}-day window)"
    ]
    # Git activity FIRST so every later finding has a denominator. The LLM
    # is instructed to anchor its top finding against shipped work.
    sections += _section_git_activity(data, caps)
    # v4 Phase 1: project ledger → command mix → freeform fraction, in that order,
    # right after "what shipped" so they share the same opening framing.
    sections += _section_project_ledger(data, caps)
    sections += _section_command_mix(data, caps)
    sections += _section_freeform_fraction(data, caps)
    # v4 Phase 3: vector aggregation. Empty until focus capture is on long
    # enough to produce real stop events; renderer omits cleanly when so.
    sections += _section_vectors(data, caps)
    sections += _section_session_outcomes(data)
    sections += _section_recurring_prompts(data, caps)
    sections += _section_command_sequences(data, caps)
    sections += _section_correlations(data, caps)
    sections += _section_resolution_chains(data, caps)
    sections += _section_stuck_by_project(data, caps)
    sections += _section_stuck_by_tool(data, caps)
    sections += _section_stuck_by_cluster(data, caps)
    sections += _section_verification_gaps(data, caps)
    sections += _section_velocity(data)
    sections += _section_abandonment_reasons(data, caps)
    sections += _section_period_comparison(data)
    sections += _section_pending_recommendations(data, caps)
    return "\n".join(sections)


def _delta_suffix(current: float | int, delta: float | int | None, *, unit: str = "") -> str:
    """Format a value with an optional week-over-week delta suffix."""
    if delta is None:
        return ""
    sign = "+" if delta >= 0 else ""
    if unit == "min":
        return f"  ({sign}{delta / 60000:.1f} min vs prior)"
    return f"  ({sign}{delta:g}{unit} vs prior)"


def _format_by_day_summary(data: CoachingData) -> str:
    """v4 Phase 1 Unit 5: render the window as a per-day timeline.

    Buckets events from `data.events` by local calendar date, names the most-
    active projects per day. Falls back to the aggregate render with a one-line
    note when the window is shorter than 2 days (per-day view is meaningless
    for ≤24h windows).

    Bucketing keys on the `date` object derived from event.ts_start so windows
    spanning a year boundary (e.g. Dec 28 → Jan 3) sort chronologically. The
    earlier implementation used a `str.strftime("%a %b %d")` key and re-parsed
    it with the current year, which mis-sorted year-boundary windows and would
    have raised on Feb-29 of a non-leap year.
    """
    from datetime import date as _date

    lines = [f"Ambient Insights — {data.date_range} (--by-day)\n"]
    if data.window_days < 2:
        lines.append("(window < 2 days; --by-day falls back to aggregate)")
        from dataclasses import replace
        aggregate = replace(data, by_day=False)
        return "\n".join(lines) + "\n" + format_terminal_summary(aggregate)

    events = data.events or []
    if not events:
        lines.append("No events recorded in this window.")
        return "\n".join(lines)

    from collections import defaultdict
    daily_ms: dict[_date, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for event in events:
        if event.duration_ms <= 0:
            continue
        day = datetime.fromtimestamp(event.ts_start / 1000).date()
        project = _project_label_for_event(event)
        daily_ms[day][project] += event.duration_ms

    if not daily_ms:
        lines.append("No qualifying events in this window.")
        return "\n".join(lines)

    for day in sorted(daily_ms.keys()):
        per_project = sorted(daily_ms[day].items(), key=lambda kv: kv[1], reverse=True)
        # Drop projects under 5 minutes from the daily line — too noisy.
        meaningful = [(p, ms) for p, ms in per_project if ms >= 5 * 60_000]
        if not meaningful:
            continue
        chunks = []
        for project, ms in meaningful:
            mins = ms // 60_000
            chunks.append(f"{project} ({mins / 60:.1f}h)" if mins >= 60 else f"{project} ({mins}min)")
        day_label = day.strftime("%a %b %d")
        lines.append(f"  {day_label}:  " + " → ".join(chunks))

    return "\n".join(lines)


def _project_label_for_event(event) -> str:
    from pathlib import PurePosixPath
    if event.type == "claude_session" and event.claude_project:
        return PurePosixPath(event.claude_project).name or "unknown"
    if event.cwd:
        name = PurePosixPath(event.cwd).name
        if name in ("", "~", "/", "tmp"):
            return "unknown"
        return name
    return "unknown"


def format_terminal_summary(data: CoachingData) -> str:
    if data.by_day:
        return _format_by_day_summary(data)

    lines = [f"Ambient Insights — {data.date_range}\n"]

    # What shipped — anchors the rest of the summary in real work.
    ga = data.git_activity
    if ga.total_commits > 0:
        lines.append(
            f"Shipped:              {ga.total_commits} commit(s) across "
            f"{len(ga.by_project)} project(s), {ga.total_lines_changed} lines changed"
        )

    # What you worked on — top 3 projects by active time, with summary lines.
    pl = data.project_ledger
    if pl is not None and pl.entries:
        for entry in pl.entries[:3]:
            minutes = entry.active_ms // 60_000
            time_str = f"{minutes / 60:.1f}h" if minutes >= 60 else f"{minutes}min"
            line = f"Worked on:            [{entry.project}] {time_str} · {entry.session_count} session(s)"
            if entry.summary:
                line += f"\n                      {entry.summary}"
            lines.append(line)

    # Freeform fraction — single-line headline with delta.
    ff = data.freeform_fraction
    if ff is not None and ff.total_prompts > 0:
        delta_str = ""
        if ff.delta_pct is not None:
            sign = "+" if ff.delta_pct >= 0 else ""
            delta_str = f"  ({sign}{ff.delta_pct * 100:.1f}pp vs prior)"
        lines.append(f"Freeform fraction:    {ff.overall_pct * 100:.0f}% of prompts{delta_str}")

    # Treat a comparison with insufficient_data_reason as no comparison at
    # all, matching _section_period_comparison's behavior. Otherwise the
    # terminal summary can render delta suffixes computed from data the
    # prompt-builder explicitly suppressed.
    cmp = data.comparison
    if cmp is not None and cmp.insufficient_data_reason:
        cmp = None

    v = data.velocity_metrics
    if v.resolved_count > 0:
        velocity_delta = cmp.velocity_delta_ms if cmp and cmp.velocity_delta_ms is not None else None
        lines.append(
            f"Resolution velocity:  {v.avg_ms / 60000:.1f} min avg "
            f"({v.resolved_count} resolved)"
            + _delta_suffix(v.avg_ms, velocity_delta, unit="min")
        )
    else:
        lines.append("Resolution velocity:  no resolved chains")

    sp = data.stuck_patterns
    stuck_delta = cmp.stuck_delta if cmp and cmp.stuck_delta is not None else None
    lines.append(
        f"Stuck episodes:       {sp.total_stuck_sessions}"
        + _delta_suffix(sp.total_stuck_sessions, stuck_delta)
    )

    avg = data.coaching_findings.avg_thrash_score
    lines.append(
        f"Thrash score:         {avg:.2f} avg"
        if avg is not None
        else "Thrash score:         n/a"
    )

    # Top repeating signals — one-liners the user can scan at a glance.
    within = [p for p in data.prompt_patterns.patterns if p.scope == "within_session"]
    if within:
        top_prompt = max(within, key=lambda p: p.count)
        lines.append(
            f"Top repeated prompt:  x{top_prompt.count} \"{_truncate(top_prompt.normalized_prompt, 60)}\""
        )
    if data.compression.sequences:
        top_seq = data.compression.sequences[0]
        seq_text = " -> ".join(_truncate(c, 30) for c in top_seq.sequence[:3])
        lines.append(f"Top command sequence: x{top_seq.count} {seq_text}")

    lines.append(f"Pending recs:         {len(data.pending_recommendations)}")

    # Verification gaps — render per bucket so all-neither workloads don't
    # surface as "100% verification gap" (the v2 failure mode this whole
    # phase exists to fix).
    vg = data.verification_gaps
    for bucket, label in (("has_tests", "Verification gaps (tests)"),
                          ("has_typecheck", "Verification gaps (typecheck)")):
        total = vg.total_fix_sessions_by_bucket.get(bucket, 0)
        if total == 0:
            continue
        bucket_gaps = vg.gaps_by_bucket.get(bucket, 0)
        if vg.low_sample_by_bucket.get(bucket, False):
            lines.append(f"{label}: {bucket_gaps}/{total} fixes (low sample)")
        else:
            rate = vg.gap_rate_by_bucket.get(bucket) or 0
            lines.append(f"{label}: {bucket_gaps}/{total} fixes ({rate * 100:.0f}%)")
    neither_total = vg.total_fix_sessions_by_bucket.get("neither", 0)
    if neither_total > 0:
        lines.append(
            f"Fixes in non-verifiable projects: {neither_total} "
            "(no test or typecheck target detected)"
        )

    # Top trigger prompt — the opening prompt of the longest stuck episode, if any
    if sp.patterns and sp.patterns[0].opening_prompts:
        trigger = sp.patterns[0].opening_prompts[0]
        lines.append(f"Top trigger prompt:   \"{_truncate(trigger, 60)}\"")

    if sp.patterns:
        top = sp.patterns[0]
        lines.append(
            f"\nTop finding: {top.project} — {top.episode_count} stuck episodes "
            f"({top.total_duration_ms / 60000:.0f} min total)"
        )
    else:
        lines.append("\nNo significant stuck patterns detected.")

    return "\n".join(lines)


def generate_insights_report(data: CoachingData, config: Config, client=None) -> str | None:
    from ambient.present.tokens import estimate_tokens

    caps = dict(_DEFAULT_CAPS)
    prompt = build_insights_prompt(data, caps)
    estimated = estimate_tokens(INSIGHTS_SYSTEM) + estimate_tokens(prompt)

    # Confidence gates suppress weak sections at assembly time, so a runaway
    # prompt is now a sign of real signal volume rather than padding. Log a
    # warning if we exceed the soft budget; do not silently shrink strong
    # sections to fit. Phase 4's baseline anomaly gate will tighten further.
    if estimated > INSIGHTS_INPUT_BUDGET:
        logger.warning(
            "INSIGHTS_PROMPT_OVER_BUDGET estimated=%d budget=%d "
            "(no shrink applied; confidence gates handle suppression)",
            estimated, INSIGHTS_INPUT_BUDGET,
        )

    try:
        from ambient.present.api import call_api
        narrative = call_api(config, INSIGHTS_SYSTEM, prompt, config.sonnet_model,
                            max_tokens=3000, client=client)
    except Exception as exc:
        logger.warning("INSIGHTS_NARRATIVE_FAILED error=%s", exc)
        return None

    date_str = datetime.now().strftime("%Y-%m-%d")
    path = config.insights_path(date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(narrative + "\n")

    return narrative
