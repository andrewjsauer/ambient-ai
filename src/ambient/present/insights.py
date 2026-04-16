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
from ambient.detect.coaching import (
    CoachingFindings,
    StuckPatternFindings,
    classify_sessions,
    group_stuck_patterns,
)
from ambient.detect.compression import CompressionFindings, detect_compression
from ambient.detect.correlator import CorrelationFindings, correlate_signals
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


logger = logging.getLogger(__name__)

INSIGHTS_INPUT_BUDGET = 30_000

# Per-section example caps, scaled down proportionally when over budget
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
    "pending_recs": 10,
}

INSIGHTS_SYSTEM = """You are a developer-workflow analyst writing a weekly coaching report.

The data you receive includes raw examples: actual prompts the developer typed, actual shell commands they ran, actual file paths they touched, actual resolution chains with the initial failing command and Claude's opening prompt. Your job is to write a coaching report grounded in those specifics — not generalities.

Report sections to produce:
1. **Top Finding** — the single most-actionable observation this week, in one paragraph. Cite at least one verbatim quote from the data (a prompt, a command, or a file path).
2. **Recurring Patterns** — the most-repeated prompts, command sequences, and workflows. Quote the normalized pattern text verbatim. Note which projects they appear in.
3. **Stuck Episode Analysis** — patterns in the stuck sessions, broken out by project, by failing tool, and by file cluster. Quote tool names and file paths from the data.
4. **Resolution Velocity** — how quickly problems close. Cite avg/median/p90 and at least one example resolved chain (the initial command, Claude's opening prompt, active time).
5. **Trend vs. Prior Week** — if the period comparison has data, state the deltas. If "insufficient data" is reported, say so explicitly and do not invent trends.
6. **Coaching Recommendations** — 2-3 specific suggestions. For each, cite the data that motivates it (count, project, chain, or file cluster). If any pending recommendation in the data matches the finding, reference its id.

Rules:
- Every claim about a pattern must be grounded in a verbatim quote from the data — a prompt, command, file path, or tool name. No generic phrasing like "you often type similar prompts" without a specific quoted example.
- When a metric is below sample threshold (marked "insufficient sample" or "low sample" in the data), hedge explicitly or omit the claim. Do not report averages or trends on insufficient samples.
- Prefer concrete counts and durations over vague language. "You ran `pytest -x && git add` 14 times" beats "you often run this sequence".
- If a section has no data (empty or zero), write one line acknowledging that and move on. Don't pad.
- Length: aim for 600-900 words total. Density over breadth.
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
    config: Config, start: datetime, end: datetime, window_days: int
) -> CoachingData:
    """Run every detector over a date window and assemble a CoachingData.

    Does not populate `.comparison`; the caller fills that in if applicable.
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
    )


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
    end = datetime.now()
    start = end - timedelta(days=window_days)
    current = _aggregate_window(config, start, end, window_days)

    if compare:
        # Prior window: equal length, ending where current window starts.
        prior_end = start
        prior_start = prior_end - timedelta(days=window_days)
        prior = _aggregate_window(config, prior_start, prior_end, window_days)
        current.comparison = compute_period_comparison(current, prior, config)

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

    lines = [f"\nRECURRING PROMPTS ({data.prompt_patterns.total_prompts} prompts analyzed):"]
    if not within and not cross:
        lines.append("  None detected above the frequency floor.")
        return lines

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
    lines = ["\nRECURRING COMMAND SEQUENCES:"]
    if not sequences:
        lines.append("  None detected above the frequency floor.")
        return lines
    for s in sequences:
        seq_text = " -> ".join(_truncate(c, 60) for c in s.sequence)
        total_min = s.total_time_ms / 60000
        lines.append(
            f"    x{s.count} {seq_text}  (total {total_min:.1f} min, gain {s.compression_gain})"
        )
    return lines


def _section_correlations(data: CoachingData, caps: dict) -> list[str]:
    lines = ["\nSHELL ↔ CLAUDE CORRELATIONS:"]
    if not data.correlations.patterns:
        lines.append("  None detected in this window.")
        return lines
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

    lines = ["\nTOP RESOLUTION CHAINS:"]
    if not data.chains:
        lines.append("  No chains in this window.")
        return lines
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


def _section_session_outcomes(data: CoachingData) -> list[str]:
    counts = data.coaching_findings.count_by_classification
    total = sum(counts.values())
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
    return lines


def _section_stuck_by_project(data: CoachingData, caps: dict) -> list[str]:
    sp = data.stuck_patterns
    lines = [f"\nSTUCK PATTERNS — BY PROJECT ({sp.total_stuck_sessions} stuck sessions):"]
    if not sp.patterns:
        lines.append("  No stuck patterns detected.")
        return lines
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
    return lines


def _section_stuck_by_tool(data: CoachingData, caps: dict) -> list[str]:
    tools = data.stuck_patterns.tool_level_patterns[: caps["stuck_tools"]]
    lines = ["\nSTUCK PATTERNS — BY FAILING TOOL:"]
    if not tools:
        lines.append("  No cross-project tool patterns.")
        return lines
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
    return lines


def _section_stuck_by_cluster(data: CoachingData, caps: dict) -> list[str]:
    clusters = data.stuck_patterns.file_cluster_patterns[: caps["stuck_clusters"]]
    lines = ["\nSTUCK PATTERNS — BY FILE CLUSTER:"]
    if not clusters:
        lines.append("  No multi-session file clusters.")
        return lines
    for c in clusters:
        lines.append(
            f"  {c.path_fragment}: {c.episode_count} stuck sessions "
            f"[{', '.join(c.projects)}], tools: {', '.join(c.failing_tools)}, "
            f"total time: {c.total_duration_ms / 60000:.1f}min"
        )
    return lines


def _section_velocity(data: CoachingData) -> list[str]:
    v = data.velocity_metrics
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
    lines = ["\nPERIOD COMPARISON (current vs. prior equal-length window):"]
    c = data.comparison
    if c is None:
        lines.append("  Not computed (compare=False).")
        return lines
    if c.insufficient_data_reason:
        lines.append(f"  Insufficient data for trend comparison: {c.insufficient_data_reason}")
        return lines
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
    lines = ["\nPENDING RECOMMENDATIONS (staged in ~/.ambient/recommendations/):"]
    if not recs:
        lines.append("  None.")
        return lines
    for r in recs:
        lines.append(f"  [{r.get('type', 'unknown')}] {r.get('id')} — {r.get('title', '')}")
    return lines


def build_insights_prompt(data: CoachingData, caps: dict | None = None) -> str:
    caps = caps or _DEFAULT_CAPS
    sections: list[str] = [
        f"COACHING DATA — {data.date_range} ({data.window_days}-day window)"
    ]
    sections += _section_session_outcomes(data)
    sections += _section_recurring_prompts(data, caps)
    sections += _section_command_sequences(data, caps)
    sections += _section_correlations(data, caps)
    sections += _section_resolution_chains(data, caps)
    sections += _section_stuck_by_project(data, caps)
    sections += _section_stuck_by_tool(data, caps)
    sections += _section_stuck_by_cluster(data, caps)
    sections += _section_velocity(data)
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


def format_terminal_summary(data: CoachingData) -> str:
    lines = [f"Ambient Insights — {data.date_range}\n"]

    cmp = data.comparison
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

    if sp.patterns:
        top = sp.patterns[0]
        lines.append(
            f"\nTop finding: {top.project} — {top.episode_count} stuck episodes "
            f"({top.total_duration_ms / 60000:.0f} min total)"
        )
    else:
        lines.append("\nNo significant stuck patterns detected.")

    return "\n".join(lines)


def _shrink_caps(caps: dict, factor: float = 0.7) -> dict:
    """Scale every section cap by a factor, floor at 1."""
    return {key: max(1, int(value * factor)) for key, value in caps.items()}


def generate_insights_report(data: CoachingData, config: Config, client=None) -> str | None:
    from ambient.present.tokens import estimate_tokens

    caps = dict(_DEFAULT_CAPS)
    prompt = build_insights_prompt(data, caps)
    estimated = estimate_tokens(INSIGHTS_SYSTEM) + estimate_tokens(prompt)
    original_caps = dict(caps)

    # Proportional trim across all sections; each pass shrinks every cap by 30%
    # and floors at 1 so the section header + at least one example survive.
    trim_iterations = 0
    while estimated > INSIGHTS_INPUT_BUDGET and any(v > 1 for v in caps.values()):
        caps = _shrink_caps(caps, factor=0.7)
        prompt = build_insights_prompt(data, caps)
        estimated = estimate_tokens(INSIGHTS_SYSTEM) + estimate_tokens(prompt)
        trim_iterations += 1
        if trim_iterations > 20:
            break  # safety: can't shrink further, send as-is

    if trim_iterations:
        logger.info(
            "PROMPT_TRIMMED call_type=insights iterations=%d caps_before=%s caps_after=%s",
            trim_iterations, original_caps, caps,
        )

    try:
        from ambient.present.api import call_api
        narrative = call_api(config, INSIGHTS_SYSTEM, prompt, config.sonnet_model,
                            max_tokens=3000, client=client)
    except Exception:
        return None

    date_str = datetime.now().strftime("%Y-%m-%d")
    path = config.insights_path(date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(narrative + "\n")

    return narrative
