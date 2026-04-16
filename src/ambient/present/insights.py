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

INSIGHTS_SYSTEM = """You are a development coach analyzing a developer's workflow data. Produce a coaching report with these sections:

1. **Resolution Velocity** — How quickly the developer resolves problems via Claude. Cite specific metrics.
2. **Stuck Episode Analysis** — Which projects/tools cause the most friction. Cite specific patterns.
3. **Thrash Patterns** — Recurring causes of stuck loops, grouped by project.
4. **Coaching Recommendations** — 2-3 specific, actionable suggestions backed by the data. Recommend CLAUDE.md rules, skills, or workflow changes.
5. **Velocity by Project** — Per-project breakdown of resolution speed.

Be direct and specific. Cite numbers from the data. Focus on what the developer should change, not just what happened. If data is sparse, say so rather than speculating."""


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


def _safe_run(fn, *args, default, label, **kwargs):
    """Run a detector, logging and substituting `default` on failure.

    Insights must never crash because one detector blew up on odd data.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("INSIGHTS_DETECTOR_FAILED detector=%s error=%s", label, exc)
        return default


def aggregate_coaching_data(config: Config, window_days: int = 7) -> CoachingData:
    end = datetime.now()
    start = end - timedelta(days=window_days)
    date_range = f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"

    events = read_events(config, start=start, end=end)

    findings = classify_sessions(events, config)
    stuck = group_stuck_patterns(findings.outcomes, events, config)

    # Build outcome map for velocity tracker
    outcome_map = {o.session_id: o.classification for o in findings.outcomes}
    chains = detect_resolution_chains(events, config, session_outcomes=outcome_map)
    velocity = compute_velocity_metrics(chains, min_chains=config.velocity_min_chains)

    # Extended detectors — each wrapped so an unexpected failure can't crash
    # the insights pipeline. All three are pure over `events` so ordering
    # and reuse are safe.
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


def build_insights_prompt(data: CoachingData) -> str:
    sections = [f"COACHING DATA — {data.date_range} ({data.window_days}-day window)\n"]

    # Session outcome distribution
    counts = data.coaching_findings.count_by_classification
    total = sum(counts.values())
    sections.append(f"SESSION OUTCOMES ({total} sessions):")
    for cls in ("productive", "friction", "quick", "abandoned"):
        n = counts.get(cls, 0)
        pct = n / total * 100 if total else 0
        sections.append(f"  {cls}: {n} ({pct:.0f}%)")

    avg_thrash = data.coaching_findings.avg_thrash_score
    if avg_thrash is not None:
        sections.append(f"  Average thrash score: {avg_thrash:.2f}")
    elif data.coaching_findings.low_sample:
        sections.append("  Average thrash score: insufficient sample")

    # Resolution velocity
    v = data.velocity_metrics
    sections.append(f"\nRESOLUTION VELOCITY ({v.total_chains} chains, {v.resolved_count} resolved):")
    if v.by_reason:
        matched = v.by_reason.get("matched_success", 0)
        idle = v.by_reason.get("idle_break", 0)
        eow = v.by_reason.get("end_of_window", 0)
        sections.append(
            f"  Closure reasons: matched-success={matched}, "
            f"idle-break={idle}, end-of-window={eow}"
        )
    if v.resolved_count > 0:
        sections.append(f"  Average active time: {v.avg_ms / 1000:.0f}s ({v.avg_ms / 60000:.1f}min)")
        sections.append(f"  Median: {v.median_ms / 1000:.0f}s")
        sections.append(f"  p90: {v.p90_ms / 1000:.0f}s")

        if v.by_project:
            sections.append("  By project:")
            for proj, pm in v.by_project.items():
                sections.append(f"    {proj}: avg {pm.avg_ms / 1000:.0f}s, {pm.resolved_count} resolved")
    else:
        sections.append("  No resolved chains in this window.")

    # Stuck patterns
    sp = data.stuck_patterns
    sections.append(f"\nSTUCK PATTERNS ({sp.total_stuck_sessions} stuck sessions):")
    if sp.patterns:
        for p in sp.patterns[:5]:
            thrash_str = (
                f"avg thrash {p.avg_thrash_score:.2f}"
                if p.avg_thrash_score is not None
                else "thrash N/A (low sample)"
            )
            sections.append(
                f"  {p.project}: {p.episode_count} episodes, "
                f"{thrash_str}, "
                f"tools: {', '.join(p.failing_tools)}, "
                f"total time: {p.total_duration_ms / 60000:.1f}min"
            )
            if p.file_cluster and p.file_cluster != ["unknown"]:
                sections.append(f"    files: {', '.join(p.file_cluster[:5])}")
    else:
        sections.append("  No stuck patterns detected.")

    return "\n".join(sections)


def format_terminal_summary(data: CoachingData) -> str:
    lines = [f"Ambient Insights — {data.date_range}\n"]

    v = data.velocity_metrics
    if v.resolved_count > 0:
        lines.append(f"Resolution velocity:  {v.avg_ms / 60000:.1f} min avg ({v.resolved_count} resolved)")
    else:
        lines.append("Resolution velocity:  no resolved chains")

    sp = data.stuck_patterns
    lines.append(f"Stuck episodes:       {sp.total_stuck_sessions}")

    avg = data.coaching_findings.avg_thrash_score
    lines.append(f"Thrash score:         {avg:.2f} avg" if avg is not None else "Thrash score:         n/a")

    if sp.patterns:
        top = sp.patterns[0]
        lines.append(f"\nTop finding: {top.project} — {top.episode_count} stuck episodes "
                     f"({top.total_duration_ms / 60000:.0f} min total)")
    else:
        lines.append("\nNo significant stuck patterns detected.")

    return "\n".join(lines)


def generate_insights_report(data: CoachingData, config: Config, client=None) -> str | None:
    from ambient.present.tokens import estimate_tokens

    prompt = build_insights_prompt(data)

    # Trim stuck patterns if prompt exceeds budget
    estimated = estimate_tokens(INSIGHTS_SYSTEM) + estimate_tokens(prompt)
    if estimated > INSIGHTS_INPUT_BUDGET and data.stuck_patterns.patterns:
        original_count = len(data.stuck_patterns.patterns)
        while len(data.stuck_patterns.patterns) > 1 and estimated > INSIGHTS_INPUT_BUDGET:
            data.stuck_patterns.patterns.pop()
            prompt = build_insights_prompt(data)
            estimated = estimate_tokens(INSIGHTS_SYSTEM) + estimate_tokens(prompt)
        logger.info(
            "PROMPT_TRIMMED call_type=insights items_before=%d items_after=%d",
            original_count, len(data.stuck_patterns.patterns),
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
