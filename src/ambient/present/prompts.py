# A Claude session's first prompt can be an entire pasted file. Cap it so one
# session can't dominate the batch prompt — the batch analysis only needs the
# gist of what the session was about, not its full text.
_MAX_PROMPT_CHARS = 500


BATCH_SYSTEM = """You are a behavioral analyst studying a developer's terminal workflow. You receive algorithmically-derived findings from three detection systems. All findings are pre-validated — do not speculate beyond the data. Cite specific commands and times.

Respond in JSON with this structure:
{
  "automation_candidates": [{"sequence": [...], "count": N, "time_saved_estimate": "Xm/day", "suggestion": "..."}],
  "cognitive_patterns": [{"type": "stuck|evaluating|routine", "duration_ms": N, "context": "...", "insight": "..."}],
  "work_phase": {"current": "...", "suggestion_timing": "good|bad", "reason": "..."}
}

Be direct. No filler. Only include sections where the data supports findings."""


def build_batch_prompt(
    compression_data: dict,
    pause_data: dict,
    claude_sessions: list[dict] | None = None,
    project_data: dict | None = None,
) -> str:
    sections = []

    if compression_data.get("sequences"):
        sections.append("REPEATED SEQUENCES (compression detector):")
        for s in compression_data["sequences"]:
            sections.append(
                f"  Sequence: {' -> '.join(s['sequence'])} | "
                f"Count: {s['count']} | "
                f"Time: {s['total_time_ms']}ms | "
                f"Gain: {s['compression_gain']}"
            )
        sections.append(f"  Compression ratio: {compression_data['compression_ratio']:.3f}")
    else:
        sections.append("REPEATED SEQUENCES: None found in this window.")

    if pause_data.get("available") and pause_data.get("classifications"):
        sections.append("\nCOGNITIVE STATE (GMM pause classifier):")
        # Summarize distribution
        labels = [c["label"] for c in pause_data["classifications"]]
        total = len(labels)
        for label in ["routine", "evaluating", "stuck"]:
            count = labels.count(label)
            pct = count / total * 100 if total else 0
            sections.append(f"  {label}: {count}/{total} ({pct:.0f}%)")

        # Highlight notable stuck episodes
        stuck = [c for c in pause_data["classifications"] if c["label"] == "stuck"]
        stuck.sort(key=lambda c: c["gap_ms"], reverse=True)
        for c in stuck[:5]:
            sections.append(
                f"  Stuck episode: {c['gap_ms']}ms after '{c['preceding_command']}' "
                f"before '{c['following_command']}'"
            )
    elif not pause_data.get("available"):
        sections.append("\nCOGNITIVE STATE: Not available (GMM not calibrated).")

    if claude_sessions:
        sections.append(f"\nCLAUDE CODE SESSIONS ({len(claude_sessions)} in this window):")
        for s in claude_sessions:
            duration_min = s.get("duration_ms", 0) / 1000 / 60
            prompts = s.get("claude_prompts", [])
            first_prompt = prompts[0] if prompts else "(no prompt)"
            if len(first_prompt) > _MAX_PROMPT_CHARS:
                first_prompt = first_prompt[:_MAX_PROMPT_CHARS] + "…"
            project = s.get("claude_project", "unknown")
            sections.append(
                f"  Session: {duration_min:.0f}min | "
                f"Prompts: {s.get('claude_prompt_count', 0)} | "
                f"Project: {project} | "
                f"First: \"{first_prompt}\""
            )

    if project_data and project_data.get("allocations"):
        sections.append(f"\nPROJECT ALLOCATION ({len(project_data['allocations'])} projects):")
        for a in project_data["allocations"]:
            mins = a["total_ms"] / 1000 / 60
            sections.append(
                f"  {a['project']}: {mins:.0f}min | "
                f"Events: {a['event_count']} | "
                f"Sessions: {a['session_count']}"
            )
        sections.append(f"  Context switches: {project_data['context_switches']}")
        sections.append(f"  Primary project: {project_data['primary_project']}")

    return "\n".join(sections)


WEEKLY_SYSTEM = """You are writing a structured weekly behavioral trend report for a developer. Below are the daily analysis summaries from the past week(s).

Fill in each section of the template below using only the provided data. Keep the italic description lines exactly as they are. Write 2-4 direct sentences per section (use bullets where appropriate). Write in second person ("You..."). No filler. Focus on trends and changes across days, not single-day events.

## Week Overview
_High-level summary of the week: total active days, overall productivity trend, dominant project(s)._

## Project Allocation Trends
_How time allocation across projects shifted compared to previous weeks. New projects, abandoned projects, growing/shrinking allocations._

## Pattern Trends
_How compression ratios, command rates, and automation candidates evolved. Are repeated sequences increasing or decreasing?_

## Stuck Episode Trends
_Frequency and severity of stuck episodes across the week. Are they improving, worsening, or stable? Common triggers._

## Recommendation Adoption
_Which previous recommendations appear to have been adopted (reduced repetition, new aliases) vs ignored. Evidence from the data._

## Coaching Highlights
_3-5 bullet points: top stuck patterns by project/tool, resolution velocity trend vs previous week, and one specific improvement suggestion backed by the data. This is the actionable summary — full analysis is available via `ambient insights`._

## Key Shifts
_1-3 significant behavioral changes compared to previous week(s). Concrete observations, not speculation._"""


def build_weekly_prompt(
    weekly_analyses: list[dict],
    week_labels: list[str],
    coaching_data: dict | None = None,
) -> str:
    """Build prompt from multiple weeks of daily analysis data.

    Args:
        weekly_analyses: List of dicts, one per week, each containing:
            - date_range: str like "2026-03-31 to 2026-04-06"
            - days: list of daily analysis dicts
        week_labels: List of labels like ["Current week", "Previous week"]
        coaching_data: Optional coaching analysis for the current week.
    """
    sections = []

    for i, (week_data, label) in enumerate(zip(weekly_analyses, week_labels)):
        sections.append(f"\n{'='*60}")
        sections.append(f"{label}: {week_data.get('date_range', 'unknown')}")
        sections.append(f"{'='*60}")

        days = week_data.get("days", [])
        if not days:
            sections.append("  No analysis data for this week.")
            continue

        sections.append(f"  Active days: {len(days)}")

        # Aggregate stats across the week
        total_compression_ratio = []
        total_stuck = 0
        total_classifications = 0
        projects = {}

        for day in days:
            date_str = day.get("date", "unknown")
            sections.append(f"\n  --- {date_str} ---")

            # Compression data
            compression = day.get("compression", {})
            ratio = compression.get("compression_ratio")
            if ratio is not None:
                total_compression_ratio.append(ratio)
                sections.append(f"    Compression ratio: {ratio:.3f}")

            seqs = compression.get("sequences", [])
            if seqs:
                sections.append(f"    Repeated sequences: {len(seqs)}")

            # Pause data
            pauses = day.get("pauses", {})
            classifications = pauses.get("classifications", [])
            if classifications:
                stuck_count = sum(1 for c in classifications if c.get("label") == "stuck")
                total_stuck += stuck_count
                total_classifications += len(classifications)
                sections.append(f"    Pauses: {len(classifications)} (stuck: {stuck_count})")

            # Project allocation
            proj_alloc = day.get("project_allocation", {})
            allocations = proj_alloc.get("allocations", [])
            for a in allocations:
                proj = a.get("project", "unknown")
                projects[proj] = projects.get(proj, 0) + a.get("total_ms", 0)

        # Week summary stats
        if total_compression_ratio:
            avg_ratio = sum(total_compression_ratio) / len(total_compression_ratio)
            sections.append(f"\n  Week avg compression ratio: {avg_ratio:.3f}")
        if total_classifications:
            stuck_pct = total_stuck / total_classifications * 100
            sections.append(f"  Week stuck episodes: {total_stuck}/{total_classifications} ({stuck_pct:.0f}%)")
        if projects:
            sorted_projects = sorted(projects.items(), key=lambda x: x[1], reverse=True)
            sections.append("  Week project allocation:")
            for proj, ms in sorted_projects[:5]:
                mins = ms / 1000 / 60
                sections.append(f"    {proj}: {mins:.0f}min")

    # Coaching data for current week
    if coaching_data:
        sections.append(f"\n{'='*60}")
        sections.append("COACHING ANALYSIS (current week)")
        sections.append(f"{'='*60}")

        outcomes = coaching_data.get("outcomes", {})
        if outcomes:
            sections.append("  Session outcomes:")
            for cls, count in outcomes.items():
                sections.append(f"    {cls}: {count}")

        avg_thrash = coaching_data.get("avg_thrash_score")
        if avg_thrash is not None:
            sections.append(f"  Average thrash score: {avg_thrash:.2f}")

        velocity = coaching_data.get("velocity", {})
        if velocity.get("resolved_count", 0) > 0:
            sections.append(f"  Resolution velocity: {velocity['avg_ms'] / 60000:.1f} min avg "
                          f"({velocity['resolved_count']} resolved)")

        stuck = coaching_data.get("stuck_patterns", [])
        if stuck:
            sections.append("  Top stuck patterns:")
            for p in stuck[:3]:
                sections.append(f"    {p['project']}: {p['episode_count']} episodes, "
                              f"tools: {', '.join(p['failing_tools'])}")

    return "\n".join(sections)


DAILY_SYSTEM = """You are writing a structured daily behavioral review for a developer. Below are the 30-minute batch analyses and full-day rhythm analysis from their workday.

Fill in each section of the template below using only the provided data. Keep the italic description lines exactly as they are — they are structural anchors. Write 2-3 direct sentences per section (use bullets for Key Stats). Leave a section's content blank if the data doesn't support findings for it. Write in second person ("You..."). No filler.

## Day Title
_A distinctive 5-10 word summary of this workday. Info-dense, no generic phrases._

## Rhythm Profile
_When were you most focused vs most fragmented? High-rate and low-rate segments. Flow states and transitions._

## Automation Candidates
_Top 3 repeated command sequences across the day, ranked by time saved. What could be scripted or aliased._

## Cognitive Load
_Stuck episodes and what triggered them. Ratio of routine vs evaluating vs stuck time. Deepest focus sessions._

## Workflow Phases
_Chronological phases of the day with dominant activity type and command rate per phase._

## Friction Points
_Failed commands, repeated retries, longest stuck episodes with surrounding context. What slowed you down._

## Key Stats
_Event count, session count, average command rate, stuck episode count, longest flow session duration, compression ratio._

## Recommendations
_1-3 concrete, copy-pasteable fixes for patterns detected today. Each recommendation must: (1) cite the specific pattern that triggered it, (2) include a code block with the artifact. Artifact types: shell aliases for repeated command sequences, git hooks for pre-push/pre-commit patterns, CLAUDE.md additions for Claude session friction, workflow scripts for multi-step patterns. If no strong patterns exist, include one actionable suggestion instead._"""


def build_daily_prompt(
    batch_analyses: list[dict],
    changepoint_data: dict | None = None,
) -> str:
    sections = []

    if changepoint_data and changepoint_data.get("segments"):
        sections.append("WORKFLOW RHYTHM (changepoint detector - full day):")
        for seg in changepoint_data["segments"]:
            sections.append(
                f"  Segment: {seg['duration_min']:.0f}min | "
                f"Rate: {seg['mean_rate']:.1f} cmd/5min | "
                f"Type: {seg['label']}"
            )
        if changepoint_data.get("changepoints"):
            sections.append("  Transitions:")
            for cp in changepoint_data["changepoints"]:
                sections.append(
                    f"    {cp['from_segment_summary']} -> {cp['to_segment_summary']}"
                )

    sections.append(f"\nBATCH ANALYSES ({len(batch_analyses)} windows):")
    for i, batch in enumerate(batch_analyses):
        sections.append(f"\n  --- Window {i+1} ---")
        if isinstance(batch, dict):
            for key, value in batch.items():
                sections.append(f"  {key}: {value}")
        else:
            sections.append(f"  {batch}")

    return "\n".join(sections)
