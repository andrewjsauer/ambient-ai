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

    return "\n".join(sections)


DAILY_SYSTEM = """You are writing a daily behavioral review for a developer. Below are the 30-minute batch analyses and full-day rhythm analysis from their workday. Synthesize into a narrative that covers:

- Overall work rhythm (when were they most focused, most fragmented?)
- Top 3 automation opportunities (most-repeated patterns across the day)
- Deepest focus sessions and what enabled them
- Biggest time sinks and what caused them
- One specific, actionable suggestion for tomorrow

Write in second person ("You..."). Be direct. No filler. Under 500 words."""


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
