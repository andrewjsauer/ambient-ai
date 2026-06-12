import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from ambient.config import Config


def cmd_start(config: Config, args):
    hooks_path = Path(__file__).parent / "capture" / "hooks.zsh"
    if not hooks_path.exists():
        print(f"Error: hooks.zsh not found at {hooks_path}", file=sys.stderr)
        sys.exit(1)

    # Check for tmux
    if not os.environ.get("TMUX"):
        print("Warning: Not running inside tmux. Hooks will work but tmux_pane will be null.")

    config.ensure_dirs()
    print(f"Ambient AI ready. Add this to your shell session:\n")
    print(f"  source {hooks_path.resolve()}\n")
    print(f"Events will be logged to: {config.logs_dir}")


def cmd_stop(config: Config, args):
    print("To stop ambient monitoring, remove or comment out the source line")
    print("from your .zshrc, or start a new shell session without it.")
    print("\nTo unset hooks in the current session:")
    print("  add-zsh-hook -d preexec _ambient_preexec")
    print("  add-zsh-hook -d precmd _ambient_precmd")


def cmd_status(config: Config, args):
    from ambient.capture.reader import read_events_today
    from ambient.daemon.launchd import is_agent_loaded
    from ambient.daemon.lock import is_locked
    from ambient.daemon.state import DaemonState

    date_str = datetime.now().strftime("%Y-%m-%d")

    # Section 1: daemon health
    daemon_loaded = is_agent_loaded()
    state = DaemonState.load(config.state_path)
    last_tick = (
        datetime.fromtimestamp(state.last_analyzed_ts / 1000).strftime("%H:%M")
        if state.last_analyzed_ts > 0
        else "never"
    )
    locked, _ = is_locked(config.lock_path)
    gmm_label = "calibrated" if config.gmm_model_path.exists() else "uncalibrated"

    print(f"Ambient · {date_str}")
    print("─" * 50)
    print(
        f"Daemon       {'running' if daemon_loaded else 'stopped'} · "
        f"last tick {last_tick} · "
        f"lock {'held' if locked else 'free'} · "
        f"GMM {gmm_label}"
    )

    # Section 2: today
    events = read_events_today(config)
    if events:
        last_event_ts = max(e.ts_end for e in events)
        last_event_str = datetime.fromtimestamp(last_event_ts / 1000).strftime("%H:%M")
        claude_count = sum(1 for e in events if e.type == "claude_session")
        # Lightweight project mix (no API)
        try:
            from ambient.detect.projects import detect_project_allocation
            alloc = detect_project_allocation(events, config)
            top_projects = []
            for a in alloc.allocations[:3]:
                minutes = a.total_ms / 1000 / 60
                if minutes >= 60:
                    top_projects.append(f"{a.project} ({minutes / 60:.1f}h)")
                else:
                    top_projects.append(f"{a.project} ({minutes:.0f}m)")
            project_line = ", ".join(top_projects) if top_projects else "—"
        except Exception:
            project_line = "—"
        print(
            f"Today        {len(events)} events · "
            f"{claude_count} claude session(s) · "
            f"last activity {last_event_str}"
        )
        print(f"Projects     {project_line}")
    else:
        print("Today        no events yet")

    # Section 3: artifacts
    summary_today = config.summary_path(date_str).exists()
    latest = _most_recent_summary(config)
    latest_label = latest[0] if latest else "none"
    rec_count = _pending_rec_count(config)
    print(
        f"Summary      today {'✓' if summary_today else '—'} · "
        f"latest {latest_label} · "
        f"pending recs {rec_count}"
    )

    # Section 4: next steps
    suggestions = []
    if not daemon_loaded:
        suggestions.append("ambient daemon-start")
    if not config.gmm_model_path.exists():
        suggestions.append("ambient calibrate")
    if latest:
        suggestions.append(f"ambient review{'' if summary_today else f' {latest[0]}'}")
    suggestions.append("ambient insights")
    print(f"Try          {' · '.join(suggestions)}")


def _pending_rec_count(config: Config) -> int:
    from ambient.present.recommender import parse_recommendation_frontmatter

    rec_dir = config.recommendations_dir
    if not rec_dir.exists():
        return 0
    count = 0
    for f in rec_dir.glob("*.md"):
        try:
            meta = parse_recommendation_frontmatter(f.read_text())
        except Exception:
            continue
        title = (meta.get("title") or "").strip()
        if not title or title in ("Skill:", "Skill: "):
            continue
        lower = title.lower()
        if "interrupted by user" in lower or "api error" in lower:
            continue
        count += 1
    return count


def cmd_stats(config: Config, args):
    from ambient.capture.reader import read_events_window, read_events_today
    from ambient.detect.compression import detect_compression
    from ambient.detect.pauses import classify
    from ambient.detect.changepoints import detect_changepoints

    # Default to today; `--window N` narrows to the last N minutes.
    explicit_window = getattr(args, "window", None)
    if explicit_window:
        events = read_events_window(config, explicit_window)
        scope_label = f"last {explicit_window} minutes"
    else:
        events = read_events_today(config)
        scope_label = "today"
    if not events:
        print(f"No events for {scope_label}.")
        return

    print(f"=== Stats for {scope_label} ({len(events)} events) ===\n")

    # Compression
    compression = detect_compression(events, config)
    print("COMPRESSION:")
    print(f"  Compression ratio: {compression.compression_ratio:.3f}")
    if compression.sequences:
        for s in compression.sequences[:10]:
            print(f"  {' -> '.join(s.sequence)} (x{s.count}, gain={s.compression_gain})")
    else:
        print("  No repeated sequences found.")

    # Pause classification
    print("\nPAUSE CLASSIFICATION:")
    pause_result = classify(events, config)
    if pause_result.available:
        labels = [c.label for c in pause_result.classifications]
        total = len(labels)
        for label in ["routine", "evaluating", "stuck"]:
            count = labels.count(label)
            pct = count / total * 100 if total else 0
            print(f"  {label}: {count}/{total} ({pct:.0f}%)")

        stuck = sorted(
            [c for c in pause_result.classifications if c.label == "stuck"],
            key=lambda c: c.gap_ms, reverse=True,
        )
        if stuck:
            print("  Top stuck episodes:")
            for c in stuck[:3]:
                print(f"    {c.gap_ms}ms after '{c.preceding_command}'")
    else:
        print(f"  Not available: {pause_result.reason}")

    # Changepoints on full day
    print("\nWORKFLOW RHYTHM (full day):")
    day_events = read_events_today(config)
    cp_result = detect_changepoints(day_events, config, pause_result)
    if cp_result.segments:
        for seg in cp_result.segments:
            print(
                f"  {seg.duration_min:.0f}min | "
                f"{seg.mean_rate:.1f} cmd/5min | "
                f"{seg.label}"
            )
        if cp_result.changepoints:
            print(f"  Transitions: {len(cp_result.changepoints)}")
    else:
        print("  Not enough data for rhythm analysis.")

    # Claude Code sessions
    claude_events = [e for e in events if e.type == "claude_session"]
    if claude_events:
        total_ms = sum(e.duration_ms for e in claude_events)
        total_min = total_ms / 1000 / 60
        print(f"\nCLAUDE CODE SESSIONS ({len(claude_events)} in window):")
        print(f"  Total time: {total_min:.0f} min")
        for e in claude_events:
            dur = e.duration_ms / 1000 / 60
            print(f"  {dur:.0f}min | {e.cwd} | {e.command[:80]}")


def cmd_analyze(config: Config, args):
    from ambient.capture.reader import read_events_window
    from ambient.detect.compression import detect_compression
    from ambient.detect.pauses import classify
    from ambient.present.narrator import narrate_batch

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        print("Set your API key or use 'ambient stats' for local-only analysis.")
        sys.exit(1)

    window = config.default_window_minutes
    events = read_events_window(config, window)
    if not events:
        print(f"No events in the last {window} minutes.")
        return

    print(f"Analyzing last {window} minutes ({len(events)} events)...")

    compression = detect_compression(events, config)
    pauses = classify(events, config)

    if not pauses.available:
        print(f"Note: {pauses.reason}")

    result = narrate_batch(compression, pauses, config)

    if result.get("analysis"):
        print("\nAnalysis:")
        print(json.dumps(result["analysis"], indent=2))
    else:
        print(f"\nAPI error: {result.get('error', 'unknown')}")
        print("Raw findings have been saved.")


def cmd_summary(config: Config, args):
    from ambient.capture.reader import read_events_today
    from ambient.detect.pauses import classify
    from ambient.detect.changepoints import detect_changepoints
    from ambient.present.narrator import narrate_daily, load_batch_analyses

    date_str = args.date if hasattr(args, "date") and args.date else datetime.now().strftime("%Y-%m-%d")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    batch_analyses = load_batch_analyses(config, date_str)
    if not batch_analyses:
        print(f"No batch analyses for {date_str}. Run 'ambient analyze' first.")
        return

    # Run changepoint detection on full day
    events = read_events_today(config)
    pause_result = classify(events, config)
    changepoints = detect_changepoints(events, config, pause_result)

    print(f"Generating daily summary for {date_str}...")
    narrative = narrate_daily(batch_analyses, changepoints, config, date_str=date_str)
    print(f"\n{narrative}")
    print(f"\nSaved to: {config.summary_path(date_str)}")


def cmd_review(config: Config, args):
    # Accept either positional `date` or `--date YYYY-MM-DD`.
    explicit_date = (
        getattr(args, "date_positional", None) or getattr(args, "date", None)
    )
    requested_date = explicit_date or datetime.now().strftime("%Y-%m-%d")
    summary_path = config.summary_path(requested_date)

    if summary_path.exists():
        print(summary_path.read_text())
        return

    # Fallback: show the most recent summary so `ambient review` never dead-ends.
    fallback = _most_recent_summary(config, before=requested_date)
    if fallback is None:
        print(f"No summary for {requested_date} and no prior summaries found.")
        print("Run 'ambient summary' (after some Claude/terminal activity) to generate one.")
        return

    fallback_date, fallback_path = fallback
    if explicit_date:
        print(f"No summary for {requested_date} — showing most recent ({fallback_date}).\n")
    else:
        print(f"No summary for today yet — showing most recent ({fallback_date}).\n")
    print(fallback_path.read_text())


def _most_recent_summary(config: Config, before: str | None = None):
    """Return (date_str, Path) for the newest summary on-or-before `before`, or None."""
    analysis_dir = config.analysis_dir
    if not analysis_dir.exists():
        return None
    candidates = []
    for p in analysis_dir.glob("summary-*.md"):
        date_part = p.stem.replace("summary-", "")
        if before and date_part > before:
            continue
        candidates.append((date_part, p))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0]


def cmd_calibrate(config: Config, args):
    from ambient.capture.reader import read_events
    from ambient.detect.pauses import calibrate

    # Read all available event files
    all_events = []
    if config.logs_dir.exists():
        for f in sorted(config.logs_dir.glob("events-*.jsonl")):
            date_str = f.stem.replace("events-", "")
            all_events.extend(read_events(config, date_str=date_str))

    if not all_events:
        print("No events found. Capture some terminal activity first.")
        return

    print(f"Calibrating GMM on {len(all_events)} events...")
    result = calibrate(all_events, config)

    if result.available:
        stats = result.calibration_stats
        print(f"\nModel saved to: {config.gmm_model_path}")
        print(f"Samples used: {stats.n_samples}")
        print("\nComponent means (ms):")
        for i, (label, mean) in enumerate(zip(
            ["routine", "evaluating", "stuck"], stats.component_means_ms
        )):
            print(f"  {label}: {mean:.0f}ms ({mean/1000:.1f}s)")
        print(f"\nBIC scores: {stats.bic_scores}")
    else:
        print(f"\n{result.reason}")


def cmd_recommendations(config: Config, args):
    from ambient.present.recommender import parse_recommendation_frontmatter

    rec_dir = config.recommendations_dir
    if not rec_dir.exists():
        print("No recommendations directory found.")
        return

    files = sorted(rec_dir.glob("*.md"))
    if not files:
        print("No pending recommendations.")
        return

    rows = []
    skipped = 0
    for f in files:
        rec_id = f.stem
        meta = parse_recommendation_frontmatter(f.read_text())
        rec_type = meta.get("type", "unknown")
        title = (meta.get("title") or "").strip()
        # Drop empty-title rows and interrupt-noise artifacts.
        if not title or title in ("Skill:", "Skill: "):
            skipped += 1
            continue
        lower_title = title.lower()
        if "interrupted by user" in lower_title or "api error" in lower_title:
            skipped += 1
            continue
        rows.append((rec_id, rec_type, title))

    if not rows:
        print("No pending recommendations.")
        if skipped:
            print(f"(Filtered {skipped} empty/noise entries.)")
        return

    id_width = 30
    type_width = 10
    print(f"{'ID':<{id_width}} {'TYPE':<{type_width}} TITLE")
    print("-" * 80)
    for rec_id, rec_type, title in rows:
        rid = rec_id if len(rec_id) <= id_width else rec_id[: id_width - 1] + "…"
        rtype = rec_type if len(rec_type) <= type_width else rec_type[: type_width - 1] + "…"
        print(f"{rid:<{id_width}} {rtype:<{type_width}} {title}")
    if skipped:
        print(f"\n({skipped} empty/noise entr{'y' if skipped == 1 else 'ies'} hidden.)")


def cmd_apply(config: Config, args):
    from ambient.present.recommender import parse_recommendation_frontmatter

    rec_id = args.recommendation_id
    rec_dir = config.recommendations_dir
    rec_path = rec_dir / f"{rec_id}.md"

    if not rec_path.exists():
        print(f"Recommendation not found: {rec_id}", file=sys.stderr)
        sys.exit(1)

    text = rec_path.read_text()
    meta = parse_recommendation_frontmatter(text)
    rec_type = meta.get("type", "unknown")

    if rec_type != "skill":
        print(f"Only skill installation supported. Type '{rec_type}' must be applied manually.")
        print(f"Use the file directly: {rec_path}")
        return

    # Extract artifact (everything after frontmatter)
    body_match = re.match(r"^---\s*\n.*?\n---\s*\n(.*)$", text, re.DOTALL)
    artifact = body_match.group(1).strip() if body_match else text

    commands_dir = Path.home() / ".claude" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)

    dest = commands_dir / f"{rec_id}.md"
    dest.write_text(artifact + "\n")

    print(f"Installed skill to: {dest}")


def cmd_projects(config: Config, args):
    from ambient.capture.reader import read_events_window, read_events
    from ambient.detect.projects import detect_project_allocation

    window = args.window if hasattr(args, "window") and args.window else None
    date = args.date if hasattr(args, "date") and args.date else None

    if date:
        events = read_events(config, date_str=date)
        label = f"date {date}"
    elif window:
        events = read_events_window(config, window)
        label = f"last {window} minutes"
    else:
        from ambient.capture.reader import read_events_today
        events = read_events_today(config)
        label = "today"

    if not events:
        print(f"No events for {label}.")
        return

    findings = detect_project_allocation(events, config)

    total_ms = sum(a.total_ms for a in findings.allocations)

    print(f"=== Project Allocation ({label}, {len(events)} events) ===\n")
    print(f"{'PROJECT':<25} {'TIME':>10} {'%':>6} {'SESSIONS':>10} {'EVENTS':>8}")
    print("-" * 63)

    for a in findings.allocations:
        minutes = a.total_ms / 1000 / 60
        pct = (a.total_ms / total_ms * 100) if total_ms else 0
        if minutes >= 60:
            time_str = f"{minutes / 60:.1f}h"
        else:
            time_str = f"{minutes:.0f}m"
        print(f"{a.project:<25} {time_str:>10} {pct:>5.0f}% {a.session_count:>10} {a.event_count:>8}")

    print(f"\nContext switches: {findings.context_switches}")
    if findings.primary_project:
        print(f"Primary project: {findings.primary_project}")


def cmd_insights(config: Config, args):
    from ambient.present.insights import (
        aggregate_coaching_data,
        format_terminal_summary,
        generate_insights_report,
    )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    window = args.window if hasattr(args, "window") and args.window else 7
    by_day = bool(getattr(args, "by_day", False))

    print(f"Analyzing last {window} days of activity...")
    data = aggregate_coaching_data(config, window_days=window)
    data.by_day = by_day

    total_sessions = sum(data.coaching_findings.count_by_classification.values())
    if total_sessions < 3:
        print("Insufficient data for coaching analysis (need at least 3 Claude sessions).")
        return

    print(format_terminal_summary(data))

    print("\nGenerating coaching report...")
    narrative = generate_insights_report(data, config)

    if narrative:
        date_str = datetime.now().strftime("%Y-%m-%d")
        print(f"\nFull report: {config.insights_path(date_str)}")
    else:
        print("\nReport generation failed (API error). Terminal summary above is still valid.")


def cmd_daemon_tick(config: Config, args):
    from ambient.daemon.tick import daemon_tick
    daemon_tick(config)


def cmd_daemon_start(config: Config, args):
    from ambient.daemon.launchd import install_agent, is_agent_loaded

    config.ensure_dirs()

    # Copy API key to dotenv for launchd context
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in current environment.", file=sys.stderr)
        print("Set your API key first, then run 'ambient daemon-start'.")
        sys.exit(1)

    # Always write current key (handles rotation)
    config.dotenv_path.write_text(f"ANTHROPIC_API_KEY={api_key}\n")
    os.chmod(config.dotenv_path, 0o600)

    if is_agent_loaded():
        print("Daemon is already running. Use 'ambient daemon-stop' first to restart.")
        return

    install_agent(config)
    print("Ambient daemon started.")
    print(f"  Analysis runs every 30 minutes")
    print(f"  API key saved to: {config.dotenv_path}")
    print(f"  Daemon log: {config.daemon_log_path}")
    print(f"\nTo stop: ambient daemon-stop")


def cmd_daemon_stop(config: Config, args):
    from ambient.daemon.launchd import is_agent_loaded, uninstall_agent

    if not is_agent_loaded():
        print("Daemon is not running.")
        return

    uninstall_agent()
    print("Ambient daemon stopped.")


def cmd_daemon_status(config: Config, args):
    from ambient.daemon.launchd import is_agent_loaded
    from ambient.daemon.lock import is_locked
    from ambient.daemon.state import DaemonState

    loaded = is_agent_loaded()
    print(f"Daemon: {'running' if loaded else 'not running'}")

    state = DaemonState.load(config.state_path)

    if state.last_analyzed_ts > 0:
        last_time = datetime.fromtimestamp(state.last_analyzed_ts / 1000).strftime("%Y-%m-%d %H:%M:%S")
        print(f"Last analysis: {last_time}")
    else:
        print("Last analysis: never")

    if state.last_summary_date:
        print(f"Last summary: {state.last_summary_date}")
    else:
        print("Last summary: never")

    if state.last_calibration_date:
        print(f"Last calibration: {state.last_calibration_date}")
    else:
        print("Last calibration: never")

    print(f"Events since calibration: {state.events_since_calibration}")

    locked, lock_info = is_locked(config.lock_path)
    if locked:
        print(f"Lock: held by PID {lock_info.get('pid', '?')} ({lock_info.get('age_minutes', 0):.0f}m)")
    else:
        print("Lock: free")

    # Event count since last analysis
    today = datetime.now().strftime("%Y-%m-%d")
    events_path = config.events_path(today)
    if events_path.exists():
        lines = sum(1 for line in open(events_path) if line.strip())
        print(f"Events today: {lines}")
    else:
        print("Events today: 0")


def cmd_focus_enable(config: Config, args):
    """Opt-in: install the NSWorkspace focus listener launchd agent."""
    from ambient.daemon.launchd import (
        install_focus_listener,
        is_focus_listener_loaded,
    )

    config.ensure_dirs()
    config.daemon_dir.mkdir(parents=True, exist_ok=True)

    if is_focus_listener_loaded():
        print("Focus listener already running. Use 'ambient focus-disable' first to restart.")
        return

    install_focus_listener(config)
    print("Focus listener enabled.")
    print(f"  Capturing app-activation events (NSWorkspace) to:")
    print(f"    {config.focus_events_path}")
    print(f"  Listener log: {config.focus_listener_log_path}")
    print(f"  Privacy contract: docs/PRIVACY.md (clauses 6, 7)")
    print(f"  Captured fields: bundle_id, app_name (localized), pid, ts")
    print(f"  NEVER captured: window title, document path, or any closed-doors field")
    print(f"\nTo stop: ambient focus-disable")


def cmd_focus_disable(config: Config, args):
    from ambient.daemon.launchd import (
        is_focus_listener_loaded,
        uninstall_focus_listener,
    )

    if not is_focus_listener_loaded():
        print("Focus listener is not running.")
        return

    uninstall_focus_listener()
    print("Focus listener stopped.")


def cmd_focus_status(config: Config, args):
    from ambient.daemon.launchd import is_focus_listener_loaded
    from ambient.daemon.lock import is_locked

    loaded = is_focus_listener_loaded()
    print(f"Focus listener: {'running' if loaded else 'not running'}")

    locked, info = is_locked(config.focus_listener_lock_path)
    if locked:
        print(f"Lock: held by PID {info.get('pid', '?')} ({info.get('age_minutes', 0):.0f}m)")
    else:
        print("Lock: free")

    if config.focus_events_path.exists():
        lines = sum(1 for line in open(config.focus_events_path) if line.strip())
        print(f"Focus events captured: {lines}")
    else:
        print("Focus events captured: 0 (no file yet)")


def cmd_focus_listener_run(config: Config, args):
    """Hidden launchd entry point. Runs the focus listener until SIGTERM."""
    from ambient.daemon.focus_listener import run
    sys.exit(run(config))


def cmd_tmux_focus_enable(config: Config, args):
    """Opt-in: install tmux pane/window focus hooks (Phase 2 Unit 8)."""
    from ambient.capture import tmux_focus

    config.ensure_dirs()
    if not tmux_focus.tmux_available():
        print("Error: tmux not found on PATH.", file=sys.stderr)
        sys.exit(1)
    try:
        tmux_focus.install_hooks(config.focus_events_path)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print("tmux focus hooks installed.")
    print(f"  Hooks: {', '.join(tmux_focus.HOOKS)}")
    print(f"  focus-events: on (prior value saved; restored on disable)")
    print(f"  Events appended to: {config.focus_events_path}")
    print(f"  Privacy contract: docs/PRIVACY.md (clauses 6, 7)")
    print(f"  Captured fields: pane_id, window_index, session_name, event, ts")
    print(f"  NEVER captured: pane_title, pane_current_command, pane_current_path")
    print(f"\nTo stop: ambient tmux-focus-disable")


def cmd_tmux_focus_disable(config: Config, args):
    from ambient.capture import tmux_focus
    if not tmux_focus.tmux_available():
        print("tmux not found on PATH; nothing to remove.")
        return
    tmux_focus.uninstall_hooks()
    print("tmux focus hooks removed (Ambient-managed only).")


def cmd_vectors(config: Config, args):
    """v4 Phase 3 diagnostic: print VectorFindings as a table.

    Used to validate the heuristic classifier and stop-event enumerator
    against real data before the deeper LLM-narrative description in the
    weekly report fully lands. Works whether or not focus capture is on —
    focus_change stops simply don't appear in the summary if no focus events
    are on disk.
    """
    from datetime import datetime, timedelta
    from ambient.capture.reader import read_events
    from ambient.detect.focus_events import read_focus_events
    from ambient.detect.pauses import classify
    from ambient.detect.vectors import (
        VectorFindings,
        detect_vectors,
        stop_reason_summary,
        top_vectors_per_project,
    )

    window = args.window if hasattr(args, "window") and args.window else 7
    end = datetime.now()
    start = end - timedelta(days=window)
    window_start_ms = int(start.timestamp() * 1000)
    window_end_ms = int(end.timestamp() * 1000)

    events = read_events(config, start=start, end=end)
    focus_events = read_focus_events(
        config.focus_events_path, since_iso=start.isoformat(),
    )
    pause_findings = classify(events, config)
    pauses = pause_findings if pause_findings.available else None

    findings = detect_vectors(
        events, focus_events, pauses, window_start_ms, window_end_ms, config,
    )

    print(f"Vectors in last {window} days: {len(findings.vectors)}")
    if not findings.vectors:
        print("  (no vectors — try running for longer or enabling focus capture)")
        return

    summary = stop_reason_summary(findings)
    total_dur = sum(d for _, _, d in summary) or 1
    chunks = []
    for reason, count, dur in summary:
        pct = dur / total_dur * 100
        chunks.append(f"{pct:.0f}% {reason} ({count})")
    print("Stop reasons: " + ", ".join(chunks))

    # Filter end_of_window vectors out of "longest" — they're tail artifacts
    # whose duration just reflects "no activity since the last stop", not a
    # real activity stretch.
    activity_only = VectorFindings(
        vectors=[v for v in findings.vectors if v.stop_reason != "end_of_window"],
    )
    include_passive = bool(getattr(args, "include_passive", False))
    label = "all longest" if include_passive else "longest with text"
    print(
        f"\nPer project (top {config.vectors_per_project} {label} vectors):"
    )
    by_project = top_vectors_per_project(activity_only, n=config.vectors_per_project)
    projects_ordered = sorted(
        by_project.items(),
        key=lambda kv: kv[1][0].duration_ms if kv[1] else 0,
        reverse=True,
    )
    # Pre-compute per-project passive totals across ALL vectors (not just the
    # top-N slice) so the rollup reflects reality.
    passive_totals: dict[str, tuple[int, int]] = {}
    for v in activity_only.vectors:
        if v.last_command_or_prompt:
            continue
        count, dur = passive_totals.get(v.project, (0, 0))
        passive_totals[v.project] = (count + 1, dur + v.duration_ms)

    for project, vectors in projects_ordered:
        total_min = sum(v.duration_ms for v in findings.vectors if v.project == project) // 60_000
        proj_count = findings.count_by_project.get(project, 0)

        text_vectors = [v for v in vectors if v.last_command_or_prompt]
        display_vectors = vectors if include_passive else text_vectors
        passive_count, passive_dur = passive_totals.get(project, (0, 0))

        print(f"  {project} ({total_min}min, {proj_count} vectors)")
        for v in display_vectors:
            mins = v.duration_ms / 60_000
            stop_label = v.stop_reason
            if v.stop_reason == "pause" and v.pause_duration_ms:
                stop_label = f"pause({v.pause_duration_ms / 60_000:.1f}m)"
            elif v.stop_reason == "exit":
                stop_label = "exit (session)"
            text = (v.last_command_or_prompt or "")[:60]
            print(f"    {mins:5.1f}m  {stop_label:14s}  {text}")
        if not include_passive and passive_count:
            print(
                f"    +{passive_count} passive focus vectors "
                f"({passive_dur / 60_000:.0f}m total — pass --include-passive to show)"
            )
        if not display_vectors and not passive_count:
            print("    (no vectors)")

    print(f"\nClassification: {dict(findings.count_by_classification)}")


def main():
    parser = argparse.ArgumentParser(
        prog="ambient",
        description="Ambient AI - Passive terminal behavioral monitor",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("start", help="Show setup instructions")
    subparsers.add_parser("stop", help="Show teardown instructions")
    subparsers.add_parser("status", help="Show current status")

    stats_parser = subparsers.add_parser("stats", help="Show raw algorithmic output (no LLM)")
    stats_parser.add_argument("--window", type=int, help="Analysis window in minutes (default: 30)")

    subparsers.add_parser("analyze", help="Run 30-min batch analysis with LLM narration")

    summary_parser = subparsers.add_parser("summary", help="Generate daily summary")
    summary_parser.add_argument("--date", help="Date to summarize (YYYY-MM-DD)")

    review_parser = subparsers.add_parser("review", help="View daily summary")
    review_parser.add_argument(
        "date_positional",
        nargs="?",
        metavar="DATE",
        help="Date to review (YYYY-MM-DD). Falls back to most recent summary if missing.",
    )
    review_parser.add_argument("--date", help="Date to review (YYYY-MM-DD)")

    subparsers.add_parser("calibrate", help="Fit GMM on accumulated event data")

    subparsers.add_parser("recommendations", help="List pending recommendations")

    apply_parser = subparsers.add_parser("apply", help="Install a staged recommendation")
    apply_parser.add_argument("recommendation_id", help="Recommendation ID to apply")

    insights_parser = subparsers.add_parser("insights", help="Generate coaching insights report")
    insights_parser.add_argument("--window", type=int, help="Analysis window in days (default: 7)")
    insights_parser.add_argument(
        "--by-day", dest="by_day", action="store_true",
        help="Render the terminal summary as a per-day timeline instead of an aggregate.",
    )

    projects_parser = subparsers.add_parser("projects", help="Show per-project time allocation")
    projects_parser.add_argument("--window", type=int, help="Analysis window in minutes")
    projects_parser.add_argument("--date", help="Date to analyze (YYYY-MM-DD)")

    # Daemon commands (flat, matching existing pattern)
    subparsers.add_parser("daemon-tick", help=argparse.SUPPRESS)  # hidden launchd entry point
    subparsers.add_parser("daemon-start", help="Start the background analysis daemon")
    subparsers.add_parser("daemon-stop", help="Stop the background analysis daemon")
    subparsers.add_parser("daemon-status", help="Show daemon status")

    # v4 Phase 2 Unit 7: focus listener (NSWorkspace app-activation capture).
    # Opt-in. See docs/PRIVACY.md before enabling.
    subparsers.add_parser(
        "focus-enable",
        help="Enable the NSWorkspace focus listener (opt-in capture; see docs/PRIVACY.md)",
    )
    subparsers.add_parser("focus-disable", help="Disable the NSWorkspace focus listener")
    subparsers.add_parser("focus-status", help="Show focus listener status")
    subparsers.add_parser(
        "focus-listener-run", help=argparse.SUPPRESS,
    )  # hidden launchd entry point

    # v4 Phase 2 Unit 8: tmux pane/window focus hooks. Same focus-events.jsonl
    # writer as Unit 7 but native tmux hooks instead of a daemon process.
    subparsers.add_parser(
        "tmux-focus-enable",
        help="Install tmux focus hooks (opt-in capture; see docs/PRIVACY.md)",
    )
    subparsers.add_parser("tmux-focus-disable", help="Remove tmux focus hooks")

    # v4 Phase 3: vector aggregation diagnostic CLI.
    vectors_parser = subparsers.add_parser(
        "vectors",
        help="Show vector aggregation table for the last N days (diagnostic)",
    )
    vectors_parser.add_argument("--window", type=int, help="Window in days (default: 7)")
    vectors_parser.add_argument(
        "--include-passive",
        dest="include_passive",
        action="store_true",
        help="Show focus_change vectors with no command/prompt text (otherwise rolled up).",
    )

    args = parser.parse_args()
    config = Config()

    try:
        from dotenv import load_dotenv

        if config.dotenv_path.exists():
            load_dotenv(config.dotenv_path)
    except ImportError:
        pass

    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "stats": cmd_stats,
        "analyze": cmd_analyze,
        "summary": cmd_summary,
        "review": cmd_review,
        "calibrate": cmd_calibrate,
        "recommendations": cmd_recommendations,
        "apply": cmd_apply,
        "insights": cmd_insights,
        "projects": cmd_projects,
        "daemon-tick": cmd_daemon_tick,
        "daemon-start": cmd_daemon_start,
        "daemon-stop": cmd_daemon_stop,
        "daemon-status": cmd_daemon_status,
        "focus-enable": cmd_focus_enable,
        "focus-disable": cmd_focus_disable,
        "focus-status": cmd_focus_status,
        "focus-listener-run": cmd_focus_listener_run,
        "tmux-focus-enable": cmd_tmux_focus_enable,
        "tmux-focus-disable": cmd_tmux_focus_disable,
        "vectors": cmd_vectors,
    }

    if args.command in commands:
        commands[args.command](config, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
