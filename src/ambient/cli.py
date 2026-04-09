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
    date_str = datetime.now().strftime("%Y-%m-%d")
    events_path = config.events_path(date_str)

    if events_path.exists():
        lines = 0
        last_line = None
        with open(events_path) as f:
            for line in f:
                if line.strip():
                    lines += 1
                    last_line = line
        if last_line:
            try:
                last_event = json.loads(last_line)
                last_ts = last_event.get("ts_end", 0)
                last_time = datetime.fromtimestamp(last_ts / 1000).strftime("%H:%M:%S")
            except (json.JSONDecodeError, KeyError):
                last_time = "unknown"
        else:
            last_time = "unknown"
        print(f"Events today: {lines}")
        print(f"Last event: {last_time}")
    else:
        print("No events captured yet.")

    # GMM status
    if config.gmm_model_path.exists():
        print("GMM: Calibrated")
    else:
        print("GMM: Not calibrated (run 'ambient calibrate')")

    # Last analysis
    analysis_path = config.analysis_path(date_str)
    if analysis_path.exists():
        print(f"Analysis: {analysis_path}")
    else:
        print("Analysis: None today")


def cmd_stats(config: Config, args):
    from ambient.capture.reader import read_events_window, read_events_today
    from ambient.detect.compression import detect_compression
    from ambient.detect.pauses import classify
    from ambient.detect.changepoints import detect_changepoints

    window = args.window if hasattr(args, "window") and args.window else config.default_window_minutes

    # Compression and pause on window
    events = read_events_window(config, window)
    if not events:
        print(f"No events in the last {window} minutes.")
        return

    print(f"=== Stats for last {window} minutes ({len(events)} events) ===\n")

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
    date_str = args.date if hasattr(args, "date") and args.date else datetime.now().strftime("%Y-%m-%d")
    summary_path = config.summary_path(date_str)

    if summary_path.exists():
        print(summary_path.read_text())
    else:
        print(f"No summary for {date_str}. Run 'ambient summary' first.")


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


def _parse_recommendation_frontmatter(text: str) -> dict:
    """Parse YAML frontmatter from a recommendation .md file."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return {}
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            value = value.strip().strip('"').strip("'")
            meta[key.strip()] = value
    return meta


def cmd_recommendations(config: Config, args):
    rec_dir = config.base_dir / "recommendations"
    if not rec_dir.exists():
        print("No recommendations directory found.")
        return

    files = sorted(rec_dir.glob("*.md"))
    if not files:
        print("No pending recommendations.")
        return

    print(f"{'ID':<30} {'TYPE':<10} TITLE")
    print("-" * 70)
    for f in files:
        rec_id = f.stem
        meta = _parse_recommendation_frontmatter(f.read_text())
        rec_type = meta.get("type", "unknown")
        title = meta.get("title", rec_id)
        print(f"{rec_id:<30} {rec_type:<10} {title}")


def cmd_apply(config: Config, args):
    rec_id = args.recommendation_id
    rec_dir = config.base_dir / "recommendations"
    rec_path = rec_dir / f"{rec_id}.md"

    if not rec_path.exists():
        print(f"Recommendation not found: {rec_id}", file=sys.stderr)
        sys.exit(1)

    text = rec_path.read_text()
    meta = _parse_recommendation_frontmatter(text)
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
        events = read_events_window(config, config.default_window_minutes)
        label = f"last {config.default_window_minutes} minutes"

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

    print(f"Analyzing last {window} days of activity...")
    data = aggregate_coaching_data(config, window_days=window)

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
    review_parser.add_argument("--date", help="Date to review (YYYY-MM-DD)")

    subparsers.add_parser("calibrate", help="Fit GMM on accumulated event data")

    subparsers.add_parser("recommendations", help="List pending recommendations")

    apply_parser = subparsers.add_parser("apply", help="Install a staged recommendation")
    apply_parser.add_argument("recommendation_id", help="Recommendation ID to apply")

    insights_parser = subparsers.add_parser("insights", help="Generate coaching insights report")
    insights_parser.add_argument("--window", type=int, help="Analysis window in days (default: 7)")

    projects_parser = subparsers.add_parser("projects", help="Show per-project time allocation")
    projects_parser.add_argument("--window", type=int, help="Analysis window in minutes")
    projects_parser.add_argument("--date", help="Date to analyze (YYYY-MM-DD)")

    # Daemon commands (flat, matching existing pattern)
    subparsers.add_parser("daemon-tick", help=argparse.SUPPRESS)  # hidden launchd entry point
    subparsers.add_parser("daemon-start", help="Start the background analysis daemon")
    subparsers.add_parser("daemon-stop", help="Stop the background analysis daemon")
    subparsers.add_parser("daemon-status", help="Show daemon status")

    args = parser.parse_args()
    config = Config()

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
    }

    if args.command in commands:
        commands[args.command](config, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
