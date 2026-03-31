import argparse
import json
import os
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
    }

    if args.command in commands:
        commands[args.command](config, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
