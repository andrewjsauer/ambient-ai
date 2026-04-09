import json
import logging
import os
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler

from ambient.config import Config
from ambient.daemon.lock import acquire_lock, release_lock
from ambient.daemon.state import DaemonState

logger = logging.getLogger(__name__)


def setup_daemon_logging(config: Config) -> None:
    config.ensure_dirs()
    handler = TimedRotatingFileHandler(
        config.daemon_log_path,
        when="midnight",
        backupCount=7,
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def _load_api_key(config: Config) -> bool:
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env loading")
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    load_dotenv(config.dotenv_path)
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _ingest_claude_sessions(config: Config, state: DaemonState) -> None:
    import time

    from ambient.daemon.session_parser import discover_session_files, parse_session_file

    session_files = discover_session_files(config.claude_projects_dir)
    if not session_files:
        return

    now_ms = int(time.time() * 1000)
    session_complete_threshold_ms = 30 * 60 * 1000  # 30 minutes

    ingested_count = 0
    config.ensure_dirs()

    for path in session_files:
        slug = path.parent.name
        session_uuid = path.stem

        if state.is_session_processed(slug, session_uuid):
            continue

        # Quick check: if file was modified recently, session is likely still active
        try:
            mtime_ms = int(path.stat().st_mtime * 1000)
            if (now_ms - mtime_ms) < session_complete_threshold_ms:
                continue
        except OSError:
            continue

        # Parse the session file for full data
        parsed = parse_session_file(path)
        if parsed is None:
            # Mark unparseable sessions as processed to avoid re-parsing every tick
            state.mark_session_processed(slug, session_uuid)
            continue

        # Double-check with parsed max timestamp (more accurate than mtime)
        if (now_ms - parsed["end_ts"]) < session_complete_threshold_ms:
            continue

        # Build enriched event dict
        event_dict = {
            "type": "claude_session",
            "ts_start": parsed["start_ts"],
            "ts_end": parsed["end_ts"],
            "duration_ms": parsed["duration_ms"],
            "command": f"claude: {parsed['prompts'][0]}" if parsed["prompts"] else "claude: (empty session)",
            "exit_code": 0,
            "cwd": parsed["project"],
            "tmux_pane": None,
            "gap_ms": None,
            "claude_session_id": parsed["session_id"],
            "claude_prompts": parsed["prompts"],
            "claude_tools": parsed["tools"],
            "claude_files": parsed["files_touched"],
            "claude_project": parsed["project"],
            "claude_prompt_count": parsed["prompt_count"],
            "claude_is_error_count": parsed["is_error_count"],
        }

        date_str = datetime.fromtimestamp(parsed["start_ts"] / 1000).strftime("%Y-%m-%d")
        events_path = config.events_path(date_str)
        with open(events_path, "a") as f:
            f.write(json.dumps(event_dict, default=str) + "\n")

        state.mark_session_processed(slug, session_uuid)
        ingested_count += 1

    if ingested_count:
        logger.info("Ingested %d Claude Code sessions", ingested_count)
        state.save(config.state_path)


def _get_new_events(config: Config, state: DaemonState) -> list:
    from ambient.capture.reader import read_events

    if state.last_analyzed_ts > 0:
        cursor_dt = datetime.fromtimestamp(state.last_analyzed_ts / 1000)
    else:
        cursor_dt = datetime.now() - timedelta(days=30)

    return read_events(config, start=cursor_dt, end=datetime.now())


def _run_analysis(config: Config, events: list, client=None) -> dict | None:
    from ambient.detect.compression import detect_compression
    from ambient.detect.pauses import classify
    from ambient.detect.projects import detect_project_allocation
    from ambient.detect.prompt_patterns import detect_prompt_patterns
    from ambient.present.narrator import narrate_batch

    compression = detect_compression(events, config)
    pauses = classify(events, config)
    project_allocation = detect_project_allocation(events, config)
    prompt_patterns = detect_prompt_patterns(events, config)

    # Notify on stuck episodes (best-effort, never crash daemon)
    try:
        from ambient.present.notify import notify_stuck
        notify_stuck(pauses.classifications, config)
    except Exception as e:
        logger.error("Stuck notification failed: %s", e)

    # Extract claude_session events for the batch prompt
    claude_sessions = [
        {
            "duration_ms": e.duration_ms,
            "claude_prompt_count": e.claude_prompt_count or 0,
            "claude_project": e.claude_project or e.cwd,
            "claude_prompts": e.claude_prompts or [e.command.removeprefix("claude: ")],
        }
        for e in events if e.type == "claude_session"
    ] or None

    return narrate_batch(compression, pauses, config, claude_sessions=claude_sessions,
                         project_allocation=project_allocation, client=client)


def _check_summaries(config: Config, state: DaemonState, client=None) -> None:
    from ambient.capture.reader import read_events
    from ambient.detect.changepoints import detect_changepoints
    from ambient.detect.compression import detect_compression
    from ambient.detect.pauses import classify
    from ambient.detect.prompt_patterns import detect_prompt_patterns
    from ambient.present.narrator import load_batch_analyses, narrate_daily
    from ambient.present.recommender import generate_recommendations

    today = datetime.now().strftime("%Y-%m-%d")

    if state.last_summary_date:
        # Scan from the day after last summary to yesterday
        start = datetime.strptime(state.last_summary_date, "%Y-%m-%d") + timedelta(days=1)
    else:
        # First run: scan last 30 days
        start = datetime.now() - timedelta(days=30)

    current = start.date()
    yesterday = (datetime.now() - timedelta(days=1)).date()

    while current <= yesterday:
        date_str = current.strftime("%Y-%m-%d")
        analysis_path = config.analysis_path(date_str)
        summary_path = config.summary_path(date_str)

        if analysis_path.exists() and not summary_path.exists():
            batch_analyses = load_batch_analyses(config, date_str)
            if batch_analyses:
                logger.info("Generating catch-up summary for %s", date_str)
                events = read_events(config, date_str=date_str)
                pause_result = classify(events, config)
                changepoints = detect_changepoints(events, config, pause_result)
                narrate_daily(batch_analyses, changepoints, config, date_str=date_str, client=client)
                state.last_summary_date = date_str
                state.save(config.state_path)

                # Generate recommendations from the day's full event data
                try:
                    compression = detect_compression(events, config)
                    prompt_patterns = detect_prompt_patterns(events, config)
                    generate_recommendations(prompt_patterns, compression, config, client=client)
                except Exception as e:
                    logger.error("Recommendation generation failed for %s: %s", date_str, e)

                # Generate coaching recommendations from session analysis
                try:
                    from ambient.detect.coaching import classify_sessions, group_stuck_patterns
                    from ambient.detect.velocity import detect_resolution_chains, compute_velocity_metrics
                    from ambient.present.recommender import generate_coaching_recommendations

                    coaching = classify_sessions(events, config)
                    stuck = group_stuck_patterns(coaching.outcomes, events)
                    outcome_map = {o.session_id: o.classification for o in coaching.outcomes}
                    chains = detect_resolution_chains(events, config, session_outcomes=outcome_map)
                    velocity = compute_velocity_metrics(chains, min_chains=config.velocity_min_chains)
                    generate_coaching_recommendations(stuck, velocity, config, client=client)
                except Exception as e:
                    logger.error("Coaching recommendation generation failed for %s: %s", date_str, e)

        current += timedelta(days=1)


def _check_weekly_summary(config: Config, state: DaemonState, client=None) -> None:
    today = datetime.now()

    # Only run on Sunday
    if today.weekday() != 6:
        return

    today_str = today.strftime("%Y-%m-%d")

    # Check if already generated this week
    if state.last_weekly_summary_date:
        last_weekly = datetime.strptime(state.last_weekly_summary_date, "%Y-%m-%d").date()
        days_since = (today.date() - last_weekly).days
        if days_since < 7:
            return

    # Gather analysis files for current + previous weeks
    from ambient.present.narrator import load_batch_analyses, narrate_weekly

    weekly_analyses = []
    week_labels = []
    num_weeks = config.weekly_min_weeks + 1  # current week + min_weeks for comparison

    for w in range(num_weeks):
        week_end = today - timedelta(weeks=w)
        week_start = week_end - timedelta(days=6)
        date_range = f"{week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}"

        days_data = []
        current_day = week_start.date()
        while current_day <= week_end.date():
            ds = current_day.strftime("%Y-%m-%d")
            analyses = load_batch_analyses(config, ds)
            if analyses:
                # Merge all batch analyses for this day into one summary
                merged = {"date": ds}
                # Take compression/pauses from first analysis that has them
                for a in analyses:
                    if "compression" in a and "compression" not in merged:
                        merged["compression"] = a["compression"]
                    if "pauses" in a and "pauses" not in merged:
                        merged["pauses"] = a["pauses"]
                    if "project_allocation" in a and "project_allocation" not in merged:
                        merged["project_allocation"] = a["project_allocation"]
                days_data.append(merged)
            current_day += timedelta(days=1)

        label = "Current week" if w == 0 else f"Week -{w}"
        weekly_analyses.append({"date_range": date_range, "days": days_data})
        week_labels.append(label)

    # Check minimum data gate
    weeks_with_data = sum(1 for w in weekly_analyses if w["days"])
    if weeks_with_data < config.weekly_min_weeks:
        logger.info(
            "Weekly summary skipped: only %d weeks with data, need %d",
            weeks_with_data, config.weekly_min_weeks,
        )
        return

    # Run coaching analysis for current week
    coaching_data = None
    try:
        from ambient.capture.reader import read_events
        from ambient.detect.coaching import classify_sessions, group_stuck_patterns
        from ambient.detect.velocity import detect_resolution_chains, compute_velocity_metrics

        week_end = today
        week_start = week_end - timedelta(days=6)
        week_events = read_events(config, start=week_start, end=week_end)

        if week_events:
            coaching = classify_sessions(week_events, config)
            stuck = group_stuck_patterns(coaching.outcomes, week_events)
            outcome_map = {o.session_id: o.classification for o in coaching.outcomes}
            chains = detect_resolution_chains(week_events, config, session_outcomes=outcome_map)
            velocity = compute_velocity_metrics(chains)

            coaching_data = {
                "outcomes": coaching.count_by_classification,
                "avg_thrash_score": coaching.avg_thrash_score,
                "velocity": {
                    "avg_ms": velocity.avg_ms,
                    "resolved_count": velocity.resolved_count,
                },
                "stuck_patterns": [
                    {
                        "project": p.project,
                        "episode_count": p.episode_count,
                        "failing_tools": p.failing_tools,
                    }
                    for p in stuck.patterns[:3]
                ],
            }
    except Exception as e:
        logger.error("Weekly coaching analysis failed: %s", e)

    logger.info("Generating weekly summary for %s", today_str)
    narrate_weekly(weekly_analyses, week_labels, config, date_str=today_str,
                   coaching_data=coaching_data, client=client)
    state.last_weekly_summary_date = today_str


def _check_recalibration(config: Config, state: DaemonState) -> None:
    from ambient.capture.reader import read_events
    from ambient.detect.pauses import calibrate

    today = datetime.now().strftime("%Y-%m-%d")

    # Check 7-day condition
    if state.last_calibration_date:
        last_cal = datetime.strptime(state.last_calibration_date, "%Y-%m-%d").date()
        days_since = (datetime.now().date() - last_cal).days
        if days_since < 7:
            return
    # No calibration date = always eligible for time condition

    # Check 200-event condition
    if state.events_since_calibration < 200:
        return

    # Read all events for calibration
    all_events = []
    if config.logs_dir.exists():
        for f in sorted(config.logs_dir.glob("events-*.jsonl")):
            ds = f.stem.replace("events-", "")
            all_events.extend(read_events(config, date_str=ds))

    if not all_events:
        return

    logger.info("Running auto-recalibration (%d events)", len(all_events))
    result = calibrate(all_events, config)
    if result.available:
        state.last_calibration_date = today
        state.events_since_calibration = 0
        logger.info("GMM recalibrated successfully")
    else:
        logger.warning("Recalibration skipped: %s", result.reason)


def daemon_tick(config: Config) -> None:
    setup_daemon_logging(config)
    logger.info("Daemon tick starting")

    # Gate 1: API key
    if not _load_api_key(config):
        logger.info("No ANTHROPIC_API_KEY available, skipping")
        return

    # Create shared API client for all calls in this tick (connection reuse + cache)
    try:
        import anthropic
        client = anthropic.Anthropic(max_retries=3)
    except ImportError:
        client = None
        logger.warning("anthropic package not installed, API calls will create individual clients")

    # Load state
    state = DaemonState.load(config.state_path)

    # Ingest Claude Code sessions from history.jsonl
    try:
        _ingest_claude_sessions(config, state)
    except Exception as e:
        logger.error("Claude history ingestion failed: %s", e)

    # Gate 2: New events (now includes any claude_session events just ingested)
    events = _get_new_events(config, state)

    # Gate 3: Lock (acquire before any work, including summary catch-up)
    if not acquire_lock(config.lock_path):
        logger.info("Lock held by another process, skipping")
        return

    try:
        if not events:
            logger.info("No new events since cursor, skipping analysis")
        else:
            # Run analysis
            logger.info("Analyzing %d events", len(events))
            _run_analysis(config, events, client=client)

            # Update cursor (exclusive: +1 so boundary event isn't re-read)
            latest_ts = max(e.ts_start for e in events)
            state.last_analyzed_ts = latest_ts + 1
            state.events_since_calibration += len(events)

            # Save state immediately after cursor update
            state.save(config.state_path)

        # Check for missing summaries (runs with or without new events)
        try:
            _check_summaries(config, state, client=client)
        except Exception as e:
            logger.error("Summary catch-up failed: %s", e)

        # Check weekly summary (after daily summaries)
        try:
            _check_weekly_summary(config, state, client=client)
        except Exception as e:
            logger.error("Weekly summary failed: %s", e)

        # Check recalibration (only meaningful when events were processed)
        try:
            _check_recalibration(config, state)
        except Exception as e:
            logger.error("Recalibration failed: %s", e)

        # Save state (summaries/recal may have updated it)
        state.save(config.state_path)

        logger.info("Daemon tick completed successfully")

    finally:
        release_lock(config.lock_path)
