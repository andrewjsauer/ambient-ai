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
    from ambient.daemon.claude_history import (
        filter_completed_sessions,
        group_into_sessions,
        read_new_history_entries,
        session_to_event,
    )

    entries, new_line_count = read_new_history_entries(
        config.claude_history_path, state.last_claude_history_line
    )
    if not entries:
        state.last_claude_history_line = max(new_line_count, state.last_claude_history_line)
        state.save(config.state_path)
        return

    sessions = group_into_sessions(entries)
    completed, _ = filter_completed_sessions(sessions)

    # Always advance cursor after reading, even if no sessions completed yet
    state.last_claude_history_line = max(new_line_count, state.last_claude_history_line)

    if not completed:
        state.save(config.state_path)
        return

    config.ensure_dirs()
    for session in completed:
        event_dict = session_to_event(session)
        date_str = datetime.fromtimestamp(session["ts_start"] / 1000).strftime("%Y-%m-%d")
        events_path = config.events_path(date_str)
        with open(events_path, "a") as f:
            f.write(json.dumps(event_dict, default=str) + "\n")

    logger.info("Ingested %d Claude Code sessions", len(completed))
    state.save(config.state_path)


def _get_new_events(config: Config, state: DaemonState) -> list:
    from ambient.capture.reader import read_events

    if state.last_analyzed_ts > 0:
        cursor_dt = datetime.fromtimestamp(state.last_analyzed_ts / 1000)
    else:
        cursor_dt = datetime.now() - timedelta(days=30)

    return read_events(config, start=cursor_dt, end=datetime.now())


def _run_analysis(config: Config, events: list) -> dict | None:
    from ambient.detect.compression import detect_compression
    from ambient.detect.pauses import classify
    from ambient.present.narrator import narrate_batch

    compression = detect_compression(events, config)
    pauses = classify(events, config)

    # Extract claude_session events for the batch prompt
    claude_sessions = [
        {"duration_ms": e.duration_ms, "claude_prompt_count": 0,
         "claude_project": e.cwd, "claude_prompts": [e.command.removeprefix("claude: ")]}
        for e in events if e.type == "claude_session"
    ] or None

    return narrate_batch(compression, pauses, config, claude_sessions=claude_sessions)


def _check_summaries(config: Config, state: DaemonState) -> None:
    from ambient.capture.reader import read_events
    from ambient.detect.changepoints import detect_changepoints
    from ambient.detect.pauses import classify
    from ambient.present.narrator import load_batch_analyses, narrate_daily

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
                narrate_daily(batch_analyses, changepoints, config, date_str=date_str)
                state.last_summary_date = date_str

        current += timedelta(days=1)


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
            _run_analysis(config, events)

            # Update cursor (exclusive: +1 so boundary event isn't re-read)
            latest_ts = max(e.ts_start for e in events)
            state.last_analyzed_ts = latest_ts + 1
            state.events_since_calibration += len(events)

            # Save state immediately after cursor update
            state.save(config.state_path)

        # Check for missing summaries (runs with or without new events)
        try:
            _check_summaries(config, state)
        except Exception as e:
            logger.error("Summary catch-up failed: %s", e)

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
