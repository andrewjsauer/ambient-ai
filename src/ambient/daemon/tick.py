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
    return narrate_batch(compression, pauses, config)


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

    # Gate 2: New events
    events = _get_new_events(config, state)
    if not events:
        logger.info("No new events since cursor, skipping analysis")
        # Still check summaries and recalibration even without new events
        _check_summaries(config, state)
        state.save(config.state_path)
        return

    # Gate 3: Lock
    if not acquire_lock(config.lock_path):
        logger.info("Lock held by another process, skipping")
        return

    try:
        # Run analysis
        logger.info("Analyzing %d events", len(events))
        result = _run_analysis(config, events)

        # Update cursor (exclusive: +1 so boundary event isn't re-read)
        latest_ts = max(e.ts_start for e in events)
        state.last_analyzed_ts = latest_ts + 1
        state.events_since_calibration += len(events)

        # Save state immediately after cursor update (before summaries/recal)
        state.save(config.state_path)

        # Check for missing summaries
        _check_summaries(config, state)

        # Check recalibration
        _check_recalibration(config, state)

        # Save state again (summaries/recal may have updated it)
        state.save(config.state_path)

        logger.info("Daemon tick completed successfully")

    finally:
        release_lock(config.lock_path)
