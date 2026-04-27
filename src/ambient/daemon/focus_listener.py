"""Focus listener daemon entry point (Phase 2 Unit 7).

Long-lived process registered as a separate launchd agent (`com.ambient.focus-listener`).
Subscribes to NSWorkspace app-activation events and writes JSONL records to
~/.ambient/focus-events.jsonl. Coexists with the existing tick daemon
(`com.ambient.daemon`) — different label, different lock, different log file.

Privacy contract: cites docs/PRIVACY.md clauses 6 and 7. Records contain only
bundle_id, app_name (localized), pid, and ts. Never window title or document
path. Off by default; user opts in via `ambient focus-enable`.
"""

from __future__ import annotations

import logging
import signal
import sys

from ambient.capture.nsworkspace_listener import (
    FocusRecord,
    append_record,
    subscribe,
)
from ambient.config import Config
from ambient.daemon.lock import acquire_lock, release_lock

logger = logging.getLogger(__name__)


def run(config: Config) -> int:
    """Main entry point. Returns process exit code.

    Acquires the focus-listener lock, configures logging, registers signal
    handlers for clean shutdown, and blocks on NSWorkspace notifications.
    """
    config.ensure_dirs()
    config.daemon_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(config)

    lock_path = config.focus_listener_lock_path
    if not acquire_lock(lock_path):
        logger.warning("focus_listener already running; exiting")
        return 1

    def _handle_signal(signum, _frame):
        logger.info("focus_listener received signal %d, shutting down", signum)
        release_lock(lock_path)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    def _on_event(record: FocusRecord) -> None:
        # Never log the record payload — privacy contract.
        append_record(record, config.focus_events_path)

    try:
        subscribe(_on_event)
    except RuntimeError as exc:
        logger.error("focus_listener failed to subscribe: %s", exc)
        release_lock(lock_path)
        return 2
    finally:
        release_lock(lock_path)
    return 0


def _configure_logging(config: Config) -> None:
    """Route logger output to focus_listener_log_path. Never includes payloads."""
    log_path = config.focus_listener_log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    root = logging.getLogger("ambient")
    root.setLevel(logging.INFO)
    # Avoid duplicating handlers if run() is called more than once in tests.
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path)
               for h in root.handlers):
        root.addHandler(handler)
