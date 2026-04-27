"""NSWorkspace app-activation listener (macOS, Phase 2 Unit 7).

Subscribes to NSWorkspaceDidActivateApplicationNotification via pyobjc and
emits a single-line JSON record per activation to ~/.ambient/focus-events.jsonl.

Privacy contract (cites docs/PRIVACY.md clauses 6, 7, and Section 5):
- Captured payload: bundle_id, app_name (localized), pid, ts.
- NEVER captured: window title, document path, or any field named in the
  closed-doors table. The build_focus_record helper accepts a notification
  payload and returns a record with exactly the four allowed fields.
- Opt-in only: this listener does not start unless the user runs
  `ambient focus-enable`. Off by default.

The pure record-builder (build_focus_record) is testable without macOS or
pyobjc. The OS subscription side (subscribe) lazy-imports pyobjc so the rest
of the module imports cleanly on Linux/CI.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FocusRecord:
    """Per-activation event. Privacy-bounded: exactly these fields, no others."""
    ts: str          # ISO 8601 UTC
    source: str      # always "nsworkspace" for this listener
    event: str       # always "app_activated" for this listener today
    bundle_id: str | None
    app_name: str | None
    pid: int | None

    def to_jsonl(self) -> str:
        return json.dumps({
            "ts": self.ts,
            "source": self.source,
            "event": self.event,
            "bundle_id": self.bundle_id,
            "app_name": self.app_name,
            "pid": self.pid,
        }, ensure_ascii=False) + "\n"


def build_focus_record(payload: dict, *, ts: datetime | None = None) -> FocusRecord:
    """Convert an NSWorkspace activation payload (dict-shaped) into a FocusRecord.

    The payload arg is the result of extracting fields from
    NSWorkspaceDidActivateApplicationNotification.userInfo['NSWorkspaceApplicationKey'].
    Pure function: takes a plain dict (testable), returns a FocusRecord.

    Allowed payload keys:
        bundle_id   (str | None) — applicationBundleIdentifier
        app_name    (str | None) — localizedName
        pid         (int | None) — processIdentifier

    Any other keys in `payload` are silently ignored — the privacy contract
    means we never let extraneous fields slip through into the JSONL output.
    """
    if ts is None:
        ts = datetime.now(timezone.utc)
    bundle_id = payload.get("bundle_id")
    app_name = payload.get("app_name")
    pid = payload.get("pid")
    return FocusRecord(
        ts=ts.isoformat(),
        source="nsworkspace",
        event="app_activated",
        bundle_id=bundle_id if isinstance(bundle_id, str) else None,
        app_name=app_name if isinstance(app_name, str) else None,
        pid=pid if isinstance(pid, int) else None,
    )


def append_record(record: FocusRecord, path: Path) -> bool:
    """Append a FocusRecord to the JSONL output path. Returns True on success.

    Failures are logged and swallowed — the listener must continue running
    even if a single write fails. Never logs the record payload itself.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(record.to_jsonl())
        return True
    except OSError as e:
        logger.warning("focus_record write failed: %s", e)
        return False


def subscribe(on_event: Callable[[FocusRecord], None]) -> None:
    """Block, subscribing to NSWorkspace notifications and invoking on_event.

    Lazy-imports pyobjc; raises RuntimeError if pyobjc is unavailable. Runs
    the AppKit notification loop indefinitely; the caller is responsible for
    the process lifecycle (signal handling, lock release on exit).
    """
    try:
        from AppKit import NSWorkspace  # type: ignore[import-not-found]
        from Foundation import NSObject  # type: ignore[import-not-found]
        import objc  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "pyobjc is required for the NSWorkspace focus listener. "
            "Install with: pip install pyobjc-framework-Cocoa"
        ) from e

    workspace = NSWorkspace.sharedWorkspace()
    notification_center = workspace.notificationCenter()

    class _Observer(NSObject):
        def appActivated_(self, notification):
            try:
                user_info = notification.userInfo()
                running_app = user_info.objectForKey_("NSWorkspaceApplicationKey")
                if running_app is None:
                    return
                payload = {
                    "bundle_id": _coerce_str(running_app.bundleIdentifier()),
                    "app_name": _coerce_str(running_app.localizedName()),
                    "pid": _coerce_int(running_app.processIdentifier()),
                }
                record = build_focus_record(payload)
                on_event(record)
            except Exception as exc:  # pragma: no cover — runtime safety
                logger.warning("focus_listener observer error: %s", exc)

    observer = _Observer.alloc().init()
    notification_center.addObserver_selector_name_object_(
        observer,
        objc.selector(observer.appActivated_, signature=b"v@:@"),
        "NSWorkspaceDidActivateApplicationNotification",
        None,
    )

    logger.info("focus_listener subscribed; entering run loop")
    # Block on the AppKit run loop until SIGTERM/SIGINT.
    from PyObjCTools import AppHelper  # type: ignore[import-not-found]
    AppHelper.runConsoleEventLoop(installInterrupt=True)


def _coerce_str(value) -> str | None:
    if value is None:
        return None
    s = str(value)
    return s if s else None


def _coerce_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
