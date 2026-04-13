"""macOS notifications for stuck episodes detected by the pause classifier."""

import logging
import subprocess
import time
from pathlib import Path

from ambient.config import Config
from ambient.detect.pauses import PauseClassification

logger = logging.getLogger(__name__)

STUCK_THRESHOLD_MS = 600_000  # 10 minutes


def notify_stuck(classifications: list[PauseClassification], config: Config) -> None:
    """Send a macOS notification if a stuck episode exceeds 10 minutes.

    Rate limited to at most one notification per call.
    """
    stuck = [c for c in classifications if c.label == "stuck" and c.gap_ms > STUCK_THRESHOLD_MS]
    if not stuck:
        return

    # Pick the worst one
    worst = max(stuck, key=lambda c: c.gap_ms)
    minutes = worst.gap_ms // 60_000
    # Escape double-quotes to prevent AppleScript injection from raw command strings
    cmd_preview = worst.preceding_command[:60].replace("\\", "\\\\").replace('"', '\\"')
    message = f"Stuck for {minutes}m after: {cmd_preview}"

    # Use the AmbientAI.app bundle so macOS attributes notifications to "Ambient AI"
    # instead of "python3.14". Falls back to raw osascript if the bundle isn't found.
    app_notify = Path(__file__).parent.parent.parent.parent / "resources" / "AmbientAI.app" / "Contents" / "MacOS" / "notify"

    try:
        if app_notify.exists():
            subprocess.run([str(app_notify), message], timeout=5, capture_output=True)
        else:
            subprocess.run(
                ["osascript", "-e", f'display notification "{message}" with title "Ambient AI"'],
                timeout=5,
                capture_output=True,
            )
    except Exception as e:
        logger.error("Failed to send stuck notification: %s", e)
