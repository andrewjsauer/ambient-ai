"""tmux focus-hook installer/uninstaller (Phase 2 Unit 8).

Installs tmux global hooks (pane-focus-in, pane-focus-out, window-focused)
that invoke scripts/tmux/ambient-focus-hook.sh, which writes a JSONL line per
event to ~/.ambient/focus-events.jsonl.

Privacy contract (cites docs/PRIVACY.md clauses 6, 7):
- Hooks invoke a shell script that captures structural identifiers only
  (event name, pane id, window index, session name) — never pane_title,
  pane_current_command, or pane_current_path.
- Off by default. User opts in via `ambient tmux-focus-enable`.
- Idempotent: install can be re-run safely; uninstall removes only Ambient-
  managed hooks via the `# ambient-managed` sentinel.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# tmux hooks we install. window-focused fires when the active window changes;
# pane-focus-{in,out} fire when focus moves between panes within a window
# (only when tmux's `focus-events` option is set on).
HOOKS = ("pane-focus-in", "pane-focus-out", "window-focused")

SENTINEL = "# ambient-managed"


def hook_script_path() -> Path:
    """Locate the shell hook script. Resolved relative to the package root."""
    # src/ambient/capture/tmux_focus.py → repo root → scripts/tmux/...
    pkg_root = Path(__file__).resolve().parents[3]
    return pkg_root / "scripts" / "tmux" / "ambient-focus-hook.sh"


def tmux_available() -> bool:
    """Return True if a tmux binary is on PATH."""
    return shutil.which("tmux") is not None


def install_hooks(focus_events_path: Path) -> None:
    """Install Ambient-managed tmux hooks. Idempotent.

    Each hook command exports AMBIENT_FOCUS_EVENTS_PATH so the shell script
    knows where to append. The sentinel comment lets uninstall_hooks remove
    only what we installed.
    """
    if not tmux_available():
        raise RuntimeError("tmux not found on PATH; nothing to install.")

    script = hook_script_path()
    if not script.exists():
        raise RuntimeError(f"tmux hook script missing: {script}")
    if not script.is_file():
        raise RuntimeError(f"tmux hook path is not a file: {script}")

    # Remove any existing Ambient-managed hooks before installing fresh ones,
    # so re-running enable doesn't stack duplicates.
    uninstall_hooks()

    for hook in HOOKS:
        # Build the run-shell payload. Quote the events path; sentinel at end.
        # Use single quotes around the outer payload so embedded path/quoting
        # stays in tmux's set-hook arg.
        cmd = (
            f'AMBIENT_FOCUS_EVENTS_PATH="{focus_events_path}" '
            f'"{script}" {hook} {SENTINEL}'
        )
        subprocess.run(
            ["tmux", "set-hook", "-g", hook, f"run-shell {_shell_quote(cmd)}"],
            check=True,
        )


def uninstall_hooks() -> None:
    """Remove only Ambient-managed hooks; leave user-managed ones alone."""
    if not tmux_available():
        return
    for hook in HOOKS:
        # Inspect current hook value; only unset if our sentinel is present.
        result = subprocess.run(
            ["tmux", "show-hook", "-g", hook],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            continue
        if SENTINEL not in result.stdout:
            continue
        subprocess.run(
            ["tmux", "set-hook", "-gu", hook],
            capture_output=True,
        )


def is_installed() -> bool:
    """Return True if at least one hook is Ambient-managed."""
    if not tmux_available():
        return False
    for hook in HOOKS:
        result = subprocess.run(
            ["tmux", "show-hook", "-g", hook],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and SENTINEL in result.stdout:
            return True
    return False


def _shell_quote(s: str) -> str:
    """Wrap `s` for safe inclusion as a single tmux set-hook payload arg."""
    # tmux set-hook accepts a single string after the hook name; we pass the
    # entire payload here. Wrap in single quotes; escape any embedded single
    # quotes by closing-quoting-reopening.
    return "'" + s.replace("'", "'\\''") + "'"
