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
import shlex
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

    Quoting: every interpolated value passes through shlex.quote so paths
    containing single/double quotes, dollar signs, backticks, or shell
    substitutions cannot break out of the run-shell payload. Adversarial
    review (adv-1) found the prior hand-rolled _shell_quote only escaped
    single quotes, leaving a real injection path on weird $HOME values.

    Append vs replace: uses `set-hook -ga` (append) so any pre-existing
    user-managed hook on the same event continues to fire alongside the
    Ambient hook. Plain `-g` (the previous code) clobbered user hooks
    silently; uninstall would leave the slot permanently empty.
    """
    if not tmux_available():
        raise RuntimeError("tmux not found on PATH; nothing to install.")

    script = hook_script_path()
    if not script.exists():
        raise RuntimeError(f"tmux hook script missing: {script}")
    if not script.is_file():
        raise RuntimeError(f"tmux hook path is not a file: {script}")

    # Remove any existing Ambient-managed hooks before installing fresh ones,
    # so re-running enable doesn't stack our own duplicates. uninstall_hooks
    # is sentinel-aware so it leaves user-managed hooks alone.
    uninstall_hooks()

    quoted_path = shlex.quote(str(focus_events_path))
    quoted_script = shlex.quote(str(script))

    for hook in HOOKS:
        cmd = (
            f"AMBIENT_FOCUS_EVENTS_PATH={quoted_path} "
            f"{quoted_script} {hook} {SENTINEL}"
        )
        # tmux's run-shell takes a single shell-string arg; wrap in shlex.quote
        # so the outer tmux-arg layer is also bulletproof.
        run_shell_arg = f"run-shell {shlex.quote(cmd)}"
        subprocess.run(
            ["tmux", "set-hook", "-ga", hook, run_shell_arg],
            check=True,
        )


def uninstall_hooks() -> None:
    """Remove Ambient-managed hooks. Skips slots without our sentinel.

    Limitation: tmux's set-hook does not support selectively removing one
    entry from an appended array — the only "remove" verb is `-gu` (unset
    the whole slot). When install_hooks added our hook via `-ga` on top of
    a pre-existing user hook, this uninstall will remove BOTH. Documented in
    the CLI help; users with custom hooks can re-add them after disable.
    """
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
