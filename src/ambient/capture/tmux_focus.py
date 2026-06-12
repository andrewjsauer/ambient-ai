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

# tmux hooks we install. session-window-changed fires when the active window
# changes within a session; pane-focus-{in,out} fire when focus moves between
# panes within a window (only when tmux's `focus-events` option is set on).
# Older docs and a draft of the plan referenced `window-focused`; that name
# does not exist in tmux 3.x — `session-window-changed` is the correct hook.
HOOKS = ("pane-focus-in", "pane-focus-out", "session-window-changed")

SENTINEL = "# ambient-managed"

# tmux user-option slot where we stash the user's prior `focus-events` value.
# pane-focus-{in,out} hooks only fire when `focus-events` is on, so install
# flips it on; uninstall restores whatever was there before so we don't leak
# state out of ambient's footprint.
PRIOR_FOCUS_EVENTS_OPT = "@ambient-prior-focus-events"


def _save_and_enable_focus_events() -> None:
    """Save the current `focus-events` value into a tmux user option, then
    set `focus-events on`. Idempotent: re-running won't overwrite the saved
    prior with our own ``on``.
    """
    saved = subprocess.run(
        ["tmux", "show-options", "-gv", PRIOR_FOCUS_EVENTS_OPT],
        capture_output=True, text=True,
    )
    if saved.returncode != 0 or not saved.stdout.strip():
        cur = subprocess.run(
            ["tmux", "show-options", "-gv", "focus-events"],
            capture_output=True, text=True,
        )
        prior = cur.stdout.strip() if cur.returncode == 0 and cur.stdout.strip() else "off"
        subprocess.run(
            ["tmux", "set-option", "-g", PRIOR_FOCUS_EVENTS_OPT, prior],
            check=True,
        )
    subprocess.run(
        ["tmux", "set-option", "-g", "focus-events", "on"],
        check=True,
    )


def _restore_focus_events() -> None:
    """Restore `focus-events` to whatever it was before install_hooks ran.
    No-op if no prior value was saved (uninstall called without a matching
    install, or the prior value was already restored).
    """
    saved = subprocess.run(
        ["tmux", "show-options", "-gv", PRIOR_FOCUS_EVENTS_OPT],
        capture_output=True, text=True,
    )
    if saved.returncode != 0 or not saved.stdout.strip():
        return
    prior = saved.stdout.strip()
    subprocess.run(
        ["tmux", "set-option", "-g", "focus-events", prior],
        capture_output=True,
    )
    subprocess.run(
        ["tmux", "set-option", "-gu", PRIOR_FOCUS_EVENTS_OPT],
        capture_output=True,
    )


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

    # pane-focus-{in,out} only fire with `focus-events on`; flip it on now and
    # remember the user's prior value so uninstall can restore it.
    try:
        _save_and_enable_focus_events()

        quoted_path = shlex.quote(str(focus_events_path))
        quoted_script = shlex.quote(str(script))

        for hook in HOOKS:
            cmd = (
                f"AMBIENT_FOCUS_EVENTS_PATH={quoted_path} "
                f"{quoted_script} {hook} {SENTINEL}"
            )
            # tmux's run-shell takes a single shell-string arg; wrap in
            # shlex.quote so the outer tmux-arg layer is also bulletproof.
            run_shell_arg = f"run-shell {shlex.quote(cmd)}"
            subprocess.run(
                ["tmux", "set-hook", "-ga", hook, run_shell_arg],
                check=True,
            )
    except subprocess.CalledProcessError as e:
        # tmux binary present but no server running: set-option/set-hook fail.
        # Raise RuntimeError so the CLI handler prints a clean message instead
        # of a traceback; uninstall_hooks already ran, so nothing is left
        # half-installed.
        raise RuntimeError(
            "tmux is installed but no server appears to be running; "
            "start a tmux session and re-run tmux-focus-enable"
        ) from e


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
    # Restore focus-events to whatever the user had before install ran.
    _restore_focus_events()


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
