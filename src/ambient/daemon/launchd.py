import os
import plistlib
import subprocess
import sys
from pathlib import Path

from ambient.config import Config

AGENT_LABEL = "com.ambient.daemon"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / "com.ambient.daemon.plist"

# v4 Phase 2 Unit 7: separate launchd agent for the focus listener. Different
# label so it coexists with the tick daemon. KeepAlive: true so it restarts on
# crash; RunAtLoad: true so opt-in starts immediately on `ambient focus-enable`.
FOCUS_LISTENER_LABEL = "com.ambient.focus-listener"
FOCUS_LISTENER_PLIST_PATH = (
    Path.home() / "Library" / "LaunchAgents" / "com.ambient.focus-listener.plist"
)


def generate_plist(config: Config) -> dict:
    return {
        "Label": AGENT_LABEL,
        "ProgramArguments": [sys.executable, "-m", "ambient.cli", "daemon-tick"],
        "StartInterval": 1800,
        "RunAtLoad": False,
        "StandardOutPath": str(config.daemon_log_path).replace(
            "daemon.log", "launchd-stdout.log"
        ),
        "StandardErrorPath": str(config.daemon_log_path).replace(
            "daemon.log", "launchd-stderr.log"
        ),
    }


def install_agent(config: Config) -> None:
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    plist = generate_plist(config)
    with open(PLIST_PATH, "wb") as f:
        plistlib.dump(plist, f)
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{uid}", str(PLIST_PATH)],
        check=True,
    )


def uninstall_agent() -> None:
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}/{AGENT_LABEL}"],
        check=True,
    )
    if PLIST_PATH.exists():
        PLIST_PATH.unlink()


def is_agent_loaded() -> bool:
    uid = os.getuid()
    result = subprocess.run(
        ["launchctl", "print", f"gui/{uid}/{AGENT_LABEL}"],
        capture_output=True,
    )
    return result.returncode == 0


# --- v4 Phase 2 Unit 7: focus listener launchd integration ---


def generate_focus_listener_plist(config: Config) -> dict:
    """Generate the launchd plist for the focus-listener daemon.

    Different label, KeepAlive (restart on crash), RunAtLoad (opt-in start
    immediately). Coexists with com.ambient.daemon.
    """
    return {
        "Label": FOCUS_LISTENER_LABEL,
        "ProgramArguments": [sys.executable, "-m", "ambient.cli", "focus-listener-run"],
        "KeepAlive": True,
        "RunAtLoad": True,
        "StandardOutPath": str(config.focus_listener_log_path).replace(
            "focus-listener.log", "focus-listener-stdout.log"
        ),
        "StandardErrorPath": str(config.focus_listener_log_path).replace(
            "focus-listener.log", "focus-listener-stderr.log"
        ),
    }


def install_focus_listener(config: Config) -> None:
    """Idempotent: writes the plist and bootstraps the agent. Re-running is safe."""
    FOCUS_LISTENER_PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    plist = generate_focus_listener_plist(config)
    with open(FOCUS_LISTENER_PLIST_PATH, "wb") as f:
        plistlib.dump(plist, f)
    uid = os.getuid()
    # Bootout first if already loaded so re-install picks up new config without
    # leaving the old agent running.
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}/{FOCUS_LISTENER_LABEL}"],
        capture_output=True,
    )
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{uid}", str(FOCUS_LISTENER_PLIST_PATH)],
        check=True,
    )


def uninstall_focus_listener() -> None:
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}/{FOCUS_LISTENER_LABEL}"],
        capture_output=True,
    )
    if FOCUS_LISTENER_PLIST_PATH.exists():
        FOCUS_LISTENER_PLIST_PATH.unlink()


def is_focus_listener_loaded() -> bool:
    uid = os.getuid()
    result = subprocess.run(
        ["launchctl", "print", f"gui/{uid}/{FOCUS_LISTENER_LABEL}"],
        capture_output=True,
    )
    return result.returncode == 0
