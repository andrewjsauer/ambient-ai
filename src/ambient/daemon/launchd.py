import os
import plistlib
import subprocess
import sys
from pathlib import Path

from ambient.config import Config

AGENT_LABEL = "com.ambient.daemon"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / "com.ambient.daemon.plist"


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
