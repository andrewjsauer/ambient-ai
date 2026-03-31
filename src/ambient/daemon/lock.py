import os
import time
from pathlib import Path

STALE_THRESHOLD_SECONDS = 60 * 60  # 60 minutes


def acquire_lock(lock_path: Path) -> bool:
    """Write current PID to lock file and return True on success.

    If lock exists and is held by a live process, return False.
    If lock exists but mtime > 60 minutes AND PID is dead, break stale lock.
    If lock file contains invalid content, treat as stale (acquirable).
    """
    if lock_path.exists():
        try:
            content = lock_path.read_text().strip()
            existing_pid = int(content)
        except (ValueError, OSError):
            # Invalid content — treat as stale, remove and acquire
            _remove_lock(lock_path)
            return _write_and_verify(lock_path)

        # Check if the process is alive
        if _pid_alive(existing_pid):
            return False

        # PID is dead — only break if mtime > 60 minutes
        try:
            age = time.time() - lock_path.stat().st_mtime
        except OSError:
            # File disappeared between checks
            return _write_and_verify(lock_path)

        if age > STALE_THRESHOLD_SECONDS:
            _remove_lock(lock_path)
            return _write_and_verify(lock_path)

        # PID is dead but lock is fresh — still break it, dead is dead
        _remove_lock(lock_path)
        return _write_and_verify(lock_path)

    return _write_and_verify(lock_path)


def release_lock(lock_path: Path) -> None:
    """Remove lock file only if held by current PID."""
    if not lock_path.exists():
        return
    try:
        content = lock_path.read_text().strip()
        pid = int(content)
    except (ValueError, OSError):
        return
    if pid == os.getpid():
        _remove_lock(lock_path)


def is_locked(lock_path: Path) -> tuple[bool, dict]:
    """Return (locked, info) where info has pid and age_minutes."""
    if not lock_path.exists():
        return False, {}
    try:
        content = lock_path.read_text().strip()
        pid = int(content)
    except (ValueError, OSError):
        return False, {}

    if not _pid_alive(pid):
        return False, {"pid": pid, "age_minutes": _age_minutes(lock_path)}

    return True, {"pid": pid, "age_minutes": _age_minutes(lock_path)}


def _write_and_verify(lock_path: Path) -> bool:
    """Write PID to lock file and re-read to verify (race detection)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    my_pid = os.getpid()
    lock_path.write_text(str(my_pid))
    # Re-read to detect two-writer race
    try:
        content = lock_path.read_text().strip()
        return int(content) == my_pid
    except (ValueError, OSError):
        return False


def _remove_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _age_minutes(lock_path: Path) -> float:
    try:
        age_seconds = time.time() - lock_path.stat().st_mtime
        return round(age_seconds / 60, 1)
    except OSError:
        return 0.0
