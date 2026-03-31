import os
import time

import pytest

from ambient.daemon.lock import acquire_lock, is_locked, release_lock


class TestAcquireLock:
    def test_succeeds_when_no_lock_exists(self, tmp_path):
        lock = tmp_path / "daemon.lock"
        assert acquire_lock(lock) is True
        assert lock.read_text().strip() == str(os.getpid())

    def test_fails_when_held_by_current_process(self, tmp_path):
        lock = tmp_path / "daemon.lock"
        lock.write_text(str(os.getpid()))
        assert acquire_lock(lock) is False

    def test_succeeds_when_stale_mtime_and_dead_pid(self, tmp_path):
        lock = tmp_path / "daemon.lock"
        dead_pid = 99999
        lock.write_text(str(dead_pid))
        # Set mtime to 2 hours ago
        old_time = time.time() - 7200
        os.utime(lock, (old_time, old_time))
        assert acquire_lock(lock) is True
        assert lock.read_text().strip() == str(os.getpid())

    def test_fails_when_stale_mtime_but_pid_alive(self, tmp_path):
        lock = tmp_path / "daemon.lock"
        # Use current PID as the alive PID
        lock.write_text(str(os.getpid()))
        old_time = time.time() - 7200
        os.utime(lock, (old_time, old_time))
        assert acquire_lock(lock) is False

    def test_succeeds_when_invalid_content(self, tmp_path):
        lock = tmp_path / "daemon.lock"
        lock.write_text("not-a-pid\ngarbage")
        assert acquire_lock(lock) is True
        assert lock.read_text().strip() == str(os.getpid())


class TestReleaseLock:
    def test_removes_lock_held_by_current_pid(self, tmp_path):
        lock = tmp_path / "daemon.lock"
        acquire_lock(lock)
        assert lock.exists()
        release_lock(lock)
        assert not lock.exists()

    def test_noop_when_held_by_different_pid(self, tmp_path):
        lock = tmp_path / "daemon.lock"
        lock.write_text("99999")
        release_lock(lock)
        assert lock.exists()
        assert lock.read_text().strip() == "99999"


class TestIsLocked:
    def test_returns_correct_status_when_locked(self, tmp_path):
        lock = tmp_path / "daemon.lock"
        acquire_lock(lock)
        locked, info = is_locked(lock)
        assert locked is True
        assert info["pid"] == os.getpid()
        assert "age_minutes" in info

    def test_returns_false_when_no_lock(self, tmp_path):
        lock = tmp_path / "daemon.lock"
        locked, info = is_locked(lock)
        assert locked is False
        assert info == {}

    def test_returns_false_when_pid_dead(self, tmp_path):
        lock = tmp_path / "daemon.lock"
        lock.write_text("99999")
        locked, info = is_locked(lock)
        assert locked is False
        assert info["pid"] == 99999
