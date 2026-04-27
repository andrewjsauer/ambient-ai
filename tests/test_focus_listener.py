"""Tests for the NSWorkspace focus listener (Phase 2 Unit 7).

Privacy-clause-citing tests verify that:
- The captured payload is restricted to bundle_id, app_name, pid, ts.
- Window titles, document paths, and other PII fields never appear in records,
  even when injected into the payload dict (clause 6 + Section 2 closed doors
  in docs/PRIVACY.md).
- The listener is off by default (clause 6).
- Failures degrade silently — listener daemon must keep running on write errors.
"""

import json
import os
from pathlib import Path

import pytest

from ambient.capture.nsworkspace_listener import (
    FocusRecord,
    append_record,
    build_focus_record,
)
from ambient.config import Config
from ambient.daemon.launchd import (
    FOCUS_LISTENER_LABEL,
    generate_focus_listener_plist,
)


def _config(tmp_path: Path) -> Config:
    cfg = Config(base_dir=tmp_path / ".ambient")
    cfg.ensure_dirs()
    cfg.daemon_dir.mkdir(parents=True, exist_ok=True)
    return cfg


# ---------- build_focus_record (privacy-clause tests) ----------

class TestBuildFocusRecord:
    def test_extracts_allowed_fields(self):
        payload = {
            "bundle_id": "com.apple.Safari",
            "app_name": "Safari",
            "pid": 12345,
        }
        record = build_focus_record(payload)
        assert record.bundle_id == "com.apple.Safari"
        assert record.app_name == "Safari"
        assert record.pid == 12345
        assert record.source == "nsworkspace"
        assert record.event == "app_activated"

    def test_window_title_in_payload_is_silently_dropped(self):
        # PRIVACY clause 6 + Section 2: window titles are never persisted.
        # Even if NSWorkspace future-versions started exposing the title in
        # userInfo, this builder ignores any field outside the allowlist.
        payload = {
            "bundle_id": "com.example.app",
            "app_name": "Example",
            "pid": 1,
            "window_title": "secrets.env — Vim",  # MUST be ignored
            "document_path": "/Users/x/secret.txt",  # MUST be ignored
        }
        record = build_focus_record(payload)
        # Field set on FocusRecord is fixed by the dataclass; serializing the
        # record must yield exactly the four payload fields.
        line = record.to_jsonl()
        parsed = json.loads(line)
        assert set(parsed.keys()) == {"ts", "source", "event", "bundle_id", "app_name", "pid"}
        assert "window_title" not in line
        assert "document_path" not in line

    def test_missing_fields_become_none(self):
        record = build_focus_record({})
        assert record.bundle_id is None
        assert record.app_name is None
        assert record.pid is None
        assert record.ts  # ts always present

    def test_wrong_type_in_payload_becomes_none(self):
        # If pyobjc returns an unexpected type (e.g. NSNumber that fails
        # int coerce), record holds None rather than crashing or coercing badly.
        record = build_focus_record({
            "bundle_id": 42,  # wrong type → None
            "app_name": ["a", "b"],  # wrong type → None
            "pid": "not-a-pid",  # wrong type → None
        })
        assert record.bundle_id is None
        assert record.app_name is None
        assert record.pid is None

    def test_to_jsonl_terminates_with_newline(self):
        record = build_focus_record({"bundle_id": "x", "app_name": "y", "pid": 1})
        line = record.to_jsonl()
        assert line.endswith("\n")
        # Single line — no embedded newlines from the JSON encoding.
        assert line.count("\n") == 1


# ---------- append_record ----------

class TestAppendRecord:
    def test_appends_to_existing_file(self, tmp_path):
        path = tmp_path / "focus-events.jsonl"
        r1 = build_focus_record({"bundle_id": "a", "app_name": "A", "pid": 1})
        r2 = build_focus_record({"bundle_id": "b", "app_name": "B", "pid": 2})
        append_record(r1, path)
        append_record(r2, path)
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["bundle_id"] == "a"
        assert json.loads(lines[1])["bundle_id"] == "b"

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "subdir" / "focus-events.jsonl"
        record = build_focus_record({"bundle_id": "x", "app_name": "X", "pid": 1})
        assert append_record(record, path) is True
        assert path.exists()

    def test_write_failure_returns_false_no_raise(self, tmp_path):
        # Path that can't be written — point at /dev/full-style failure.
        # On macOS we simulate with a directory path (write to a path that's a directory).
        bad_path = tmp_path / "is-a-dir"
        bad_path.mkdir()
        record = build_focus_record({"bundle_id": "x", "app_name": "X", "pid": 1})
        result = append_record(record, bad_path)
        assert result is False  # Did not raise; returned False so caller continues.


# ---------- launchd plist generation ----------

class TestFocusListenerPlist:
    def test_plist_uses_correct_label(self, tmp_path):
        plist = generate_focus_listener_plist(_config(tmp_path))
        assert plist["Label"] == FOCUS_LISTENER_LABEL
        assert plist["Label"] != "com.ambient.daemon"  # different from tick agent

    def test_plist_keepalive_for_long_lived_process(self, tmp_path):
        plist = generate_focus_listener_plist(_config(tmp_path))
        # Phase 2 review: KeepAlive is now conditional — restart on crash but
        # NOT on clean exit (REL-01 / adv-5: prevents crash loop on missing
        # pyobjc and respects `ambient focus-disable` SIGTERM).
        assert plist["KeepAlive"] == {"SuccessfulExit": False, "Crashed": True}
        assert plist["RunAtLoad"] is True
        assert plist["ThrottleInterval"] == 30  # respawn rate cap

    def test_plist_log_paths_use_explicit_config_fields(self, tmp_path):
        # Earlier code derived stdout/stderr paths via str.replace on the
        # focus_listener_log_path Config field — silent no-op if the user
        # customized that path (K2 P1). Now they come from explicit fields.
        from ambient.config import Config
        cfg = Config(base_dir=tmp_path / ".ambient")
        # Simulate a user who renamed the log file.
        cfg.focus_listener_log_path = tmp_path / ".ambient" / "renamed.log"
        cfg.focus_listener_stdout_path = tmp_path / ".ambient" / "out.log"
        cfg.focus_listener_stderr_path = tmp_path / ".ambient" / "err.log"
        plist = generate_focus_listener_plist(cfg)
        assert plist["StandardOutPath"] == str(cfg.focus_listener_stdout_path)
        assert plist["StandardErrorPath"] == str(cfg.focus_listener_stderr_path)
        # The three paths must be distinct.
        assert plist["StandardOutPath"] != plist["StandardErrorPath"]
        assert plist["StandardOutPath"] != str(cfg.focus_listener_log_path)

    def test_plist_program_args_invoke_focus_listener_run(self, tmp_path):
        plist = generate_focus_listener_plist(_config(tmp_path))
        args = plist["ProgramArguments"]
        assert "ambient.cli" in args
        assert "focus-listener-run" in args

    def test_plist_log_paths_use_focus_listener_basename(self, tmp_path):
        plist = generate_focus_listener_plist(_config(tmp_path))
        # The launchd stdout/stderr go to focus-listener-{stdout,stderr}.log
        # (default naming), not the tick daemon's launchd-{stdout,stderr}.log.
        assert "focus-listener-stdout" in plist["StandardOutPath"]
        assert "focus-listener-stderr" in plist["StandardErrorPath"]


# ---------- Off-by-default config ----------

class TestOffByDefault:
    def test_focus_capture_off_by_default(self, tmp_path):
        cfg = _config(tmp_path)
        # Privacy clause 6: opt-in per signal class.
        assert cfg.focus_capture_enabled is False

    def test_focus_paths_under_ambient_base_dir(self, tmp_path):
        cfg = _config(tmp_path)
        assert cfg.focus_events_path.parent == cfg.base_dir
        assert cfg.focus_listener_log_path.parent == cfg.base_dir
        assert cfg.focus_listener_lock_path.parent == cfg.daemon_dir


# ---------- subscribe (lazy pyobjc import) ----------

class TestSubscribe:
    def test_subscribe_raises_helpful_error_without_pyobjc(self):
        # We can't easily install pyobjc-uninstall in CI; instead patch the
        # importer to simulate ImportError.
        import builtins
        from unittest.mock import patch
        from ambient.capture import nsworkspace_listener

        original_import = builtins.__import__

        def block_appkit(name, *args, **kwargs):
            if name == "AppKit":
                raise ImportError("no AppKit on this platform")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", block_appkit):
            with pytest.raises(RuntimeError) as exc_info:
                nsworkspace_listener.subscribe(lambda r: None)
            err = str(exc_info.value).lower()
            assert "pyobjc" in err
            # Helpful install hint — not just "pyobjc missing".
            assert "pip install" in err


class TestLiveObserverPrivacy:
    """Phase 2 review testing-002: privacy contract was being asserted only
    against build_focus_record (a tautology against the dataclass). These
    tests exercise the live boundary — record_from_running_app — with a
    realistic mock NSRunningApplication that exposes forbidden accessors.
    """

    def test_record_only_calls_allowlisted_accessors(self):
        from unittest.mock import MagicMock
        from ambient.capture.nsworkspace_listener import record_from_running_app

        # Mock an NSRunningApplication-like object with extra forbidden
        # methods. spec lets us assert at the end that only allowed ones
        # were called.
        running_app = MagicMock(spec=[
            "bundleIdentifier", "localizedName", "processIdentifier",
            # Forbidden — exists on real NSRunningApplication but must
            # never be touched by this code path:
            "executableURL", "bundleURL", "windowTitle", "ownsMenuBar",
            "isFinishedLaunching", "icon", "isHidden", "isActive",
        ])
        running_app.bundleIdentifier.return_value = "com.apple.Safari"
        running_app.localizedName.return_value = "Safari"
        running_app.processIdentifier.return_value = 12345
        # The forbidden methods would return PII if called.
        running_app.executableURL.return_value = "/Applications/Safari.app"
        running_app.windowTitle.return_value = "secrets.env — Vim"
        running_app.bundleURL.return_value = "/Applications/Safari.app"

        record = record_from_running_app(running_app)

        # Only the three allowlisted accessors were called.
        running_app.bundleIdentifier.assert_called_once()
        running_app.localizedName.assert_called_once()
        running_app.processIdentifier.assert_called_once()
        # NONE of the forbidden ones were called.
        running_app.executableURL.assert_not_called()
        running_app.windowTitle.assert_not_called()
        running_app.bundleURL.assert_not_called()
        running_app.ownsMenuBar.assert_not_called()
        running_app.isFinishedLaunching.assert_not_called()
        running_app.icon.assert_not_called()

        # Record contains the four allowed fields.
        line = record.to_jsonl()
        parsed = json.loads(line)
        assert set(parsed.keys()) == {
            "ts", "source", "event", "bundle_id", "app_name", "pid"
        }
        assert "windowTitle" not in line
        assert "executableURL" not in line
        assert "secrets.env" not in line


class TestFocusListenerRunLifecycle:
    """Phase 2 review testing-001: focus_listener.run() was uncovered."""

    def test_run_acquires_lock_then_releases_on_exit(self, tmp_path):
        from unittest.mock import patch
        from ambient.daemon import focus_listener as fl

        cfg = _config(tmp_path)
        # subscribe() does the OS work; mock it to return immediately so run()
        # falls through its try/finally and releases the lock.
        with patch.object(fl, "subscribe", return_value=None):
            exit_code = fl.run(cfg)
        assert exit_code == 0
        # Lock file should NOT be held after a clean run (release_lock removes it).
        assert not cfg.focus_listener_lock_path.exists()

    def test_run_releases_lock_when_subscribe_raises_runtime_error(self, tmp_path):
        from unittest.mock import patch
        from ambient.daemon import focus_listener as fl

        cfg = _config(tmp_path)
        with patch.object(fl, "subscribe", side_effect=RuntimeError("no pyobjc")):
            exit_code = fl.run(cfg)
        assert exit_code == 2  # error path
        # Even on failure, the lock must be released so the next attempt
        # (e.g. after pip install pyobjc) can acquire it.
        assert not cfg.focus_listener_lock_path.exists()

    def test_run_returns_1_when_already_locked(self, tmp_path):
        from unittest.mock import patch
        from ambient.daemon import focus_listener as fl

        cfg = _config(tmp_path)
        # Pre-acquire the lock with a live PID (this process).
        cfg.focus_listener_lock_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.focus_listener_lock_path.write_text(str(os.getpid()))

        with patch.object(fl, "subscribe") as mock_subscribe:
            exit_code = fl.run(cfg)
        assert exit_code == 1
        # Subscribe was never called — the lock guard short-circuited.
        mock_subscribe.assert_not_called()
