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
        assert plist["KeepAlive"] is True
        assert plist["RunAtLoad"] is True

    def test_plist_program_args_invoke_focus_listener_run(self, tmp_path):
        plist = generate_focus_listener_plist(_config(tmp_path))
        args = plist["ProgramArguments"]
        assert "ambient.cli" in args
        assert "focus-listener-run" in args

    def test_plist_log_paths_use_focus_listener_basename(self, tmp_path):
        plist = generate_focus_listener_plist(_config(tmp_path))
        # The launchd stdout/stderr go to focus-listener-{stdout,stderr}.log,
        # not the tick daemon's launchd-{stdout,stderr}.log.
        assert "focus-listener" in plist["StandardOutPath"]
        assert "focus-listener" in plist["StandardErrorPath"]


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
            assert "pyobjc" in str(exc_info.value).lower()
