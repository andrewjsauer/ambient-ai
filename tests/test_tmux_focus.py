"""Tests for tmux focus hooks (Phase 2 Unit 8)."""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from ambient.capture import tmux_focus


# ---------- Hook script payload tests ----------

class TestHookScript:
    """End-to-end tests against the shell script that tmux invokes.

    The script is small and self-contained, so we can run it directly with
    fake env vars and assert the JSONL output is well-formed and contains
    only the privacy-allowlisted fields.
    """

    def test_script_emits_valid_jsonl_line(self, tmp_path):
        script = tmux_focus.hook_script_path()
        events_path = tmp_path / "focus-events.jsonl"
        env = {
            **os.environ,
            "AMBIENT_FOCUS_EVENTS_PATH": str(events_path),
            "TMUX_PANE": "%17",
            # Force the inner `if [ -n "${TMUX:-}" ]` branch to skip — we don't
            # have a real tmux server in tests.
            "TMUX": "",
        }
        result = subprocess.run(
            ["sh", str(script), "pane-focus-in"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert events_path.exists()
        line = events_path.read_text().strip()
        record = json.loads(line)
        assert record["source"] == "tmux"
        assert record["event"] == "pane-focus-in"
        assert record["pane_id"] == "%17"

    def test_script_record_has_only_allowlisted_fields(self, tmp_path):
        # PRIVACY clause 7: tmux records carry structural identifiers only.
        # No pane_title, no pane_current_command, no pane_current_path.
        script = tmux_focus.hook_script_path()
        events_path = tmp_path / "focus-events.jsonl"
        env = {
            **os.environ,
            "AMBIENT_FOCUS_EVENTS_PATH": str(events_path),
            "TMUX_PANE": "%1",
            "TMUX": "",
        }
        subprocess.run(
            ["sh", str(script), "window-focused"],
            env=env, capture_output=True, check=True,
        )
        record = json.loads(events_path.read_text().strip())
        forbidden = {"pane_title", "pane_current_command", "pane_current_path", "window_title"}
        assert set(record.keys()) & forbidden == set()
        # Allowlist must be exactly these six.
        assert set(record.keys()) == {
            "ts", "source", "event", "pane_id", "window_index", "session_name"
        }

    def test_script_appends_does_not_overwrite(self, tmp_path):
        script = tmux_focus.hook_script_path()
        events_path = tmp_path / "focus-events.jsonl"
        env = {
            **os.environ,
            "AMBIENT_FOCUS_EVENTS_PATH": str(events_path),
            "TMUX_PANE": "%1",
            "TMUX": "",
        }
        subprocess.run(["sh", str(script), "pane-focus-in"], env=env, check=True)
        subprocess.run(["sh", str(script), "pane-focus-out"], env=env, check=True)
        lines = events_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "pane-focus-in"
        assert json.loads(lines[1])["event"] == "pane-focus-out"

    def test_script_creates_parent_directory(self, tmp_path):
        script = tmux_focus.hook_script_path()
        events_path = tmp_path / "subdir" / "focus-events.jsonl"
        env = {
            **os.environ,
            "AMBIENT_FOCUS_EVENTS_PATH": str(events_path),
            "TMUX_PANE": "%1",
            "TMUX": "",
        }
        subprocess.run(["sh", str(script), "pane-focus-in"], env=env, check=True)
        assert events_path.exists()


# ---------- install/uninstall logic ----------

class TestInstall:
    def test_tmux_unavailable_raises(self, tmp_path):
        with patch.object(tmux_focus, "tmux_available", return_value=False):
            with pytest.raises(RuntimeError, match="tmux not found"):
                tmux_focus.install_hooks(tmp_path / "events.jsonl")

    def test_install_runs_set_hook_per_hook(self, tmp_path):
        with patch.object(tmux_focus, "tmux_available", return_value=True), \
             patch.object(tmux_focus, "uninstall_hooks") as mock_uninstall, \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
            tmux_focus.install_hooks(tmp_path / "events.jsonl")
            # uninstall is called once at the start (idempotency)
            mock_uninstall.assert_called_once()
            # one set-hook call per hook in HOOKS
            assert mock_run.call_count == len(tmux_focus.HOOKS)
            for call, expected_hook in zip(mock_run.call_args_list, tmux_focus.HOOKS):
                args = call.args[0]
                assert args[0] == "tmux"
                assert args[1] == "set-hook"
                assert args[2] == "-g"
                assert args[3] == expected_hook
                # The hook payload must reference the events path and the sentinel.
                payload = args[4]
                assert "ambient-focus-hook.sh" in payload
                assert tmux_focus.SENTINEL in payload

    def test_uninstall_only_removes_ambient_managed_hooks(self):
        # show-hook returns the hook command; uninstall must SKIP user-managed
        # hooks (no SENTINEL in stdout) and unset Ambient-managed ones.
        responses = {
            "pane-focus-in": "pane-focus-in -> 'run-shell ambient-focus-hook.sh # ambient-managed'",
            "pane-focus-out": "pane-focus-out -> 'run-shell user-custom-thing'",  # NOT ambient
            "window-focused": "window-focused -> 'run-shell ambient-focus-hook.sh # ambient-managed'",
        }
        run_log = []

        def fake_run(cmd, *a, **kw):
            run_log.append(tuple(cmd))
            if len(cmd) >= 3 and cmd[1] == "show-hook":
                hook_name = cmd[3]
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=responses.get(hook_name, ""), stderr=""
                )
            # set-hook -gu
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        with patch.object(tmux_focus, "tmux_available", return_value=True), \
             patch("subprocess.run", side_effect=fake_run):
            tmux_focus.uninstall_hooks()

        unset_calls = [c for c in run_log if len(c) >= 3 and c[1:3] == ("set-hook", "-gu")]
        unset_hooks = [c[3] for c in unset_calls]
        assert "pane-focus-in" in unset_hooks
        assert "window-focused" in unset_hooks
        assert "pane-focus-out" not in unset_hooks  # user-managed; not removed


class TestHookScriptPath:
    def test_path_is_executable(self):
        script = tmux_focus.hook_script_path()
        assert script.exists()
        assert script.is_file()
        assert os.access(script, os.X_OK), f"{script} should be executable"


class TestPrivacyContract:
    def test_hook_payload_never_references_pane_title_or_path(self):
        # Static check: the hook script must not reference pane_title,
        # pane_current_command, or pane_current_path. These fields would let
        # tmux interpolate window-title-equivalent data into our records.
        script_text = tmux_focus.hook_script_path().read_text()
        # The script can mention these in COMMENTS (privacy contract docstring),
        # but never as `#{pane_title}` interpolations passed to display-message.
        forbidden_interpolations = [
            "#{pane_title}",
            "#{pane_current_command}",
            "#{pane_current_path}",
            "#{pane_current_pid}",
        ]
        for token in forbidden_interpolations:
            assert token not in script_text, (
                f"Privacy violation: hook script must not interpolate {token}"
            )
