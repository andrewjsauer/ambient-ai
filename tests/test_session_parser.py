import json
import tempfile
from pathlib import Path

import pytest

from ambient.daemon.session_parser import (
    discover_session_files,
    parse_session_file,
)


def _write_jsonl(path: Path, entries: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _user_prompt(text: str, ts: str = "2026-04-08T10:00:00Z") -> dict:
    return {
        "type": "user",
        "message": {"role": "user", "content": text},
        "timestamp": ts,
        "uuid": "u1",
        "sessionId": "sess-1",
        "cwd": "/projects/test",
    }


def _user_tool_result(is_error=None, ts: str = "2026-04-08T10:01:00Z") -> dict:
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "result text",
                    "is_error": is_error,
                }
            ],
        },
        "timestamp": ts,
        "uuid": "u2",
        "sessionId": "sess-1",
        "cwd": "/projects/test",
    }


def _assistant_tool_use(name: str, inp: dict, ts: str = "2026-04-08T10:00:30Z") -> dict:
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me read that file."},
                {"type": "tool_use", "id": "toolu_1", "name": name, "input": inp},
            ],
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        },
        "timestamp": ts,
        "uuid": "a1",
        "sessionId": "sess-1",
        "cwd": "/projects/test",
    }


def _meta_user(ts: str = "2026-04-08T09:59:00Z") -> dict:
    return {
        "type": "user",
        "isMeta": True,
        "message": {"role": "user", "content": "<local-command-caveat>...</local-command-caveat>"},
        "timestamp": ts,
        "uuid": "m1",
        "sessionId": "sess-1",
        "cwd": "/projects/test",
    }


class TestParseSessionFile:
    def test_extracts_prompts_and_tools(self, tmp_path):
        path = tmp_path / "sess.jsonl"
        _write_jsonl(path, [
            _user_prompt("fix the failing tests"),
            _assistant_tool_use("Read", {"file_path": "/src/foo.py"}),
            _user_tool_result(is_error=None),
            _user_prompt("yes, apply that"),
            _assistant_tool_use("Edit", {"file_path": "/src/foo.py", "old_string": "x", "new_string": "y"}),
            _user_tool_result(is_error=None),
        ])

        result = parse_session_file(path)
        assert result is not None
        assert result["prompts"] == ["fix the failing tests", "yes, apply that"]
        assert result["prompt_count"] == 2
        assert len(result["tools"]) == 2
        assert result["tools"][0]["name"] == "Read"
        assert "/src/foo.py" in result["files_touched"]

    def test_ran_test_true_when_session_runs_tests(self, tmp_path):
        """A Bash tool call running a test/build counts as in-session
        verification — the signal the shell-hook stream can't see."""
        path = tmp_path / "sess.jsonl"
        _write_jsonl(path, [
            _user_prompt("fix the charge-refund bug"),
            _assistant_tool_use("Edit", {"file_path": "/src/billing.py", "old_string": "a", "new_string": "b"}),
            _assistant_tool_use("Bash", {"command": "python -m pytest tests/test_billing.py -k refund"}),
        ])
        result = parse_session_file(path)
        assert result["ran_test"] is True

    def test_verification_resolved_on_fail_then_pass(self, tmp_path):
        """A verification that failed then later passed (in-session red→green
        fix loop) sets verification_resolved — the signal velocity uses."""
        def bash(tid, ts):
            return {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": "pytest"}}]},
                "timestamp": ts, "uuid": tid, "sessionId": "s", "cwd": "/projects/test"}

        def result(tid, is_error, ts):
            return {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tid, "is_error": is_error, "content": "x"}]},
                "timestamp": ts, "uuid": "r" + tid, "sessionId": "s", "cwd": "/projects/test"}

        path = tmp_path / "sess.jsonl"
        _write_jsonl(path, [
            _user_prompt("the test is failing"),
            bash("t1", "2026-04-08T10:00:10Z"),
            result("t1", True, "2026-04-08T10:00:20Z"),     # red
            _assistant_tool_use("Edit", {"file_path": "/src/x.py", "old_string": "a", "new_string": "b"}),
            bash("t2", "2026-04-08T10:01:00Z"),
            result("t2", False, "2026-04-08T10:01:10Z"),    # green
        ])
        result_data = parse_session_file(path)
        assert result_data["verification_resolved"] is True

    def test_verification_resolved_false_when_pass_only(self, tmp_path):
        """A test that passed without ever failing is not a fix loop."""
        def bash(tid, ts):
            return {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": "pytest"}}]},
                "timestamp": ts, "uuid": tid, "sessionId": "s", "cwd": "/projects/test"}

        def result(tid, is_error, ts):
            return {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tid, "is_error": is_error, "content": "x"}]},
                "timestamp": ts, "uuid": "r" + tid, "sessionId": "s", "cwd": "/projects/test"}

        path = tmp_path / "sess.jsonl"
        _write_jsonl(path, [
            _user_prompt("run the tests"),
            bash("t1", "2026-04-08T10:00:10Z"),
            result("t1", False, "2026-04-08T10:00:20Z"),
        ])
        result_data = parse_session_file(path)
        assert result_data["ran_test"] is True
        assert result_data["verification_resolved"] is False

    def test_verification_resolved_false_when_fail_only(self, tmp_path):
        """A test that failed and never passed is not a resolved fix loop."""
        def bash(tid, ts):
            return {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": "pytest"}}]},
                "timestamp": ts, "uuid": tid, "sessionId": "s", "cwd": "/projects/test"}

        def result(tid, is_error, ts):
            return {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tid, "is_error": is_error, "content": "x"}]},
                "timestamp": ts, "uuid": "r" + tid, "sessionId": "s", "cwd": "/projects/test"}

        path = tmp_path / "sess.jsonl"
        _write_jsonl(path, [
            _user_prompt("the test fails"),
            bash("t1", "2026-04-08T10:00:10Z"),
            result("t1", True, "2026-04-08T10:00:20Z"),
            bash("t2", "2026-04-08T10:01:00Z"),
            result("t2", True, "2026-04-08T10:01:10Z"),
        ])
        result_data = parse_session_file(path)
        assert result_data["ran_test"] is True
        assert result_data["verification_resolved"] is False

    def test_ran_test_false_without_test_command(self, tmp_path):
        """Editing and running a non-verification command does not count."""
        path = tmp_path / "sess.jsonl"
        _write_jsonl(path, [
            _user_prompt("rename the helper"),
            _assistant_tool_use("Edit", {"file_path": "/src/util.py", "old_string": "a", "new_string": "b"}),
            _assistant_tool_use("Bash", {"command": "ls -la && git status"}),
        ])
        result = parse_session_file(path)
        assert result["ran_test"] is False

    def test_extracts_file_paths_from_tool_inputs(self, tmp_path):
        path = tmp_path / "sess.jsonl"
        _write_jsonl(path, [
            _user_prompt("read files"),
            _assistant_tool_use("Read", {"file_path": "/src/models/user.py"}),
            _user_tool_result(),
            _assistant_tool_use("Edit", {"file_path": "/src/views/home.html"}),
            _user_tool_result(),
        ])

        result = parse_session_file(path)
        assert "/src/models/user.py" in result["files_touched"]
        assert "/src/views/home.html" in result["files_touched"]

    def test_counts_errors(self, tmp_path):
        path = tmp_path / "sess.jsonl"
        _write_jsonl(path, [
            _user_prompt("try this"),
            _assistant_tool_use("Bash", {"command": "false"}),
            _user_tool_result(is_error=True, ts="2026-04-08T10:01:00Z"),
            _assistant_tool_use("Bash", {"command": "true"}),
            _user_tool_result(is_error=None, ts="2026-04-08T10:02:00Z"),
            _user_tool_result(is_error=True, ts="2026-04-08T10:03:00Z"),
        ])

        result = parse_session_file(path)
        assert result["is_error_count"] == 2

    def test_skips_meta_messages(self, tmp_path):
        path = tmp_path / "sess.jsonl"
        _write_jsonl(path, [
            _meta_user(),
            _user_prompt("real prompt"),
        ])

        result = parse_session_file(path)
        assert result["prompts"] == ["real prompt"]
        assert result["prompt_count"] == 1

    def test_handles_string_content(self, tmp_path):
        """User message content as plain string (not list of blocks)."""
        path = tmp_path / "sess.jsonl"
        _write_jsonl(path, [_user_prompt("hello world")])

        result = parse_session_file(path)
        assert result["prompts"] == ["hello world"]

    def test_handles_list_content_with_text_blocks(self, tmp_path):
        """User message content as list with text blocks."""
        path = tmp_path / "sess.jsonl"
        entry = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "do the thing"}],
            },
            "timestamp": "2026-04-08T10:00:00Z",
            "sessionId": "sess-1",
            "cwd": "/proj",
        }
        _write_jsonl(path, [entry])

        result = parse_session_file(path)
        assert result["prompts"] == ["do the thing"]

    def test_skips_xml_tagged_content(self, tmp_path):
        """User messages starting with < are system injections, skip them."""
        path = tmp_path / "sess.jsonl"
        _write_jsonl(path, [
            _user_prompt("<command-name>/clear</command-name>"),
            _user_prompt("actual prompt"),
        ])

        result = parse_session_file(path)
        assert result["prompts"] == ["actual prompt"]

    def test_malformed_line_skipped(self, tmp_path):
        path = tmp_path / "sess.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps(_user_prompt("before")) + "\n")
            f.write("not valid json\n")
            f.write(json.dumps(_user_prompt("after", ts="2026-04-08T10:05:00Z")) + "\n")

        result = parse_session_file(path)
        assert result["prompts"] == ["before", "after"]

    def test_empty_file(self, tmp_path):
        path = tmp_path / "sess.jsonl"
        path.write_text("")

        result = parse_session_file(path)
        assert result is None

    def test_session_duration(self, tmp_path):
        path = tmp_path / "sess.jsonl"
        _write_jsonl(path, [
            _user_prompt("start", ts="2026-04-08T10:00:00Z"),
            _user_prompt("end", ts="2026-04-08T10:30:00Z"),
        ])

        result = parse_session_file(path)
        assert result["duration_ms"] == 30 * 60 * 1000

    def test_tool_result_only_messages_not_counted_as_prompts(self, tmp_path):
        """User messages with only tool_result blocks should not add prompts."""
        path = tmp_path / "sess.jsonl"
        _write_jsonl(path, [
            _user_prompt("actual prompt"),
            _user_tool_result(is_error=None),
            _user_tool_result(is_error=None, ts="2026-04-08T10:02:00Z"),
        ])

        result = parse_session_file(path)
        assert result["prompt_count"] == 1


class TestCommandClassifier:
    """_classify_command must match the program actually run, not a substring."""

    def test_real_test_invocations(self):
        from ambient.daemon.session_parser import _classify_command
        for cmd in ["pytest", "python -m pytest tests/ -k refund", "poetry run pytest",
                    "npx jest", "pnpm exec vitest", "go test ./...", "cargo test",
                    "npm test", "npm run test:e2e", "cd app && yarn test",
                    "FORCE_COLOR=1 pytest -q", "make test"]:
            assert _classify_command(cmd) == "test", cmd

    def test_typecheck_invocations(self):
        from ambient.daemon.session_parser import _classify_command
        for cmd in ["tsc --noEmit", "mypy src", "npm run build", "go build ./...",
                    "cargo check", "npm run typecheck"]:
            assert _classify_command(cmd) == "typecheck", cmd

    def test_false_positives_return_none(self):
        from ambient.daemon.session_parser import _classify_command
        for cmd in ['git commit -m "fix pytest"', 'echo "run pytest"', "cat pytest.ini",
                    "# go test", "git log --grep ruff", "make test-data", "grep mypy src/",
                    "ruff check .", "eslint .", "cargo clippy", "ls && echo done"]:
            assert _classify_command(cmd) is None, cmd


class TestDiscoverSessionFiles:
    def test_finds_jsonl_in_project_dirs(self, tmp_path):
        proj = tmp_path / "projects" / "-slug-one"
        proj.mkdir(parents=True)
        (proj / "sess-1.jsonl").write_text("{}")
        (proj / "sess-2.jsonl").write_text("{}")

        files = discover_session_files(tmp_path / "projects")
        assert len(files) == 2

    def test_skips_subdirectories(self, tmp_path):
        proj = tmp_path / "projects" / "-slug"
        proj.mkdir(parents=True)
        (proj / "sess.jsonl").write_text("{}")

        sub = proj / "subagents"
        sub.mkdir()
        (sub / "agent.jsonl").write_text("{}")

        mem = proj / "memory"
        mem.mkdir()
        (mem / "notes.md").write_text("notes")

        files = discover_session_files(tmp_path / "projects")
        assert len(files) == 1
        assert files[0].name == "sess.jsonl"

    def test_missing_dir(self, tmp_path):
        files = discover_session_files(tmp_path / "nonexistent")
        assert files == []
