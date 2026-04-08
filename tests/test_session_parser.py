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
