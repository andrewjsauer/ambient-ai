"""Tests for the command-mix detector and the shared session-walk helper."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ambient.config import Config
from ambient.detect._claude_session_walk import walk_prompts
from ambient.detect.command_mix import (
    CommandMixFindings,
    ProjectMix,
    detect_command_mix,
)


# ---------- Fixture helpers ----------

def _user_line(ts: str, content_text: str, session_id: str = "sess-1") -> str:
    """Build a single Claude Code JSONL line for a user message."""
    obj = {
        "type": "user",
        "sessionId": session_id,
        "timestamp": ts,
        "message": {"content": content_text},
    }
    return json.dumps(obj) + "\n"


def _user_line_blocks(ts: str, blocks: list[dict], session_id: str = "sess-1") -> str:
    obj = {
        "type": "user",
        "sessionId": session_id,
        "timestamp": ts,
        "message": {"content": blocks},
    }
    return json.dumps(obj) + "\n"


def _slash_body(command: str, args: str = "") -> str:
    return (
        f"<command-message>{command.lstrip('/')}</command-message>\n"
        f"<command-name>{command}</command-name>\n"
        f"<command-args>{args}</command-args>"
    )


def _make_projects_dir(tmp_path: Path, layout: dict[str, list[str]]) -> Path:
    """layout = {project_slug: [jsonl_line, ...]}."""
    root = tmp_path / "projects"
    root.mkdir()
    for slug, lines in layout.items():
        slug_dir = root / slug
        slug_dir.mkdir()
        (slug_dir / f"{slug}-session.jsonl").write_text("".join(lines), encoding="utf-8")
    return root


def _config(tmp_path: Path, **overrides) -> Config:
    cfg = Config(base_dir=tmp_path / ".ambient")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


WINDOW_START = datetime(2026, 4, 20, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 4, 27, tzinfo=timezone.utc)
T_INSIDE = "2026-04-22T15:00:00Z"
T_BEFORE = "2026-04-19T15:00:00Z"
T_AFTER = "2026-04-28T15:00:00Z"


# ---------- _claude_session_walk tests ----------

class TestWalkPrompts:
    def test_yields_freeform_and_slash_prompts(self, tmp_path):
        root = _make_projects_dir(tmp_path, {
            "proj-a": [
                _user_line(T_INSIDE, "freeform message"),
                _user_line(T_INSIDE, _slash_body("/ship")),
            ],
        })
        recs = list(walk_prompts(root, WINDOW_START, WINDOW_END))
        assert len(recs) == 2
        commands = sorted(r.slash_command or "" for r in recs)
        assert commands == ["", "/ship"]

    def test_excludes_subagents_dir(self, tmp_path):
        root = _make_projects_dir(tmp_path, {
            "subagents": [_user_line(T_INSIDE, "in subagents")],
            "proj-a": [_user_line(T_INSIDE, "in proj")],
        })
        recs = list(walk_prompts(root, WINDOW_START, WINDOW_END))
        assert [r.project for r in recs] == ["proj-a"]

    def test_filters_outside_window(self, tmp_path):
        root = _make_projects_dir(tmp_path, {
            "proj-a": [
                _user_line(T_BEFORE, "before"),
                _user_line(T_INSIDE, "inside"),
                _user_line(T_AFTER, "after"),
            ],
        })
        recs = list(walk_prompts(root, WINDOW_START, WINDOW_END))
        assert [r.text for r in recs] == ["inside"]

    def test_no_window_yields_all(self, tmp_path):
        root = _make_projects_dir(tmp_path, {
            "proj-a": [
                _user_line(T_BEFORE, "a"),
                _user_line(T_INSIDE, "b"),
                _user_line(T_AFTER, "c"),
            ],
        })
        recs = list(walk_prompts(root, None, None))
        assert {r.text for r in recs} == {"a", "b", "c"}

    def test_skips_tool_output_echoes(self, tmp_path):
        root = _make_projects_dir(tmp_path, {
            "proj-a": [
                _user_line(T_INSIDE, "<bash-stdout>build output</bash-stdout>"),
                _user_line(T_INSIDE, "<local-command-stdout>foo</local-command-stdout>"),
                _user_line(T_INSIDE, "<system-reminder>do this</system-reminder>"),
                _user_line(T_INSIDE, "real prompt"),
            ],
        })
        recs = list(walk_prompts(root, WINDOW_START, WINDOW_END))
        assert [r.text for r in recs] == ["real prompt"]

    def test_strips_appended_system_reminder_from_real_prompt(self, tmp_path):
        # Real prompts with an appended <system-reminder> block must keep the
        # human-typed text and drop the trailing reminder. Earlier filter only
        # caught body-starts-with prefixes and let appended blocks pollute the
        # text fed downstream to Haiku.
        body = "fix the bug in tick.py\n<system-reminder>do this carefully</system-reminder>"
        root = _make_projects_dir(tmp_path, {"proj-a": [_user_line(T_INSIDE, body)]})
        recs = list(walk_prompts(root, WINDOW_START, WINDOW_END))
        assert len(recs) == 1
        assert recs[0].text == "fix the bug in tick.py"

    def test_strips_multiple_appended_blocks(self, tmp_path):
        body = (
            "real prompt\n"
            "<system-reminder>one</system-reminder>\n"
            "<bash-stdout>two</bash-stdout>"
        )
        root = _make_projects_dir(tmp_path, {"proj-a": [_user_line(T_INSIDE, body)]})
        recs = list(walk_prompts(root, WINDOW_START, WINDOW_END))
        assert recs[0].text == "real prompt"

    def test_handles_content_block_list(self, tmp_path):
        root = _make_projects_dir(tmp_path, {
            "proj-a": [
                _user_line_blocks(T_INSIDE, [
                    {"type": "text", "text": "block-a"},
                    {"type": "tool_result", "is_error": False},  # ignored
                    {"type": "text", "text": "block-b"},
                ]),
            ],
        })
        recs = list(walk_prompts(root, WINDOW_START, WINDOW_END))
        assert sorted(r.text for r in recs) == ["block-a", "block-b"]

    def test_skips_malformed_json(self, tmp_path):
        root = tmp_path / "projects"
        root.mkdir()
        slug = root / "proj-a"
        slug.mkdir()
        (slug / "s.jsonl").write_text(
            "this is not json\n" + _user_line(T_INSIDE, "real one"),
            encoding="utf-8",
        )
        recs = list(walk_prompts(root, WINDOW_START, WINDOW_END))
        assert [r.text for r in recs] == ["real one"]

    def test_missing_projects_dir_yields_nothing(self, tmp_path):
        root = tmp_path / "does-not-exist"
        recs = list(walk_prompts(root, WINDOW_START, WINDOW_END))
        assert recs == []

    def test_handles_naive_datetime_bounds(self, tmp_path):
        root = _make_projects_dir(tmp_path, {
            "proj-a": [_user_line(T_INSIDE, "x")],
        })
        # Naive bounds should be treated as UTC, not raise.
        naive_start = datetime(2026, 4, 20)
        naive_end = datetime(2026, 4, 27)
        recs = list(walk_prompts(root, naive_start, naive_end))
        assert len(recs) == 1


# ---------- ProjectMix dataclass ----------

class TestProjectMix:
    def test_empty_mix_total_is_zero(self):
        m = ProjectMix()
        assert m.total == 0

    def test_empty_mix_ratios_all_zero(self):
        ratios = ProjectMix().ratios
        assert all(v == 0.0 for v in ratios.values())
        assert set(ratios.keys()) == {
            "planning", "execution", "review", "design", "meta", "other", "freeform"
        }

    def test_add_increments_correct_field(self):
        m = ProjectMix()
        m.add("planning")
        m.add("planning")
        m.add("freeform")
        assert m.planning_count == 2
        assert m.freeform_count == 1
        assert m.total == 3

    def test_unknown_category_falls_to_other(self):
        m = ProjectMix()
        m.add("not-a-real-category")
        assert m.other_count == 1
        assert m.total == 1

    def test_ratios_normalize_to_one(self):
        m = ProjectMix(planning_count=1, execution_count=1, freeform_count=2)
        ratios = m.ratios
        assert ratios["planning"] == 0.25
        assert ratios["execution"] == 0.25
        assert ratios["freeform"] == 0.5
        assert sum(ratios.values()) == pytest.approx(1.0)


# ---------- detect_command_mix ----------

class TestDetectCommandMix:
    def test_aggregates_overall_and_per_project(self, tmp_path):
        root = _make_projects_dir(tmp_path, {
            "proj-a": [
                _user_line(T_INSIDE, _slash_body("/compound-engineering:ce-plan")),
                _user_line(T_INSIDE, _slash_body("/ship")),
                _user_line(T_INSIDE, "freeform"),
            ] * 4,  # 12 prompts → above default floor of 10
            "proj-b": [
                _user_line(T_INSIDE, _slash_body("/compound-engineering:ce-review")),
            ] * 12,
        })
        cfg = _config(tmp_path)
        result = detect_command_mix(root, WINDOW_START, WINDOW_END, cfg)
        assert isinstance(result, CommandMixFindings)
        assert result.overall.total == 12 + 12
        assert result.overall.planning_count == 4
        assert result.overall.execution_count == 4
        assert result.overall.review_count == 12
        assert result.overall.freeform_count == 4
        assert "proj-a" in result.per_project
        assert "proj-b" in result.per_project
        assert result.per_project["proj-b"].review_count == 12

    def test_drops_projects_below_floor_from_per_project(self, tmp_path):
        root = _make_projects_dir(tmp_path, {
            "proj-a": [_user_line(T_INSIDE, _slash_body("/ship"))] * 12,
            "proj-quiet": [_user_line(T_INSIDE, "x")] * 3,  # below default floor
        })
        cfg = _config(tmp_path)
        result = detect_command_mix(root, WINDOW_START, WINDOW_END, cfg)
        assert "proj-a" in result.per_project
        assert "proj-quiet" not in result.per_project
        # but quiet project is still in overall
        assert result.overall.freeform_count == 3
        assert result.overall.execution_count == 12

    def test_excludes_subagents(self, tmp_path):
        root = _make_projects_dir(tmp_path, {
            "subagents": [_user_line(T_INSIDE, _slash_body("/ship"))] * 50,
            "proj-a": [_user_line(T_INSIDE, _slash_body("/ship"))] * 12,
        })
        cfg = _config(tmp_path)
        result = detect_command_mix(root, WINDOW_START, WINDOW_END, cfg)
        assert result.overall.execution_count == 12  # subagents not counted
        assert "subagents" not in result.per_project

    def test_window_filters_apply(self, tmp_path):
        root = _make_projects_dir(tmp_path, {
            "proj-a": [
                _user_line(T_BEFORE, _slash_body("/ship")),
                *[_user_line(T_INSIDE, _slash_body("/ship"))] * 12,
                _user_line(T_AFTER, _slash_body("/ship")),
            ],
        })
        cfg = _config(tmp_path)
        result = detect_command_mix(root, WINDOW_START, WINDOW_END, cfg)
        assert result.overall.execution_count == 12  # neither outside-window prompt

    def test_user_overrides_take_effect(self, tmp_path):
        root = _make_projects_dir(tmp_path, {
            "proj-a": [_user_line(T_INSIDE, _slash_body("/my-cmd"))] * 12,
        })
        cfg = _config(tmp_path, slash_taxonomy_overrides={"/my-cmd": "review"})
        result = detect_command_mix(root, WINDOW_START, WINDOW_END, cfg)
        assert result.overall.review_count == 12
        assert result.overall.other_count == 0

    def test_window_iso_strings_populated(self, tmp_path):
        root = _make_projects_dir(tmp_path, {})
        cfg = _config(tmp_path)
        result = detect_command_mix(root, WINDOW_START, WINDOW_END, cfg)
        assert result.window_start_iso.startswith("2026-04-20")
        assert result.window_end_iso.startswith("2026-04-27")

    def test_empty_projects_dir(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        cfg = _config(tmp_path)
        result = detect_command_mix(root, WINDOW_START, WINDOW_END, cfg)
        assert result.overall.total == 0
        assert result.per_project == {}

    def test_floor_zero_keeps_all_projects(self, tmp_path):
        root = _make_projects_dir(tmp_path, {
            "proj-quiet": [_user_line(T_INSIDE, "x")],
        })
        cfg = _config(tmp_path, command_mix_min_prompts=0)
        result = detect_command_mix(root, WINDOW_START, WINDOW_END, cfg)
        assert "proj-quiet" in result.per_project
