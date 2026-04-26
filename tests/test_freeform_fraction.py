"""Tests for the freeform-fraction detector."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ambient.config import Config
from ambient.detect.command_mix import detect_command_mix
from ambient.detect.freeform_fraction import (
    FreeformFraction,
    detect_freeform_fraction,
)


def _user_line(ts: str, content_text: str, session_id: str = "sess-1") -> str:
    obj = {
        "type": "user",
        "sessionId": session_id,
        "timestamp": ts,
        "message": {"content": content_text},
    }
    return json.dumps(obj) + "\n"


def _slash_body(command: str) -> str:
    return (
        f"<command-message>{command.lstrip('/')}</command-message>\n"
        f"<command-name>{command}</command-name>\n"
        f"<command-args></command-args>"
    )


def _make_projects_dir(tmp_path: Path, layout: dict[str, list[str]]) -> Path:
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


CUR_START = datetime(2026, 4, 20, tzinfo=timezone.utc)
CUR_END = datetime(2026, 4, 27, tzinfo=timezone.utc)
PRIOR_START = datetime(2026, 4, 13, tzinfo=timezone.utc)
PRIOR_END = datetime(2026, 4, 20, tzinfo=timezone.utc)
T_CUR = "2026-04-22T15:00:00Z"
T_PRIOR = "2026-04-15T15:00:00Z"


class TestDetectFreeformFraction:
    def test_overall_pct_basic(self, tmp_path):
        # 8 freeform of 10 total → 0.8
        lines = [_user_line(T_CUR, "freeform")] * 8 + [
            _user_line(T_CUR, _slash_body("/ship"))
        ] * 2
        root = _make_projects_dir(tmp_path, {"proj-a": lines})
        cfg = _config(tmp_path)
        result = detect_freeform_fraction(root, CUR_START, CUR_END, cfg)
        assert isinstance(result, FreeformFraction)
        assert result.overall_pct == pytest.approx(0.8)
        assert result.total_prompts == 10

    def test_zero_prompts_in_window_returns_zero(self, tmp_path):
        root = _make_projects_dir(tmp_path, {"proj-a": []})
        cfg = _config(tmp_path)
        result = detect_freeform_fraction(root, CUR_START, CUR_END, cfg)
        assert result.overall_pct == 0.0
        assert result.total_prompts == 0
        assert result.prior_window_pct is None
        assert result.delta_pct is None

    def test_prior_window_delta(self, tmp_path):
        # Prior: 9 freeform of 10 → 0.9
        # Current: 7 freeform of 10 → 0.7
        # Delta: -0.2
        prior_lines = [_user_line(T_PRIOR, "f")] * 9 + [_user_line(T_PRIOR, _slash_body("/ship"))]
        cur_lines = [_user_line(T_CUR, "f")] * 7 + [_user_line(T_CUR, _slash_body("/ship"))] * 3
        root = _make_projects_dir(tmp_path, {"proj-a": prior_lines + cur_lines})
        cfg = _config(tmp_path)
        result = detect_freeform_fraction(
            root, CUR_START, CUR_END, cfg,
            prior_window_start=PRIOR_START,
            prior_window_end=PRIOR_END,
        )
        assert result.overall_pct == pytest.approx(0.7)
        assert result.prior_window_pct == pytest.approx(0.9)
        assert result.delta_pct == pytest.approx(-0.2)
        assert result.prior_total_prompts == 10

    def test_prior_window_with_no_data_yields_none(self, tmp_path):
        cur_lines = [_user_line(T_CUR, "f")] * 5
        root = _make_projects_dir(tmp_path, {"proj-a": cur_lines})
        cfg = _config(tmp_path)
        result = detect_freeform_fraction(
            root, CUR_START, CUR_END, cfg,
            prior_window_start=PRIOR_START,
            prior_window_end=PRIOR_END,
        )
        assert result.overall_pct == pytest.approx(1.0)
        assert result.prior_window_pct is None
        assert result.delta_pct is None

    def test_per_project_floor_drops_quiet_projects(self, tmp_path):
        layout = {
            "proj-loud": [_user_line(T_CUR, "f")] * 25,
            "proj-quiet": [_user_line(T_CUR, "f")] * 5,
        }
        root = _make_projects_dir(tmp_path, layout)
        cfg = _config(tmp_path)
        result = detect_freeform_fraction(root, CUR_START, CUR_END, cfg)
        assert "proj-loud" in result.per_project
        assert "proj-quiet" not in result.per_project
        # but quiet project is still in overall total
        assert result.total_prompts == 30

    def test_per_project_pct_is_correct(self, tmp_path):
        layout = {
            "proj-mixed": (
                [_user_line(T_CUR, "f")] * 15
                + [_user_line(T_CUR, _slash_body("/ship"))] * 5
            ),  # 75% freeform, 20 total → above default floor
        }
        root = _make_projects_dir(tmp_path, layout)
        cfg = _config(tmp_path)
        result = detect_freeform_fraction(root, CUR_START, CUR_END, cfg)
        assert result.per_project["proj-mixed"] == pytest.approx(0.75)

    def test_excludes_subagents_dir(self, tmp_path):
        layout = {
            "subagents": [_user_line(T_CUR, "f")] * 100,
            "proj-a": [_user_line(T_CUR, "f")] * 25,
        }
        root = _make_projects_dir(tmp_path, layout)
        cfg = _config(tmp_path)
        result = detect_freeform_fraction(root, CUR_START, CUR_END, cfg)
        assert result.total_prompts == 25  # subagents excluded

    def test_window_iso_strings_populated(self, tmp_path):
        root = _make_projects_dir(tmp_path, {})
        cfg = _config(tmp_path)
        result = detect_freeform_fraction(root, CUR_START, CUR_END, cfg)
        assert result.window_start_iso.startswith("2026-04-20")
        assert result.window_end_iso.startswith("2026-04-27")

    def test_floor_zero_keeps_quiet_projects(self, tmp_path):
        layout = {"proj-quiet": [_user_line(T_CUR, "f")] * 3}
        root = _make_projects_dir(tmp_path, layout)
        cfg = _config(tmp_path, freeform_fraction_min_prompts=0)
        result = detect_freeform_fraction(root, CUR_START, CUR_END, cfg)
        assert "proj-quiet" in result.per_project


class TestIntegrationWithCommandMix:
    """Both detectors run on the same data and produce non-contradictory totals."""

    def test_freeform_count_matches_command_mix_overall(self, tmp_path):
        layout = {
            "proj-a": (
                [_user_line(T_CUR, "f")] * 12
                + [_user_line(T_CUR, _slash_body("/ship"))] * 4
                + [_user_line(T_CUR, _slash_body("/compound-engineering:ce-plan"))] * 4
            ),
        }
        root = _make_projects_dir(tmp_path, layout)
        cfg = _config(tmp_path)
        ff = detect_freeform_fraction(root, CUR_START, CUR_END, cfg)
        cm = detect_command_mix(root, CUR_START, CUR_END, cfg)
        # Same denominator
        assert ff.total_prompts == cm.overall.total
        # Same freeform count
        expected_freeform = cm.overall.freeform_count
        assert int(ff.overall_pct * ff.total_prompts) == expected_freeform
