import json
from unittest.mock import MagicMock, patch

import pytest

from ambient.config import Config
from ambient.detect.compression import CompressionFindings, RepeatedSequence
from ambient.detect.pauses import PauseFindings, PauseClassification
from ambient.present.narrator import narrate_batch, narrate_daily, load_batch_analyses
from ambient.present.prompts import build_batch_prompt, build_daily_prompt, DAILY_SYSTEM


@pytest.fixture
def config(tmp_path):
    return Config(base_dir=tmp_path)


def _sample_compression():
    return CompressionFindings(
        sequences=[
            RepeatedSequence(
                sequence=("git add .", "git commit -m wip", "git push"),
                count=5,
                total_time_ms=15000,
                compression_gain=15,
            ),
        ],
        compression_ratio=0.65,
    )


def _sample_pauses():
    return PauseFindings(
        available=True,
        classifications=[
            PauseClassification(
                gap_ms=2000, label="routine",
                probabilities={"routine": 0.9, "evaluating": 0.08, "stuck": 0.02},
                preceding_command="git status", following_command="vim file.py",
            ),
            PauseClassification(
                gap_ms=45000, label="stuck",
                probabilities={"routine": 0.01, "evaluating": 0.1, "stuck": 0.89},
                preceding_command="pytest tests/", following_command="vim parser.py",
            ),
        ],
    )


def _unavailable_pauses():
    return PauseFindings(available=False, reason="not_calibrated")


def test_batch_prompt_with_findings():
    prompt = build_batch_prompt(
        {"sequences": [{"sequence": ["a", "b"], "count": 3, "total_time_ms": 600, "compression_gain": 6}],
         "compression_ratio": 0.7},
        {"available": True, "classifications": [
            {"label": "routine", "gap_ms": 1000, "preceding_command": "a", "following_command": "b"},
            {"label": "stuck", "gap_ms": 30000, "preceding_command": "c", "following_command": "d"},
        ]},
    )
    assert "a -> b" in prompt
    assert "routine: 1/2" in prompt
    assert "stuck: 1/2" in prompt


def test_batch_prompt_without_pauses():
    prompt = build_batch_prompt(
        {"sequences": [], "compression_ratio": 1.0},
        {"available": False},
    )
    assert "Not available" in prompt
    assert "None found" in prompt


def test_daily_system_has_structured_template():
    sections = [
        "## Day Title",
        "## Rhythm Profile",
        "## Automation Candidates",
        "## Cognitive Load",
        "## Workflow Phases",
        "## Friction Points",
        "## Key Stats",
        "## Actionable Insight",
    ]
    for section in sections:
        assert section in DAILY_SYSTEM, f"Missing section: {section}"
    # Each section has an italic description
    assert DAILY_SYSTEM.count("_") >= 16  # at least 8 pairs of underscores


def test_daily_prompt_with_changepoints():
    prompt = build_daily_prompt(
        batch_analyses=[{"summary": "window 1"}],
        changepoint_data={"segments": [
            {"duration_min": 60, "mean_rate": 8.0, "label": "high-rate, git-focused"},
        ], "changepoints": []},
    )
    assert "high-rate, git-focused" in prompt
    assert "Window 1" in prompt


@patch("ambient.present.narrator._call_api")
def test_narrate_batch_success(mock_api, config):
    mock_api.return_value = json.dumps({
        "automation_candidates": [{"sequence": ["a", "b"], "suggestion": "alias it"}],
        "cognitive_patterns": [],
        "work_phase": {"current": "coding", "suggestion_timing": "bad", "reason": "in flow"},
    })

    result = narrate_batch(_sample_compression(), _sample_pauses(), config)

    assert result["analysis"] is not None
    assert "automation_candidates" in result["analysis"]
    assert result["compression"] is not None

    # Verify saved to JSONL
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")
    analyses = load_batch_analyses(config, date_str)
    assert len(analyses) == 1


@patch("ambient.present.narrator._call_api")
def test_narrate_batch_api_failure(mock_api, config):
    mock_api.side_effect = Exception("API key not set")

    result = narrate_batch(_sample_compression(), _sample_pauses(), config)

    # Raw findings should be preserved
    assert result["compression"] is not None
    assert result["pauses"] is not None
    assert result["analysis"] is None
    assert "API key not set" in result["error"]

    # Should still save to JSONL
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")
    analyses = load_batch_analyses(config, date_str)
    assert len(analyses) == 1


def test_empty_findings_prompt():
    prompt = build_batch_prompt(
        {"sequences": [], "compression_ratio": 1.0},
        {"available": True, "classifications": []},
    )
    assert "None found" in prompt


@patch("ambient.present.narrator._call_api")
def test_narrate_daily(mock_api, config):
    mock_api.return_value = "You had a productive morning..."

    result = narrate_daily(
        batch_analyses=[{"summary": "window 1"}],
        changepoints=None,
        config=config,
    )

    assert "productive morning" in result
    # Verify summary was saved
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")
    assert config.summary_path(date_str).exists()


@patch("ambient.present.narrator._call_api")
def test_narrate_daily_respects_date_param(mock_api, config):
    mock_api.return_value = "Summary for a past date."

    result = narrate_daily(
        batch_analyses=[{"summary": "window 1"}],
        changepoints=None,
        config=config,
        date_str="2026-03-29",
    )

    assert "past date" in result
    # Should save to the specified date, not today
    assert config.summary_path("2026-03-29").exists()
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    if today != "2026-03-29":
        assert not config.summary_path(today).exists()
