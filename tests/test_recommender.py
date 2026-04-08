from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from ambient.config import Config
from ambient.present.recommender import (
    Recommendation,
    generate_recommendations,
    _generate_alias_recommendation,
    _slugify,
)


@dataclass
class FakePromptPattern:
    normalized_prompt: str
    raw_examples: list[str]
    count: int
    projects: list[str]


@dataclass
class FakePromptPatternFindings:
    patterns: list[FakePromptPattern] = field(default_factory=list)
    total_prompts: int = 0


@dataclass
class FakeRepeatedSequence:
    sequence: tuple[str, ...]
    count: int
    total_time_ms: int
    compression_gain: int = 0


@dataclass
class FakeCompressionFindings:
    sequences: list[FakeRepeatedSequence] = field(default_factory=list)
    compression_ratio: float = 1.0


class TestSlugify:
    def test_basic(self):
        assert _slugify("commit and push") == "commit-and-push"

    def test_special_chars(self):
        assert _slugify("fix the test!") == "fix-the-test"

    def test_truncation(self):
        result = _slugify("a" * 100)
        assert len(result) <= 50


class TestAliasRecommendation:
    def test_generates_alias(self):
        seq = {"sequence": ["git add .", "git commit", "git push"], "count": 5, "total_time_ms": 10000}
        rec = _generate_alias_recommendation(seq)
        assert rec is not None
        assert rec.type == "alias"
        assert "alias ggg=" in rec.artifact
        assert "git add . && git commit && git push" in rec.artifact

    def test_skips_single_command(self):
        seq = {"sequence": ["git status"], "count": 10, "total_time_ms": 5000}
        rec = _generate_alias_recommendation(seq)
        assert rec is None


class TestGenerateRecommendations:
    @patch("ambient.present.api.call_api", return_value="---\ndescription: Auto commit\n---\n# Auto Commit\nCommits and pushes.")
    def test_generates_skill_from_prompt_pattern(self, mock_api, tmp_path):
        config = Config(base_dir=tmp_path)
        patterns = FakePromptPatternFindings(
            patterns=[
                FakePromptPattern(
                    normalized_prompt="commit and push this",
                    raw_examples=["commit and push this", "commit and push"],
                    count=8,
                    projects=["my-project"],
                )
            ],
            total_prompts=20,
        )

        result = generate_recommendations(patterns, None, config)
        assert len(result.recommendations) == 1
        rec = result.recommendations[0]
        assert rec.type == "skill"
        assert "commit" in rec.id
        mock_api.assert_called_once()

    @patch("ambient.present.api.call_api", return_value="skill content")
    def test_writes_recommendation_files(self, mock_api, tmp_path):
        config = Config(base_dir=tmp_path)
        patterns = FakePromptPatternFindings(
            patterns=[
                FakePromptPattern(
                    normalized_prompt="deploy to staging",
                    raw_examples=["deploy to staging"],
                    count=6,
                    projects=["app"],
                )
            ],
        )

        generate_recommendations(patterns, None, config)
        rec_dir = tmp_path / "recommendations"
        assert rec_dir.exists()
        files = list(rec_dir.glob("*.md"))
        assert len(files) == 1
        content = files[0].read_text()
        assert "type: skill" in content

    def test_generates_alias_from_compression(self, tmp_path):
        config = Config(base_dir=tmp_path)
        compression = FakeCompressionFindings(
            sequences=[
                FakeRepeatedSequence(
                    sequence=("git add .", "git commit -m fix", "git push"),
                    count=7,
                    total_time_ms=15000,
                )
            ]
        )

        result = generate_recommendations(None, compression, config)
        assert len(result.recommendations) == 1
        rec = result.recommendations[0]
        assert rec.type == "alias"

    def test_skips_patterns_below_threshold(self, tmp_path):
        config = Config(base_dir=tmp_path)
        patterns = FakePromptPatternFindings(
            patterns=[
                FakePromptPattern(
                    normalized_prompt="rarely typed",
                    raw_examples=["rarely typed"],
                    count=3,  # below 5 threshold
                    projects=["app"],
                )
            ],
        )

        result = generate_recommendations(patterns, None, config)
        assert len(result.recommendations) == 0

    @patch("ambient.present.api.call_api", side_effect=Exception("API error"))
    def test_api_failure_skips_recommendation(self, mock_api, tmp_path):
        config = Config(base_dir=tmp_path)
        patterns = FakePromptPatternFindings(
            patterns=[
                FakePromptPattern(
                    normalized_prompt="failing pattern",
                    raw_examples=["failing pattern"],
                    count=10,
                    projects=["app"],
                )
            ],
        )

        result = generate_recommendations(patterns, None, config)
        assert len(result.recommendations) == 0

    def test_no_patterns_no_recommendations(self, tmp_path):
        config = Config(base_dir=tmp_path)
        result = generate_recommendations(None, None, config)
        assert len(result.recommendations) == 0

    @patch("ambient.present.api.call_api", return_value="skill content")
    def test_idempotent_overwrite(self, mock_api, tmp_path):
        config = Config(base_dir=tmp_path)
        patterns = FakePromptPatternFindings(
            patterns=[
                FakePromptPattern(
                    normalized_prompt="same pattern",
                    raw_examples=["same pattern"],
                    count=5,
                    projects=["app"],
                )
            ],
        )

        generate_recommendations(patterns, None, config)
        generate_recommendations(patterns, None, config)
        rec_dir = tmp_path / "recommendations"
        files = list(rec_dir.glob("*.md"))
        assert len(files) == 1  # overwritten, not duplicated
