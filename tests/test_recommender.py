from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from ambient.config import Config
from ambient.present.recommender import (
    COACHING_RULE_SYSTEM,
    Recommendation,
    generate_coaching_recommendations,
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


@dataclass
class FakeStuckPattern:
    project: str
    file_cluster: list[str]
    failing_tools: list[str]
    episode_count: int
    avg_thrash_score: float | None
    total_duration_ms: int
    session_ids: list[str] = field(default_factory=list)


@dataclass
class FakeStuckPatternFindings:
    patterns: list[FakeStuckPattern] = field(default_factory=list)
    total_stuck_sessions: int = 0


@dataclass
class FakeProjectVelocity:
    avg_ms: int = 0
    resolved_count: int = 0


@dataclass
class FakeVelocityMetrics:
    avg_ms: int = 0
    by_project: dict = field(default_factory=dict)


def _make_pattern(episode_count: int) -> FakeStuckPatternFindings:
    return FakeStuckPatternFindings(
        patterns=[
            FakeStuckPattern(
                project="app",
                file_cluster=["src/foo.py"],
                failing_tools=["pytest"],
                episode_count=episode_count,
                avg_thrash_score=0.6,
                total_duration_ms=600000,
            )
        ]
    )


class TestCoachingRecommendations:
    def test_system_prompt_contains_whitelist_language(self):
        assert "Do not invent" in COACHING_RULE_SYSTEM
        assert "ambient insights" in COACHING_RULE_SYSTEM

    @patch("ambient.present.api.call_api", return_value="- be careful in app")
    def test_pattern_6_emits_rec_with_low_confidence(self, mock_api, tmp_path):
        config = Config(base_dir=tmp_path)
        stuck = _make_pattern(6)
        recs = generate_coaching_recommendations(stuck, None, config)
        assert len(recs) == 1
        # captured user prompt is positional arg 2 of call_api
        _, args, kwargs = mock_api.mock_calls[0]
        user_prompt = args[2]
        assert "Sample size: 6" in user_prompt
        assert "confidence: low" in user_prompt

    @patch("ambient.present.api.call_api", return_value="- be careful in app")
    def test_pattern_4_below_gate_no_rec(self, mock_api, tmp_path):
        config = Config(base_dir=tmp_path)
        stuck = _make_pattern(4)
        recs = generate_coaching_recommendations(stuck, None, config)
        assert recs == []
        mock_api.assert_not_called()

    @patch("ambient.present.api.call_api", return_value="- rule")
    def test_pattern_10_medium_confidence(self, mock_api, tmp_path):
        config = Config(base_dir=tmp_path)
        stuck = _make_pattern(10)
        generate_coaching_recommendations(stuck, None, config)
        user_prompt = mock_api.mock_calls[0].args[2]
        assert "confidence: medium" in user_prompt

    @patch("ambient.present.api.call_api", return_value="- rule")
    def test_pattern_20_high_confidence(self, mock_api, tmp_path):
        config = Config(base_dir=tmp_path)
        stuck = _make_pattern(20)
        generate_coaching_recommendations(stuck, None, config)
        user_prompt = mock_api.mock_calls[0].args[2]
        assert "confidence: high" in user_prompt

    def test_velocity_outlier_resolved_4_skipped(self, tmp_path):
        config = Config(base_dir=tmp_path)
        vm = FakeVelocityMetrics(
            avg_ms=60000,
            by_project={"app": FakeProjectVelocity(avg_ms=300000, resolved_count=4)},
        )
        recs = generate_coaching_recommendations(None, vm, config)
        assert recs == []

    def test_velocity_outlier_resolved_5_emits_with_sample_phrase(self, tmp_path):
        config = Config(base_dir=tmp_path)
        vm = FakeVelocityMetrics(
            avg_ms=60000,
            by_project={"app": FakeProjectVelocity(avg_ms=300000, resolved_count=5)},
        )
        recs = generate_coaching_recommendations(None, vm, config)
        assert len(recs) == 1
        assert "5 resolved chains" in recs[0].artifact
        assert "low sample" in recs[0].artifact
