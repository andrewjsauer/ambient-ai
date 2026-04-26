"""Tests for the slash-command taxonomy module."""

import pytest

from ambient.detect.slash_taxonomy import (
    classify_slash_command,
    extract_slash_command,
)


class TestExtractSlashCommand:
    def test_extracts_simple_command(self):
        text = "<command-name>/ship</command-name>\n<command-args></command-args>"
        assert extract_slash_command(text) == "/ship"

    def test_extracts_namespaced_command(self):
        text = (
            "<command-message>compound-engineering:ce-plan</command-message>\n"
            "<command-name>/compound-engineering:ce-plan</command-name>\n"
            "<command-args>foo bar</command-args>"
        )
        assert extract_slash_command(text) == "/compound-engineering:ce-plan"

    def test_returns_none_when_no_marker(self):
        assert extract_slash_command("just a regular prompt with no slash") is None

    def test_returns_none_for_empty_input(self):
        assert extract_slash_command("") is None

    def test_returns_none_for_none_input(self):
        assert extract_slash_command(None) is None  # type: ignore[arg-type]

    def test_strips_whitespace_inside_markers(self):
        text = "<command-name>  /ship  </command-name>"
        assert extract_slash_command(text) == "/ship"

    def test_first_marker_wins(self):
        text = (
            "<command-name>/ship</command-name>\n"
            "<command-name>/clear</command-name>"
        )
        assert extract_slash_command(text) == "/ship"


class TestClassifySlashCommand:
    def test_classifies_ship_as_execution(self):
        assert classify_slash_command("/ship") == "execution"

    def test_classifies_work_as_execution(self):
        assert classify_slash_command("/work") == "execution"

    def test_classifies_ce_work_as_execution(self):
        assert classify_slash_command("/compound-engineering:ce-work") == "execution"

    def test_classifies_ce_plan_as_planning(self):
        assert classify_slash_command("/compound-engineering:ce-plan") == "planning"

    def test_classifies_colon_form_ce_plan_as_planning(self):
        # Deprecated namespacing variant — same category.
        assert classify_slash_command("/compound-engineering:ce:plan") == "planning"

    def test_classifies_ce_brainstorm_as_planning(self):
        assert classify_slash_command("/compound-engineering:ce-brainstorm") == "planning"

    def test_classifies_ce_review_as_review(self):
        assert classify_slash_command("/compound-engineering:ce-review") == "review"

    def test_classifies_aeo_audit_as_review(self):
        assert classify_slash_command("/aeo-audit") == "review"

    def test_classifies_frontend_design_as_design(self):
        assert classify_slash_command("/compound-engineering:frontend-design") == "design"

    def test_classifies_clear_as_meta(self):
        assert classify_slash_command("/clear") == "meta"

    def test_classifies_model_as_meta(self):
        assert classify_slash_command("/model") == "meta"

    def test_unknown_command_is_other(self):
        assert classify_slash_command("/totally-made-up-command") == "other"

    def test_none_command_is_other(self):
        assert classify_slash_command(None) == "other"

    def test_empty_command_is_other(self):
        assert classify_slash_command("") == "other"

    def test_strips_trailing_colon(self):
        # Some real prompts have "/ship:" — should match "/ship".
        assert classify_slash_command("/ship:") == "execution"

    def test_strips_whitespace(self):
        assert classify_slash_command("  /ship  ") == "execution"

    def test_adds_leading_slash_if_missing(self):
        # Defensive — extract_slash_command always returns leading slash, but be lenient.
        assert classify_slash_command("ship") == "execution"


class TestOverrides:
    def test_user_override_reclassifies_known_command(self):
        # Forcing /ship to be classified as planning (artificial example).
        result = classify_slash_command("/ship", overrides={"/ship": "planning"})
        assert result == "planning"

    def test_user_override_classifies_unknown_command(self):
        result = classify_slash_command(
            "/my-custom-cmd", overrides={"/my-custom-cmd": "review"}
        )
        assert result == "review"

    def test_invalid_override_category_falls_through_to_builtin(self):
        # Garbage category in the override should not crash; built-in wins.
        result = classify_slash_command(
            "/ship", overrides={"/ship": "not-a-real-category"}
        )
        assert result == "execution"

    def test_override_does_not_affect_unrelated_commands(self):
        result = classify_slash_command(
            "/ship", overrides={"/clear": "planning"}
        )
        assert result == "execution"

    def test_empty_overrides_behaves_like_default(self):
        assert classify_slash_command("/ship", overrides={}) == "execution"

    def test_none_overrides_behaves_like_default(self):
        assert classify_slash_command("/ship", overrides=None) == "execution"


class TestCategoryCoverage:
    """Smoke checks that each category has at least one canonical member."""

    @pytest.mark.parametrize("cmd,expected", [
        ("/compound-engineering:ce-plan", "planning"),
        ("/ship", "execution"),
        ("/compound-engineering:ce-review", "review"),
        ("/compound-engineering:frontend-design", "design"),
        ("/clear", "meta"),
        ("/banana", "other"),
    ])
    def test_each_category_has_a_member(self, cmd: str, expected: str):
        assert classify_slash_command(cmd) == expected
