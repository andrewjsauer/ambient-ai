"""Tests for project capability detection."""

import json

import pytest

from ambient.detect.project_capabilities import (
    Capabilities,
    clear_capability_cache,
    detect_capabilities,
)


@pytest.fixture(autouse=True)
def _reset_capability_cache():
    """Per-test isolation. setup_function alone wouldn't reset between class
    methods, which could let cache state leak across tests."""
    clear_capability_cache()
    yield
    clear_capability_cache()


class TestDetectCapabilities:
    def test_package_json_with_test_script(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({
            "scripts": {"test": "vitest", "dev": "next dev"},
        }))
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_tests is True
        assert caps.has_dev_server is True
        assert "vitest" in caps.evidence["has_tests"]
        assert "next dev" in caps.evidence["has_dev_server"]

    def test_package_json_with_typecheck_script(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({
            "scripts": {"typecheck": "tsc --noEmit"},
        }))
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_typecheck is True
        assert "tsc --noEmit" in caps.evidence["has_typecheck"]

    def test_package_json_no_test_script(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({
            "scripts": {"build": "next build"},
        }))
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_tests is False
        assert caps.has_dev_server is False
        assert caps.evidence == {}

    def test_makefile_test_and_dev_targets(self, tmp_path):
        (tmp_path / "Makefile").write_text(
            "test:\n\tpytest\n\ndev:\n\tuvicorn app:app --reload\n"
        )
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_tests is True
        assert caps.has_dev_server is True
        assert "Makefile" in caps.evidence["has_tests"]

    def test_makefile_no_test_target(self, tmp_path):
        (tmp_path / "Makefile").write_text("build:\n\tcc -o app app.c\n")
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_tests is False

    def test_pyproject_with_pytest_section(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\nminversion = \"8.0\"\n"
        )
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_tests is True
        assert "pyproject.toml" in caps.evidence["has_tests"]

    def test_pyproject_commented_out_pytest_does_not_match(self, tmp_path):
        """Anchor at start-of-line so commented-out config doesn't count."""
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname = \"x\"\n# [tool.pytest.ini_options] -- commented out\n"
        )
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_tests is False

    def test_pyproject_pytest_in_string_literal_does_not_match(self, tmp_path):
        """A doc-string mention of '[tool.pytest' shouldn't fire the probe."""
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname = \"x\"\ndescription = \"see [tool.pytest] in docs\"\n"
        )
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_tests is False

    def test_pyproject_without_pytest_falls_back_to_tests_dir(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = \"x\"\n")
        (tmp_path / "tests").mkdir()
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_tests is True
        assert "Python project" in caps.evidence["has_tests"]

    def test_setup_py_also_gates_tests_dir(self, tmp_path):
        """An older Python project with setup.py + tests/ should also count."""
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()\n")
        (tmp_path / "tests").mkdir()
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_tests is True

    def test_bare_tests_dir_without_python_signal_does_not_count(self, tmp_path):
        """A JS/Go/etc. project with only a tests/ folder must not be classified
        as has_tests — the tests/ heuristic is Python-specific."""
        (tmp_path / "tests").mkdir()
        # No pyproject.toml, no setup.py, no package.json
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_tests is False

    def test_cargo_toml_implies_tests(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname = \"x\"\n")
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_tests is True
        assert "Cargo.toml" in caps.evidence["has_tests"]

    def test_tsconfig_implies_typecheck(self, tmp_path):
        (tmp_path / "tsconfig.json").write_text("{}")
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_typecheck is True
        assert caps.evidence["has_typecheck"] == "tsconfig.json present"

    def test_combined_signals(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({
            "scripts": {"test": "jest", "dev": "vite"},
        }))
        (tmp_path / "tsconfig.json").write_text("{}")
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_tests is True
        assert caps.has_dev_server is True
        assert caps.has_typecheck is True

    def test_nonexistent_cwd_returns_empty(self):
        caps = detect_capabilities("/nonexistent/path/that/does/not/exist")
        assert caps == Capabilities()

    def test_none_cwd_returns_empty(self):
        caps = detect_capabilities(None)
        assert caps == Capabilities()

    def test_empty_cwd_returns_empty(self):
        caps = detect_capabilities("")
        assert caps == Capabilities()

    def test_malformed_package_json_does_not_raise(self, tmp_path):
        (tmp_path / "package.json").write_text("{not json")
        # Falls through gracefully; no other capabilities present so empty result.
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_tests is False
        assert caps.has_dev_server is False

    def test_malformed_package_json_does_not_block_makefile_probe(self, tmp_path):
        (tmp_path / "package.json").write_text("{not json")
        (tmp_path / "Makefile").write_text("test:\n\tpytest\n")
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_tests is True
        assert "Makefile" in caps.evidence["has_tests"]

    def test_package_json_scripts_not_dict(self, tmp_path):
        # Defensive: scripts could be malformed even in valid JSON
        (tmp_path / "package.json").write_text(json.dumps({"scripts": "oops"}))
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_tests is False

    def test_permission_error_on_dir_check_does_not_raise(self, tmp_path, monkeypatch):
        """If Path.is_dir() raises PermissionError (e.g. mode 700 parent owned
        by another user, expired NFS ACL), detect_capabilities returns empty
        rather than propagating per its docstring contract."""
        from pathlib import Path
        original_is_dir = Path.is_dir

        def raising_is_dir(self):
            if str(self) == str(tmp_path):
                raise PermissionError("simulated EACCES")
            return original_is_dir(self)

        monkeypatch.setattr(Path, "is_dir", raising_is_dir)
        caps = detect_capabilities(str(tmp_path))
        assert caps == Capabilities()

    def test_permission_error_on_file_probe_does_not_raise(self, tmp_path, monkeypatch):
        """is_file() inside a probe must also tolerate PermissionError."""
        from pathlib import Path
        original_is_file = Path.is_file

        def raising_is_file(self):
            if self.name == "package.json":
                raise PermissionError("simulated EACCES")
            return original_is_file(self)

        monkeypatch.setattr(Path, "is_file", raising_is_file)
        # Other probes should still complete; result is empty since no metadata exists.
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_tests is False  # didn't crash

    def test_first_evidence_wins(self, tmp_path):
        # package.json should win over Makefile when both exist with test capability
        (tmp_path / "package.json").write_text(json.dumps({
            "scripts": {"test": "vitest"},
        }))
        (tmp_path / "Makefile").write_text("test:\n\tpytest\n")
        caps = detect_capabilities(str(tmp_path))
        assert caps.has_tests is True
        assert "package.json" in caps.evidence["has_tests"]


class TestCache:
    def test_cache_returns_same_instance(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({
            "scripts": {"test": "vitest"},
        }))
        first = detect_capabilities(str(tmp_path))
        second = detect_capabilities(str(tmp_path))
        assert first is second

    def test_cache_persists_until_cleared(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({
            "scripts": {"test": "vitest"},
        }))
        first = detect_capabilities(str(tmp_path))
        # Remove the file; cached result should still come back
        (tmp_path / "package.json").unlink()
        second = detect_capabilities(str(tmp_path))
        assert second is first
        assert second.has_tests is True

    def test_clear_cache_forces_reprobe(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({
            "scripts": {"test": "vitest"},
        }))
        first = detect_capabilities(str(tmp_path))
        assert first.has_tests is True
        (tmp_path / "package.json").unlink()
        clear_capability_cache()
        second = detect_capabilities(str(tmp_path))
        assert second is not first
        assert second.has_tests is False
