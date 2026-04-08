import json
import subprocess
import sys
from pathlib import Path

import pytest

from ambient.config import Config


@pytest.fixture
def config(tmp_path):
    c = Config(base_dir=tmp_path)
    c.ensure_dirs()
    return c


def _write_fixture_events(config, date_str="2026-03-30", count=20):
    path = config.events_path(date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in range(count):
            event = {
                "ts_start": 1711800000000 + i * 5000,
                "ts_end": 1711800000000 + i * 5000 + 100,
                "duration_ms": 100,
                "command": f"cmd_{i}",
                "exit_code": 0,
                "cwd": "/tmp",
                "tmux_pane": "%0",
                "gap_ms": 4000 if i > 0 else None,
            }
            f.write(json.dumps(event) + "\n")


def test_ambient_command_exists():
    result = subprocess.run(
        [sys.executable, "-m", "ambient.cli", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Ambient AI" in result.stdout


def test_start_prints_source_line():
    result = subprocess.run(
        [sys.executable, "-m", "ambient.cli", "start"],
        capture_output=True, text=True,
    )
    assert "source" in result.stdout
    assert "hooks.zsh" in result.stdout


def test_status_no_data(config):
    from ambient.cli import cmd_status
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        cmd_status(config, type("Args", (), {})())

    output = f.getvalue()
    assert "No events captured yet" in output
    assert "Not calibrated" in output


def test_status_with_data(config):
    from ambient.cli import cmd_status
    import io
    from contextlib import redirect_stdout
    from datetime import datetime

    date_str = datetime.now().strftime("%Y-%m-%d")
    _write_fixture_events(config, date_str)

    f = io.StringIO()
    with redirect_stdout(f):
        cmd_status(config, type("Args", (), {})())

    output = f.getvalue()
    assert "Events today: 20" in output


def test_review_no_summary(config):
    from ambient.cli import cmd_review
    import io
    from contextlib import redirect_stdout

    args = type("Args", (), {"date": "2026-03-30"})()

    f = io.StringIO()
    with redirect_stdout(f):
        cmd_review(config, args)

    assert "No summary" in f.getvalue()


def test_review_with_summary(config):
    from ambient.cli import cmd_review
    import io
    from contextlib import redirect_stdout

    config.summary_path("2026-03-30").write_text("# Great day\nYou were productive.")
    args = type("Args", (), {"date": "2026-03-30"})()

    f = io.StringIO()
    with redirect_stdout(f):
        cmd_review(config, args)

    assert "Great day" in f.getvalue()


# --- Unit 8: recommendations and apply ---

def _write_recommendation(config, rec_id, rec_type="skill", title="Test Skill", artifact="# My Skill\nDo stuff"):
    rec_dir = config.base_dir / "recommendations"
    rec_dir.mkdir(parents=True, exist_ok=True)
    content = f"""---
type: {rec_type}
title: "{title}"
rationale: "You do this a lot."
source_pattern: "test pattern"
---

{artifact}
"""
    path = rec_dir / f"{rec_id}.md"
    path.write_text(content)
    return path


def test_recommendations_empty(config):
    from ambient.cli import cmd_recommendations
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        cmd_recommendations(config, type("Args", (), {})())

    assert "No recommendations" in f.getvalue()


def test_recommendations_lists_files(config):
    from ambient.cli import cmd_recommendations
    import io
    from contextlib import redirect_stdout

    _write_recommendation(config, "skill-deploy", "skill", "Deploy Skill")
    _write_recommendation(config, "alias-gp", "alias", "Git Push Alias")

    f = io.StringIO()
    with redirect_stdout(f):
        cmd_recommendations(config, type("Args", (), {})())

    output = f.getvalue()
    assert "alias-gp" in output
    assert "skill-deploy" in output
    assert "skill" in output
    assert "alias" in output


def test_apply_skill(config, tmp_path):
    from ambient.cli import cmd_apply
    import io
    from contextlib import redirect_stdout

    _write_recommendation(config, "skill-test", "skill", "Test Skill", "# Test\nStep 1: do thing")

    # Override HOME so we don't write to real ~/.claude/commands
    commands_dir = tmp_path / ".claude" / "commands"
    import ambient.cli as cli_module
    original_home = Path.home

    try:
        # Monkey-patch Path.home to use tmp_path
        Path.home = staticmethod(lambda: tmp_path)

        f = io.StringIO()
        with redirect_stdout(f):
            cmd_apply(config, type("Args", (), {"recommendation_id": "skill-test"})())

        output = f.getvalue()
        assert "Installed skill to" in output

        dest = commands_dir / "skill-test.md"
        assert dest.exists()
        content = dest.read_text()
        assert "# Test" in content
        assert "Step 1: do thing" in content
    finally:
        Path.home = original_home


def test_apply_nonexistent(config):
    from ambient.cli import cmd_apply

    with pytest.raises(SystemExit):
        cmd_apply(config, type("Args", (), {"recommendation_id": "no-such-rec"})())


def test_apply_non_skill(config):
    from ambient.cli import cmd_apply
    import io
    from contextlib import redirect_stdout

    _write_recommendation(config, "alias-gp", "alias", "Git Push Alias", 'alias gp="git push"')

    f = io.StringIO()
    with redirect_stdout(f):
        cmd_apply(config, type("Args", (), {"recommendation_id": "alias-gp"})())

    output = f.getvalue()
    assert "Only skill installation supported" in output


# --- Unit 11: projects ---

def _write_project_events(config, date_str="2026-03-30"):
    """Write events across multiple projects for testing."""
    path = config.events_path(date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_ts = 1711800000000
    events = []
    # 10 events in project-alpha (50ms each)
    for i in range(10):
        events.append({
            "ts_start": base_ts + i * 5000,
            "ts_end": base_ts + i * 5000 + 50,
            "duration_ms": 50,
            "command": f"make build",
            "exit_code": 0,
            "cwd": "/home/user/project-alpha",
            "tmux_pane": "%0",
            "gap_ms": 4000 if i > 0 else None,
        })
    # 5 events in project-beta (100ms each)
    for i in range(5):
        events.append({
            "ts_start": base_ts + (10 + i) * 5000,
            "ts_end": base_ts + (10 + i) * 5000 + 100,
            "duration_ms": 100,
            "command": "cargo test",
            "exit_code": 0,
            "cwd": "/home/user/project-beta",
            "tmux_pane": "%0",
            "gap_ms": 4000,
        })
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_projects_no_events(config):
    from ambient.cli import cmd_projects
    import io
    from contextlib import redirect_stdout

    args = type("Args", (), {"window": None, "date": "2026-01-01"})()

    f = io.StringIO()
    with redirect_stdout(f):
        cmd_projects(config, args)

    assert "No events" in f.getvalue()


def test_projects_with_data(config):
    from ambient.cli import cmd_projects
    import io
    from contextlib import redirect_stdout

    _write_project_events(config, "2026-03-30")

    args = type("Args", (), {"window": None, "date": "2026-03-30"})()

    f = io.StringIO()
    with redirect_stdout(f):
        cmd_projects(config, args)

    output = f.getvalue()
    assert "project-alpha" in output
    assert "project-beta" in output
    assert "Context switches" in output
    assert "Primary project" in output
