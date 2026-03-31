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
