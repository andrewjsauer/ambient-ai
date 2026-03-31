import os
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import pytest

from ambient.config import Config
from ambient.daemon.state import DaemonState


@pytest.fixture
def config(tmp_path):
    c = Config(base_dir=tmp_path)
    c.ensure_dirs()
    return c


class TestDaemonStart:
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key-123"})
    @patch("ambient.daemon.launchd.is_agent_loaded", return_value=False)
    @patch("ambient.daemon.launchd.install_agent")
    def test_copies_api_key_and_installs(self, mock_install, mock_loaded, config):
        from ambient.cli import cmd_daemon_start

        args = type("Args", (), {})()
        out = StringIO()
        with redirect_stdout(out):
            cmd_daemon_start(config, args)

        assert config.dotenv_path.read_text() == "ANTHROPIC_API_KEY=sk-test-key-123\n"
        mock_install.assert_called_once_with(config)
        assert "started" in out.getvalue().lower()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key-123"})
    @patch("ambient.daemon.launchd.is_agent_loaded", return_value=False)
    @patch("ambient.daemon.launchd.install_agent")
    def test_overwrites_existing_env_file(self, mock_install, mock_loaded, config):
        from ambient.cli import cmd_daemon_start

        config.dotenv_path.write_text("ANTHROPIC_API_KEY=old-key\n")
        args = type("Args", (), {})()
        with redirect_stdout(StringIO()):
            cmd_daemon_start(config, args)

        assert "sk-test-key-123" in config.dotenv_path.read_text()

    @patch.dict(os.environ, {}, clear=True)
    def test_errors_when_no_api_key(self, config):
        from ambient.cli import cmd_daemon_start

        os.environ.pop("ANTHROPIC_API_KEY", None)
        args = type("Args", (), {})()
        with pytest.raises(SystemExit):
            cmd_daemon_start(config, args)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"})
    @patch("ambient.daemon.launchd.is_agent_loaded", return_value=True)
    @patch("ambient.daemon.launchd.install_agent")
    def test_warns_when_already_running(self, mock_install, mock_loaded, config):
        from ambient.cli import cmd_daemon_start

        args = type("Args", (), {})()
        out = StringIO()
        with redirect_stdout(out):
            cmd_daemon_start(config, args)

        mock_install.assert_not_called()
        assert "already running" in out.getvalue().lower()


class TestDaemonStop:
    @patch("ambient.daemon.launchd.is_agent_loaded", return_value=True)
    @patch("ambient.daemon.launchd.uninstall_agent")
    def test_stops_running_daemon(self, mock_uninstall, mock_loaded, config):
        from ambient.cli import cmd_daemon_stop

        args = type("Args", (), {})()
        out = StringIO()
        with redirect_stdout(out):
            cmd_daemon_stop(config, args)

        mock_uninstall.assert_called_once()
        assert "stopped" in out.getvalue().lower()

    @patch("ambient.daemon.launchd.is_agent_loaded", return_value=False)
    def test_graceful_when_not_running(self, mock_loaded, config):
        from ambient.cli import cmd_daemon_stop

        args = type("Args", (), {})()
        out = StringIO()
        with redirect_stdout(out):
            cmd_daemon_stop(config, args)

        assert "not running" in out.getvalue().lower()


class TestDaemonStatus:
    @patch("ambient.daemon.launchd.is_agent_loaded", return_value=False)
    @patch("ambient.daemon.lock.is_locked", return_value=(False, {}))
    def test_shows_not_running(self, mock_locked, mock_loaded, config):
        from ambient.cli import cmd_daemon_status

        args = type("Args", (), {})()
        out = StringIO()
        with redirect_stdout(out):
            cmd_daemon_status(config, args)

        output = out.getvalue()
        assert "not running" in output
        assert "never" in output

    @patch("ambient.daemon.launchd.is_agent_loaded", return_value=True)
    @patch("ambient.daemon.lock.is_locked", return_value=(False, {}))
    def test_shows_running_with_state(self, mock_locked, mock_loaded, config):
        from ambient.cli import cmd_daemon_status

        state = DaemonState(
            last_analyzed_ts=1711870984117,
            last_summary_date="2026-03-30",
            last_calibration_date="2026-03-24",
            events_since_calibration=150,
        )
        state.save(config.state_path)

        args = type("Args", (), {})()
        out = StringIO()
        with redirect_stdout(out):
            cmd_daemon_status(config, args)

        output = out.getvalue()
        assert "running" in output
        assert "2026-03-30" in output
        assert "150" in output
