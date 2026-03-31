import plistlib
import sys
from unittest.mock import patch, MagicMock

import pytest

from ambient.config import Config
from ambient.daemon.launchd import (
    AGENT_LABEL,
    generate_plist,
    install_agent,
    uninstall_agent,
    is_agent_loaded,
)


@pytest.fixture
def config(tmp_path):
    c = Config(base_dir=tmp_path)
    c.ensure_dirs()
    return c


class TestGeneratePlist:
    def test_program_arguments(self, config):
        plist = generate_plist(config)
        assert plist["ProgramArguments"] == [
            sys.executable,
            "-m",
            "ambient.cli",
            "daemon-tick",
        ]

    def test_uses_sys_executable(self, config):
        plist = generate_plist(config)
        assert plist["ProgramArguments"][0] == sys.executable

    def test_label(self, config):
        plist = generate_plist(config)
        assert plist["Label"] == "com.ambient.daemon"

    def test_start_interval(self, config):
        plist = generate_plist(config)
        assert plist["StartInterval"] == 1800

    def test_run_at_load_false(self, config):
        plist = generate_plist(config)
        assert plist["RunAtLoad"] is False

    def test_stdout_path(self, config):
        plist = generate_plist(config)
        expected = str(config.daemon_dir / "launchd-stdout.log")
        assert plist["StandardOutPath"] == expected

    def test_stderr_path(self, config):
        plist = generate_plist(config)
        expected = str(config.daemon_dir / "launchd-stderr.log")
        assert plist["StandardErrorPath"] == expected

    def test_plist_roundtrip(self, config, tmp_path):
        plist = generate_plist(config)
        plist_file = tmp_path / "test.plist"
        with open(plist_file, "wb") as f:
            plistlib.dump(plist, f)
        with open(plist_file, "rb") as f:
            loaded = plistlib.load(f)
        assert loaded == plist


class TestInstallAgent:
    @patch("ambient.daemon.launchd.subprocess.run")
    def test_writes_plist_and_calls_bootstrap(self, mock_run, config, tmp_path, monkeypatch):
        import ambient.daemon.launchd as launchd_mod

        fake_plist = tmp_path / "com.ambient.daemon.plist"
        monkeypatch.setattr(launchd_mod, "PLIST_PATH", fake_plist)

        install_agent(config)

        assert fake_plist.exists()
        with open(fake_plist, "rb") as f:
            written = plistlib.load(f)
        assert written["Label"] == AGENT_LABEL

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "launchctl"
        assert cmd[1] == "bootstrap"
        assert "gui/" in cmd[2]
        assert str(fake_plist) == cmd[3]


class TestUninstallAgent:
    @patch("ambient.daemon.launchd.subprocess.run")
    def test_calls_bootout_and_removes_plist(self, mock_run, tmp_path, monkeypatch):
        import ambient.daemon.launchd as launchd_mod

        fake_plist = tmp_path / "com.ambient.daemon.plist"
        fake_plist.write_bytes(b"placeholder")
        monkeypatch.setattr(launchd_mod, "PLIST_PATH", fake_plist)

        uninstall_agent()

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "launchctl"
        assert cmd[1] == "bootout"
        assert AGENT_LABEL in cmd[2]

        assert not fake_plist.exists()


class TestIsAgentLoaded:
    @patch("ambient.daemon.launchd.subprocess.run")
    def test_returns_true_when_loaded(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assert is_agent_loaded() is True

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "launchctl"
        assert cmd[1] == "print"
        assert AGENT_LABEL in cmd[2]

    @patch("ambient.daemon.launchd.subprocess.run")
    def test_returns_false_when_not_loaded(self, mock_run):
        mock_run.return_value = MagicMock(returncode=113)
        assert is_agent_loaded() is False
