import json

import pytest

from ambient.config import Config
from ambient.daemon.state import DaemonState


@pytest.fixture
def config(tmp_path):
    c = Config(base_dir=tmp_path)
    c.ensure_dirs()
    return c


class TestConfig:
    def test_daemon_paths_derived_from_base_dir(self, config, tmp_path):
        assert config.daemon_dir == tmp_path / "daemon"
        assert config.lock_path == tmp_path / "daemon" / "daemon.lock"
        assert config.state_path == tmp_path / "daemon" / "state.json"
        assert config.daemon_log_path == tmp_path / "daemon" / "daemon.log"
        assert config.dotenv_path == tmp_path / ".env"

    def test_ensure_dirs_creates_daemon_dir(self, config):
        assert config.daemon_dir.is_dir()


class TestDaemonState:
    def test_round_trip(self, config):
        state = DaemonState(
            last_analyzed_ts=1711870984117,
            last_summary_date="2026-03-30",
            last_calibration_date="2026-03-24",
            events_since_calibration=450,
        )
        state.save(config.state_path)
        loaded = DaemonState.load(config.state_path)
        assert loaded.last_analyzed_ts == 1711870984117
        assert loaded.last_summary_date == "2026-03-30"
        assert loaded.last_calibration_date == "2026-03-24"
        assert loaded.events_since_calibration == 450

    def test_load_returns_defaults_when_file_missing(self, config):
        state = DaemonState.load(config.state_path)
        assert state.last_analyzed_ts == 0
        assert state.last_summary_date == ""
        assert state.last_calibration_date == ""
        assert state.events_since_calibration == 0

    def test_load_handles_corrupt_json(self, config):
        config.state_path.write_text("not valid json{{{")
        state = DaemonState.load(config.state_path)
        assert state.last_analyzed_ts == 0

    def test_load_handles_partial_json(self, config):
        config.state_path.write_text(json.dumps({"last_analyzed_ts": 999}))
        state = DaemonState.load(config.state_path)
        assert state.last_analyzed_ts == 999
        assert state.events_since_calibration == 0

    def test_save_is_atomic(self, config):
        state = DaemonState(last_analyzed_ts=100)
        state.save(config.state_path)
        assert config.state_path.exists()
        # No tmp files left behind
        tmp_files = list(config.daemon_dir.glob("*.tmp"))
        assert tmp_files == []
