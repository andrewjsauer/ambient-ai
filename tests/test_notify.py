import logging
from unittest.mock import patch

import pytest

from ambient.config import Config
from ambient.detect.pauses import PauseClassification
from ambient.present.notify import notify_stuck


@pytest.fixture
def config(tmp_path):
    return Config(base_dir=tmp_path)


def _make_classification(label="stuck", gap_ms=700_000):
    return PauseClassification(
        gap_ms=gap_ms,
        label=label,
        probabilities={"routine": 0.01, "evaluating": 0.09, "stuck": 0.90},
        preceding_command="pytest tests/",
        following_command="vim parser.py",
        ts_start=1000,
    )


@patch("ambient.present.notify.subprocess.run")
def test_stuck_notification_triggered(mock_run, config):
    """Happy path: stuck classification with gap > 10 min triggers subprocess call."""
    classifications = [_make_classification(label="stuck", gap_ms=700_000)]
    notify_stuck(classifications, config)
    mock_run.assert_called_once()
    args = mock_run.call_args
    assert "osascript" in args[0][0][0]
    assert "Ambient" in args[0][0][2]


@patch("ambient.present.notify.subprocess.run")
def test_multiple_stuck_only_one_notification(mock_run, config):
    """Multiple stuck episodes produce at most one notification."""
    classifications = [
        _make_classification(label="stuck", gap_ms=700_000),
        _make_classification(label="stuck", gap_ms=900_000),
        _make_classification(label="stuck", gap_ms=800_000),
    ]
    notify_stuck(classifications, config)
    assert mock_run.call_count == 1
    # Should pick the worst (900_000ms)
    call_script = mock_run.call_args[0][0][2]
    assert "15m" in call_script  # 900_000 // 60_000 = 15


@patch("ambient.present.notify.subprocess.run")
def test_no_stuck_no_notification(mock_run, config):
    """No stuck classifications means no notification and no error."""
    classifications = [
        _make_classification(label="routine", gap_ms=2000),
        _make_classification(label="evaluating", gap_ms=30_000),
    ]
    notify_stuck(classifications, config)
    mock_run.assert_not_called()


@patch("ambient.present.notify.subprocess.run")
def test_stuck_below_threshold_no_notification(mock_run, config):
    """Stuck classification below 10 min threshold should not trigger."""
    classifications = [_make_classification(label="stuck", gap_ms=300_000)]
    notify_stuck(classifications, config)
    mock_run.assert_not_called()


@patch("ambient.present.notify.subprocess.run")
def test_osascript_failure_caught(mock_run, config, caplog):
    """osascript failure is caught and logged, doesn't crash."""
    mock_run.side_effect = OSError("osascript not found")
    classifications = [_make_classification(label="stuck", gap_ms=700_000)]
    with caplog.at_level(logging.ERROR):
        notify_stuck(classifications, config)
    assert "Failed to send stuck notification" in caplog.text


@patch("ambient.present.notify.subprocess.run")
def test_empty_classifications_no_error(mock_run, config):
    """Empty list is handled gracefully."""
    notify_stuck([], config)
    mock_run.assert_not_called()
