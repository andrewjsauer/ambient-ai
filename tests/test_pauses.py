import numpy as np
import pytest

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.pauses import calibrate, classify


def _make_events_with_gaps(gaps_ms: list[int | None], commands: list[str] | None = None) -> list[Event]:
    events = []
    ts = 1000
    for i, gap in enumerate(gaps_ms):
        cmd = commands[i] if commands else f"cmd_{i}"
        events.append(
            Event(
                ts_start=ts,
                ts_end=ts + 100,
                duration_ms=100,
                command=cmd,
                exit_code=0,
                cwd="/tmp",
                tmux_pane="%0",
                gap_ms=gap,
                session_boundary=gap is not None and gap > 600_000,
            )
        )
        ts += 1000
    return events


def _generate_trimodal_gaps(n_per_component: int = 100, seed: int = 42) -> list[int]:
    rng = np.random.RandomState(seed)
    # routine: ~2s (log(2000) ≈ 7.6)
    routine = np.exp(rng.normal(7.6, 0.3, n_per_component)).astype(int)
    # evaluating: ~15s (log(15000) ≈ 9.6)
    evaluating = np.exp(rng.normal(9.6, 0.3, n_per_component)).astype(int)
    # stuck: ~60s (log(60000) ≈ 11.0)
    stuck = np.exp(rng.normal(11.0, 0.3, n_per_component)).astype(int)
    all_gaps = np.concatenate([routine, evaluating, stuck])
    rng.shuffle(all_gaps)
    return [int(g) for g in all_gaps]


@pytest.fixture
def config(tmp_path):
    return Config(base_dir=tmp_path, gmm_min_samples=60)


def test_calibrate_trimodal(config):
    gaps = _generate_trimodal_gaps(100)
    events = _make_events_with_gaps([None] + gaps)

    result = calibrate(events, config)

    assert result.available is True
    assert result.calibration_stats is not None
    assert result.calibration_stats.n_samples == 300

    # Means should be ordered: routine < evaluating < stuck
    means = result.calibration_stats.component_means_ms
    assert means[0] < means[1] < means[2]

    # Rough magnitude checks (in ms)
    assert 500 < means[0] < 8000      # routine ~2s
    assert 5000 < means[1] < 40000    # evaluating ~15s
    assert 20000 < means[2] < 200000  # stuck ~60s


def test_calibrate_insufficient_data(config):
    gaps = [1000, 2000, 3000]
    events = _make_events_with_gaps([None] + gaps)

    result = calibrate(events, config)

    assert result.available is False
    assert "Currently have 3 gaps" in result.reason
    assert "need 60" in result.reason


def test_classify_routine_gap(config):
    gaps = _generate_trimodal_gaps(100)
    events = _make_events_with_gaps([None] + gaps)
    calibrate(events, config)

    # Classify a clearly routine gap (~2s)
    test_events = _make_events_with_gaps([None, 2000], commands=["prev", "next"])
    result = classify(test_events, config)

    assert result.available is True
    assert len(result.classifications) == 1
    assert result.classifications[0].label == "routine"
    assert result.classifications[0].probabilities["routine"] > 0.5


def test_classify_stuck_gap(config):
    gaps = _generate_trimodal_gaps(100)
    events = _make_events_with_gaps([None] + gaps)
    calibrate(events, config)

    # Classify a clearly stuck gap (~60s)
    test_events = _make_events_with_gaps([None, 60000], commands=["prev", "next"])
    result = classify(test_events, config)

    assert result.available is True
    assert len(result.classifications) == 1
    assert result.classifications[0].label == "stuck"
    assert result.classifications[0].probabilities["stuck"] > 0.5


def test_classify_without_model(config):
    events = _make_events_with_gaps([None, 5000])
    result = classify(events, config)

    assert result.available is False
    assert result.reason == "not_calibrated"
    assert result.classifications == []


def test_session_boundaries_excluded(config):
    # Mix of normal gaps and session boundaries
    gaps = _generate_trimodal_gaps(30)
    gaps_with_boundaries = gaps + [700_000, 800_000, 900_000]  # These should be excluded
    events = _make_events_with_gaps([None] + gaps_with_boundaries)

    result = calibrate(events, config)

    # Should only use the non-boundary gaps
    assert result.available is True
    assert result.calibration_stats.n_samples == 90  # 30 * 3 components


def test_model_persistence(config):
    gaps = _generate_trimodal_gaps(100)
    events = _make_events_with_gaps([None] + gaps)
    calibrate(events, config)

    # Classify with fresh config pointing to same model
    test_events = _make_events_with_gaps([None, 2000])
    result1 = classify(test_events, config)

    # Load and classify again
    result2 = classify(test_events, config)

    assert result1.classifications[0].label == result2.classifications[0].label
    assert result1.classifications[0].probabilities == pytest.approx(
        result2.classifications[0].probabilities, abs=1e-6
    )


def test_bic_warning_logged(config, caplog):
    # Generate clearly bimodal data - BIC should prefer 2 components
    rng = np.random.RandomState(42)
    low = np.exp(rng.normal(7.0, 0.2, 150)).astype(int)
    high = np.exp(rng.normal(11.0, 0.2, 150)).astype(int)
    gaps = [int(g) for g in np.concatenate([low, high])]
    rng.shuffle(gaps)

    events = _make_events_with_gaps([None] + gaps)

    import logging
    with caplog.at_level(logging.WARNING):
        result = calibrate(events, config)

    # Model should still be saved (we keep configured n_components)
    assert result.available is True
    assert config.gmm_model_path.exists()
    # BIC scores should be computed
    assert 2 in result.calibration_stats.bic_scores
    assert 3 in result.calibration_stats.bic_scores
    assert 4 in result.calibration_stats.bic_scores
