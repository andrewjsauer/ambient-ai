import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

from ambient.capture.reader import Event
from ambient.config import Config

logger = logging.getLogger(__name__)

LABELS = ["routine", "evaluating", "stuck"]


@dataclass
class PauseClassification:
    gap_ms: int
    label: str
    probabilities: dict[str, float]
    preceding_command: str
    following_command: str


@dataclass
class CalibrationStats:
    n_samples: int
    component_means_ms: list[float]
    component_stds_ms: list[float]
    bic_scores: dict[int, float]


@dataclass
class PauseFindings:
    available: bool
    reason: str | None = None
    calibration_stats: CalibrationStats | None = None
    classifications: list[PauseClassification] = field(default_factory=list)


def _extract_gaps(events: list[Event], session_boundary_ms: int) -> list[int]:
    gaps = []
    for e in events:
        if e.gap_ms is None:
            continue
        if e.session_boundary or e.gap_ms > session_boundary_ms:
            continue
        if e.gap_ms <= 0:
            continue
        gaps.append(e.gap_ms)
    return gaps


def calibrate(events: list[Event], config: Config) -> PauseFindings:
    gaps = _extract_gaps(events, config.session_boundary_ms)

    if len(gaps) == 0:
        return PauseFindings(
            available=False,
            reason=f"No valid gaps found (all filtered as None, <=0, or session boundaries). "
            f"Need at least {config.gmm_min_samples} gaps to calibrate.",
        )

    if len(gaps) < config.gmm_min_samples:
        needed = config.gmm_min_samples - len(gaps)
        est_hours = max(1, needed // 30)
        return PauseFindings(
            available=False,
            reason=f"Currently have {len(gaps)} gaps, need {config.gmm_min_samples}. "
            f"Work for approximately {est_hours} more hour(s) and re-run.",
        )

    log_gaps = np.log(np.array(gaps, dtype=float)).reshape(-1, 1)

    # Fit the GMM
    gmm = GaussianMixture(
        n_components=config.gmm_n_components,
        covariance_type=config.gmm_covariance_type,
        n_init=config.gmm_n_init,
        random_state=42,
    )
    gmm.fit(log_gaps)

    if not gmm.converged_:
        logger.warning("GMM did not converge after %d iterations", gmm.n_iter_)

    # Sort components by mean (ascending) to assign labels
    means = gmm.means_.flatten()
    order = np.argsort(means)
    label_map = {order[i]: LABELS[i] for i in range(len(LABELS))}

    # BIC validation for 2, 3, 4 components
    bic_scores = {}
    for n in [2, 3, 4]:
        test_gmm = GaussianMixture(
            n_components=n,
            covariance_type=config.gmm_covariance_type,
            n_init=config.gmm_n_init,
            random_state=42,
        )
        test_gmm.fit(log_gaps)
        bic_scores[n] = float(test_gmm.bic(log_gaps))

    best_n = min(bic_scores, key=bic_scores.get)
    if best_n != config.gmm_n_components:
        logger.warning(
            "BIC suggests %d components (BIC=%.1f) over %d (BIC=%.1f). "
            "Keeping %d as configured.",
            best_n, bic_scores[best_n],
            config.gmm_n_components, bic_scores[config.gmm_n_components],
            config.gmm_n_components,
        )

    # Save model and label map
    config.ensure_dirs()
    model_data = {"gmm": gmm, "label_map": label_map}
    joblib.dump(model_data, config.gmm_model_path)

    # Compute stats in original ms space
    sorted_means = means[order]
    variances = gmm.covariances_.flatten()[order]
    stds = np.sqrt(variances)

    stats = CalibrationStats(
        n_samples=len(gaps),
        component_means_ms=[float(np.exp(m)) for m in sorted_means],
        component_stds_ms=[float(np.exp(s)) for s in stds],
        bic_scores=bic_scores,
    )

    return PauseFindings(available=True, calibration_stats=stats)


def classify(events: list[Event], config: Config) -> PauseFindings:
    if not config.gmm_model_path.exists():
        return PauseFindings(
            available=False,
            reason="not_calibrated",
        )

    try:
        model_data = joblib.load(config.gmm_model_path)
        gmm: GaussianMixture = model_data["gmm"]
        label_map: dict[int, str] = model_data["label_map"]
    except Exception as e:
        logger.warning("Failed to load GMM model: %s", e)
        return PauseFindings(
            available=False,
            reason=f"model_corrupted: {e}",
        )

    classifications = []
    for i, event in enumerate(events):
        if event.gap_ms is None or event.gap_ms <= 0:
            continue
        if event.session_boundary or event.gap_ms > config.session_boundary_ms:
            continue

        log_gap = np.log(float(event.gap_ms)).reshape(1, -1)
        probs = gmm.predict_proba(log_gap)[0]

        prob_dict = {label_map[j]: float(probs[j]) for j in range(len(probs))}
        label = max(prob_dict, key=prob_dict.get)

        preceding = events[i - 1].command if i > 0 else ""
        following = event.command

        classifications.append(
            PauseClassification(
                gap_ms=event.gap_ms,
                label=label,
                probabilities=prob_dict,
                preceding_command=preceding,
                following_command=following,
            )
        )

    return PauseFindings(available=True, classifications=classifications)
