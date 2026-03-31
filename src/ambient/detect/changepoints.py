import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import ruptures

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.pauses import PauseFindings

logger = logging.getLogger(__name__)

# Command category classification keywords
CATEGORY_PATTERNS = {
    "test": ["pytest", "test", "rspec", "jest", "mocha", "cargo test"],
    "build": ["make", "build", "compile", "webpack", "cargo build", "npm run build"],
    "git": ["git"],
    "edit": ["vim", "nvim", "nano", "emacs", "code"],
    "claude": ["claude"],
}


def _categorize_command(cmd: str) -> str:
    cmd_lower = cmd.lower()
    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if cmd_lower.startswith(pattern) or f" {pattern}" in cmd_lower:
                return category
    return "other"


@dataclass
class Segment:
    start_ts: int
    end_ts: int
    duration_min: float
    mean_rate: float  # commands per bucket
    dominant_category: str
    pause_distribution: dict[str, float] | None
    label: str


@dataclass
class Changepoint:
    ts: int
    from_segment_summary: str
    to_segment_summary: str


@dataclass
class ChangepointFindings:
    segments: list[Segment]
    changepoints: list[Changepoint]


def _bucket_events(
    events: list[Event], bucket_minutes: int
) -> tuple[list[int], list[list[Event]], list[int]]:
    if not events:
        return [], [], []

    min_ts = min(e.ts_start for e in events)
    max_ts = max(e.ts_end for e in events)
    bucket_ms = bucket_minutes * 60 * 1000

    n_buckets = max(1, (max_ts - min_ts) // bucket_ms + 1)
    counts = [0] * n_buckets
    bucketed_events: list[list[Event]] = [[] for _ in range(n_buckets)]
    bucket_starts = [min_ts + i * bucket_ms for i in range(n_buckets)]

    for e in events:
        idx = min((e.ts_start - min_ts) // bucket_ms, n_buckets - 1)
        counts[idx] += 1
        bucketed_events[idx].append(e)

    return counts, bucketed_events, bucket_starts


def _segment_label(
    mean_rate: float,
    dominant_category: str,
    pause_dist: dict[str, float] | None,
) -> str:
    # Rate description
    if mean_rate >= 5:
        rate_desc = "high-rate"
    elif mean_rate >= 2:
        rate_desc = "moderate-rate"
    else:
        rate_desc = "low-rate"

    # Pause description
    pause_desc = ""
    if pause_dist:
        dominant_pause = max(pause_dist, key=pause_dist.get)
        if pause_dist.get("stuck", 0) > 0.3:
            pause_desc = ", stuck-heavy"
        elif pause_dist.get("evaluating", 0) > 0.4:
            pause_desc = ", evaluating-heavy"

    return f"{rate_desc}{pause_desc}, {dominant_category}-focused"


def detect_changepoints(
    events: list[Event],
    config: Config,
    pause_findings: PauseFindings | None = None,
) -> ChangepointFindings:
    if not events:
        return ChangepointFindings(segments=[], changepoints=[])

    counts, bucketed_events, bucket_starts = _bucket_events(events, config.bucket_minutes)
    n_buckets = len(counts)
    bucket_ms = config.bucket_minutes * 60 * 1000

    # Need at least min_size * 2 buckets for a single changepoint
    if n_buckets < config.pelt_min_size * 2:
        # Return single segment covering everything
        segment = _build_segment(
            events, bucket_starts[0],
            bucket_starts[-1] + bucket_ms,
            np.mean(counts), config, pause_findings,
        )
        return ChangepointFindings(segments=[segment], changepoints=[])

    # Run changepoint detection
    signal = np.array(counts, dtype=float).reshape(-1, 1)
    # Use median absolute deviation for robust scale estimate,
    # then apply BIC-like penalty: 2 * log(n) * scale^2
    median_val = np.median(signal)
    mad = max(np.median(np.abs(signal - median_val)), 1.0)
    penalty = 2 * np.log(n_buckets) * mad ** 2

    algo = ruptures.Pelt(model=config.pelt_model, min_size=config.pelt_min_size, jump=1)
    algo.fit(signal)
    change_indices = algo.predict(pen=penalty)

    # Remove sentinel (last element is always n_buckets)
    if change_indices and change_indices[-1] == n_buckets:
        change_indices = change_indices[:-1]

    # Build segments
    boundaries = [0] + change_indices + [n_buckets]
    segments = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]

        seg_events = []
        for j in range(start_idx, end_idx):
            seg_events.extend(bucketed_events[j])

        seg_counts = counts[start_idx:end_idx]
        start_ts = bucket_starts[start_idx]
        end_ts = bucket_starts[min(end_idx, n_buckets) - 1] + bucket_ms

        segment = _build_segment(
            seg_events, start_ts, end_ts,
            float(np.mean(seg_counts)), config, pause_findings,
        )
        segments.append(segment)

    # Build changepoint descriptions
    cps = []
    for i, cp_idx in enumerate(change_indices):
        cp_ts = bucket_starts[cp_idx]
        from_seg = segments[i].label if i < len(segments) else ""
        to_seg = segments[i + 1].label if i + 1 < len(segments) else ""
        cps.append(Changepoint(ts=cp_ts, from_segment_summary=from_seg, to_segment_summary=to_seg))

    return ChangepointFindings(segments=segments, changepoints=cps)


def _build_segment(
    events: list[Event],
    start_ts: int,
    end_ts: int,
    mean_rate: float,
    config: Config,
    pause_findings: PauseFindings | None,
) -> Segment:
    duration_min = (end_ts - start_ts) / 60_000

    # Dominant category
    if events:
        categories = [_categorize_command(e.command) for e in events]
        cat_counts = Counter(categories)
        dominant_category = cat_counts.most_common(1)[0][0]
    else:
        dominant_category = "other"

    # Pause distribution (optional) — filter to pauses within this segment's time range
    pause_dist = None
    if pause_findings and pause_findings.available and pause_findings.classifications:
        relevant_pauses = [
            c for c in pause_findings.classifications
            if start_ts <= c.ts_start <= end_ts
        ]
        if relevant_pauses:
            label_counts = Counter(c.label for c in relevant_pauses)
            total = sum(label_counts.values())
            pause_dist = {label: count / total for label, count in label_counts.items()}

    label = _segment_label(mean_rate, dominant_category, pause_dist)

    return Segment(
        start_ts=start_ts,
        end_ts=end_ts,
        duration_min=duration_min,
        mean_rate=mean_rate,
        dominant_category=dominant_category,
        pause_distribution=pause_dist,
        label=label,
    )
