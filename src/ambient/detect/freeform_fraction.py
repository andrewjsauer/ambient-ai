"""Freeform-fraction detector.

Reports the percentage of user prompts that bypassed any structured slash
command, plus a per-project breakdown and a delta vs the prior window.

Why this exists: the inventory revealed that 78.5% of user prompts in
~/.claude/projects are freeform (no slash command). That number is signal,
not prescription — a high freeform fraction means most work bypasses
structured workflows, which is a fact about how a developer actually works,
not a problem to fix. Trends over time tell a more interesting story than
the absolute number.

Read-only contract: walks the JSONL files only; never modifies them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ambient.config import Config
from ambient.detect._claude_session_walk import walk_prompts

logger = logging.getLogger(__name__)


@dataclass
class FreeformFraction:
    """Result of `detect_freeform_fraction`.

    `overall_pct` is in 0.0–1.0 (multiply by 100 for display).
    `delta_pct` is `overall_pct - prior_window_pct` when both windows have data.
    `per_project` is the same shape as `overall_pct`, restricted to projects
    above `freeform_fraction_min_prompts`.
    `total_prompts` and `prior_total_prompts` are sample-size denominators
    so callers can hedge low-sample reports.
    """

    overall_pct: float = 0.0
    prior_window_pct: float | None = None
    delta_pct: float | None = None
    per_project: dict[str, float] = field(default_factory=dict)
    total_prompts: int = 0
    prior_total_prompts: int = 0
    window_start_iso: str = ""
    window_end_iso: str = ""


def detect_freeform_fraction(
    claude_projects_dir: Path,
    window_start: datetime,
    window_end: datetime,
    config: Config,
    prior_window_start: datetime | None = None,
    prior_window_end: datetime | None = None,
) -> FreeformFraction:
    """Compute freeform fraction for the window, plus prior-window delta if provided."""
    floor = max(0, getattr(config, "freeform_fraction_min_prompts", 20))

    current_total, current_freeform, per_project_counts = _walk_window(
        claude_projects_dir, window_start, window_end
    )
    overall_pct = (current_freeform / current_total) if current_total else 0.0

    prior_pct: float | None = None
    prior_total = 0
    if prior_window_start is not None and prior_window_end is not None:
        prior_total, prior_freeform, _ = _walk_window(
            claude_projects_dir, prior_window_start, prior_window_end
        )
        if prior_total:
            prior_pct = prior_freeform / prior_total

    delta = (overall_pct - prior_pct) if prior_pct is not None else None

    per_project = {
        project: counts.freeform / counts.total
        for project, counts in per_project_counts.items()
        if counts.total >= floor
    }

    return FreeformFraction(
        overall_pct=overall_pct,
        prior_window_pct=prior_pct,
        delta_pct=delta,
        per_project=per_project,
        total_prompts=current_total,
        prior_total_prompts=prior_total,
        window_start_iso=window_start.isoformat(),
        window_end_iso=window_end.isoformat(),
    )


@dataclass
class _Counts:
    total: int = 0
    freeform: int = 0


def _walk_window(
    claude_projects_dir: Path,
    start: datetime,
    end: datetime,
) -> tuple[int, int, dict[str, _Counts]]:
    """Return (total_prompts, freeform_prompts, per_project_counts) for a window."""
    total = 0
    freeform = 0
    per_project: dict[str, _Counts] = {}

    try:
        for record in walk_prompts(claude_projects_dir, start, end):
            counts = per_project.setdefault(record.project, _Counts())
            counts.total += 1
            total += 1
            if record.slash_command is None:
                counts.freeform += 1
                freeform += 1
    except Exception:  # pragma: no cover — defensive only
        logger.warning("freeform_fraction walk failed; returning partial", exc_info=True)

    return total, freeform, per_project
