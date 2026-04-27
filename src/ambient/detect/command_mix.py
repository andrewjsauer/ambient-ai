"""Command-mix detector.

For each project (and overall) within the insights window, count user prompts
by intent category — planning, execution, review, design, meta, freeform — and
compute ratios. Sourced from `~/.claude/projects/*.jsonl` via the shared
`_claude_session_walk` helper. Excludes the `subagents/` directory.

Why this exists: the v3 weekly report grounded findings in shipped work (git)
but had no view of *intent*. "Was this week mostly planning, mostly shipping,
or mostly review?" was unanswerable from the existing pipeline. This detector
makes that question one detector call.

Read-only contract: walks the JSONL files only; never modifies them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ambient.config import Config
from ambient.detect._claude_session_walk import PromptRecord, walk_prompts
from ambient.detect.slash_taxonomy import classify_slash_command

logger = logging.getLogger(__name__)


@dataclass
class ProjectMix:
    """Per-project (or overall) intent counts plus normalized ratios."""

    planning_count: int = 0
    execution_count: int = 0
    review_count: int = 0
    design_count: int = 0
    meta_count: int = 0
    other_count: int = 0       # unknown slash commands; tracked separately from freeform
    freeform_count: int = 0    # prompts with no slash command

    @property
    def total(self) -> int:
        return (
            self.planning_count
            + self.execution_count
            + self.review_count
            + self.design_count
            + self.meta_count
            + self.other_count
            + self.freeform_count
        )

    @property
    def ratios(self) -> dict[str, float]:
        """Per-category share of `total`. Empty mix returns all zeros."""
        n = self.total
        if n == 0:
            return {
                "planning": 0.0,
                "execution": 0.0,
                "review": 0.0,
                "design": 0.0,
                "meta": 0.0,
                "other": 0.0,
                "freeform": 0.0,
            }
        return {
            "planning": self.planning_count / n,
            "execution": self.execution_count / n,
            "review": self.review_count / n,
            "design": self.design_count / n,
            "meta": self.meta_count / n,
            "other": self.other_count / n,
            "freeform": self.freeform_count / n,
        }

    def add(self, category: str) -> None:
        attr = f"{category}_count"
        if not hasattr(self, attr):
            attr = "other_count"
        setattr(self, attr, getattr(self, attr) + 1)


@dataclass
class CommandMixFindings:
    """Result of `detect_command_mix`.

    `per_project` only contains projects whose total prompts ≥ the floor;
    quieter projects are still rolled into `overall`.
    """

    per_project: dict[str, ProjectMix] = field(default_factory=dict)
    overall: ProjectMix = field(default_factory=ProjectMix)
    window_start_iso: str = ""
    window_end_iso: str = ""


def detect_command_mix(
    claude_projects_dir: Path,
    window_start: datetime,
    window_end: datetime,
    config: Config,
    *,
    prompts: list[PromptRecord] | None = None,
) -> CommandMixFindings:
    """Compute per-project + overall command-mix counts within a window.

    When `prompts` is provided, skip the JSONL walk and use the pre-materialized
    list. The orchestrator (insights._aggregate_window) walks once per window
    and shares the list across all Phase 1 detectors to avoid redundant I/O.
    """
    floor = config.command_mix_min_prompts
    overrides = config.slash_taxonomy_overrides or None

    if prompts is None:
        prompts = list(walk_prompts(claude_projects_dir, window_start, window_end))

    raw: dict[str, ProjectMix] = {}
    overall = ProjectMix()

    for record in prompts:
        if record.slash_command is None:
            category = "freeform"
        else:
            category = classify_slash_command(record.slash_command, overrides=overrides)
        mix = raw.setdefault(record.project, ProjectMix())
        mix.add(category)
        overall.add(category)

    per_project = {p: mix for p, mix in raw.items() if mix.total >= floor}

    return CommandMixFindings(
        per_project=per_project,
        overall=overall,
        window_start_iso=window_start.isoformat(),
        window_end_iso=window_end.isoformat(),
    )
