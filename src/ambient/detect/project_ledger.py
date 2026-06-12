"""Project ledger detector — what you worked on, per project, per window.

For each project that crosses an active-time floor, builds a ledger entry with:
  - active time (gap-based today via projects.detect_project_allocation;
    Phase 2 Unit 9 swaps this to attention-weighted)
  - session count
  - top files touched
  - representative recent prompts (capped, truncated)
  - LLM-generated one-line summary of what the user worked on (Haiku)

Cost: per-project Haiku call, sequential, ~$0.10-0.20/week at the inventory's
volumes (10-15 active projects × 1 call each). The system block is below the
2048-token threshold for Haiku prompt caching, so cache_control on the system
block is a no-op today; if cost grows, batch projects or move per-project
context into the cached system block. When the API call fails, the entry still
renders with `summary=None` — never blocks the rest of the report.

Read-only contract: walks events + ~/.claude/projects/*.jsonl only. Never
modifies any files. Never sends events outside the existing API boundary.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect._claude_session_walk import PromptRecord, walk_prompts
from ambient.detect.projects import detect_project_allocation

logger = logging.getLogger(__name__)

TimeBasis = Literal["command_span", "attention_weighted"]

LEDGER_SUMMARY_SYSTEM = (
    "You are summarizing what a developer worked on in a single project over one window. "
    "Output one sentence, no more than 25 words, no preamble, no quotation marks. "
    "Quote the developer's distinctive wording verbatim where natural; paraphrase generic prompts. "
    "Do not invent specifics — file names, function names, technologies — that are not present in the input. "
    "Do not pad. Do not editorialize. If the prompts are too varied to summarize, say so plainly."
)


@dataclass
class ProjectLedgerEntry:
    project: str
    active_ms: int
    session_count: int
    top_files: list[str] = field(default_factory=list)
    representative_prompts: list[str] = field(default_factory=list)
    summary: str | None = None


@dataclass
class ProjectLedger:
    entries: list[ProjectLedgerEntry] = field(default_factory=list)
    window_start_iso: str = ""
    window_end_iso: str = ""
    time_basis: TimeBasis = "command_span"


def detect_project_ledger(
    events: list[Event],
    claude_projects_dir: Path,
    window_start: datetime,
    window_end: datetime,
    config: Config,
    *,
    api_client=None,
    skip_summaries: bool = False,
    prompts: list[PromptRecord] | None = None,
    attention_intervals: list[tuple[datetime, datetime]] | None = None,
) -> ProjectLedger:
    """Build a project ledger for the window.

    Args:
        events: shell + claude_session events used to derive time, sessions, files.
        claude_projects_dir: ~/.claude/projects/ directory for prompt aggregation.
        window_start, window_end: window bounds for prompt filtering.
        config: project_ledger_* fields control floors, caps, and truncation.
        api_client: optional anthropic.Anthropic instance (pre-created, retry enabled).
            When None, the per-project Haiku calls create a fresh client each call.
        skip_summaries: when True, populate entries but skip the Haiku call. Useful
            for tests and for cost-bounded re-runs.
        prompts: when provided, skip the JSONL walk and use this pre-materialized
            list. The orchestrator walks once per window across all Phase 1 detectors.
    """
    min_active_ms = config.project_ledger_min_active_ms
    top_files_n = config.project_ledger_top_files_n
    max_prompts = config.project_ledger_summary_max_prompts
    truncate_chars = config.project_ledger_summary_truncate_chars

    # Time + session counts: reuse projects.detect_project_allocation.
    # When attention_intervals is provided (Phase 2 Unit 9 capture is on),
    # the allocation flips to attention-weighted; ProjectLedger.time_basis
    # below mirrors the underlying ProjectFindings.time_basis.
    proj_findings = detect_project_allocation(
        events, config, attention_intervals=attention_intervals
    )
    allocations = proj_findings.allocations

    # Per-project file frequency from event metadata.
    files_per_project = _aggregate_files(events, top_files_n)

    # Build basename → set[full cwd] from events. Claude Code stores per-session
    # JSONL under a slug derived as `cwd.replace("/", "-")`, so we can encode
    # each cwd directly into the slug Claude Code uses. This avoids the earlier
    # tail-of-slug heuristic that broke any project with a hyphenated basename
    # (e.g. ambient-ai → tail "ai").
    cwds_by_basename = _cwds_by_basename(events)

    prompts_by_slug = _aggregate_prompts(
        claude_projects_dir, window_start, window_end, max_prompts, truncate_chars,
        prompts=prompts,
    )

    entries: list[ProjectLedgerEntry] = []
    for alloc in allocations:
        if alloc.total_ms < min_active_ms:
            continue
        prompts = _prompts_for_project(
            alloc.project, cwds_by_basename, prompts_by_slug, max_prompts
        )
        entries.append(
            ProjectLedgerEntry(
                project=alloc.project,
                active_ms=alloc.total_ms,
                session_count=alloc.session_count,
                top_files=files_per_project.get(alloc.project, []),
                representative_prompts=prompts,
            )
        )

    if not skip_summaries:
        for entry in entries:
            if not entry.representative_prompts:
                continue
            entry.summary = _summarize(entry, config, api_client)

    return ProjectLedger(
        entries=entries,
        window_start_iso=window_start.isoformat(),
        window_end_iso=window_end.isoformat(),
        time_basis=proj_findings.time_basis,
    )


# --- helpers ---

def _aggregate_files(events: list[Event], top_n: int) -> dict[str, list[str]]:
    """Count file-path frequency per project; return top_n basenames per project."""
    if top_n == 0:
        return {}
    counters: dict[str, Counter] = {}
    for event in events:
        project = _derive_project(event)
        if not project:
            continue
        counter = counters.setdefault(project, Counter())
        files = getattr(event, "claude_files", None) or []
        for path in files:
            if not path:
                continue
            base = PurePosixPath(path).name
            if base:
                counter[base] += 1
    return {p: [name for name, _ in c.most_common(top_n)] for p, c in counters.items() if c}


def _derive_project(event: Event) -> str:
    """Mirror of projects._derive_project so files aggregation matches allocation keys."""
    if event.type == "claude_session" and event.claude_project:
        path = PurePosixPath(event.claude_project)
        return path.name or "unknown"
    if event.cwd:
        path = PurePosixPath(event.cwd)
        name = path.name
        if name in ("", "~", "/", "tmp"):
            return "unknown"
        return name
    return "unknown"


def _aggregate_prompts(
    claude_projects_dir: Path,
    window_start: datetime,
    window_end: datetime,
    max_prompts: int,
    truncate_chars: int,
    *,
    prompts: list[PromptRecord] | None = None,
) -> dict[str, list[str]]:
    """Per-slug prompt list, most-recent first, capped + truncated.

    When `prompts` is provided, skip the JSONL walk.
    """
    if prompts is None:
        prompts = list(walk_prompts(claude_projects_dir, window_start, window_end))

    by_slug: dict[str, list[tuple[datetime, str]]] = {}
    for record in prompts:
        text = record.text
        if truncate_chars and len(text) > truncate_chars:
            text = text[:truncate_chars]
        by_slug.setdefault(record.project, []).append((record.ts, text))

    out: dict[str, list[str]] = {}
    for slug, items in by_slug.items():
        items.sort(key=lambda x: x[0], reverse=True)
        out[slug] = [t for _, t in items[:max_prompts]] if max_prompts else []
    return out


def _cwds_by_basename(events: list[Event]) -> dict[str, set[str]]:
    """Build basename → set[full cwd] from events so we can encode the right slug.

    The basename is what `projects._derive_project` returns. Multiple cwds may
    share a basename (e.g. ~/work/foo/api and ~/play/bar/api both have basename
    "api"); the ledger handles that by encoding *every* candidate cwd to its
    Claude Code slug and unioning the prompts.
    """
    by_basename: dict[str, set[str]] = {}
    for event in events:
        cwd = None
        if event.type == "claude_session" and event.claude_project:
            cwd = event.claude_project
        elif event.cwd:
            cwd = event.cwd
        if not cwd:
            continue
        basename = PurePosixPath(cwd).name
        if not basename or basename in ("~", "/", "tmp"):
            continue
        by_basename.setdefault(basename, set()).add(cwd)
    return by_basename


def _cwd_to_slug(cwd: str) -> str:
    """Encode a cwd path into Claude Code's per-project slug format.

    Claude Code stores per-session JSONL under ~/.claude/projects/<slug>/
    where the slug is the absolute cwd with `/` replaced by `-`. So
    `/Users/you/projects/my-app` becomes `-Users-you-projects-my-app`.
    """
    return cwd.replace("/", "-")


def _prompts_for_project(
    basename: str,
    cwds_by_basename: dict[str, set[str]],
    prompts_by_slug: dict[str, list[str]],
    max_prompts: int,
) -> list[str]:
    """Collect prompts for a project basename across all candidate cwds."""
    cwds = cwds_by_basename.get(basename, set())
    if not cwds:
        return []
    collected: list[str] = []
    for cwd in cwds:
        slug = _cwd_to_slug(cwd)
        collected.extend(prompts_by_slug.get(slug, []))
    if max_prompts and len(collected) > max_prompts:
        return collected[:max_prompts]
    return collected


def _summarize(entry: ProjectLedgerEntry, config: Config, client) -> str | None:
    """Single Haiku call per project. Failures are logged and return None."""
    if not entry.representative_prompts:
        return None
    body_lines = [f"Project: {entry.project}"]
    body_lines.append(
        f"Active time: {entry.active_ms // 60_000} minutes across "
        f"{entry.session_count} sessions"
    )
    if entry.top_files:
        body_lines.append("Top files: " + ", ".join(entry.top_files))
    body_lines.append("")
    body_lines.append("Recent user prompts (most-recent first):")
    for i, prompt in enumerate(entry.representative_prompts, 1):
        body_lines.append(f"{i}. {prompt}")

    prompt_text = "\n".join(body_lines)

    try:
        from ambient.present.api import call_api

        response = call_api(
            config,
            LEDGER_SUMMARY_SYSTEM,
            prompt_text,
            config.haiku_model,
            max_tokens=120,
            client=client,
        )
        return response.strip().strip('"').strip()
    except Exception as e:
        logger.warning("project_ledger summary failed for %s: %s", entry.project, e)
        return None
