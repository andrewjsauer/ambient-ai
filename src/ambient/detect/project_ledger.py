"""Project ledger detector — what you worked on, per project, per window.

For each project that crosses an active-time floor, builds a ledger entry with:
  - active time (gap-based today via projects.detect_project_allocation;
    Phase 2 Unit 9 swaps this to attention-weighted)
  - session count
  - top files touched
  - representative recent prompts (capped, truncated)
  - LLM-generated one-line summary of what the user worked on (Haiku)

The summary call is per-project and prompt-cached; cost is ~$0.05/week at the
inventory's volumes. When the API call fails, the entry still renders with
`summary=None` — never blocks the rest of the report.

Read-only contract: walks events + ~/.claude/projects/*.jsonl only. Never
modifies any files. Never sends events outside the existing API boundary.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect._claude_session_walk import walk_prompts
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
    """
    min_active_ms = max(0, getattr(config, "project_ledger_min_active_ms", 600_000))
    top_files_n = max(0, getattr(config, "project_ledger_top_files_n", 5))
    max_prompts = max(0, getattr(config, "project_ledger_summary_max_prompts", 30))
    truncate_chars = max(0, getattr(config, "project_ledger_summary_truncate_chars", 500))

    # Time + session counts: reuse projects.detect_project_allocation.
    allocations = detect_project_allocation(events, config).allocations

    # Per-project file frequency from event metadata.
    files_per_project = _aggregate_files(events, top_files_n)

    # Per-project prompts (most-recent first, capped, truncated) keyed by the
    # JSONL slug — which differs from projects.detect_project_allocation's
    # cwd-basename keys. We resolve the slug→display-name mismatch by matching
    # tail-of-slug against the allocation project name.
    prompts_by_slug = _aggregate_prompts(
        claude_projects_dir, window_start, window_end, max_prompts, truncate_chars
    )

    entries: list[ProjectLedgerEntry] = []
    for alloc in allocations:
        if alloc.total_ms < min_active_ms:
            continue
        slug_match = _match_slug(alloc.project, prompts_by_slug.keys())
        prompts = prompts_by_slug.get(slug_match, []) if slug_match else []
        entries.append(
            ProjectLedgerEntry(
                project=alloc.project,
                active_ms=alloc.total_ms,
                session_count=alloc.session_count,
                top_files=files_per_project.get(alloc.project, []),
                representative_prompts=prompts,
            )
        )

    if not skip_summaries and _api_available():
        for entry in entries:
            if not entry.representative_prompts:
                continue
            entry.summary = _summarize(entry, config, api_client)

    return ProjectLedger(
        entries=entries,
        window_start_iso=window_start.isoformat(),
        window_end_iso=window_end.isoformat(),
        time_basis="command_span",
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
) -> dict[str, list[str]]:
    """Per-slug prompt list, most-recent first, capped + truncated."""
    by_slug: dict[str, list[tuple[datetime, str]]] = {}
    for record in walk_prompts(claude_projects_dir, window_start, window_end):
        text = record.text
        if truncate_chars and len(text) > truncate_chars:
            text = text[:truncate_chars]
        by_slug.setdefault(record.project, []).append((record.ts, text))

    out: dict[str, list[str]] = {}
    for slug, items in by_slug.items():
        items.sort(key=lambda x: x[0], reverse=True)
        out[slug] = [t for _, t in items[:max_prompts]] if max_prompts else []
    return out


def _match_slug(project_name: str, slugs) -> str | None:
    """Match an allocation project (basename of cwd) to a JSONL slug.

    Slugs in ~/.claude/projects/ are dash-encoded full paths
    (e.g. -Users-you-projects-my-app). The trailing
    segment matches the cwd basename. We pick the slug whose tail-segment
    equals the project name; if multiple match, prefer the longest slug
    (deepest path is most specific).
    """
    if not project_name:
        return None
    candidates = []
    for slug in slugs:
        tail = slug.rsplit("-", 1)[-1] if slug else ""
        if tail == project_name:
            candidates.append(slug)
    if not candidates:
        return None
    return max(candidates, key=len)


def _api_available() -> bool:
    """Return True if the Anthropic API is callable in this environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


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
