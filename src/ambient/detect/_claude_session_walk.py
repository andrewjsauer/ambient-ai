"""Shared helper for detectors that need per-prompt access to Claude Code session JSONL.

Why this exists separately from `daemon/session_parser.py`:

`session_parser.parse_session_file` is shaped for session-level summarization —
it returns a single dict per session with aggregated `prompts`, `tools`, etc.
Crucially, its `_extract_user_message` filters out any prompt body starting with
`<` (which would otherwise leak tool-result echoes into the prompts list). That
filter also drops `<command-name>/foo</command-name>` slash-command invocations,
which the v4 Phase 1 detectors specifically need.

This helper does the orthogonal job: walk every prompt across every session,
yield per-prompt records *with timestamps* (for window filtering), preserve the
slash-command marker, and skip tool-output echoes by exact prefix.

Read-only contract: never modifies the JSONL files or the projects directory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from ambient.detect.slash_taxonomy import extract_slash_command

logger = logging.getLogger(__name__)

# Tool-output echoes that arrive as user-message content; these are not real prompts.
_TOOL_OUTPUT_PREFIXES = (
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<bash-stdout>",
    "<bash-stderr>",
    "<system-reminder>",
    "<tool_use_error>",
)

# Subdirectory of ~/.claude/projects/ that holds subagent invocations rather than
# user-driven sessions. Always excluded from detector aggregations.
_SUBAGENTS_SLUG = "subagents"


@dataclass(frozen=True)
class PromptRecord:
    """A single user prompt extracted from a Claude Code session JSONL."""

    ts: datetime
    project: str        # JSONL parent directory name (slug); used for per-project grouping
    session_id: str
    text: str           # Full prompt text (may include slash-command markers)
    slash_command: str | None  # e.g. "/ship", or None for freeform prompts


def walk_prompts(
    projects_dir: Path,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> Iterator[PromptRecord]:
    """Yield user prompts from every session JSONL under `projects_dir`.

    Excludes the `subagents/` directory. Skips tool-output echoes. Filters by
    window when bounds are provided (inclusive of both ends). Malformed JSON
    lines are logged and skipped; partial files yield whatever they can.

    Timestamps are timezone-aware (UTC for `Z`-suffixed inputs); window bounds
    should be timezone-aware to match. Naive bounds are treated as UTC.
    """
    if not projects_dir.is_dir():
        return

    start_aware = _ensure_tz(window_start)
    end_aware = _ensure_tz(window_end)

    for slug_dir in projects_dir.iterdir():
        if not slug_dir.is_dir():
            continue
        if slug_dir.name == _SUBAGENTS_SLUG:
            continue

        for jsonl_path in slug_dir.glob("*.jsonl"):
            yield from _walk_one_file(jsonl_path, slug_dir.name, start_aware, end_aware)


def _walk_one_file(
    path: Path,
    project: str,
    start: datetime | None,
    end: datetime | None,
) -> Iterator[PromptRecord]:
    session_id: str | None = None
    try:
        with open(path, encoding="utf-8") as fh:
            for line_num, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed JSON at line %d in %s", line_num, path)
                    continue

                if session_id is None and obj.get("sessionId"):
                    session_id = obj["sessionId"]

                if obj.get("type") != "user" or obj.get("isMeta"):
                    continue

                ts = _parse_ts(obj.get("timestamp"))
                if ts is None:
                    continue
                if start is not None and ts < start:
                    continue
                if end is not None and ts > end:
                    continue

                for text in _iter_user_text_blocks(obj):
                    if _is_tool_output_echo(text):
                        continue
                    cleaned = _strip_appended_tag_blocks(text)
                    if not cleaned:
                        # Body was entirely tag-block content — skip.
                        continue
                    yield PromptRecord(
                        ts=ts,
                        project=project,
                        session_id=session_id or path.stem,
                        text=cleaned,
                        slash_command=extract_slash_command(cleaned),
                    )
    except OSError as e:
        logger.warning("Cannot read session file %s: %s", path, e)


def _iter_user_text_blocks(obj: dict) -> Iterator[str]:
    """Yield each text body inside a user message (string or content-block list)."""
    msg = obj.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        text = content.strip()
        if text:
            yield text
        return
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = (block.get("text") or "").strip()
        if text:
            yield text


def _is_tool_output_echo(text: str) -> bool:
    """Return True if `text` looks like a tool-output echo wrapped in user content.

    Catches both whole-message echoes (`<bash-stdout>...`) and prompts whose
    body is *entirely* a system-reminder block. Real prompts with an appended
    reminder (e.g. "fix the bug\\n<system-reminder>...</system-reminder>") are
    NOT echoes — they get stripped of the trailing reminder via _strip_tags.
    """
    return text.startswith(_TOOL_OUTPUT_PREFIXES)


_TAG_BLOCK_RE = None  # lazy-compiled below


def _strip_appended_tag_blocks(text: str) -> str:
    """Strip trailing <system-reminder>...</system-reminder> and similar blocks.

    Claude Code injects system reminders into user-message bodies; if the user
    typed a real prompt and a reminder was appended, the prompt should be
    counted as freeform but the reminder should not pollute the text fed to
    Haiku. Strip from the right; preserve everything before any trailing block.
    """
    import re
    global _TAG_BLOCK_RE
    if _TAG_BLOCK_RE is None:
        # Match a trailing block: <tag>...</tag> at the end of the string,
        # possibly preceded by whitespace. Tag names limited to known noise.
        _TAG_BLOCK_RE = re.compile(
            r"\s*<(system-reminder|local-command-stdout|local-command-stderr|bash-stdout|bash-stderr|tool_use_error)>"
            r".*?</\1>\s*$",
            re.DOTALL,
        )
    stripped = text
    # Repeat until no trailing tag block remains (multiple appended blocks possible).
    while True:
        new = _TAG_BLOCK_RE.sub("", stripped).rstrip()
        if new == stripped:
            return stripped
        stripped = new


def _parse_ts(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _ensure_tz(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        from datetime import timezone
        return dt.replace(tzinfo=timezone.utc)
    return dt
