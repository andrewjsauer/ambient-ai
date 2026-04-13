"""Parse Claude Code per-session JSONL files for deep conversation capture."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# File path pattern for extraction from tool_use inputs
_FILE_PATH_RE = re.compile(r'(?:^|["\s])(/[\w./-]+\.[\w]+)')


def parse_session_file(path: Path, skip_lines: int = 0) -> dict | None:
    """Parse a single session JSONL file, extracting prompts, tool calls, and outcomes.

    Args:
        path: Path to the session JSONL file.
        skip_lines: Number of lines to skip from the start (for incremental parsing).
            When > 0, only extracts data from lines after the skip point.
            Session ID and project are still read from early lines for context.

    Returns a dict with session metadata and extracted data, or None if the file
    cannot be parsed at all. Includes 'total_lines' for tracking incremental state.
    """
    prompts: list[str] = []
    tools: list[dict] = []
    files_touched: set[str] = set()
    is_error_count = 0
    session_id = None
    project = None
    max_ts: datetime | None = None
    min_ts: datetime | None = None
    total_lines = 0

    try:
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                total_lines = line_num
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON at line %d in %s", line_num, path)
                    continue

                entry_type = d.get("type")
                if not entry_type:
                    continue

                # Always track session ID and project (even in skipped lines)
                if not session_id and d.get("sessionId"):
                    session_id = d["sessionId"]
                if not project and d.get("cwd"):
                    project = d["cwd"]

                # Track timestamps for session duration
                ts_str = d.get("timestamp")
                if ts_str and isinstance(ts_str, str):
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if min_ts is None or ts < min_ts:
                            min_ts = ts
                        if max_ts is None or ts > max_ts:
                            max_ts = ts
                    except (ValueError, TypeError):
                        pass

                # Skip already-processed lines for content extraction
                if line_num <= skip_lines:
                    continue

                if entry_type == "user" and not d.get("isMeta"):
                    _extract_user_message(d, prompts)
                    is_error_count += _count_errors_in_user(d)

                elif entry_type == "assistant":
                    _extract_assistant_tools(d, tools, files_touched)

    except OSError as e:
        logger.warning("Cannot read session file %s: %s", path, e)
        return None

    if min_ts is None:
        return None

    start_ms = int(min_ts.timestamp() * 1000)
    end_ms = int(max_ts.timestamp() * 1000) if max_ts else start_ms
    duration_ms = end_ms - start_ms

    return {
        "session_id": session_id or path.stem,
        "project": project or "",
        "prompts": prompts,
        "tools": tools,
        "files_touched": sorted(files_touched),
        "is_error_count": is_error_count,
        "prompt_count": len(prompts),
        "start_ts": start_ms,
        "end_ts": end_ms,
        "duration_ms": duration_ms,
        "total_lines": total_lines,
    }


def _count_errors_in_user(d: dict) -> int:
    """Count is_error=True tool_results in a user message."""
    content = d.get("message", {}).get("content", "")
    if isinstance(content, str):
        return 0
    count = 0
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            if block.get("is_error") is True:
                count += 1
    return count


def _extract_user_message(d: dict, prompts: list[str]):
    """Extract prompt text from a user message."""
    content = d.get("message", {}).get("content", "")
    if isinstance(content, str):
        text = content.strip()
        if text and not text.startswith("<"):
            prompts.append(text)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text and not text.startswith("<"):
                        prompts.append(text)


def _extract_assistant_tools(d: dict, tools: list[dict], files_touched: set[str]):
    """Extract tool_use blocks from an assistant message."""
    content = d.get("message", {}).get("content", [])
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name = block.get("name", "")
        inp = block.get("input", {})

        # Extract file paths from common tool inputs
        tool_files = []
        for key in ("file_path", "path", "command"):
            val = inp.get(key, "")
            if isinstance(val, str):
                matches = _FILE_PATH_RE.findall(val)
                tool_files.extend(matches)

        tools.append({"name": name, "files": tool_files})
        files_touched.update(tool_files)


def discover_session_files(projects_dir: Path) -> list[Path]:
    """Find all per-session JSONL files under the Claude projects directory.

    Uses non-recursive glob per project directory to exclude subagents/,
    memory/, and other subdirectories.
    """
    session_files = []
    if not projects_dir.is_dir():
        return session_files

    for slug_dir in projects_dir.iterdir():
        if not slug_dir.is_dir():
            continue
        for jsonl_file in slug_dir.glob("*.jsonl"):
            session_files.append(jsonl_file)

    return session_files
