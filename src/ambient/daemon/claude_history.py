import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SESSION_COMPLETE_THRESHOLD_MS = 30 * 60 * 1000  # 30 minutes


def read_new_history_entries(path: Path, start_line: int) -> tuple[list[dict], int]:
    if not path.exists():
        return [], start_line

    entries = []
    line_count = 0
    try:
        with open(path) as f:
            for i, line in enumerate(f):
                line_count = i + 1
                if i < start_line:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("sessionId"):
                        entries.append(entry)
                except (json.JSONDecodeError, KeyError):
                    logger.debug("Skipping malformed history line %d", i + 1)
    except OSError as e:
        logger.warning("Failed to read history file: %s", e)
        return [], start_line

    return entries, line_count


def group_into_sessions(entries: list[dict]) -> list[dict]:
    sessions: dict[str, dict] = {}
    for entry in entries:
        sid = entry["sessionId"]
        display = entry.get("display", "")
        ts = entry.get("timestamp", 0)
        project = entry.get("project", "")

        if sid not in sessions:
            sessions[sid] = {
                "session_id": sid,
                "project": project,
                "prompts": [],
                "ts_start": ts,
                "ts_end": ts,
                "prompt_count": 0,
            }

        session = sessions[sid]
        session["prompts"].append(display[:100] if display else "")
        session["prompt_count"] += 1
        if ts < session["ts_start"]:
            session["ts_start"] = ts
        if ts > session["ts_end"]:
            session["ts_end"] = ts

    return list(sessions.values())


def filter_completed_sessions(
    sessions: list[dict], now_ms: int | None = None
) -> tuple[list[dict], list[dict]]:
    if now_ms is None:
        now_ms = int(datetime.now().timestamp() * 1000)

    cutoff = now_ms - SESSION_COMPLETE_THRESHOLD_MS
    completed = []
    in_progress = []

    for session in sessions:
        if session["ts_end"] < cutoff:
            completed.append(session)
        else:
            in_progress.append(session)

    return completed, in_progress


def session_to_event(session: dict) -> dict:
    duration_ms = session["ts_end"] - session["ts_start"]
    first_prompt = session["prompts"][0] if session["prompts"] else ""
    command_preview = f"claude: {first_prompt}" if first_prompt else "claude: (session)"

    return {
        "type": "claude_session",
        "ts_start": session["ts_start"],
        "ts_end": session["ts_end"],
        "duration_ms": max(duration_ms, 0),
        "command": command_preview,
        "exit_code": 0,
        "cwd": session["project"],
        "tmux_pane": None,
        "gap_ms": None,
        "session_boundary": False,
        "claude_session_id": session["session_id"],
        "claude_prompt_count": session["prompt_count"],
        "claude_project": session["project"],
        "claude_prompts": session["prompts"],
    }
