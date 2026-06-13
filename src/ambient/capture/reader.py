import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

from ambient.config import Config

logger = logging.getLogger(__name__)


@dataclass
class Event:
    ts_start: int
    ts_end: int
    duration_ms: int
    command: str
    exit_code: int
    cwd: str
    tmux_pane: str | None
    gap_ms: int | None
    session_boundary: bool = False
    type: str = "command"
    # Claude session fields (optional, None for shell command events)
    claude_session_id: str | None = None
    claude_prompts: list[str] | None = None
    claude_tools: list[dict] | None = None
    claude_files: list[str] | None = None
    claude_project: str | None = None
    claude_prompt_count: int | None = None
    claude_is_error_count: int | None = None
    claude_ran_verification: bool = False
    claude_verification_resolved: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            ts_start=d["ts_start"],
            ts_end=d["ts_end"],
            duration_ms=d["duration_ms"],
            command=d["command"],
            exit_code=d["exit_code"],
            cwd=d["cwd"],
            tmux_pane=d.get("tmux_pane"),
            gap_ms=d.get("gap_ms"),
            session_boundary=d.get("session_boundary", False),
            type=d.get("type", "command"),
            claude_session_id=d.get("claude_session_id"),
            claude_prompts=d.get("claude_prompts"),
            claude_tools=d.get("claude_tools"),
            claude_files=d.get("claude_files"),
            claude_project=d.get("claude_project"),
            claude_prompt_count=d.get("claude_prompt_count"),
            claude_is_error_count=d.get("claude_is_error_count"),
            claude_ran_verification=d.get("claude_ran_verification", False),
            claude_verification_resolved=d.get("claude_verification_resolved", False),
        )


def _read_file(path: Path) -> Iterator[Event]:
    try:
        f = open(path)
    except FileNotFoundError:
        return
    with f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                yield Event.from_dict(d)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning("Skipping malformed line %d in %s: %s", line_num, path, e)


def date_range(start: datetime, end: datetime) -> list[str]:
    """YYYY-MM-DD strings for every calendar day from start to end inclusive."""
    dates = []
    current = start.date()
    end_date = end.date()
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def read_events(
    config: Config,
    start: datetime | None = None,
    end: datetime | None = None,
    date_str: str | None = None,
) -> list[Event]:
    if date_str and not start and not end:
        path = config.events_path(date_str)
        return list(_read_file(path))

    if start and end:
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        date_strs = date_range(start, end)
    elif date_str:
        date_strs = [date_str]
        start_ms = None
        end_ms = None
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        return list(_read_file(config.events_path(today)))

    events = []
    for ds in date_strs:
        for event in _read_file(config.events_path(ds)):
            if start_ms is not None and end_ms is not None:
                if event.ts_start < start_ms or event.ts_start > end_ms:
                    continue
            events.append(event)
    return events


def read_events_window(config: Config, window_minutes: int = 30) -> list[Event]:
    now = datetime.now()
    start = now - timedelta(minutes=window_minutes)
    return read_events(config, start=start, end=now)


def read_events_today(config: Config) -> list[Event]:
    return read_events(config, date_str=datetime.now().strftime("%Y-%m-%d"))
