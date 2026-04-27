import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DaemonState:
    last_analyzed_ts: int = 0  # epoch ms, exclusive lower bound
    last_summary_date: str = ""  # YYYY-MM-DD, last date a summary was generated
    last_calibration_date: str = ""  # YYYY-MM-DD
    events_since_calibration: int = 0
    last_notification_ts: int = 0  # epoch ms, last stuck notification sent
    last_weekly_summary_date: str = ""  # YYYY-MM-DD, last weekly summary generated
    last_claude_history_line: int = 0  # line number cursor for ~/.claude/history.jsonl (legacy)
    # Maps project slug -> {session_uuid: ingested_at_epoch_ms}
    processed_sessions: dict[str, dict[str, int]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "DaemonState":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            # Handle legacy format where processed_sessions was list-based
            raw_sessions = data.get("processed_sessions", {})
            if isinstance(raw_sessions, dict):
                # Normalize: if values are lists (legacy), convert to dict with current timestamp
                processed = {}
                for slug, sessions in raw_sessions.items():
                    if isinstance(sessions, list):
                        now_ms = int(time.time() * 1000)
                        processed[slug] = {s: now_ms for s in sessions}
                    elif isinstance(sessions, dict):
                        processed[slug] = sessions
                    else:
                        processed[slug] = {}
            else:
                processed = {}
            return cls(
                last_analyzed_ts=data.get("last_analyzed_ts", 0),
                last_summary_date=data.get("last_summary_date", ""),
                last_calibration_date=data.get("last_calibration_date", ""),
                events_since_calibration=data.get("events_since_calibration", 0),
                last_notification_ts=data.get("last_notification_ts", 0),
                last_weekly_summary_date=data.get("last_weekly_summary_date", ""),
                last_claude_history_line=data.get("last_claude_history_line", 0),
                processed_sessions=processed,
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return cls()

    def save(self, path: Path) -> None:
        # Clean up processed sessions older than 30 days
        self._cleanup_old_sessions()

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(asdict(self), f, indent=2)
                f.write("\n")
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _cleanup_old_sessions(self) -> None:
        """Remove processed session entries older than 30 days."""
        cutoff_ms = int(time.time() * 1000) - (30 * 24 * 60 * 60 * 1000)
        for slug in list(self.processed_sessions):
            sessions = self.processed_sessions[slug]
            cleaned = {}
            for uuid, entry in sessions.items():
                if isinstance(entry, dict):
                    if entry.get("ingested_at", 0) > cutoff_ms:
                        cleaned[uuid] = entry
                elif isinstance(entry, (int, float)):
                    # Legacy format (bare int = ingested_at_ms)
                    if entry > cutoff_ms:
                        cleaned[uuid] = entry
            self.processed_sessions[slug] = cleaned
            if not self.processed_sessions[slug]:
                del self.processed_sessions[slug]

    def is_session_processed(self, slug: str, session_uuid: str) -> bool:
        return session_uuid in self.processed_sessions.get(slug, {})

    def get_session_line_count(self, slug: str, session_uuid: str) -> int:
        """Return the line count last seen for this session, or 0 if never processed."""
        entry = self.processed_sessions.get(slug, {}).get(session_uuid)
        if entry is None:
            return 0
        if isinstance(entry, dict):
            return entry.get("line_count", 0)
        # Legacy format (bare int = ingested_at_ms): treat as fully processed with unknown lines
        return -1

    def mark_session_processed(self, slug: str, session_uuid: str, line_count: int = 0) -> None:
        if slug not in self.processed_sessions:
            self.processed_sessions[slug] = {}
        self.processed_sessions[slug][session_uuid] = {
            "ingested_at": int(time.time() * 1000),
            "line_count": line_count,
        }
