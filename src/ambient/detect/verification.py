"""Verification-gap detector: find Edit/Write tool calls in Claude sessions
that were not followed by a verifying test command in the same cwd within
a configurable window.

This surfaces the DORA "Rework Rate"-style signal at an individual level:
fixes that shipped without re-running the tests that would have proven them.
Unique to ambient-ai because it requires both the Claude session stream and
the shell event stream for the same developer.
"""

from dataclasses import dataclass, field

from ambient.capture.reader import Event
from ambient.config import Config


@dataclass
class VerificationGap:
    session_id: str
    project: str
    session_end_ts: int
    session_cwd: str
    edited_files: list[str]


@dataclass
class VerificationGapFindings:
    gaps: list[VerificationGap] = field(default_factory=list)
    total_fix_sessions: int = 0
    gap_rate: float | None = None  # gaps / total_fix_sessions; None under low sample
    low_sample: bool = False


_FIX_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})


def _is_fix_session(event: Event) -> bool:
    """A fix session is a claude_session that modified code via Edit/Write tools."""
    if event.type != "claude_session":
        return False
    for tool in event.claude_tools or []:
        if tool.get("name") in _FIX_TOOLS:
            return True
    return False


def _edited_files(event: Event) -> list[str]:
    seen: list[str] = []
    for tool in event.claude_tools or []:
        if tool.get("name") in _FIX_TOOLS:
            for f in tool.get("files") or []:
                if f and f not in seen:
                    seen.append(f)
    return seen


def _matches_test_pattern(command: str, patterns: list[str]) -> bool:
    cmd = command.strip().lower()
    for pat in patterns:
        if cmd.startswith(pat.lower()):
            return True
    return False


def detect_verification_gaps(
    events: list[Event], config: Config
) -> VerificationGapFindings:
    """Find fix sessions with no subsequent verifying test run.

    A fix session is verified when a command matching one of
    `verification_test_command_patterns` fires in the same cwd within
    `verification_gap_window_ms` after the session ends.
    """
    if not events:
        return VerificationGapFindings()

    window_ms = config.verification_gap_window_ms
    patterns = config.verification_test_command_patterns

    gaps: list[VerificationGap] = []
    total_fix_sessions = 0

    # Pre-index command events for quick per-session scanning. Sorting is
    # cheap; correlator.py follows the same pattern.
    command_events = [e for e in events if e.type == "command"]

    for event in events:
        if not _is_fix_session(event):
            continue
        total_fix_sessions += 1

        verified = False
        for cmd in command_events:
            gap = cmd.ts_start - event.ts_end
            if gap < 0 or gap > window_ms:
                continue
            if cmd.cwd != event.cwd:
                continue
            if _matches_test_pattern(cmd.command, patterns):
                verified = True
                break

        if not verified:
            gaps.append(VerificationGap(
                session_id=event.claude_session_id or "",
                project=event.claude_project or event.cwd or "unknown",
                session_end_ts=event.ts_end,
                session_cwd=event.cwd or "",
                edited_files=_edited_files(event),
            ))

    low_sample = total_fix_sessions < config.verification_min_fix_sessions
    gap_rate = None if low_sample else len(gaps) / total_fix_sessions if total_fix_sessions else None

    return VerificationGapFindings(
        gaps=gaps,
        total_fix_sessions=total_fix_sessions,
        gap_rate=gap_rate,
        low_sample=low_sample,
    )
