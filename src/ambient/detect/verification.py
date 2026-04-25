"""Verification-gap detector: find Edit/Write tool calls in Claude sessions
that were not followed by a verifying test command in the same cwd within
a configurable window.

This surfaces the DORA "Rework Rate"-style signal at an individual level:
fixes that shipped without re-running the tests that would have proven them.
Unique to ambient-ai because it requires both the Claude session stream and
the shell event stream for the same developer.

The detector is project-capability aware: each fix session is bucketed into
`has_tests` (project has a real test target), `has_typecheck` (project has
a typecheck/build but no tests), or `neither` (no detectable verification
capability). Gap rates are reported per-bucket so a project with no tests
is not conflated with a project that skipped tests.
"""

from dataclasses import dataclass, field
from typing import Literal

from ambient.capture.reader import Event
from ambient.config import Config
from ambient.detect.project_capabilities import detect_capabilities

Bucket = Literal["has_tests", "has_typecheck", "neither"]
_BUCKETS: tuple[Bucket, ...] = ("has_tests", "has_typecheck", "neither")


@dataclass
class VerificationGap:
    session_id: str
    project: str
    session_end_ts: int
    session_cwd: str
    edited_files: list[str]
    bucket: Bucket = "has_tests"


@dataclass
class VerificationGapFindings:
    gaps: list[VerificationGap] = field(default_factory=list)
    total_fix_sessions: int = 0
    gap_rate: float | None = None  # gaps / total_fix_sessions; None under low sample
    low_sample: bool = False
    # Per-bucket breakdowns. Keys are members of `Bucket`. `gap_rate_by_bucket[b]`
    # is None when that bucket is below `verification_min_fix_sessions`
    # (per-bucket low sample) and unconditionally None for the `neither` bucket
    # (no verification capability — see detect_verification_gaps docstring).
    total_fix_sessions_by_bucket: dict[Bucket, int] = field(default_factory=dict)
    gaps_by_bucket: dict[Bucket, int] = field(default_factory=dict)
    gap_rate_by_bucket: dict[Bucket, float | None] = field(default_factory=dict)
    low_sample_by_bucket: dict[Bucket, bool] = field(default_factory=dict)


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


def _matches_pattern(command: str, patterns: list[str]) -> bool:
    cmd = command.strip().lower()
    for pat in patterns:
        if cmd.startswith(pat.lower()):
            return True
    return False


def _bucket_for(cwd: str | None) -> Bucket:
    """Bucket a session by the verification capabilities of its cwd.

    Defensive against any unexpected probe failure: if detect_capabilities
    raises, this returns 'neither' rather than propagating, so a single
    pathological cwd can't fail-stop the whole verification detector via
    the surrounding _safe_run wrapper.
    """
    try:
        caps = detect_capabilities(cwd)
    except Exception as e:  # noqa: BLE001 — last-resort containment
        import logging
        logging.getLogger(__name__).warning(
            "verification: capability probe failed for cwd=%r, defaulting to 'neither': %s",
            cwd, e,
        )
        return "neither"
    if caps.has_tests:
        return "has_tests"
    if caps.has_typecheck:
        return "has_typecheck"
    return "neither"


def _patterns_for_bucket(bucket: Bucket, config: Config) -> list[str]:
    """Which command patterns count as verification for this bucket."""
    if bucket == "has_tests":
        return config.verification_test_command_patterns
    if bucket == "has_typecheck":
        return config.verification_typecheck_command_patterns
    # `neither` projects have no verification capability; nothing counts.
    return []


def detect_verification_gaps(
    events: list[Event], config: Config
) -> VerificationGapFindings:
    """Find fix sessions with no subsequent verifying command.

    For `has_tests` projects, a fix session is verified when a command matching
    `verification_test_command_patterns` fires in the same cwd within
    `verification_gap_window_ms`. For `has_typecheck` projects, same logic but
    against `verification_typecheck_command_patterns`. For `neither` projects,
    no verification is possible; sessions are recorded with `bucket="neither"`
    and `gap_rate_by_bucket["neither"]` is always None — every neither session
    technically lands in `gaps` but framing those as "skipped verification" is
    a category error the rendering layer must avoid.
    """
    if not events:
        return VerificationGapFindings()

    window_ms = config.verification_gap_window_ms

    gaps: list[VerificationGap] = []
    total_fix_sessions = 0

    # Pre-index command events for quick per-session scanning. Sorting is
    # cheap; correlator.py follows the same pattern.
    command_events = [e for e in events if e.type == "command"]

    # Per-bucket tallies start at zero so even buckets with no sessions appear
    # in the output dicts (downstream code can rely on every bucket being keyed).
    by_bucket_total = {b: 0 for b in _BUCKETS}
    by_bucket_gaps = {b: 0 for b in _BUCKETS}

    for event in events:
        if not _is_fix_session(event):
            continue
        total_fix_sessions += 1

        bucket = _bucket_for(event.cwd)
        by_bucket_total[bucket] += 1
        patterns = _patterns_for_bucket(bucket, config)

        verified = False
        if patterns:
            for cmd in command_events:
                gap = cmd.ts_start - event.ts_end
                if gap < 0 or gap > window_ms:
                    continue
                if cmd.cwd != event.cwd:
                    continue
                if _matches_pattern(cmd.command, patterns):
                    verified = True
                    break

        if not verified:
            by_bucket_gaps[bucket] += 1
            gaps.append(VerificationGap(
                session_id=event.claude_session_id or "",
                project=event.claude_project or event.cwd or "unknown",
                session_end_ts=event.ts_end,
                session_cwd=event.cwd or "",
                edited_files=_edited_files(event),
                bucket=bucket,
            ))

    min_n = config.verification_min_fix_sessions
    low_sample = total_fix_sessions < min_n
    gap_rate = (
        None
        if low_sample or total_fix_sessions == 0
        else len(gaps) / total_fix_sessions
    )

    gap_rate_by_bucket: dict[str, float | None] = {}
    low_sample_by_bucket: dict[str, bool] = {}
    for b, total in by_bucket_total.items():
        bucket_low = total < min_n
        low_sample_by_bucket[b] = bucket_low
        # `neither` projects have no verification capability by definition —
        # publishing a rate (even 1.0) misleads any downstream consumer that
        # treats it as a confident signal. Suppress unconditionally.
        if b == "neither":
            gap_rate_by_bucket[b] = None
            continue
        if bucket_low or total == 0:
            gap_rate_by_bucket[b] = None
        else:
            gap_rate_by_bucket[b] = by_bucket_gaps[b] / total

    return VerificationGapFindings(
        gaps=gaps,
        total_fix_sessions=total_fix_sessions,
        gap_rate=gap_rate,
        low_sample=low_sample,
        total_fix_sessions_by_bucket=by_bucket_total,
        gaps_by_bucket=by_bucket_gaps,
        gap_rate_by_bucket=gap_rate_by_bucket,
        low_sample_by_bucket=low_sample_by_bucket,
    )
