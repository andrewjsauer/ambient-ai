"""Git activity detector.

Pulls per-project commit data for the insights window using `git log`.

Why this exists: ambient-ai's prior weekly reports had no view of what the
developer actually shipped — every finding was process-meta with no
denominator. This detector reads `git log --since --until` for every
git root touched during the window, so the report can open with "you
shipped X commits across Y projects" and every later finding (verification
gaps, stuck patterns) can be framed against real work.

Read-only by contract: `git log` and `git rev-parse --show-toplevel` only.
No commits, branches, or remotes are touched. All subprocess calls have a
strict timeout; failures degrade silently to an empty list for that project.
"""

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ambient.capture.reader import Event
from ambient.config import Config

logger = logging.getLogger(__name__)


@dataclass
class GitCommit:
    sha: str
    ts_iso: str          # author-date in ISO 8601, repo-local timezone
    author: str
    subject: str
    files_changed: int
    insertions: int
    deletions: int


@dataclass
class GitActivityFindings:
    by_project: dict[str, list[GitCommit]] = field(default_factory=dict)
    total_commits: int = 0
    total_lines_changed: int = 0


_GIT_TIMEOUT_SECONDS = 5
# Cap total wall-clock spent in detect_git_activity. Worst-case sequential
# run of N timeouts (rev-parse + log) could be ~150s for 15 stalled NFS
# repos; this budget caps that and returns partial results instead.
_DETECTOR_BUDGET_SECONDS = 60
# Commit format: SHA|author-date-ISO|author-name|subject. Subject is last so
# pipe characters in the subject (rare but legal) don't break parsing.
_LOG_FORMAT = "%H|%aI|%an|%s"

# Force English git output regardless of system locale. Without this,
# `git log --shortstat` translates "files changed" / "insertions" / "deletions"
# and our keyword-based parser silently zeros the counts.
_GIT_ENV = {**os.environ, "LANG": "C", "LC_ALL": "C"}


def _safe_cwd_for_git(cwd: str) -> bool:
    """Reject cwds that git could parse as a flag.

    `git -C <cwd>` interprets any cwd that starts with `-` as a global git
    option (e.g. `--exec-path=/tmp/evil`). Legitimate macOS shell cwds are
    always absolute paths starting with `/`, so anything else is unsafe.
    """
    return bool(cwd) and cwd.startswith("/")


def _git_root(cwd: str) -> str | None:
    """Return the git toplevel for cwd, or None if cwd isn't in a repo.

    Bounded by `_GIT_TIMEOUT_SECONDS`. Suppresses all subprocess errors so a
    single bad cwd never propagates out of the detector. Refuses cwds git
    could parse as flags.
    """
    if not _safe_cwd_for_git(cwd):
        logger.warning("git_activity: refusing unsafe cwd=%r (must be absolute path)", cwd)
        return None
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, errors="replace",
            env=_GIT_ENV,
            timeout=_GIT_TIMEOUT_SECONDS, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("git_activity: rev-parse failed for cwd=%r: %s", cwd, e)
        return None
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    return root or None


def _parse_shortstat(line: str) -> tuple[int, int, int]:
    """Parse `git log --shortstat` line: ' 2 files changed, 14 insertions(+), 3 deletions(-)'.

    Any of the three counts may be absent (e.g. insertions-only commit).
    Returns (files_changed, insertions, deletions).
    """
    files_changed = insertions = deletions = 0
    for part in line.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            num_str, label = part.split(" ", 1)
            num = int(num_str)
        except (ValueError, IndexError):
            continue
        if "file" in label:
            files_changed = num
        elif "insertion" in label:
            insertions = num
        elif "deletion" in label:
            deletions = num
    return files_changed, insertions, deletions


def _git_log_for_root(root: str, start: datetime, end: datetime, max_commits: int) -> list[GitCommit]:
    """Run `git log --since --until --shortstat --no-merges` for one git root.

    Returns parsed commits, or [] on any failure.
    """
    args = [
        "git", "-C", root, "log",
        f"--since={start.isoformat()}",
        f"--until={end.isoformat()}",
        f"--pretty=format:{_LOG_FORMAT}",
        "--shortstat",
        "--no-merges",
        f"-n{max_commits}",
    ]
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, errors="replace",
            env=_GIT_ENV,
            timeout=_GIT_TIMEOUT_SECONDS, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("git_activity: log failed for root=%r: %s", root, e)
        return []
    if result.returncode != 0:
        logger.warning(
            "git_activity: log nonzero exit for root=%r rc=%d: %s",
            root, result.returncode, result.stderr.strip()[:200],
        )
        return []
    return _parse_log_output(result.stdout)


_SHA_HEX_RE = re.compile(r"^[0-9a-f]{7,40}\|")


def _parse_log_output(output: str) -> list[GitCommit]:
    """Parse the alternating header / shortstat lines from git log output.

    git log --shortstat with our format emits:
        <SHA>|<ISO>|<author>|<subject>
         N files changed, X insertions(+), Y deletions(-)
        <blank>
        <SHA>|<ISO>|<author>|<subject>
         N files changed, X insertions(+), Y deletions(-)

    A header line without a following shortstat (e.g., empty commit) yields
    a record with zero counts. Header detection uses a SHA hex prefix match
    rather than a pipe-count heuristic, so a shortstat line that happens to
    contain pipes can never be misclassified as a header.
    """
    commits: list[GitCommit] = []
    pending: GitCommit | None = None

    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _SHA_HEX_RE.match(line):
            # Flush previous header that never got a shortstat.
            if pending is not None:
                commits.append(pending)
            sha, ts_iso, author, subject = line.split("|", 3)
            pending = GitCommit(
                sha=sha, ts_iso=ts_iso, author=author, subject=subject,
                files_changed=0, insertions=0, deletions=0,
            )
        elif pending is not None and "changed" in line:
            files_changed, insertions, deletions = _parse_shortstat(line)
            pending.files_changed = files_changed
            pending.insertions = insertions
            pending.deletions = deletions
            commits.append(pending)
            pending = None

    if pending is not None:
        commits.append(pending)

    return commits


def _resolve_root_cached(cwd: str, known_roots: list[str]) -> str | None:
    """If `cwd` is inside an already-discovered git root, return that root
    without invoking `git rev-parse`. Otherwise call `_git_root(cwd)`.

    Same monorepo with N subdirectories pays O(N) parent-prefix checks and
    only one rev-parse, instead of N rev-parse subprocess calls.
    """
    cwd_path = Path(cwd) if cwd else None
    if cwd_path is not None:
        for root in known_roots:
            try:
                root_path = Path(root)
            except (ValueError, OSError):
                continue
            if cwd_path == root_path or root_path in cwd_path.parents:
                return root
    return _git_root(cwd)


def _unique_project_label(root: str, taken: set[str]) -> str:
    """Derive a unique project label from a git root path.

    Default is the basename. If two roots share a basename (rare, e.g.
    `~/work/auth` vs `~/personal/auth`), suffix with the parent directory
    name so each root keeps a distinct entry. Avoids the merge-sort + cap
    accounting bugs that would otherwise come with collapsing them.
    """
    p = Path(root)
    label = p.name
    if label and label not in taken:
        return label
    # Walk up parents until a unique label emerges, or fall back to a hash
    # suffix if even the full path collides (essentially impossible in
    # practice but keeps the function total).
    for parent in p.parents:
        candidate = f"{label} ({parent.name})" if parent.name else label
        if candidate and candidate not in taken:
            return candidate
    return f"{label or 'unknown'}#{abs(hash(root)) % 10000}"


def detect_git_activity(
    events: list[Event],
    start: datetime,
    end: datetime,
    config: Config,
) -> GitActivityFindings:
    """Walk unique cwds in `events`, find each git root, and read its log
    for the [start, end] window.

    Each project's commits are sorted newest-first (git log default). Caps
    each repo at `config.git_activity_max_commits`. Idempotent: never writes
    anything to disk or to git state.
    """
    cwds: set[str] = {e.cwd for e in events if e.cwd}
    if not cwds:
        return GitActivityFindings()

    # Group cwds by their git root. Multiple cwds can map to the same root
    # (e.g., subdirectories of a repo) so we only `git log` once per root.
    # `_resolve_root_cached` short-circuits the rev-parse subprocess when a
    # parent of cwd is already known to be a git root, so N cwds under one
    # monorepo cost one rev-parse instead of N.
    known_roots: list[str] = []
    roots: dict[str, str] = {}  # root -> project label (guaranteed unique)
    for cwd in cwds:
        root = _resolve_root_cached(cwd, known_roots)
        if root and root not in roots:
            known_roots.append(root)
            roots[root] = _unique_project_label(root, set(roots.values()))

    if not roots:
        return GitActivityFindings()

    by_project: dict[str, list[GitCommit]] = {}
    total_commits = 0
    total_lines_changed = 0
    max_commits = config.git_activity_max_commits
    deadline = time.monotonic() + _DETECTOR_BUDGET_SECONDS

    for root, project in roots.items():
        if time.monotonic() > deadline:
            logger.warning(
                "git_activity: budget %ds exhausted; returning partial results "
                "(%d/%d projects scanned)",
                _DETECTOR_BUDGET_SECONDS, len(by_project), len(roots),
            )
            break
        commits = _git_log_for_root(root, start, end, max_commits)
        if not commits:
            continue
        by_project[project] = commits
        total_commits += len(commits)
        total_lines_changed += sum(c.insertions + c.deletions for c in commits)

    return GitActivityFindings(
        by_project=by_project,
        total_commits=total_commits,
        total_lines_changed=total_lines_changed,
    )
