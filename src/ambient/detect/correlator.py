from dataclasses import dataclass, field

from ambient.capture.reader import Event

# Default time window for correlating events (5 minutes in ms)
CORRELATION_WINDOW_MS = 300_000

# Commands where non-zero exit codes are normal/benign, not errors
BENIGN_NONZERO_COMMANDS = frozenset({
    "grep",
    "egrep",
    "fgrep",
    "rg",
    "test",
    "diff",
    "cmp",
    "false",
})


@dataclass
class CorrelationPattern:
    pattern_type: str
    count: int
    examples: list[dict] = field(default_factory=list)


@dataclass
class CorrelationFindings:
    patterns: list[CorrelationPattern] = field(default_factory=list)
    total_correlations: int = 0


def _base_command(command: str) -> str:
    """Extract the base command name (first token) from a full command string."""
    parts = command.strip().split()
    return parts[0] if parts else command


def _is_benign_nonzero(event: Event) -> bool:
    """Check if a non-zero exit code is benign for this command."""
    return _base_command(event.command) in BENIGN_NONZERO_COMMANDS


def correlate_signals(
    events: list[Event],
    correlation_window_ms: int = CORRELATION_WINDOW_MS,
) -> CorrelationFindings:
    if not events:
        return CorrelationFindings()

    commands = [e for e in events if e.type == "command"]
    claude_sessions = [e for e in events if e.type == "claude_session"]

    error_then_claude_examples: list[dict] = []
    claude_then_retry_examples: list[dict] = []
    claude_then_success_examples: list[dict] = []

    # Pattern 1: error_then_claude
    # Shell command with non-zero exit followed by Claude session within window
    for cmd in commands:
        if cmd.exit_code == 0 or _is_benign_nonzero(cmd):
            continue
        for cs in claude_sessions:
            gap = cs.ts_start - cmd.ts_end
            if 0 <= gap <= correlation_window_ms:
                error_then_claude_examples.append({
                    "command": cmd.command,
                    "exit_code": cmd.exit_code,
                    "claude_session_start": cs.ts_start,
                    "gap_ms": gap,
                })
                break  # one match per failed command

    # Pattern 2: claude_then_retry
    # Claude session followed by a shell command matching one from before the session
    for cs in claude_sessions:
        # Collect commands that happened shortly before this Claude session (within window)
        pre_commands = {
            c.command for c in commands
            if cs.ts_start - correlation_window_ms <= c.ts_end <= cs.ts_start
        }
        if not pre_commands:
            continue
        # Look for post-session commands that match a pre-session command
        for cmd in commands:
            gap = cmd.ts_start - cs.ts_end
            if 0 <= gap <= correlation_window_ms and cmd.command in pre_commands:
                claude_then_retry_examples.append({
                    "command": cmd.command,
                    "claude_session_end": cs.ts_end,
                    "retry_start": cmd.ts_start,
                    "gap_ms": gap,
                })
                break  # one match per Claude session

    # Pattern 3: claude_then_success
    # Claude session followed by a shell command with exit_code=0 within window
    for cs in claude_sessions:
        for cmd in commands:
            gap = cmd.ts_start - cs.ts_end
            if 0 <= gap <= correlation_window_ms and cmd.exit_code == 0:
                claude_then_success_examples.append({
                    "command": cmd.command,
                    "claude_session_end": cs.ts_end,
                    "success_start": cmd.ts_start,
                    "gap_ms": gap,
                })
                break  # one match per Claude session

    patterns: list[CorrelationPattern] = []
    for ptype, examples in [
        ("error_then_claude", error_then_claude_examples),
        ("claude_then_retry", claude_then_retry_examples),
        ("claude_then_success", claude_then_success_examples),
    ]:
        if examples:
            patterns.append(CorrelationPattern(
                pattern_type=ptype,
                count=len(examples),
                examples=examples,
            ))

    total = sum(p.count for p in patterns)
    return CorrelationFindings(patterns=patterns, total_correlations=total)
