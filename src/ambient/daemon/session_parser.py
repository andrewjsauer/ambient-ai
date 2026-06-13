"""Parse Claude Code per-session JSONL files for deep conversation capture."""

import json
import logging
import re
import shlex
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# File path pattern for extraction from tool_use inputs
_FILE_PATH_RE = re.compile(r'(?:^|["\s])(/[\w./-]+\.[\w]+)')

# When Claude runs a test or typecheck/build command via its Bash tool, the
# session is verifying its own work. Heavy Claude Code users run these inside
# the session rather than in a separate shell, so the shell-hook stream never
# sees them — this is what made the verification-gap detector report ~100%
# gaps on real data. We persist only booleans per session (ran a test? ran a
# typecheck? did a red test go green?), never the command text.
#
# Classification is STRUCTURAL, not a substring match: we look at the actual
# program each command segment runs (after stripping env vars, wrappers, and
# runner prefixes like `python -m` / `poetry run` / `npx`), so a tool name
# inside a commit message, echo, filename, or `--grep` argument does NOT
# count. Categories mirror the two config buckets in detect/verification.py:
# `test` ↔ verification_test_command_patterns, `typecheck` ↔
# verification_typecheck_command_patterns. Lint (ruff/eslint/flake8) is NOT
# verification — it is absent from both config lists — so it classifies to
# None and credits nothing.
_VERIFY_DIRECT = {
    "pytest": "test", "jest": "test", "vitest": "test", "mocha": "test",
    "rspec": "test", "tox": "test", "nox": "test", "ava": "test",
    "tsc": "typecheck", "mypy": "typecheck", "pyright": "typecheck",
}
_VERIFY_LINT = {"ruff", "eslint", "flake8", "golangci-lint", "pylint",
                "standard", "biome", "prettier", "black", "isort"}
_RUN_WRAPPERS = {"sudo", "time", "env", "nice", "command", "xvfb-run",
                 "stdbuf", "nohup", "exec"}
_RUNNER_RUN = {"poetry", "uv", "pdm", "rye", "pipenv", "bundle"}  # <runner> run <tool>
_PKG_MGRS = {"npm", "yarn", "pnpm", "bun"}  # <pm> test | <pm> run <script>


def _classify_token_list(tokens: list[str]) -> str | None:
    """Classify a single command's tokens as 'test', 'typecheck', or None."""
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if "=" in t and t.split("=", 1)[0].isidentifier():  # FOO=bar prefix
            i += 1
            continue
        if t in _RUN_WRAPPERS:
            i += 1
            continue
        break
    if i >= len(tokens):
        return None
    head, rest = tokens[i], tokens[i + 1:]

    # Unwrap runner prefixes to find the real tool.
    if head in ("python", "python3", "py") and rest[:1] == ["-m"] and len(rest) >= 2:
        head, rest = rest[1], rest[2:]
    elif head in _RUNNER_RUN and rest[:1] == ["run"] and len(rest) >= 2:
        head, rest = rest[1], rest[2:]
    elif head == "npx" and rest:
        head, rest = rest[0], rest[1:]
    elif head in ("pnpm", "yarn") and rest[:2] and rest[0] in ("exec", "dlx") and len(rest) >= 2:
        head, rest = rest[1], rest[2:]

    head = head.rsplit("/", 1)[-1]  # strip ./node_modules/.bin/jest etc.
    sub = rest[0] if rest else ""

    if head in _VERIFY_DIRECT:
        return _VERIFY_DIRECT[head]
    if head in _VERIFY_LINT:
        return None
    if head == "go":
        return "test" if sub == "test" else "typecheck" if sub in ("build", "vet") else None
    if head == "cargo":
        return "test" if sub == "test" else "typecheck" if sub in ("build", "check") else None
    if head == "make":
        return "test" if sub == "test" else "typecheck" if sub in ("check", "typecheck", "build") else None
    if head in ("rake", "rails") or head.endswith("rails"):
        return "test" if sub == "test" else None
    if head in ("mix", "deno"):
        return "test" if sub == "test" else None
    if head in _PKG_MGRS:
        if sub == "test" or sub.startswith("test:"):
            return "test"
        if sub == "run" and len(rest) >= 2:
            script = rest[1]
            if script == "test" or script.startswith("test:"):
                return "test"
            if script in ("build", "typecheck", "type-check", "tsc", "compile"):
                return "typecheck"
        return None
    return None


_SHELL_SPLIT_RE = re.compile(r"&&|\|\||;|\||\n")


def _classify_command(cmd: str) -> str | None:
    """Classify a Bash command string. 'test' wins over 'typecheck' when both
    appear (a session that ran tests verified more strongly)."""
    if not isinstance(cmd, str):
        return None
    best: str | None = None
    for seg in _SHELL_SPLIT_RE.split(cmd):
        seg = seg.strip()
        if not seg or seg.startswith("#"):
            continue
        try:
            tokens = shlex.split(seg)
        except ValueError:
            tokens = seg.split()
        cat = _classify_token_list(tokens)
        if cat == "test":
            return "test"
        if cat == "typecheck":
            best = "typecheck"
    return best


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
    categories_run: set[str] = set()      # {"test", "typecheck"} the session ran via Bash
    pending_verif: dict[str, str] = {}    # verification Bash tool_use id -> category
    verif_results: list[tuple[str, bool]] = []  # (category, is_error) per result, in order
    is_error_count = 0
    session_id = None
    project = None
    max_ts: datetime | None = None
    min_ts: datetime | None = None
    new_min_ts: datetime | None = None  # first timestamp in the new (non-skipped) portion
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

                # Track the first timestamp in the new portion
                if new_min_ts is None and ts_str:
                    try:
                        new_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        new_min_ts = new_ts
                    except (ValueError, TypeError):
                        pass

                if entry_type == "user" and not d.get("isMeta"):
                    _extract_user_message(d, prompts)
                    is_error_count += _count_errors_in_user(d)
                    _record_verification_results(d, pending_verif, verif_results)

                elif entry_type == "assistant":
                    _extract_assistant_tools(d, tools, files_touched,
                                             pending_verif, categories_run)

    except OSError as e:
        logger.warning("Cannot read session file %s: %s", path, e)
        return None

    if min_ts is None:
        return None

    start_ms = int(min_ts.timestamp() * 1000)
    end_ms = int(max_ts.timestamp() * 1000) if max_ts else start_ms
    duration_ms = end_ms - start_ms

    # For incremental parsing, provide the start of the new portion
    new_start_ms = int(new_min_ts.timestamp() * 1000) if new_min_ts else start_ms
    new_duration_ms = end_ms - new_start_ms

    return {
        "session_id": session_id or path.stem,
        "project": project or "",
        "prompts": prompts,
        "tools": tools,
        "files_touched": sorted(files_touched),
        "ran_test": "test" in categories_run,
        "ran_typecheck": "typecheck" in categories_run,
        "verification_resolved": _verification_resolved(verif_results),
        "is_error_count": is_error_count,
        "prompt_count": len(prompts),
        "start_ts": start_ms,
        "end_ts": end_ms,
        "duration_ms": duration_ms,
        "new_start_ts": new_start_ms,
        "new_duration_ms": new_duration_ms,
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


def _extract_assistant_tools(d: dict, tools: list[dict], files_touched: set[str],
                             pending_verif: dict[str, str], categories_run: set[str]) -> None:
    """Extract tool_use blocks from an assistant message.

    For each Bash call that runs a test or typecheck/build command, records its
    category in `categories_run` and maps its tool_use id -> category in
    `pending_verif` so the matching tool_result (pass/fail) can be correlated
    in the user message that follows. The command text is classified and
    discarded — only the category (and later, the boolean outcome) propagates.
    """
    content = d.get("message", {}).get("content", [])
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name = block.get("name", "")
        inp = block.get("input", {})

        if name == "Bash":
            category = _classify_command(inp.get("command", ""))
            if category:
                categories_run.add(category)
                tid = block.get("id")
                if tid:
                    pending_verif[tid] = category

        # Extract file paths from common tool inputs
        tool_files = []
        for key in ("file_path", "path", "command"):
            val = inp.get(key, "")
            if isinstance(val, str):
                matches = _FILE_PATH_RE.findall(val)
                tool_files.extend(matches)

        tools.append({"name": name, "files": tool_files})
        files_touched.update(tool_files)


def _record_verification_results(d: dict, pending_verif: dict[str, str],
                                 verif_results: list[tuple[str, bool]]) -> None:
    """Correlate tool_results in a user message with pending verification
    commands, appending each verification's (category, is_error) in order."""
    content = d.get("message", {}).get("content", "")
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        tid = block.get("tool_use_id")
        category = pending_verif.pop(tid, None) if tid else None
        if category:
            verif_results.append((category, block.get("is_error") is True))


def _verification_resolved(results: list[tuple[str, bool]]) -> bool:
    """True when a verification failed and a later one in the SAME category
    passed — an in-session fix loop (red→green). Per-category so a pytest
    failure followed by an unrelated typecheck pass is not a false resolution.
    `results` is the ordered list of (category, is_error)."""
    failed: set[str] = set()
    for category, is_err in results:
        if is_err:
            failed.add(category)
        elif category in failed:
            return True
    return False


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
