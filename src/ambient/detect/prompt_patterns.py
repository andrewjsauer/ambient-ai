import re
from collections import defaultdict
from dataclasses import dataclass

from ambient.capture.reader import Event
from ambient.config import Config

# Prompts matching these patterns are filtered as noise
NOISE_PATTERNS = frozenset({"clear", "yes", "ok", "y", "n", "no", "q", "exit", "quit"})
MIN_PROMPT_LENGTH = 3

# Substrings (lowercased) that mark a prompt as Claude-Code chrome rather than
# user intent — these should never seed a "skill" recommendation.
_CHROME_SUBSTRINGS = (
    "[request interrupted by user]",
    "request interrupted by user",
    "api error",
    "(no content)",
)

# Regex to strip file paths like /foo/bar.py or ./src/thing.ts
_FILE_PATH_RE = re.compile(r"[./]?/[\w./-]+\.\w+")

# Regex to strip leading slash-command prefix: "/commit message" -> "message"
_SLASH_CMD_RE = re.compile(r"^/\S+\s*")

# Collapse runs of whitespace
_WHITESPACE_RE = re.compile(r"\s+")

MAX_NORMALIZED_LENGTH = 200


@dataclass
class PromptPattern:
    normalized_prompt: str
    raw_examples: list[str]
    count: int
    projects: list[str]
    scope: str = "within_session"  # "within_session" | "cross_session"


@dataclass
class PromptPatternFindings:
    patterns: list[PromptPattern]
    total_prompts: int


def _normalize(prompt: str) -> str:
    """Normalization pipeline: lowercase, strip slash-command prefix, strip file paths, collapse whitespace, truncate."""
    text = prompt.lower()
    text = _SLASH_CMD_RE.sub("", text)
    text = _FILE_PATH_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text[:MAX_NORMALIZED_LENGTH]


def _is_noise(prompt: str) -> bool:
    stripped = prompt.strip()
    if len(stripped) < MIN_PROMPT_LENGTH:
        return True
    lower = stripped.lower()
    if lower in NOISE_PATTERNS:
        return True
    if any(marker in lower for marker in _CHROME_SUBSTRINGS):
        return True
    return False


def detect_prompt_patterns(
    events: list[Event], config: Config
) -> PromptPatternFindings:
    min_freq = getattr(config, "prompt_pattern_min_frequency", 3)
    max_ngram = getattr(config, "prompt_pattern_max_length", 4)
    cross_session_max_gap_ms = getattr(
        config, "prompt_pattern_cross_session_max_gap_ms", 86_400_000
    )

    # First pass: collect per-session data.
    # Each entry: (pairs, project, session_start_ts, session_end_ts, session_id).
    # ts and id are kept for the cross-session pass; within-session pass ignores them.
    session_data: list[
        tuple[list[tuple[str, str]], str | None, int, int, str]
    ] = []
    total_prompts = 0

    for event in events:
        if event.type != "claude_session":
            continue
        if not event.claude_prompts:
            continue
        project = event.claude_project
        pairs: list[tuple[str, str]] = []
        for raw in event.claude_prompts:
            if _is_noise(raw):
                continue
            total_prompts += 1
            pairs.append((raw, _normalize(raw)))
        if pairs:
            session_data.append(
                (pairs, project, event.ts_start, event.ts_end, event.claude_session_id or "")
            )

    if total_prompts == 0:
        return PromptPatternFindings(patterns=[], total_prompts=0)

    # --- Single-prompt frequency counts (cross-session) ---
    single_counts: dict[str, int] = defaultdict(int)
    single_projects: dict[str, set[str]] = defaultdict(set)
    single_examples: dict[str, list[str]] = defaultdict(list)

    for pairs, project, *_ in session_data:
        for raw, norm in pairs:
            single_counts[norm] += 1
            if project:
                single_projects[norm].add(project)
            if len(single_examples[norm]) < 5:
                single_examples[norm].append(raw)

    # --- N-gram counts (within individual sessions only) ---
    ngram_counts: dict[tuple[str, ...], int] = defaultdict(int)
    ngram_projects: dict[tuple[str, ...], set[str]] = defaultdict(set)
    ngram_examples: dict[tuple[str, ...], list[str]] = defaultdict(list)

    for pairs, project, *_ in session_data:
        norms = [norm for _, norm in pairs]
        for window_size in range(2, max_ngram + 1):
            for i in range(len(norms) - window_size + 1):
                gram = tuple(norms[i : i + window_size])
                ngram_counts[gram] += 1
                if project:
                    ngram_projects[gram].add(project)
                example = " -> ".join(gram)
                if len(ngram_examples[gram]) < 3:
                    ngram_examples[gram].append(example)

    # --- Cross-session n-grams: per-project flattened stream with time-gap sentinels ---
    # A cross-session n-gram requires at least 2 distinct session_ids in its window,
    # so within-session repetition is not double-counted here.
    cross_counts: dict[tuple[str, ...], int] = defaultdict(int)
    cross_projects: dict[tuple[str, ...], set[str]] = defaultdict(set)
    cross_examples: dict[tuple[str, ...], list[str]] = defaultdict(list)

    by_project_sessions: dict[str, list] = defaultdict(list)
    for entry in session_data:
        pairs, project, ts_start, ts_end, session_id = entry
        if not project:
            continue  # cross-session requires a known project scope
        by_project_sessions[project].append(entry)

    for project, project_sessions in by_project_sessions.items():
        project_sessions.sort(key=lambda s: s[2])  # by ts_start
        # Build a flat stream: (norm, session_id) with None as time-gap sentinel
        stream: list[tuple[str, str] | None] = []
        prev_end: int | None = None
        for pairs, _proj, ts_start, ts_end, session_id in project_sessions:
            if prev_end is not None and ts_start - prev_end > cross_session_max_gap_ms:
                stream.append(None)
            for _raw, norm in pairs:
                stream.append((norm, session_id))
            prev_end = ts_end

        for window_size in range(2, max_ngram + 1):
            for i in range(len(stream) - window_size + 1):
                window = stream[i : i + window_size]
                if any(item is None for item in window):
                    continue  # sentinel inside window
                session_ids_in_window = {item[1] for item in window}
                if len(session_ids_in_window) < 2:
                    continue  # entirely within one session — handled above
                gram = tuple(item[0] for item in window)
                cross_counts[gram] += 1
                cross_projects[gram].add(project)
                if len(cross_examples[gram]) < 3:
                    cross_examples[gram].append(" -> ".join(gram))

    # --- Build results ---
    patterns: list[PromptPattern] = []

    for norm, count in single_counts.items():
        if count >= min_freq:
            patterns.append(
                PromptPattern(
                    normalized_prompt=norm,
                    raw_examples=single_examples[norm][:5],
                    count=count,
                    projects=sorted(single_projects.get(norm, set())),
                    scope="within_session",
                )
            )

    for gram, count in ngram_counts.items():
        if count >= min_freq:
            patterns.append(
                PromptPattern(
                    normalized_prompt=" -> ".join(gram),
                    raw_examples=ngram_examples[gram][:5],
                    count=count,
                    projects=sorted(ngram_projects.get(gram, set())),
                    scope="within_session",
                )
            )

    for gram, count in cross_counts.items():
        if count >= min_freq:
            patterns.append(
                PromptPattern(
                    normalized_prompt=" -> ".join(gram),
                    raw_examples=cross_examples[gram][:5],
                    count=count,
                    projects=sorted(cross_projects.get(gram, set())),
                    scope="cross_session",
                )
            )

    patterns.sort(key=lambda p: p.count, reverse=True)

    return PromptPatternFindings(patterns=patterns, total_prompts=total_prompts)
