"""Slash-command taxonomy.

Classifies Claude Code slash commands into intent categories so detectors
can answer "how much planning vs execution vs review" without re-deriving
the mapping. The taxonomy was validated against a large inventory of real
Claude Code sessions during development.

Categories:
    planning   — ce-plan, ce-brainstorm, ce-ideate, deepen-plan, value-realization
    execution  — ce-work, /work, /ship, /cleanup-*
    review     — ce-review, aeo-audit, delight-auditor
    design     — frontend-design, ui-ux-pro-max, extract-design
    meta       — /clear, /model, /login, /effort, /insights
    other      — anything unrecognized

Two namespacing variants exist in real data:
    /compound-engineering:ce-plan   (hyphenated)
    /compound-engineering:ce:plan   (deprecated colon-form, but still appears)

Both normalize to the same category. Per-user overrides via
`Config.slash_taxonomy_overrides` let advanced users reclassify custom commands
without code changes.
"""

import re
from typing import Literal

SlashCategory = Literal["planning", "execution", "review", "design", "meta", "other"]

_PLANNING: frozenset[str] = frozenset({
    "/compound-engineering:ce-plan",
    "/compound-engineering:ce:plan",
    "/compound-engineering:ce-brainstorm",
    "/compound-engineering:ce:brainstorm",
    "/compound-engineering:ce-ideate",
    "/compound-engineering:ce:ideate",
    "/compound-engineering:deepen-plan",
    "/compound-engineering:ce-deepen-plan",
    "/plan",
    "/value-realization",
})

_EXECUTION: frozenset[str] = frozenset({
    "/compound-engineering:ce-work",
    "/compound-engineering:ce:work",
    "/work",
    "/ship",
    "/cleanup-branch",
    "/cleanup-tree",
    "/cleanup",
    "/commit",
    "/commit-commands:commit",
    "/commit-commands:commit-push-pr",
})

_REVIEW: frozenset[str] = frozenset({
    "/compound-engineering:ce-review",
    "/compound-engineering:ce:review",
    "/review",
    "/aeo-audit",
    "/delight-auditor",
    "/security-review",
    "/code-review:code-review",
})

_DESIGN: frozenset[str] = frozenset({
    "/compound-engineering:frontend-design",
    "/frontend-design:frontend-design",
    "/ui-ux-pro-max",
    "/extract-design",
})

_META: frozenset[str] = frozenset({
    "/clear",
    "/model",
    "/login",
    "/insights",
    "/effort",
    "/init",
})

_COMMAND_NAME_RE = re.compile(r"<command-name>\s*(/[^<\s]+)\s*</command-name>")


def extract_slash_command(prompt_text: str) -> str | None:
    """Extract the slash command from a Claude Code user-prompt body.

    Claude Code wraps slash invocations in `<command-name>/foo</command-name>`
    markers in the JSONL message body. Returns the command (with leading `/`)
    or None if no marker is present.
    """
    if not prompt_text:
        return None
    m = _COMMAND_NAME_RE.search(prompt_text)
    if not m:
        return None
    return m.group(1).strip()


def classify_slash_command(
    command: str | None,
    overrides: dict[str, str] | None = None,
) -> SlashCategory:
    """Classify a slash command into one of six categories.

    Args:
        command: Slash command including the leading `/`. None or empty → "other".
        overrides: Per-user reclassification map; overrides[command] = category.
            Override values are validated against the SlashCategory literal;
            invalid categories fall through to the built-in classification.

    Behavior:
    - Strips trailing whitespace and trailing colons.
    - Normalizes the deprecated colon-form (`ce:plan`) to its canonical
      hyphenated equivalent (`ce-plan`) for built-in classifications, so
      both variants land in the same category.
    - Unknown commands return "other".
    """
    if not command:
        return "other"
    cmd = command.strip().rstrip(":")
    if not cmd.startswith("/"):
        cmd = "/" + cmd

    if overrides:
        forced = overrides.get(cmd)
        if forced in ("planning", "execution", "review", "design", "meta", "other"):
            return forced  # type: ignore[return-value]

    if cmd in _PLANNING:
        return "planning"
    if cmd in _EXECUTION:
        return "execution"
    if cmd in _REVIEW:
        return "review"
    if cmd in _DESIGN:
        return "design"
    if cmd in _META:
        return "meta"
    return "other"
