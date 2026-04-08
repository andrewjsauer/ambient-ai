"""Recommendation engine: generates actionable artifacts from detector findings."""

import logging
import re
from dataclasses import dataclass, field



from ambient.config import Config

logger = logging.getLogger(__name__)

SKILL_GENERATION_SYSTEM = """You are generating a Claude Code skill definition for a developer who repeatedly performs a specific action. Create a concise, useful skill that automates the detected pattern.

Return ONLY the skill content in this format:
---
description: [One-line description of what this skill does]
---

# [Skill Name]

[2-3 sentences describing what this skill does and when to use it.]

## Steps

[Numbered steps the skill should perform. Be specific and actionable.]

Do not include meta-commentary. The output should be a complete, valid skill file."""


@dataclass
class Recommendation:
    id: str
    type: str  # "skill", "alias", "claude_md"
    title: str
    rationale: str
    artifact: str  # the actual skill/alias/rule content
    source_pattern: str


@dataclass
class RecommendationFindings:
    recommendations: list[Recommendation] = field(default_factory=list)


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:50].strip("-")


def generate_recommendations(
    prompt_patterns: object | None,
    compression_findings: object | None,
    config: Config,
) -> RecommendationFindings:
    """Generate recommendation artifacts from detector findings.

    Uses Haiku to draft skill definitions from repeated prompt patterns.
    Generates alias suggestions from repeated shell command sequences.
    """
    recommendations = []

    # Skill recommendations from prompt patterns
    if prompt_patterns:
        patterns = _get_patterns(prompt_patterns)
        for pattern in patterns:
            if pattern["count"] < 5:  # higher bar for skill recommendations
                continue
            rec = _generate_skill_recommendation(pattern, config)
            if rec:
                recommendations.append(rec)

    # Alias recommendations from compression findings
    if compression_findings:
        sequences = _get_sequences(compression_findings)
        for seq in sequences:
            if seq["count"] < 5:
                continue
            rec = _generate_alias_recommendation(seq)
            if rec:
                recommendations.append(rec)

    # Write recommendations to staging directory
    findings = RecommendationFindings(recommendations=recommendations)
    if recommendations:
        _write_recommendations(recommendations, config)

    return findings


def _get_patterns(prompt_patterns) -> list[dict]:
    """Extract pattern dicts from PromptPatternFindings."""
    if hasattr(prompt_patterns, "patterns"):
        return [
            {
                "normalized_prompt": p.normalized_prompt,
                "raw_examples": p.raw_examples,
                "count": p.count,
                "projects": p.projects,
            }
            for p in prompt_patterns.patterns
        ]
    return []


def _get_sequences(compression_findings) -> list[dict]:
    """Extract sequence dicts from CompressionFindings."""
    if hasattr(compression_findings, "sequences"):
        return [
            {
                "sequence": list(s.sequence),
                "count": s.count,
                "total_time_ms": s.total_time_ms,
            }
            for s in compression_findings.sequences
        ]
    return []


def _generate_skill_recommendation(pattern: dict, config: Config) -> Recommendation | None:
    """Generate a skill recommendation from a repeated prompt pattern."""
    normalized = pattern["normalized_prompt"]
    examples = pattern.get("raw_examples", [])[:3]
    count = pattern["count"]

    prompt = f"""A developer repeatedly types this kind of prompt in Claude Code ({count} times):

Pattern: "{normalized}"

Examples of actual prompts:
{chr(10).join(f'- "{ex}"' for ex in examples)}

Generate a Claude Code skill that automates this action."""

    try:
        from ambient.present.api import call_api
        artifact = call_api(config, SKILL_GENERATION_SYSTEM, prompt, config.haiku_model)
    except Exception as e:
        logger.warning("Skill generation failed for pattern '%s': %s", normalized, e)
        return None

    slug = _slugify(normalized)
    return Recommendation(
        id=f"skill-{slug}",
        type="skill",
        title=f"Skill: {normalized[:60]}",
        rationale=f"You typed this {count} times across sessions. A skill can automate it.",
        artifact=artifact,
        source_pattern=normalized,
    )


def _generate_alias_recommendation(seq: dict) -> Recommendation | None:
    """Generate a shell alias from a repeated command sequence."""
    commands = seq["sequence"]
    count = seq["count"]
    time_ms = seq.get("total_time_ms", 0)

    if len(commands) < 2:
        return None

    # Build a reasonable alias name from the commands
    parts = []
    for cmd in commands:
        first_word = cmd.split()[0] if cmd.split() else cmd
        # Use first letter of each command word
        parts.append(first_word[0] if first_word else "x")
    alias_name = "".join(parts)

    # Build the alias command
    chained = " && ".join(commands)
    artifact = f'alias {alias_name}="{chained}"'

    return Recommendation(
        id=f"alias-{alias_name}",
        type="alias",
        title=f"Alias: {alias_name} ({' -> '.join(commands)})",
        rationale=f"You run this sequence {count} times (total {time_ms / 1000:.0f}s). An alias saves keystrokes.",
        artifact=artifact,
        source_pattern=" -> ".join(commands),
    )


def _write_recommendations(recommendations: list[Recommendation], config: Config) -> None:
    """Write recommendation files to the staging directory."""
    rec_dir = config.base_dir / "recommendations"
    rec_dir.mkdir(parents=True, exist_ok=True)

    for rec in recommendations:
        path = rec_dir / f"{rec.id}.md"
        content = f"""---
type: {rec.type}
title: "{rec.title}"
rationale: "{rec.rationale}"
source_pattern: "{rec.source_pattern}"
---

{rec.artifact}
"""
        path.write_text(content)
        logger.info("Wrote recommendation: %s", path)
