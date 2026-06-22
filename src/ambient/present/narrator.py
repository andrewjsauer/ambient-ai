import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ambient.config import Config
from ambient.detect.changepoints import ChangepointFindings
from ambient.detect.compression import CompressionFindings
from ambient.detect.pauses import PauseFindings
from ambient.present.prompts import (
    BATCH_SYSTEM,
    DAILY_SYSTEM,
    WEEKLY_SYSTEM,
    build_batch_prompt,
    build_daily_prompt,
    build_weekly_prompt,
)
from ambient.present.tokens import estimate_tokens

logger = logging.getLogger(__name__)

# Token budgets for input prompts (conservative, tunable via observation of TOKEN_USAGE logs).
BATCH_INPUT_BUDGET = 8_000
DAILY_INPUT_BUDGET = 50_000
WEEKLY_INPUT_BUDGET = 80_000


def _findings_to_dict(findings) -> dict:
    if hasattr(findings, "__dataclass_fields__"):
        return asdict(findings)
    return findings


def _call_api(config: Config, system: str, prompt: str, model: str,
              max_tokens: int = 2048, client=None) -> str:
    from ambient.present.api import call_api
    return call_api(config, system, prompt, model, max_tokens=max_tokens, client=client)


def narrate_batch(
    compression: CompressionFindings,
    pauses: PauseFindings,
    config: Config,
    claude_sessions: list[dict] | None = None,
    project_allocation: object | None = None,
    client=None,
) -> dict:
    compression_dict = _findings_to_dict(compression)
    pause_dict = _findings_to_dict(pauses)
    project_dict = _findings_to_dict(project_allocation) if project_allocation else None

    # Trim oldest claude sessions if the prompt would exceed budget (mirrors
    # narrate_daily/narrate_weekly). The detector findings are bounded; the
    # claude_sessions list carries verbatim prompt text and is what actually
    # blows the budget, so drop oldest sessions until it fits.
    sessions = list(claude_sessions) if claude_sessions else None
    prompt = build_batch_prompt(compression_dict, pause_dict, sessions, project_dict)
    while sessions and estimate_tokens(BATCH_SYSTEM) + estimate_tokens(prompt) > BATCH_INPUT_BUDGET:
        logger.info(
            "PROMPT_TRIMMED call_type=batch sessions_before=%d sessions_after=%d",
            len(sessions), len(sessions) - 1,
        )
        sessions.pop(0)  # drop oldest session
        prompt = build_batch_prompt(compression_dict, pause_dict, sessions or None, project_dict)

    # If still over budget with no sessions left to drop, the detector findings
    # themselves are oversized — warn but proceed (truncating them would distort
    # the analysis).
    estimated = estimate_tokens(BATCH_SYSTEM) + estimate_tokens(prompt)
    if estimated > BATCH_INPUT_BUDGET:
        logger.warning(
            "PROMPT_OVERBUDGET call_type=batch estimated=%d budget=%d",
            estimated, BATCH_INPUT_BUDGET,
        )

    # Build raw findings for preservation
    raw_findings = {
        "timestamp": datetime.now().isoformat(),
        "compression": compression_dict,
        "pauses": pause_dict,
    }
    if project_dict:
        raw_findings["project_allocation"] = project_dict

    try:
        response_text = _call_api(config, BATCH_SYSTEM, prompt, config.haiku_model,
                                  max_tokens=1024, client=client)
        try:
            analysis = json.loads(response_text)
        except json.JSONDecodeError:
            analysis = {"raw_narrative": response_text}

        result = {**raw_findings, "analysis": analysis}
    except Exception as e:
        logger.error("API call failed: %s. Raw findings preserved.", e)
        result = {**raw_findings, "analysis": None, "error": str(e)}

    # Save to analysis JSONL
    config.ensure_dirs()
    date_str = datetime.now().strftime("%Y-%m-%d")
    analysis_path = config.analysis_path(date_str)
    with open(analysis_path, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")

    return result


def narrate_daily(
    batch_analyses: list[dict],
    changepoints: ChangepointFindings | None,
    config: Config,
    date_str: str | None = None,
    client=None,
) -> str | None:
    changepoint_dict = _findings_to_dict(changepoints) if changepoints else None

    # Trim oldest batch analyses if prompt would exceed budget
    trimmed = list(batch_analyses)
    while len(trimmed) > 1:
        test_prompt = build_daily_prompt(trimmed, changepoint_dict)
        if estimate_tokens(DAILY_SYSTEM) + estimate_tokens(test_prompt) <= DAILY_INPUT_BUDGET:
            break
        logger.info(
            "PROMPT_TRIMMED call_type=daily items_before=%d items_after=%d",
            len(trimmed), len(trimmed) - 1,
        )
        trimmed.pop(0)  # drop oldest window

    prompt = build_daily_prompt(trimmed, changepoint_dict)

    try:
        narrative = _call_api(config, DAILY_SYSTEM, prompt, config.sonnet_model,
                              max_tokens=3000, client=client)
    except Exception as e:
        # Write nothing on failure: a placeholder at the canonical path would
        # satisfy the daemon's summary_path.exists() gate (and could overwrite
        # a previously generated good summary), permanently blocking the retry.
        # Returning None lets the caller leave state unadvanced and retry.
        logger.error("Daily summary API call failed: %s", e)
        return None

    # Save summary
    config.ensure_dirs()
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    summary_path = config.summary_path(date_str)
    summary_path.write_text(narrative)

    return narrative


def load_batch_analyses(config: Config, date_str: str) -> list[dict]:
    path = config.analysis_path(date_str)
    if not path.exists():
        return []
    analyses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    analyses.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return analyses


def narrate_weekly(
    weekly_analyses: list[dict],
    week_labels: list[str],
    config: Config,
    date_str: str | None = None,
    coaching_data: dict | None = None,
    client=None,
) -> str | None:
    """Generate a weekly trend summary from multiple weeks of daily analysis data.

    Returns the narrative on success, None on API failure (nothing written)."""
    # Trim oldest weeks if prompt would exceed budget
    trimmed_analyses = list(weekly_analyses)
    trimmed_labels = list(week_labels)
    while len(trimmed_analyses) > 1:
        test_prompt = build_weekly_prompt(trimmed_analyses, trimmed_labels, coaching_data=coaching_data)
        if estimate_tokens(WEEKLY_SYSTEM) + estimate_tokens(test_prompt) <= WEEKLY_INPUT_BUDGET:
            break
        logger.info(
            "PROMPT_TRIMMED call_type=weekly items_before=%d items_after=%d",
            len(trimmed_analyses), len(trimmed_analyses) - 1,
        )
        trimmed_analyses.pop()  # drop oldest week (last in list)
        trimmed_labels.pop()

    prompt = build_weekly_prompt(trimmed_analyses, trimmed_labels, coaching_data=coaching_data)

    try:
        narrative = _call_api(config, WEEKLY_SYSTEM, prompt, config.sonnet_model,
                              max_tokens=4000, client=client)
    except Exception as e:
        # Same contract as narrate_daily: no placeholder, no state advance —
        # the caller retries on a later tick.
        logger.error("Weekly summary API call failed: %s", e)
        return None

    config.ensure_dirs()
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    weekly_path = config.weekly_summary_path(date_str)
    weekly_path.write_text(narrative)

    return narrative
