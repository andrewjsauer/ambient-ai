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
    build_batch_prompt,
    build_daily_prompt,
)

logger = logging.getLogger(__name__)


def _findings_to_dict(findings) -> dict:
    if hasattr(findings, "__dataclass_fields__"):
        return asdict(findings)
    return findings


def _call_api(config: Config, system: str, prompt: str, model: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    if not response.content:
        raise RuntimeError("API returned empty response content")
    return response.content[0].text


def narrate_batch(
    compression: CompressionFindings,
    pauses: PauseFindings,
    config: Config,
    claude_sessions: list[dict] | None = None,
) -> dict:
    compression_dict = _findings_to_dict(compression)
    pause_dict = _findings_to_dict(pauses)

    prompt = build_batch_prompt(compression_dict, pause_dict, claude_sessions)

    # Build raw findings for preservation
    raw_findings = {
        "timestamp": datetime.now().isoformat(),
        "compression": compression_dict,
        "pauses": pause_dict,
    }

    try:
        response_text = _call_api(config, BATCH_SYSTEM, prompt, config.haiku_model)
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
) -> str:
    changepoint_dict = _findings_to_dict(changepoints) if changepoints else None
    prompt = build_daily_prompt(batch_analyses, changepoint_dict)

    try:
        narrative = _call_api(config, DAILY_SYSTEM, prompt, config.sonnet_model)
    except Exception as e:
        logger.error("Daily summary API call failed: %s", e)
        narrative = (
            f"# Daily Summary (API unavailable)\n\n"
            f"API error: {e}\n\n"
            f"Raw data: {len(batch_analyses)} batch analyses available.\n"
        )

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
