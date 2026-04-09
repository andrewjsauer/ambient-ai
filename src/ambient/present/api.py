"""Shared Anthropic API client for narrator and recommender modules."""

import logging

from ambient.config import Config

logger = logging.getLogger(__name__)


def call_api(
    config: Config,
    system: str,
    prompt: str,
    model: str,
    max_tokens: int = 2048,
    client=None,
) -> str:
    """Call the Anthropic Messages API and return the response text.

    Args:
        config: Application configuration.
        system: System prompt text (converted to content-block format internally).
        prompt: User message text.
        model: Model identifier.
        max_tokens: Maximum output tokens (varies by call type).
        client: Optional pre-created anthropic.Anthropic instance for reuse.
                If None, creates a new client with retry enabled.
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    if client is None:
        client = anthropic.Anthropic(max_retries=3)

    # Use content-block format for system message, enabling future prompt caching.
    # cache_control is included unconditionally; the API ignores it when the block
    # is below the minimum cache threshold (1024 tokens for Sonnet, 2048 for Haiku).
    system_blocks = [
        {
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_blocks,
        messages=[{"role": "user", "content": prompt}],
    )
    if not response.content:
        raise RuntimeError("API returned empty response content")

    # Structured token usage logging
    usage = response.usage
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    logger.info(
        "TOKEN_USAGE model=%s input=%d output=%d cache_read=%d cache_write=%d",
        model,
        usage.input_tokens,
        usage.output_tokens,
        cache_read,
        cache_write,
    )

    return response.content[0].text
