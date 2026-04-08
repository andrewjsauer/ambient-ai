"""Shared Anthropic API client for narrator and recommender modules."""

from ambient.config import Config


def call_api(config: Config, system: str, prompt: str, model: str) -> str:
    """Call the Anthropic Messages API and return the response text."""
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
