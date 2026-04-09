"""Token estimation utilities for prompt budgeting."""

# Conservative ratio: ~3.5 chars per token for English text, with 1.15x safety margin.
# Structured/JSON-heavy text tokenizes at higher density (~2.5-3 chars/token);
# the safety margin partially compensates.
_CHARS_PER_TOKEN = 3.5
_SAFETY_MARGIN = 1.15

# Overhead for API message structure (role tags, separators, etc.)
_MESSAGE_OVERHEAD_TOKENS = 50


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string.

    Uses a character-based heuristic with conservative safety margin.
    Not exact, but sufficient for budgeting decisions.
    """
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN * _SAFETY_MARGIN))


def estimate_message_tokens(system: str, prompt: str) -> int:
    """Estimate total token count for a full API call (system + user message + overhead)."""
    return estimate_tokens(system) + estimate_tokens(prompt) + _MESSAGE_OVERHEAD_TOKENS
