"""Tests for token estimation utilities."""

from ambient.present.tokens import estimate_tokens, estimate_message_tokens


def test_estimate_tokens_english_text():
    """100 chars of English -> roughly 33 tokens (100 / 3.5 * 1.15 ≈ 33)."""
    text = "a" * 100
    result = estimate_tokens(text)
    assert 25 <= result <= 45


def test_estimate_tokens_empty_string():
    assert estimate_tokens("") == 0


def test_estimate_tokens_returns_positive_for_any_input():
    assert estimate_tokens("x") >= 1
    assert estimate_tokens("hello world") >= 1


def test_estimate_tokens_long_text_proportional():
    short = estimate_tokens("a" * 100)
    long = estimate_tokens("a" * 100_000)
    # Long text should be roughly 1000x the short text estimate
    ratio = long / short
    assert 900 <= ratio <= 1100


def test_estimate_tokens_structured_text_conservative():
    """JSON/structured text with delimiters should still get a reasonable estimate."""
    structured = '{"key": "value", "count": 42, "items": [1, 2, 3]}'
    result = estimate_tokens(structured)
    # Structured text tokenizes denser, but our estimate should still be > 0
    # and roughly in the right ballpark (real tokens ~20-25 for this)
    assert result >= 10


def test_estimate_tokens_non_ascii():
    """Non-ASCII text should get a conservative estimate."""
    text = "こんにちは世界" * 10  # 70 chars of Japanese
    result = estimate_tokens(text)
    assert result >= 1


def test_estimate_message_tokens_includes_overhead():
    system = "You are a helpful assistant."
    prompt = "Hello!"
    result = estimate_message_tokens(system, prompt)
    # Should be more than just the sum of text estimates due to overhead
    text_only = estimate_tokens(system) + estimate_tokens(prompt)
    assert result > text_only
    assert result == text_only + 50  # exact overhead constant


def test_estimate_message_tokens_empty_inputs():
    result = estimate_message_tokens("", "")
    assert result == 50  # just the overhead


def test_estimate_message_tokens_with_real_prompts():
    """Test against a realistic system prompt size."""
    # Roughly the size of BATCH_SYSTEM (~450 chars)
    system = "x" * 450
    # Roughly a medium batch prompt (~2000 chars)
    prompt = "x" * 2000
    result = estimate_message_tokens(system, prompt)
    # Should be in a reasonable range
    assert 500 < result < 1500
