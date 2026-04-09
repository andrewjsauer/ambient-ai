"""Tests for the enhanced API client."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from ambient.config import Config
from ambient.present.api import call_api


@pytest.fixture
def config():
    return Config()


def _mock_response(text="Hello", input_tokens=100, output_tokens=50):
    """Create a mock API response with usage data."""
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    response.usage = MagicMock(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    # Simulate missing cache fields (common for non-cached calls)
    response.usage.cache_read_input_tokens = 0
    response.usage.cache_creation_input_tokens = 0
    return response


def test_call_api_returns_text(config):
    client = MagicMock()
    client.messages.create.return_value = _mock_response("test output")

    result = call_api(config, "system prompt", "user prompt", "model-id", client=client)

    assert result == "test output"


def test_call_api_uses_content_block_system_format(config):
    client = MagicMock()
    client.messages.create.return_value = _mock_response()

    call_api(config, "You are helpful.", "Hello", "model-id", client=client)

    call_args = client.messages.create.call_args
    system_arg = call_args.kwargs["system"]

    # System should be a list of content blocks, not a string
    assert isinstance(system_arg, list)
    assert len(system_arg) == 1
    assert system_arg[0]["type"] == "text"
    assert system_arg[0]["text"] == "You are helpful."
    assert system_arg[0]["cache_control"] == {"type": "ephemeral"}


def test_call_api_passes_max_tokens(config):
    client = MagicMock()
    client.messages.create.return_value = _mock_response()

    call_api(config, "sys", "prompt", "model-id", max_tokens=4000, client=client)

    call_args = client.messages.create.call_args
    assert call_args.kwargs["max_tokens"] == 4000


def test_call_api_default_max_tokens(config):
    client = MagicMock()
    client.messages.create.return_value = _mock_response()

    call_api(config, "sys", "prompt", "model-id", client=client)

    call_args = client.messages.create.call_args
    assert call_args.kwargs["max_tokens"] == 2048


def test_call_api_reuses_provided_client(config):
    client = MagicMock()
    client.messages.create.return_value = _mock_response()

    call_api(config, "sys", "prompt", "model-id", client=client)
    call_api(config, "sys", "prompt2", "model-id", client=client)

    assert client.messages.create.call_count == 2


def test_call_api_creates_client_with_retries_when_none(config):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_response()

    with patch("anthropic.Anthropic", return_value=mock_client) as mock_cls:
        call_api(config, "sys", "prompt", "model-id")
        mock_cls.assert_called_once_with(max_retries=3)


def test_call_api_logs_token_usage(config, caplog):
    client = MagicMock()
    response = _mock_response(input_tokens=150, output_tokens=75)
    client.messages.create.return_value = response

    with caplog.at_level(logging.INFO, logger="ambient.present.api"):
        call_api(config, "sys", "prompt", "test-model", client=client)

    assert any("TOKEN_USAGE" in record.message for record in caplog.records)
    usage_record = next(r for r in caplog.records if "TOKEN_USAGE" in r.message)
    assert "model=test-model" in usage_record.message
    assert "input=150" in usage_record.message
    assert "output=75" in usage_record.message


def test_call_api_logs_cache_fields_when_present(config, caplog):
    client = MagicMock()
    response = _mock_response()
    response.usage.cache_read_input_tokens = 500
    response.usage.cache_creation_input_tokens = 200
    client.messages.create.return_value = response

    with caplog.at_level(logging.INFO, logger="ambient.present.api"):
        call_api(config, "sys", "prompt", "model-id", client=client)

    usage_record = next(r for r in caplog.records if "TOKEN_USAGE" in r.message)
    assert "cache_read=500" in usage_record.message
    assert "cache_write=200" in usage_record.message


def test_call_api_handles_missing_cache_fields(config, caplog):
    """When cache fields are missing from usage, log 0 without crashing."""
    client = MagicMock()
    response = _mock_response()
    # Simulate missing attributes
    del response.usage.cache_read_input_tokens
    del response.usage.cache_creation_input_tokens
    client.messages.create.return_value = response

    with caplog.at_level(logging.INFO, logger="ambient.present.api"):
        call_api(config, "sys", "prompt", "model-id", client=client)

    usage_record = next(r for r in caplog.records if "TOKEN_USAGE" in r.message)
    assert "cache_read=0" in usage_record.message
    assert "cache_write=0" in usage_record.message


def test_call_api_raises_on_empty_response(config):
    client = MagicMock()
    response = MagicMock()
    response.content = []
    client.messages.create.return_value = response

    with pytest.raises(RuntimeError, match="empty response content"):
        call_api(config, "sys", "prompt", "model-id", client=client)


def test_call_api_propagates_api_errors(config):
    client = MagicMock()
    client.messages.create.side_effect = Exception("rate limited")

    with pytest.raises(Exception, match="rate limited"):
        call_api(config, "sys", "prompt", "model-id", client=client)
