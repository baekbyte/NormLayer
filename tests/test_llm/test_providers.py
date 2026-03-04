"""Tests for LLM providers — lazy imports, env var fallback, sync wrapper."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from normlayer.llm.providers import AnthropicProvider, BaseLLMProvider, OpenAIProvider


# ---------------------------------------------------------------------------
# MockProvider for testing BaseLLMProvider
# ---------------------------------------------------------------------------

class MockProvider(BaseLLMProvider):
    """Concrete provider for testing the base class sync wrapper."""

    def __init__(self, response: str = "mock response") -> None:
        self.response = response

    async def async_complete(self, prompt: str, system: str = "") -> str:
        return self.response


class TestBaseLLMProvider:
    def test_sync_complete_calls_async(self):
        provider = MockProvider(response="hello")
        result = provider.sync_complete("test prompt")
        assert result == "hello"

    def test_sync_complete_with_system(self):
        provider = MockProvider(response="sys response")
        result = provider.sync_complete("test", system="be helpful")
        assert result == "sys response"

    @pytest.mark.asyncio
    async def test_async_complete(self):
        provider = MockProvider(response="async hello")
        result = await provider.async_complete("test")
        assert result == "async hello"


class TestAnthropicProvider:
    def test_api_key_from_param(self):
        provider = AnthropicProvider(api_key="test-key")
        assert provider.api_key == "test-key"

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            provider = AnthropicProvider()
            assert provider.api_key == "env-key"

    def test_lazy_import_error(self):
        provider = AnthropicProvider(api_key="test")
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="AnthropicProvider requires"):
                provider._get_client()


class TestOpenAIProvider:
    def test_api_key_from_param(self):
        provider = OpenAIProvider(api_key="test-key")
        assert provider.api_key == "test-key"

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            provider = OpenAIProvider()
            assert provider.api_key == "env-key"

    def test_lazy_import_error(self):
        provider = OpenAIProvider(api_key="test")
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="OpenAIProvider requires"):
                provider._get_client()
