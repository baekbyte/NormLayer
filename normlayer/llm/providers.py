"""LLM provider abstraction — Anthropic and OpenAI implementations."""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    Subclasses must implement ``async_complete``. A synchronous ``sync_complete``
    wrapper is provided by default using ``asyncio.run()``.
    """

    @abstractmethod
    async def async_complete(self, prompt: str, system: str = "") -> str:
        """Send a prompt to the LLM and return the response text.

        Args:
            prompt: The user/human message text.
            system: Optional system prompt.

        Returns:
            The LLM's response as a string.
        """
        ...

    def sync_complete(self, prompt: str, system: str = "") -> str:
        """Synchronous wrapper around ``async_complete``.

        Args:
            prompt: The user/human message text.
            system: Optional system prompt.

        Returns:
            The LLM's response as a string.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already in an async context — create a new thread to avoid
            # blocking the event loop.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run, self.async_complete(prompt, system)
                ).result()
        return asyncio.run(self.async_complete(prompt, system))


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider.

    Requires the ``anthropic`` package (``pip install normlayer[anthropic]``).
    API key is read from ``api_key`` param or ``ANTHROPIC_API_KEY`` env var.

    Args:
        api_key: Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
        model: Model identifier (default ``"claude-sonnet-4-20250514"``).
        temperature: Sampling temperature (default ``0.0``).
        max_tokens: Maximum response tokens (default ``1024``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Any = None
        self._async_client: Any = None

    def _get_client(self) -> Any:
        """Lazily import and create the synchronous Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "AnthropicProvider requires 'anthropic'. "
                    "Install it with: pip install normlayer[anthropic]"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _get_async_client(self) -> Any:
        """Lazily import and create the async Anthropic client."""
        if self._async_client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "AnthropicProvider requires 'anthropic'. "
                    "Install it with: pip install normlayer[anthropic]"
                ) from exc
            self._async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._async_client

    async def async_complete(self, prompt: str, system: str = "") -> str:
        """Send prompt to Claude and return response text.

        Args:
            prompt: The user message.
            system: Optional system prompt.

        Returns:
            The assistant's text response.
        """
        client = self._get_async_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = await client.messages.create(**kwargs)
        return str(response.content[0].text)

    def sync_complete(self, prompt: str, system: str = "") -> str:
        """Send prompt to Claude synchronously.

        Args:
            prompt: The user message.
            system: Optional system prompt.

        Returns:
            The assistant's text response.
        """
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return str(response.content[0].text)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider.

    Requires the ``openai`` package (``pip install normlayer[openai]``).
    API key is read from ``api_key`` param or ``OPENAI_API_KEY`` env var.

    Args:
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        model: Model identifier (default ``"gpt-4o"``).
        temperature: Sampling temperature (default ``0.0``).
        max_tokens: Maximum response tokens (default ``1024``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Any = None
        self._async_client: Any = None

    def _get_client(self) -> Any:
        """Lazily import and create the synchronous OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError(
                    "OpenAIProvider requires 'openai'. "
                    "Install it with: pip install normlayer[openai]"
                ) from exc
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def _get_async_client(self) -> Any:
        """Lazily import and create the async OpenAI client."""
        if self._async_client is None:
            try:
                import openai
            except ImportError as exc:
                raise ImportError(
                    "OpenAIProvider requires 'openai'. "
                    "Install it with: pip install normlayer[openai]"
                ) from exc
            self._async_client = openai.AsyncOpenAI(api_key=self.api_key)
        return self._async_client

    async def async_complete(self, prompt: str, system: str = "") -> str:
        """Send prompt to OpenAI and return response text.

        Args:
            prompt: The user message.
            system: Optional system prompt.

        Returns:
            The assistant's text response.
        """
        client = self._get_async_client()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        return str(response.choices[0].message.content)

    def sync_complete(self, prompt: str, system: str = "") -> str:
        """Send prompt to OpenAI synchronously.

        Args:
            prompt: The user message.
            system: Optional system prompt.

        Returns:
            The assistant's text response.
        """
        client = self._get_client()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        return str(response.choices[0].message.content)
