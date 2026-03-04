"""LLMJudge — shared LLM evaluation engine for policy judgments."""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from typing import Any

from normlayer.llm.cache import JudgmentCache
from normlayer.llm.prompts import JUDGE_SYSTEM_PROMPT
from normlayer.llm.providers import BaseLLMProvider


@dataclass
class LLMResponse:
    """Parsed response from an LLM judge evaluation.

    Attributes:
        violated: Whether the policy was violated.
        violation_score: Confidence score in [0, 1].
        severity: Severity level of the violation.
        reasoning: Brief explanation from the LLM.
        raw: The raw response string from the LLM.
    """

    violated: bool
    violation_score: float
    severity: str
    reasoning: str
    raw: str


def _parse_json_response(text: str) -> dict[str, Any]:
    """Parse JSON from LLM response, handling code fences and extra text.

    Tries in order:
    1. Direct ``json.loads`` on the full text.
    2. Extract from markdown code fences (```json ... ``` or ``` ... ```).
    3. Find the first ``{...}`` block.

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If no valid JSON can be extracted.
    """
    text = text.strip()

    # Try direct parse
    try:
        return dict(json.loads(text))
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            return dict(json.loads(fence_match.group(1).strip()))
        except (json.JSONDecodeError, TypeError):
            pass

    # Try finding first {...} block
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        try:
            return dict(json.loads(brace_match.group(0)))
        except (json.JSONDecodeError, TypeError):
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")


class LLMJudge:
    """Shared LLM evaluation engine for NormLayer policies.

    Handles prompt formatting, LLM invocation, JSON response parsing,
    and optional caching. Not a policy itself — consumed by ``LLMPolicy``
    and ``llm_enhanced()``.

    Args:
        provider: An LLM provider instance (e.g. ``AnthropicProvider``).
        cache: Optional ``JudgmentCache`` instance. Pass ``None`` to disable.
        system_prompt: Override the default judge system prompt.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        cache: JudgmentCache | None = None,
        system_prompt: str = JUDGE_SYSTEM_PROMPT,
    ) -> None:
        self.provider = provider
        self.cache = cache if cache is not None else JudgmentCache()
        self.system_prompt = system_prompt

    def _parse_response(self, raw: str) -> LLMResponse:
        """Parse raw LLM output into a structured LLMResponse.

        Args:
            raw: Raw string from the LLM provider.

        Returns:
            Parsed LLMResponse.

        Raises:
            ValueError: If JSON parsing fails.
        """
        data = _parse_json_response(raw)
        return LLMResponse(
            violated=bool(data.get("violated", False)),
            violation_score=float(min(1.0, max(0.0, data.get("violation_score", 0.0)))),
            severity=str(data.get("severity", "medium")),
            reasoning=str(data.get("reasoning", "")),
            raw=raw,
        )

    def judge(self, prompt: str) -> LLMResponse:
        """Evaluate a prompt synchronously with caching.

        Args:
            prompt: The formatted evaluation prompt.

        Returns:
            Parsed LLMResponse.
        """
        cached = self.cache.get(prompt)
        if isinstance(cached, LLMResponse):
            return cached

        try:
            raw = self.provider.sync_complete(prompt, system=self.system_prompt)
            response = self._parse_response(raw)
        except Exception as exc:
            warnings.warn(
                f"LLMJudge evaluation failed: {exc}. Returning non-violation default.",
                stacklevel=2,
            )
            response = LLMResponse(
                violated=False,
                violation_score=0.0,
                severity="low",
                reasoning=f"LLM evaluation failed: {exc}",
                raw="",
            )

        self.cache.put(prompt, response)
        return response

    async def async_judge(self, prompt: str) -> LLMResponse:
        """Evaluate a prompt asynchronously with caching.

        Args:
            prompt: The formatted evaluation prompt.

        Returns:
            Parsed LLMResponse.
        """
        cached = self.cache.get(prompt)
        if isinstance(cached, LLMResponse):
            return cached

        try:
            raw = await self.provider.async_complete(prompt, system=self.system_prompt)
            response = self._parse_response(raw)
        except Exception as exc:
            warnings.warn(
                f"LLMJudge evaluation failed: {exc}. Returning non-violation default.",
                stacklevel=2,
            )
            response = LLMResponse(
                violated=False,
                violation_score=0.0,
                severity="low",
                reasoning=f"LLM evaluation failed: {exc}",
                raw="",
            )

        self.cache.put(prompt, response)
        return response
