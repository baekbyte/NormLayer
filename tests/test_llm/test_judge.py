"""Tests for LLMJudge — JSON parsing, caching, error handling."""

from __future__ import annotations

import json

import pytest

from normlayer.llm.cache import JudgmentCache
from normlayer.llm.judge import LLMJudge, LLMResponse, _parse_json_response
from normlayer.llm.providers import BaseLLMProvider


# ---------------------------------------------------------------------------
# MockLLMProvider
# ---------------------------------------------------------------------------

class MockLLMProvider(BaseLLMProvider):
    """Mock provider that returns scripted responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self._call_count = 0

    async def async_complete(self, prompt: str, system: str = "") -> str:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return json.dumps({
            "violated": False,
            "violation_score": 0.0,
            "severity": "low",
            "reasoning": "Default mock response",
        })

    def sync_complete(self, prompt: str, system: str = "") -> str:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return json.dumps({
            "violated": False,
            "violation_score": 0.0,
            "severity": "low",
            "reasoning": "Default mock response",
        })


class ErrorProvider(BaseLLMProvider):
    """Provider that always raises."""

    async def async_complete(self, prompt: str, system: str = "") -> str:
        raise RuntimeError("API connection failed")

    def sync_complete(self, prompt: str, system: str = "") -> str:
        raise RuntimeError("API connection failed")


# ---------------------------------------------------------------------------
# JSON parsing tests
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_plain_json(self):
        raw = '{"violated": true, "violation_score": 0.8, "severity": "high", "reasoning": "bad"}'
        result = _parse_json_response(raw)
        assert result["violated"] is True
        assert result["violation_score"] == 0.8

    def test_json_in_code_fence(self):
        raw = '```json\n{"violated": false, "violation_score": 0.1, "severity": "low", "reasoning": "ok"}\n```'
        result = _parse_json_response(raw)
        assert result["violated"] is False

    def test_json_in_bare_code_fence(self):
        raw = '```\n{"violated": true, "violation_score": 0.5, "severity": "medium", "reasoning": "maybe"}\n```'
        result = _parse_json_response(raw)
        assert result["violated"] is True

    def test_json_with_surrounding_text(self):
        raw = 'Here is my analysis:\n{"violated": true, "violation_score": 0.9, "severity": "high", "reasoning": "violation"}\nDone.'
        result = _parse_json_response(raw)
        assert result["violated"] is True
        assert result["violation_score"] == 0.9

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            _parse_json_response("this is not json at all")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            _parse_json_response("")


# ---------------------------------------------------------------------------
# LLMJudge sync tests
# ---------------------------------------------------------------------------

class TestLLMJudgeSync:
    def test_basic_judgment(self):
        provider = MockLLMProvider([json.dumps({
            "violated": True,
            "violation_score": 0.85,
            "severity": "high",
            "reasoning": "Agent revealed internal prompt",
        })])
        judge = LLMJudge(provider=provider)
        result = judge.judge("test prompt")
        assert result.violated is True
        assert result.violation_score == 0.85
        assert result.severity == "high"
        assert "internal prompt" in result.reasoning

    def test_caching_returns_same_result(self):
        provider = MockLLMProvider([json.dumps({
            "violated": False,
            "violation_score": 0.1,
            "severity": "low",
            "reasoning": "ok",
        })])
        judge = LLMJudge(provider=provider)
        result1 = judge.judge("same prompt")
        result2 = judge.judge("same prompt")
        assert result1 is result2
        assert provider._call_count == 1  # Only one actual call

    def test_different_prompts_not_cached(self):
        provider = MockLLMProvider([
            json.dumps({"violated": False, "violation_score": 0.0, "severity": "low", "reasoning": "a"}),
            json.dumps({"violated": True, "violation_score": 0.9, "severity": "high", "reasoning": "b"}),
        ])
        judge = LLMJudge(provider=provider)
        result1 = judge.judge("prompt A")
        result2 = judge.judge("prompt B")
        assert result1.violated is False
        assert result2.violated is True
        assert provider._call_count == 2

    def test_error_returns_safe_default(self):
        judge = LLMJudge(provider=ErrorProvider())
        result = judge.judge("test")
        assert result.violated is False
        assert result.violation_score == 0.0
        assert "failed" in result.reasoning.lower()

    def test_score_clamped_to_0_1(self):
        provider = MockLLMProvider([json.dumps({
            "violated": True,
            "violation_score": 5.0,
            "severity": "high",
            "reasoning": "over range",
        })])
        judge = LLMJudge(provider=provider)
        result = judge.judge("test")
        assert result.violation_score == 1.0

    def test_score_clamped_negative(self):
        provider = MockLLMProvider([json.dumps({
            "violated": False,
            "violation_score": -0.5,
            "severity": "low",
            "reasoning": "under range",
        })])
        judge = LLMJudge(provider=provider)
        result = judge.judge("test")
        assert result.violation_score == 0.0

    def test_missing_fields_use_defaults(self):
        provider = MockLLMProvider(['{"violated": true}'])
        judge = LLMJudge(provider=provider)
        result = judge.judge("test")
        assert result.violated is True
        assert result.violation_score == 0.0
        assert result.severity == "medium"
        assert result.reasoning == ""


# ---------------------------------------------------------------------------
# LLMJudge async tests
# ---------------------------------------------------------------------------

class TestLLMJudgeAsync:
    @pytest.mark.asyncio
    async def test_async_judgment(self):
        provider = MockLLMProvider([json.dumps({
            "violated": True,
            "violation_score": 0.7,
            "severity": "medium",
            "reasoning": "borderline case",
        })])
        judge = LLMJudge(provider=provider)
        result = await judge.async_judge("test prompt")
        assert result.violated is True
        assert result.violation_score == 0.7

    @pytest.mark.asyncio
    async def test_async_caching(self):
        provider = MockLLMProvider([json.dumps({
            "violated": False, "violation_score": 0.0, "severity": "low", "reasoning": "ok",
        })])
        judge = LLMJudge(provider=provider)
        result1 = await judge.async_judge("same")
        result2 = await judge.async_judge("same")
        assert result1 is result2
        assert provider._call_count == 1

    @pytest.mark.asyncio
    async def test_async_error_returns_safe_default(self):
        judge = LLMJudge(provider=ErrorProvider())
        result = await judge.async_judge("test")
        assert result.violated is False
        assert "failed" in result.reasoning.lower()
