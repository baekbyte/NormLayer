"""Tests for llm_enhanced() — two-tier heuristic + LLM wrapper."""

from __future__ import annotations

import json
from typing import Any

import pytest

from normlayer.base_policy import AgentMessage, BasePolicy, HandlerType, PolicyResult
from normlayer.llm.enhanced import _EnhancedPolicy, llm_enhanced
from normlayer.llm.judge import LLMJudge
from normlayer.llm.providers import BaseLLMProvider
from normlayer.testing import MockMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockLLMProvider(BaseLLMProvider):
    """Mock provider returning scripted JSON responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self._call_count = 0

    async def async_complete(self, prompt: str, system: str = "") -> str:
        return self._next()

    def sync_complete(self, prompt: str, system: str = "") -> str:
        return self._next()

    def _next(self) -> str:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return json.dumps({
            "violated": False, "violation_score": 0.0,
            "severity": "low", "reasoning": "default",
        })


class FailingProvider(BaseLLMProvider):
    async def async_complete(self, prompt: str, system: str = "") -> str:
        raise RuntimeError("LLM down")

    def sync_complete(self, prompt: str, system: str = "") -> str:
        raise RuntimeError("LLM down")


class FakeHeuristicPolicy(BasePolicy):
    """Heuristic policy that returns a configurable violation score."""

    name: str = "FakeHeuristic"

    def __init__(
        self, score: float, passed: bool | None = None, handler: HandlerType = "warn",
    ) -> None:
        super().__init__(handler=handler)
        self.score = score
        self._passed = passed if passed is not None else (score == 0.0)

    def evaluate(self, message: AgentMessage, context: dict[str, Any]) -> PolicyResult:
        return PolicyResult(
            passed=self._passed,
            violation_score=self.score,
            policy_name=self.name,
            agent_id=message.sender,
            handler=self.handler,
            severity="medium",
            details=f"Heuristic score: {self.score}",
        )


def _msg(content: str = "test message", sender: str = "agent_a") -> AgentMessage:
    return MockMessage(content=content, sender=sender).to_agent_message()


# ---------------------------------------------------------------------------
# Non-borderline: heuristic result used directly
# ---------------------------------------------------------------------------

class TestEnhancedNonBorderline:
    def test_score_below_borderline_uses_heuristic(self):
        """Score 0.1 is below borderline (0.3, 0.8) — LLM not called."""
        heuristic = FakeHeuristicPolicy(score=0.1, passed=True)
        provider = MockLLMProvider()
        judge = LLMJudge(provider=provider, cache=None)
        enhanced = llm_enhanced(heuristic, judge, borderline_range=(0.3, 0.8))

        result = enhanced.evaluate(_msg(), {})
        assert result.passed is True
        assert result.violation_score == 0.1
        assert provider._call_count == 0  # LLM never called
        assert "FakeHeuristic+LLM" in result.policy_name

    def test_score_above_borderline_uses_heuristic(self):
        """Score 0.95 is above borderline (0.3, 0.8) — LLM not called."""
        heuristic = FakeHeuristicPolicy(score=0.95, passed=False)
        provider = MockLLMProvider()
        judge = LLMJudge(provider=provider, cache=None)
        enhanced = llm_enhanced(heuristic, judge, borderline_range=(0.3, 0.8))

        result = enhanced.evaluate(_msg(), {})
        assert result.passed is False
        assert result.violation_score == 0.95
        assert provider._call_count == 0


# ---------------------------------------------------------------------------
# Borderline: LLM invoked
# ---------------------------------------------------------------------------

class TestEnhancedBorderline:
    def test_borderline_calls_llm(self):
        """Score 0.5 is in borderline (0.3, 0.8) — LLM is invoked."""
        heuristic = FakeHeuristicPolicy(score=0.5, passed=False)
        provider = MockLLMProvider([json.dumps({
            "violated": True, "violation_score": 0.8,
            "severity": "high", "reasoning": "LLM confirms violation",
        })])
        judge = LLMJudge(provider=provider, cache=None)
        enhanced = llm_enhanced(heuristic, judge, borderline_range=(0.3, 0.8))

        result = enhanced.evaluate(_msg(), {})
        assert result.passed is False
        assert result.violation_score == 0.8
        assert provider._call_count == 1
        assert "LLM confirms" in result.details

    def test_borderline_llm_overrides_to_pass(self):
        """LLM can override heuristic to pass on borderline case."""
        heuristic = FakeHeuristicPolicy(score=0.5, passed=False)
        provider = MockLLMProvider([json.dumps({
            "violated": False, "violation_score": 0.1,
            "severity": "low", "reasoning": "Actually fine on review",
        })])
        judge = LLMJudge(provider=provider, cache=None)
        enhanced = llm_enhanced(heuristic, judge, borderline_range=(0.3, 0.8))

        result = enhanced.evaluate(_msg(), {})
        assert result.passed is True
        assert result.violation_score == 0.1

    def test_exact_boundary_low(self):
        """Score exactly at lower boundary (0.3) is borderline."""
        heuristic = FakeHeuristicPolicy(score=0.3, passed=True)
        provider = MockLLMProvider([json.dumps({
            "violated": False, "violation_score": 0.1,
            "severity": "low", "reasoning": "ok",
        })])
        judge = LLMJudge(provider=provider, cache=None)
        enhanced = llm_enhanced(heuristic, judge, borderline_range=(0.3, 0.8))

        enhanced.evaluate(_msg(), {})
        assert provider._call_count == 1  # LLM was called

    def test_exact_boundary_high(self):
        """Score exactly at upper boundary (0.8) is borderline."""
        heuristic = FakeHeuristicPolicy(score=0.8, passed=False)
        provider = MockLLMProvider([json.dumps({
            "violated": True, "violation_score": 0.9,
            "severity": "high", "reasoning": "confirmed",
        })])
        judge = LLMJudge(provider=provider, cache=None)
        enhanced = llm_enhanced(heuristic, judge, borderline_range=(0.3, 0.8))

        enhanced.evaluate(_msg(), {})
        assert provider._call_count == 1


# ---------------------------------------------------------------------------
# Handler override
# ---------------------------------------------------------------------------

class TestEnhancedHandler:
    def test_handler_override(self):
        heuristic = FakeHeuristicPolicy(score=0.1, handler="warn")
        provider = MockLLMProvider()
        judge = LLMJudge(provider=provider, cache=None)
        enhanced = llm_enhanced(heuristic, judge, handler="block")

        result = enhanced.evaluate(_msg(), {})
        assert result.handler == "block"

    def test_handler_inherits_from_base(self):
        heuristic = FakeHeuristicPolicy(score=0.1, handler="escalate")
        provider = MockLLMProvider()
        judge = LLMJudge(provider=provider, cache=None)
        enhanced = llm_enhanced(heuristic, judge)

        result = enhanced.evaluate(_msg(), {})
        assert result.handler == "escalate"


# ---------------------------------------------------------------------------
# LLM failure fallback
# ---------------------------------------------------------------------------

class TestEnhancedFallback:
    def test_llm_failure_falls_back_to_heuristic(self):
        """On LLM error, borderline case falls back to heuristic result."""
        heuristic = FakeHeuristicPolicy(score=0.5, passed=False)
        judge = LLMJudge(provider=FailingProvider(), cache=None)
        enhanced = llm_enhanced(heuristic, judge, borderline_range=(0.3, 0.8))

        result = enhanced.evaluate(_msg(), {})
        assert result.passed is False
        assert result.violation_score == 0.5
        assert "fallback" in result.details.lower()


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------

class TestEnhancedAsync:
    @pytest.mark.asyncio
    async def test_async_borderline(self):
        heuristic = FakeHeuristicPolicy(score=0.5, passed=False)
        provider = MockLLMProvider([json.dumps({
            "violated": True, "violation_score": 0.85,
            "severity": "high", "reasoning": "confirmed async",
        })])
        judge = LLMJudge(provider=provider, cache=None)
        enhanced = llm_enhanced(heuristic, judge, borderline_range=(0.3, 0.8))

        result = await enhanced.async_evaluate(_msg(), {})
        assert result.passed is False
        assert result.violation_score == 0.85

    @pytest.mark.asyncio
    async def test_async_non_borderline(self):
        heuristic = FakeHeuristicPolicy(score=0.1, passed=True)
        provider = MockLLMProvider()
        judge = LLMJudge(provider=provider, cache=None)
        enhanced = llm_enhanced(heuristic, judge, borderline_range=(0.3, 0.8))

        result = await enhanced.async_evaluate(_msg(), {})
        assert result.passed is True
        assert provider._call_count == 0
