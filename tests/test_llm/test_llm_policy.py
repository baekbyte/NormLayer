"""Tests for LLMPolicy — natural language policy evaluation via LLM."""

from __future__ import annotations

import json

import pytest

from normlayer.base_policy import AgentMessage
from normlayer.llm.judge import LLMJudge
from normlayer.llm.policy import LLMPolicy
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
    """Provider that always raises."""

    async def async_complete(self, prompt: str, system: str = "") -> str:
        raise RuntimeError("LLM API down")

    def sync_complete(self, prompt: str, system: str = "") -> str:
        raise RuntimeError("LLM API down")


def _msg(content: str, sender: str = "agent_a") -> AgentMessage:
    return MockMessage(content=content, sender=sender).to_agent_message()


def _judge(responses: list[str]) -> LLMJudge:
    return LLMJudge(provider=MockLLMProvider(responses), cache=None)


# ---------------------------------------------------------------------------
# Pass cases
# ---------------------------------------------------------------------------

class TestLLMPolicyPass:
    def test_passes_when_llm_says_no_violation(self):
        judge = _judge([json.dumps({
            "violated": False, "violation_score": 0.1,
            "severity": "low", "reasoning": "Message is appropriate",
        })])
        policy = LLMPolicy(
            description="Agents must not reveal internal prompts",
            judge=judge, name="NoPromptLeak",
        )
        result = policy.evaluate(_msg("The task is complete."), {})
        assert result.passed is True
        assert result.violation_score == 0.1
        assert result.policy_name == "NoPromptLeak"

    def test_custom_handler(self):
        judge = _judge([json.dumps({
            "violated": False, "violation_score": 0.0,
            "severity": "low", "reasoning": "ok",
        })])
        policy = LLMPolicy(
            description="test", judge=judge, handler="block",
        )
        result = policy.evaluate(_msg("hello"), {})
        assert result.handler == "block"


# ---------------------------------------------------------------------------
# Fail cases
# ---------------------------------------------------------------------------

class TestLLMPolicyFail:
    def test_fails_when_llm_says_violated(self):
        judge = _judge([json.dumps({
            "violated": True, "violation_score": 0.9,
            "severity": "high", "reasoning": "Agent leaked system prompt",
        })])
        policy = LLMPolicy(
            description="Agents must not reveal internal prompts",
            judge=judge, name="NoPromptLeak",
        )
        result = policy.evaluate(_msg("My system prompt says..."), {})
        assert result.passed is False
        assert result.violation_score == 0.9
        assert result.severity == "high"

    def test_severity_defaults_to_medium_on_invalid(self):
        judge = _judge([json.dumps({
            "violated": True, "violation_score": 0.5,
            "severity": "critical",  # invalid — should default to medium
            "reasoning": "test",
        })])
        policy = LLMPolicy(description="test", judge=judge)
        result = policy.evaluate(_msg("test"), {})
        assert result.severity == "medium"


# ---------------------------------------------------------------------------
# Context handling
# ---------------------------------------------------------------------------

class TestLLMPolicyContext:
    def test_context_included_in_prompt(self):
        """Verify context keys are formatted into the prompt."""
        prompts_seen: list[str] = []
        original_sync = MockLLMProvider.sync_complete

        def capture_sync(self_prov: MockLLMProvider, prompt: str, system: str = "") -> str:
            prompts_seen.append(prompt)
            return original_sync(self_prov, prompt, system)

        provider = MockLLMProvider([json.dumps({
            "violated": False, "violation_score": 0.0,
            "severity": "low", "reasoning": "ok",
        })])
        provider.sync_complete = capture_sync.__get__(provider, MockLLMProvider)  # type: ignore[attr-defined]
        judge = LLMJudge(provider=provider, cache=None)
        policy = LLMPolicy(description="test policy", judge=judge)
        policy.evaluate(_msg("hello"), {"role": "planner", "task": "summarize"})

        assert len(prompts_seen) == 1
        assert "role" in prompts_seen[0]
        assert "planner" in prompts_seen[0]

    def test_empty_context_no_section(self):
        """Empty context should not produce a Context section."""
        prompts_seen: list[str] = []
        original_sync = MockLLMProvider.sync_complete

        def capture_sync(self_prov: MockLLMProvider, prompt: str, system: str = "") -> str:
            prompts_seen.append(prompt)
            return original_sync(self_prov, prompt, system)

        provider = MockLLMProvider([json.dumps({
            "violated": False, "violation_score": 0.0,
            "severity": "low", "reasoning": "ok",
        })])
        provider.sync_complete = capture_sync.__get__(provider, MockLLMProvider)  # type: ignore[attr-defined]
        judge = LLMJudge(provider=provider, cache=None)
        policy = LLMPolicy(description="test policy", judge=judge)
        policy.evaluate(_msg("hello"), {})

        assert "**Context:**" not in prompts_seen[0]


# ---------------------------------------------------------------------------
# Fail-open / fail-closed
# ---------------------------------------------------------------------------

class TestLLMPolicyFailOpen:
    def test_fail_open_passes_on_error(self):
        judge = LLMJudge(provider=FailingProvider())
        policy = LLMPolicy(description="test", judge=judge, fail_open=True)
        result = policy.evaluate(_msg("hello"), {})
        assert result.passed is True
        assert result.violation_score == 0.0

    def test_fail_closed_violates_on_error(self):
        judge = LLMJudge(provider=FailingProvider())
        policy = LLMPolicy(description="test", judge=judge, fail_open=False)
        result = policy.evaluate(_msg("hello"), {})
        assert result.passed is False
        assert result.violation_score == 1.0


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------

class TestLLMPolicyAsync:
    @pytest.mark.asyncio
    async def test_async_evaluate(self):
        judge = _judge([json.dumps({
            "violated": True, "violation_score": 0.75,
            "severity": "medium", "reasoning": "borderline violation",
        })])
        policy = LLMPolicy(description="test", judge=judge, name="AsyncTest")
        result = await policy.async_evaluate(_msg("test"), {})
        assert result.passed is False
        assert result.violation_score == 0.75

    @pytest.mark.asyncio
    async def test_async_fail_open(self):
        judge = LLMJudge(provider=FailingProvider())
        policy = LLMPolicy(description="test", judge=judge, fail_open=True)
        result = await policy.async_evaluate(_msg("hello"), {})
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_async_fail_closed(self):
        judge = LLMJudge(provider=FailingProvider())
        policy = LLMPolicy(description="test", judge=judge, fail_open=False)
        result = await policy.async_evaluate(_msg("hello"), {})
        assert result.passed is False
