"""Tests for the ResponseProportionality policy."""

import pytest

from normlayer.base_policy import AgentMessage
from normlayer.policies.response_proportionality import ResponseProportionality
from normlayer.testing import MockMessage


def _msg(content: str, sender: str = "agent_a") -> MockMessage:
    return MockMessage(content=content, sender=sender)


def _trigger(content: str, sender: str = "agent_b") -> AgentMessage:
    """Create a triggering AgentMessage."""
    return AgentMessage(content=content, sender=sender)


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------


class TestResponseProportionalityPasses:
    def test_passes_without_triggering_message(self):
        """Policy should pass when no triggering_message is in context."""
        policy = ResponseProportionality()
        msg = _msg("A response.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is True
        assert result.violation_score == 0.0

    def test_passes_with_zero_length_trigger(self):
        """Policy should pass when the triggering message is empty."""
        policy = ResponseProportionality()
        msg = _msg("Some response.").to_agent_message()
        result = policy.evaluate(msg, context={"triggering_message": _trigger("")})
        assert result.passed is True

    def test_passes_within_default_bounds(self):
        """A response within ratio 0.1–5.0 should pass."""
        policy = ResponseProportionality()
        trigger = _trigger("Hello, how are you?")  # 19 chars
        msg = _msg("I am doing fine, thanks!").to_agent_message()  # 24 chars
        result = policy.evaluate(msg, context={"triggering_message": trigger})
        assert result.passed is True

    def test_passes_at_exact_max_boundary(self):
        """A response at exactly max_ratio should pass."""
        policy = ResponseProportionality(max_ratio=5.0)
        trigger = _trigger("ab")  # 2 chars
        msg = _msg("a" * 10).to_agent_message()  # ratio = 5.0 exactly
        result = policy.evaluate(msg, context={"triggering_message": trigger})
        assert result.passed is True

    def test_passes_at_exact_min_boundary(self):
        """A response at exactly min_ratio should pass."""
        policy = ResponseProportionality(min_ratio=0.1)
        trigger = _trigger("a" * 100)  # 100 chars
        msg = _msg("a" * 10).to_agent_message()  # ratio = 0.1 exactly
        result = policy.evaluate(msg, context={"triggering_message": trigger})
        assert result.passed is True

    def test_passes_with_custom_ratios(self):
        """Custom max/min ratios should be respected."""
        policy = ResponseProportionality(max_ratio=10.0, min_ratio=0.01)
        trigger = _trigger("hi")  # 2 chars
        msg = _msg("a" * 20).to_agent_message()  # ratio = 10.0
        result = policy.evaluate(msg, context={"triggering_message": trigger})
        assert result.passed is True

    def test_passes_equal_length(self):
        """Identical-length messages should pass (ratio = 1.0)."""
        policy = ResponseProportionality()
        trigger = _trigger("hello")
        msg = _msg("world").to_agent_message()
        result = policy.evaluate(msg, context={"triggering_message": trigger})
        assert result.passed is True


# ---------------------------------------------------------------------------
# Violations — too long
# ---------------------------------------------------------------------------


class TestResponseProportionalityTooLong:
    def test_violation_too_long(self):
        """A response exceeding max_ratio should be flagged."""
        policy = ResponseProportionality(max_ratio=5.0)
        trigger = _trigger("hi")  # 2 chars
        msg = _msg("a" * 11).to_agent_message()  # ratio = 5.5
        result = policy.evaluate(msg, context={"triggering_message": trigger})
        assert result.passed is False
        assert result.violation_score > 0.0
        assert "long" in result.details

    def test_high_severity_when_ratio_exceeds_double_max(self):
        """Severity should be 'high' when ratio > 2 * max_ratio."""
        policy = ResponseProportionality(max_ratio=2.0)
        trigger = _trigger("hi")  # 2 chars
        msg = _msg("a" * 10).to_agent_message()  # ratio = 5.0 > 2*2.0
        result = policy.evaluate(msg, context={"triggering_message": trigger})
        assert result.passed is False
        assert result.severity == "high"

    def test_medium_severity_just_over_max(self):
        """Severity should be 'medium' when ratio is just over max."""
        policy = ResponseProportionality(max_ratio=5.0)
        trigger = _trigger("hi")  # 2 chars
        msg = _msg("a" * 11).to_agent_message()  # ratio = 5.5, < 10.0
        result = policy.evaluate(msg, context={"triggering_message": trigger})
        assert result.passed is False
        assert result.severity == "medium"

    def test_violation_score_capped_at_one(self):
        """Violation score should never exceed 1.0."""
        policy = ResponseProportionality(max_ratio=1.0)
        trigger = _trigger("a")
        msg = _msg("a" * 1000).to_agent_message()  # huge ratio
        result = policy.evaluate(msg, context={"triggering_message": trigger})
        assert result.violation_score <= 1.0


# ---------------------------------------------------------------------------
# Violations — too short
# ---------------------------------------------------------------------------


class TestResponseProportionalityTooShort:
    def test_violation_too_short(self):
        """A response below min_ratio should be flagged."""
        policy = ResponseProportionality(min_ratio=0.1)
        trigger = _trigger("a" * 100)  # 100 chars
        msg = _msg("a" * 5).to_agent_message()  # ratio = 0.05
        result = policy.evaluate(msg, context={"triggering_message": trigger})
        assert result.passed is False
        assert "short" in result.details

    def test_empty_response_violates(self):
        """An empty response to a non-empty trigger should violate min_ratio."""
        policy = ResponseProportionality(min_ratio=0.1)
        trigger = _trigger("Please provide a detailed analysis.")
        msg = _msg("").to_agent_message()  # ratio = 0.0
        result = policy.evaluate(msg, context={"triggering_message": trigger})
        assert result.passed is False
        assert result.violation_score > 0.0

    def test_high_severity_when_ratio_below_half_min(self):
        """Severity should be 'high' when ratio < min_ratio / 2."""
        policy = ResponseProportionality(min_ratio=0.5)
        trigger = _trigger("a" * 100)
        msg = _msg("a" * 10).to_agent_message()  # ratio = 0.1 < 0.25
        result = policy.evaluate(msg, context={"triggering_message": trigger})
        assert result.passed is False
        assert result.severity == "high"

    def test_medium_severity_just_under_min(self):
        """Severity should be 'medium' when ratio is just under min_ratio."""
        policy = ResponseProportionality(min_ratio=0.5)
        trigger = _trigger("a" * 100)
        msg = _msg("a" * 40).to_agent_message()  # ratio = 0.4, > 0.25
        result = policy.evaluate(msg, context={"triggering_message": trigger})
        assert result.passed is False
        assert result.severity == "medium"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestResponseProportionalityMetadata:
    def test_policy_name_in_result(self):
        """Result should carry the correct policy name."""
        policy = ResponseProportionality()
        msg = _msg("hello").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.policy_name == "ResponseProportionality"

    def test_handler_default_is_warn(self):
        """Default handler should be 'warn'."""
        policy = ResponseProportionality()
        msg = _msg("hello").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.handler == "warn"

    def test_handler_is_forwarded(self):
        """Custom handler should appear in the result."""
        policy = ResponseProportionality(max_ratio=1.0, handler="block")
        trigger = _trigger("a")
        msg = _msg("aa").to_agent_message()  # ratio = 2.0 > 1.0
        result = policy.evaluate(msg, context={"triggering_message": trigger})
        assert result.handler == "block"

    def test_agent_id_in_result(self):
        """Result should carry the correct agent_id."""
        policy = ResponseProportionality()
        msg = _msg("hello", sender="my_agent").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.agent_id == "my_agent"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestResponseProportionalityEdge:
    def test_single_char_messages(self):
        """Single character trigger and response should work."""
        policy = ResponseProportionality()
        trigger = _trigger("x")
        msg = _msg("y").to_agent_message()  # ratio = 1.0
        result = policy.evaluate(msg, context={"triggering_message": trigger})
        assert result.passed is True
