"""Tests for the LoopDetection policy."""

import pytest

from normlayer.policies.loop_detection import LoopDetection
from normlayer.testing import MockAgent, MockMessage


def _msg(content: str, sender: str = "agent_a") -> MockMessage:
    return MockMessage(content=content, sender=sender)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_history(*contents: str, sender: str = "agent_a") -> list:
    """Return a list of AgentMessages for use as context["history"]."""
    return [MockMessage(content=c, sender=sender).to_agent_message() for c in contents]


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------

class TestLoopDetectionPasses:
    def test_passes_with_no_history(self):
        """Policy should pass when there is no conversation history."""
        policy = LoopDetection()
        msg = _msg("Let me assign task A to the worker.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is True
        assert result.violation_score == 0.0

    def test_passes_with_empty_history_key(self):
        """Policy should pass when history is an empty list."""
        policy = LoopDetection()
        msg = _msg("Let me assign task A.").to_agent_message()
        result = policy.evaluate(msg, context={"history": []})
        assert result.passed is True

    def test_passes_with_dissimilar_messages(self):
        """Policy should pass when the sender's prior messages are all different."""
        policy = LoopDetection(max_repetitions=3, similarity_threshold=0.85)
        history = make_history(
            "I will plan the roadmap.",
            "The budget needs review.",
            "Let's schedule a sync tomorrow.",
        )
        msg = _msg("Task A has been delegated to the executor.").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is True

    def test_passes_below_max_repetitions(self):
        """One or two similar messages should not trigger when threshold is 3."""
        policy = LoopDetection(max_repetitions=3, similarity_threshold=0.85)
        repeated = "The task is complete."
        history = make_history(repeated, repeated)  # 2 similar, threshold is 3
        msg = _msg(repeated).to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is True

    def test_only_checks_same_sender(self):
        """Similar messages from a different sender should not count against this agent."""
        policy = LoopDetection(max_repetitions=2, similarity_threshold=0.85)
        repeated = "The task is complete."
        # Three repetitions, but all from a different sender
        history = make_history(repeated, repeated, repeated, sender="agent_b")
        msg = _msg(repeated, sender="agent_a").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is True


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------

class TestLoopDetectionViolations:
    def test_detects_exact_repetitions(self):
        """Three identical messages from the same sender should trigger a violation."""
        policy = LoopDetection(max_repetitions=3, similarity_threshold=0.85)
        repeated = "I need more information before I can proceed."
        history = make_history(repeated, repeated, repeated)
        msg = _msg(repeated).to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is False
        assert result.violation_score > 0.0
        assert "agent_a" in result.details

    def test_detects_near_duplicate_repetitions(self):
        """Near-identical messages (high similarity) should also trigger."""
        policy = LoopDetection(max_repetitions=2, similarity_threshold=0.85)
        base = "I need more information before proceeding."
        history = make_history(
            "I need more information before proceeding!",
            "I need more information before proceeding...",
        )
        msg = _msg(base).to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is False

    def test_violation_score_is_capped_at_one(self):
        """violation_score must never exceed 1.0."""
        policy = LoopDetection(max_repetitions=2, similarity_threshold=0.85)
        repeated = "Same message over and over."
        history = make_history(*([repeated] * 10))  # 10 identical past messages
        msg = _msg(repeated).to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.violation_score <= 1.0

    def test_handler_is_forwarded(self):
        """The handler on the policy instance should appear in the result."""
        policy = LoopDetection(max_repetitions=1, similarity_threshold=0.5, handler="block")
        repeated = "Stuck in a loop."
        history = make_history(repeated)
        msg = _msg(repeated).to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is False
        assert result.handler == "block"

    def test_policy_name_in_result(self):
        """Result should carry the correct policy name."""
        policy = LoopDetection()
        msg = _msg("hello").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.policy_name == "LoopDetection"


# ---------------------------------------------------------------------------
# Similarity helper
# ---------------------------------------------------------------------------

class TestLoopDetectionSimilarity:
    def test_identical_strings_score_one(self):
        assert LoopDetection._similarity("hello", "hello") == 1.0

    def test_empty_strings_score_one(self):
        assert LoopDetection._similarity("", "") == 1.0

    def test_completely_different_strings_score_near_zero(self):
        score = LoopDetection._similarity("aaa", "zzz")
        assert score < 0.5

    def test_window_size_limits_history_checked(self):
        """Only the most recent `window_size` messages should be considered."""
        policy = LoopDetection(max_repetitions=2, similarity_threshold=0.9, window_size=2)
        repeated = "I am stuck."
        # 10 identical messages in history, but window_size=2 → only 2 checked
        history = make_history(*([repeated] * 10))
        msg = _msg(repeated).to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        # Window of 2 + current message: similar_count = 2 → triggers at max_repetitions=2
        assert result.passed is False
