"""Tests for the CoalitionConsistency policy."""

import pytest

from normlayer.base_policy import AgentMessage
from normlayer.policies.coalition_consistency import CoalitionConsistency
from normlayer.testing import MockMessage


def _msg(
    content: str,
    sender: str = "agent_a",
    recipient: str | None = None,
) -> MockMessage:
    return MockMessage(content=content, sender=sender, recipient=recipient)


def _agent_msg(
    content: str,
    sender: str = "agent_a",
    recipient: str | None = None,
) -> AgentMessage:
    """Create an AgentMessage directly (used for history entries)."""
    return AgentMessage(content=content, sender=sender, recipient=recipient)


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------


class TestCoalitionConsistencyPasses:
    def test_passes_with_no_coalitions(self):
        """Policy should pass when no coalitions are defined."""
        policy = CoalitionConsistency()
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is True
        assert result.violation_score == 0.0

    def test_passes_with_no_history(self):
        """Policy should pass when history is absent."""
        policy = CoalitionConsistency(
            coalitions={"team": ["agent_a", "agent_b"]}
        )
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is True

    def test_passes_with_empty_history(self):
        """Policy should pass when history is empty."""
        policy = CoalitionConsistency(
            coalitions={"team": ["agent_a", "agent_b"]}
        )
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={"history": []})
        assert result.passed is True

    def test_passes_sender_not_in_any_coalition(self):
        """Policy should pass when sender is not in any coalition."""
        policy = CoalitionConsistency(
            coalitions={"team": ["agent_b", "agent_c"]}
        )
        msg = _msg("Hello", sender="agent_a").to_agent_message()
        history = [
            _agent_msg("Hi there", sender="agent_a", recipient="agent_b"),
        ]
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is True

    def test_passes_only_in_group_messages(self):
        """Policy should pass when sender only has in-group messages (can't compare)."""
        policy = CoalitionConsistency(
            coalitions={"team": ["agent_a", "agent_b"]}
        )
        history = [
            _agent_msg("Great job!", sender="agent_a", recipient="agent_b"),
            _agent_msg("Well done!", sender="agent_a", recipient="agent_b"),
        ]
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is True

    def test_passes_only_out_group_messages(self):
        """Policy should pass when sender only has out-group messages."""
        policy = CoalitionConsistency(
            coalitions={"team": ["agent_a", "agent_b"]}
        )
        history = [
            _agent_msg("Hello outsider", sender="agent_a", recipient="agent_c"),
        ]
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is True

    def test_passes_consistent_behavior(self):
        """Policy should pass when agent treats both groups similarly."""
        policy = CoalitionConsistency(
            coalitions={"team": ["agent_a", "agent_b"]},
            consistency_threshold=0.5,
        )
        history = [
            _agent_msg("Good work on the task", sender="agent_a", recipient="agent_b"),
            _agent_msg("Good work on the task", sender="agent_a", recipient="agent_c"),
        ]
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is True

    def test_passes_with_context_override(self):
        """Runtime context coalitions should override constructor coalitions."""
        policy = CoalitionConsistency(
            coalitions={"team": ["agent_a", "agent_b"]}
        )
        # Override with empty coalitions → should pass
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={"coalitions": {}})
        assert result.passed is True

    def test_passes_no_recipient_messages_ignored(self):
        """Messages without a recipient should not be classified."""
        policy = CoalitionConsistency(
            coalitions={"team": ["agent_a", "agent_b"]}
        )
        history = [
            _agent_msg("I agree", sender="agent_a", recipient=None),
            _agent_msg("I disagree", sender="agent_a", recipient=None),
        ]
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is True


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------


class TestCoalitionConsistencyViolations:
    def test_violation_length_disparity(self):
        """Large length difference between groups should trigger violation."""
        policy = CoalitionConsistency(
            coalitions={"team": ["agent_a", "agent_b"]},
            consistency_threshold=0.3,
        )
        history = [
            # Long messages to in-group
            _agent_msg("a" * 200, sender="agent_a", recipient="agent_b"),
            # Short messages to out-group
            _agent_msg("a" * 10, sender="agent_a", recipient="agent_c"),
        ]
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is False
        assert result.violation_score > 0.0

    def test_violation_sentiment_disparity(self):
        """Different sentiment toward groups should trigger violation."""
        policy = CoalitionConsistency(
            coalitions={"team": ["agent_a", "agent_b"]},
            consistency_threshold=0.3,
        )
        # Same-ish length but very different sentiment
        history = [
            _agent_msg("Great job, I agree with you completely", sender="agent_a", recipient="agent_b"),
            _agent_msg("Great job, I agree with you completely", sender="agent_a", recipient="agent_b"),
            _agent_msg("This is a neutral statement here ok", sender="agent_a", recipient="agent_c"),
            _agent_msg("This is a neutral statement here ok", sender="agent_a", recipient="agent_c"),
        ]
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is False

    def test_violation_combined_disparity(self):
        """Both length and sentiment disparity should trigger."""
        policy = CoalitionConsistency(
            coalitions={"team": ["agent_a", "agent_b"]},
            consistency_threshold=0.2,
        )
        history = [
            _agent_msg("Great job, I approve of this good work!", sender="agent_a", recipient="agent_b"),
            _agent_msg("No", sender="agent_a", recipient="agent_c"),
        ]
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is False

    def test_violation_custom_threshold(self):
        """Lower threshold should catch smaller disparities."""
        policy = CoalitionConsistency(
            coalitions={"team": ["agent_a", "agent_b"]},
            consistency_threshold=0.1,
        )
        history = [
            _agent_msg("I agree with this plan for the task", sender="agent_a", recipient="agent_b"),
            _agent_msg("Here is my analysis of the task now", sender="agent_a", recipient="agent_c"),
        ]
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is False


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestCoalitionConsistencyMetadata:
    def test_policy_name_in_result(self):
        """Result should carry the correct policy name."""
        policy = CoalitionConsistency()
        msg = _msg("hello").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.policy_name == "CoalitionConsistency"

    def test_default_handler_is_warn(self):
        """Default handler should be 'warn'."""
        policy = CoalitionConsistency()
        msg = _msg("hello").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.handler == "warn"

    def test_handler_is_forwarded(self):
        """Custom handler should appear in the result."""
        policy = CoalitionConsistency(handler="block")
        msg = _msg("hello").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.handler == "block"

    def test_high_severity_above_double_threshold(self):
        """Severity should be 'high' when discrepancy > 2 * threshold."""
        policy = CoalitionConsistency(
            coalitions={"team": ["agent_a", "agent_b"]},
            consistency_threshold=0.1,
        )
        history = [
            _agent_msg("a" * 200, sender="agent_a", recipient="agent_b"),
            _agent_msg("a" * 10, sender="agent_a", recipient="agent_c"),
        ]
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is False
        assert result.severity == "high"

    def test_medium_severity_just_over_threshold(self):
        """Severity should be 'medium' when discrepancy is just over threshold."""
        policy = CoalitionConsistency(
            coalitions={"team": ["agent_a", "agent_b"]},
            consistency_threshold=0.5,
        )
        # Create a modest length disparity just over 0.5
        history = [
            _agent_msg("a" * 100, sender="agent_a", recipient="agent_b"),
            _agent_msg("a" * 30, sender="agent_a", recipient="agent_c"),
        ]
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is False
        assert result.severity == "medium"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestCoalitionConsistencyEdge:
    def test_agent_in_multiple_coalitions(self):
        """Agent in multiple coalitions should have all co-members as in-group."""
        policy = CoalitionConsistency(
            coalitions={
                "team_1": ["agent_a", "agent_b"],
                "team_2": ["agent_a", "agent_c"],
            },
            consistency_threshold=0.3,
        )
        # Both agent_b and agent_c are in-group; agent_d is out-group
        history = [
            _agent_msg("Good job, I agree!", sender="agent_a", recipient="agent_b"),
            _agent_msg("Good job, I agree!", sender="agent_a", recipient="agent_c"),
            _agent_msg("Good job, I agree!", sender="agent_a", recipient="agent_d"),
        ]
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        # Consistent across all groups → should pass
        assert result.passed is True

    def test_single_member_coalition(self):
        """A coalition with only the sender has no in-group peers."""
        policy = CoalitionConsistency(
            coalitions={"solo": ["agent_a"]}
        )
        history = [
            _agent_msg("Hello there", sender="agent_a", recipient="agent_b"),
        ]
        msg = _msg("Hello").to_agent_message()
        # No in-group (only sender), so all messages are out-group, can't compare
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is True

    def test_empty_content_messages(self):
        """Empty messages should not cause errors."""
        policy = CoalitionConsistency(
            coalitions={"team": ["agent_a", "agent_b"]},
            consistency_threshold=0.5,
        )
        history = [
            _agent_msg("", sender="agent_a", recipient="agent_b"),
            _agent_msg("", sender="agent_a", recipient="agent_c"),
        ]
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is True

    def test_violation_score_capped_at_one(self):
        """Violation score should never exceed 1.0."""
        policy = CoalitionConsistency(
            coalitions={"team": ["agent_a", "agent_b"]},
            consistency_threshold=0.01,
        )
        history = [
            _agent_msg("a" * 1000, sender="agent_a", recipient="agent_b"),
            _agent_msg("a", sender="agent_a", recipient="agent_c"),
        ]
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.violation_score <= 1.0

    def test_messages_from_other_senders_ignored(self):
        """Only the evaluated sender's messages should be analyzed."""
        policy = CoalitionConsistency(
            coalitions={"team": ["agent_a", "agent_b"]},
            consistency_threshold=0.3,
        )
        history = [
            # Different sender's messages should be ignored
            _agent_msg("a" * 200, sender="agent_b", recipient="agent_a"),
            _agent_msg("a" * 5, sender="agent_b", recipient="agent_c"),
            # agent_a's consistent messages
            _agent_msg("hello there!", sender="agent_a", recipient="agent_b"),
            _agent_msg("hello there!", sender="agent_a", recipient="agent_c"),
        ]
        msg = _msg("Hello").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is True
