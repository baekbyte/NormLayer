"""Tests for the EscalateOnConflict policy."""

import pytest

from normlayer.policies.escalate_on_conflict import EscalateOnConflict
from normlayer.testing import MockMessage


def _msg(content: str, sender: str = "agent_a") -> MockMessage:
    return MockMessage(content=content, sender=sender)


def make_history(
    *contents: str, sender: str = "agent_a"
) -> list:
    """Return a list of AgentMessages for use as context['history']."""
    return [MockMessage(content=c, sender=sender).to_agent_message() for c in contents]


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------


class TestEscalateOnConflictPasses:
    def test_passes_with_no_history(self):
        """Policy should pass when there is no conversation history."""
        policy = EscalateOnConflict()
        msg = _msg("Sounds good to me.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is True
        assert result.violation_score == 0.0

    def test_passes_with_empty_history(self):
        """Policy should pass when history is empty."""
        policy = EscalateOnConflict()
        msg = _msg("Let's proceed.").to_agent_message()
        result = policy.evaluate(msg, context={"history": []})
        assert result.passed is True

    def test_passes_with_agreeable_history(self):
        """Non-conflicting messages should not trigger."""
        policy = EscalateOnConflict(conflict_threshold=3)
        history = make_history(
            "Sounds good to me.",
            "I agree with the plan.",
            "Let's move forward.",
        )
        msg = _msg("Great work everyone.").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is True

    def test_passes_below_threshold(self):
        """Fewer conflicts than threshold should pass."""
        policy = EscalateOnConflict(conflict_threshold=3)
        history = make_history(
            "I disagree with that approach.",
            "That seems fine to me.",
        )
        msg = _msg("Let me think about it.").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is True

    def test_conflicts_from_other_sender_ignored(self):
        """Conflicts from a different sender should not count."""
        policy = EscalateOnConflict(conflict_threshold=2)
        history = make_history(
            "I disagree completely.",
            "You are wrong about this.",
            "That is incorrect.",
            sender="agent_b",
        )
        msg = _msg("Sounds good.").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is True

    def test_passes_with_empty_content(self):
        """Empty messages should not be counted as conflicting."""
        policy = EscalateOnConflict(conflict_threshold=2)
        history = make_history("", "", "")
        msg = _msg("").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is True


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------


class TestEscalateOnConflictViolations:
    def test_violation_at_threshold(self):
        """Exactly threshold conflicts should trigger violation."""
        policy = EscalateOnConflict(conflict_threshold=3)
        history = make_history(
            "I disagree with this.",
            "That is wrong.",
        )
        # Current message adds a third conflict
        msg = _msg("I reject this proposal.").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is False
        assert result.violation_score > 0.0

    def test_violation_above_threshold(self):
        """More than threshold conflicts should trigger."""
        policy = EscalateOnConflict(conflict_threshold=2)
        history = make_history(
            "I disagree.",
            "You are wrong.",
            "That is invalid.",
        )
        msg = _msg("I oppose this.").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is False

    def test_current_message_pushes_over(self):
        """A non-conflicting history + conflicting current message at threshold=1."""
        policy = EscalateOnConflict(conflict_threshold=1)
        msg = _msg("I disagree with everything.").to_agent_message()
        result = policy.evaluate(msg, context={"history": []})
        assert result.passed is False

    def test_phrase_detection(self):
        """Conflict phrases (multi-word) should be detected."""
        policy = EscalateOnConflict(conflict_threshold=1)
        msg = _msg("I think that is not what we agreed on.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is False

    def test_mixed_history_hits_threshold(self):
        """Mix of conflicting and non-conflicting messages from the sender."""
        policy = EscalateOnConflict(conflict_threshold=3)
        history = make_history(
            "I agree with the plan.",
            "That is wrong, we need a new approach.",
            "Sounds good.",
            "I disagree strongly.",
        )
        msg = _msg("I oppose this decision.").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.passed is False

    def test_keyword_case_insensitivity(self):
        """Conflict keywords should be matched case-insensitively."""
        policy = EscalateOnConflict(conflict_threshold=1)
        msg = _msg("I DISAGREE with this approach.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is False

    def test_violation_score_capped_at_one(self):
        """Violation score should never exceed 1.0."""
        policy = EscalateOnConflict(conflict_threshold=1)
        history = make_history(
            "I disagree.",
            "You are wrong.",
            "That is invalid.",
            "I oppose.",
            "I reject.",
        )
        msg = _msg("I refuse to accept.").to_agent_message()
        result = policy.evaluate(msg, context={"history": history})
        assert result.violation_score <= 1.0


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestEscalateOnConflictMetadata:
    def test_policy_name_in_result(self):
        """Result should carry the correct policy name."""
        policy = EscalateOnConflict()
        msg = _msg("hello").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.policy_name == "EscalateOnConflict"

    def test_default_handler_is_escalate(self):
        """Default handler should be 'escalate'."""
        policy = EscalateOnConflict()
        msg = _msg("hello").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.handler == "escalate"

    def test_custom_handler_forwarded(self):
        """Custom handler should appear in the result."""
        policy = EscalateOnConflict(conflict_threshold=1, handler="block")
        msg = _msg("I disagree.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.handler == "block"

    def test_severity_is_high(self):
        """Escalation violations should always be high severity."""
        policy = EscalateOnConflict(conflict_threshold=1)
        msg = _msg("I disagree.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.severity == "high"

    def test_supervisor_in_details(self):
        """When 'to' is set, the supervisor should appear in violation details."""
        policy = EscalateOnConflict(conflict_threshold=1, to="supervisor_agent")
        msg = _msg("I disagree completely.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is False
        assert "supervisor_agent" in result.details

    def test_no_supervisor_in_details_when_to_is_none(self):
        """When 'to' is None, details should not mention a supervisor."""
        policy = EscalateOnConflict(conflict_threshold=1, to=None)
        msg = _msg("I disagree.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is False
        assert "supervisor" not in result.details.lower()

    def test_agent_id_in_result(self):
        """Result should carry the correct agent_id."""
        policy = EscalateOnConflict()
        msg = _msg("hello", sender="my_agent").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.agent_id == "my_agent"
