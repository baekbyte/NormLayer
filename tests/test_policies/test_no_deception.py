"""Tests for the NoDeception policy.

The sentence-transformers model is never loaded in this test suite.
``score_faithfulness`` is monkeypatched to return controlled float values,
keeping tests fast, deterministic, and free of network calls.
"""

import pytest

from normlayer.policies.no_deception import NoDeception
from normlayer.testing import MockMessage


def _msg(content: str, sender: str = "summarizer_agent") -> object:
    return MockMessage(content=content, sender=sender).to_agent_message()


# ---------------------------------------------------------------------------
# Passes — no original_message in context
# ---------------------------------------------------------------------------

class TestNoDeceptionNoOriginal:
    def test_passes_when_no_original_message(self):
        """If original_message is absent from context, the policy skips evaluation."""
        policy = NoDeception(threshold=0.8)
        msg = _msg("The worker said the task is complete.")
        result = policy.evaluate(msg, context={})
        assert result.passed is True
        assert result.violation_score == 0.0

    def test_passes_when_original_is_empty_string(self):
        """Empty string is falsy — treated the same as absent."""
        policy = NoDeception(threshold=0.8)
        msg = _msg("Some restatement.")
        result = policy.evaluate(msg, context={"original_message": ""})
        assert result.passed is True


# ---------------------------------------------------------------------------
# Passes — faithful restatement (monkeypatched score ≥ threshold)
# ---------------------------------------------------------------------------

class TestNoDeceptionFaithful:
    def test_passes_when_score_above_threshold(self, monkeypatch):
        policy = NoDeception(threshold=0.8)
        monkeypatch.setattr(policy, "score_faithfulness", lambda orig, summ: 0.95)

        msg = _msg("The task was completed successfully.")
        result = policy.evaluate(
            msg, context={"original_message": "Task A is done."}
        )
        assert result.passed is True
        assert result.violation_score < 0.1  # 1 - 0.95 = 0.05

    def test_passes_exactly_at_threshold(self, monkeypatch):
        """Score exactly equal to threshold should pass (≥, not >)."""
        policy = NoDeception(threshold=0.8)
        monkeypatch.setattr(policy, "score_faithfulness", lambda orig, summ: 0.8)

        msg = _msg("Faithful restatement.")
        result = policy.evaluate(
            msg, context={"original_message": "Original text."}
        )
        assert result.passed is True


# ---------------------------------------------------------------------------
# Violations — unfaithful restatement (monkeypatched score < threshold)
# ---------------------------------------------------------------------------

class TestNoDeceptionUnfaithful:
    def test_fires_when_score_below_threshold(self, monkeypatch):
        policy = NoDeception(threshold=0.8)
        monkeypatch.setattr(policy, "score_faithfulness", lambda orig, summ: 0.5)

        msg = _msg("Totally different claim.")
        result = policy.evaluate(
            msg, context={"original_message": "The task is assigned to agent B."}
        )
        assert result.passed is False
        assert result.violation_score > 0.0
        assert "summarizer_agent" in result.details
        assert "0.500" in result.details

    def test_violation_score_equals_one_minus_similarity(self, monkeypatch):
        """violation_score should be 1 - similarity, clamped to [0, 1]."""
        policy = NoDeception(threshold=0.8)
        monkeypatch.setattr(policy, "score_faithfulness", lambda orig, summ: 0.3)

        msg = _msg("Misleading summary.")
        result = policy.evaluate(
            msg, context={"original_message": "Original text."}
        )
        assert abs(result.violation_score - 0.7) < 1e-6

    def test_high_severity_when_violation_score_above_half(self, monkeypatch):
        """violation_score > 0.5 should produce severity='high'."""
        policy = NoDeception(threshold=0.8)
        monkeypatch.setattr(policy, "score_faithfulness", lambda orig, summ: 0.2)

        msg = _msg("Completely wrong summary.")
        result = policy.evaluate(
            msg, context={"original_message": "Real message."}
        )
        assert result.passed is False
        assert result.severity == "high"

    def test_medium_severity_when_violation_score_at_or_below_half(self, monkeypatch):
        """violation_score ≤ 0.5 should produce severity='medium'."""
        policy = NoDeception(threshold=0.8)
        # similarity=0.6, violation_score=0.4 → medium
        monkeypatch.setattr(policy, "score_faithfulness", lambda orig, summ: 0.6)

        msg = _msg("Slightly off summary.")
        result = policy.evaluate(
            msg, context={"original_message": "Real message."}
        )
        assert result.passed is False
        assert result.severity == "medium"

    def test_handler_forwarded_to_result(self, monkeypatch):
        policy = NoDeception(threshold=0.8, handler="escalate")
        monkeypatch.setattr(policy, "score_faithfulness", lambda orig, summ: 0.1)

        msg = _msg("Wrong claim.")
        result = policy.evaluate(
            msg, context={"original_message": "The real claim."}
        )
        assert result.passed is False
        assert result.handler == "escalate"


# ---------------------------------------------------------------------------
# Metadata / bookkeeping
# ---------------------------------------------------------------------------

class TestNoDeceptionMetadata:
    def test_policy_name_in_result(self):
        policy = NoDeception()
        msg = _msg("summary")
        result = policy.evaluate(msg, context={})
        assert result.policy_name == "NoDeception"

    def test_agent_id_in_result(self, monkeypatch):
        policy = NoDeception(threshold=0.8)
        monkeypatch.setattr(policy, "score_faithfulness", lambda orig, summ: 0.9)
        msg = _msg("Summary.", sender="relay_agent")
        result = policy.evaluate(
            msg, context={"original_message": "Original."}
        )
        assert result.agent_id == "relay_agent"

    def test_custom_threshold_respected(self, monkeypatch):
        """A lower threshold should allow more divergence without violation."""
        policy = NoDeception(threshold=0.5)
        monkeypatch.setattr(policy, "score_faithfulness", lambda orig, summ: 0.6)

        msg = _msg("Somewhat different summary.")
        result = policy.evaluate(
            msg, context={"original_message": "Original."}
        )
        assert result.passed is True  # 0.6 ≥ 0.5

    def test_model_loaded_lazily(self):
        """Model should not be loaded until score_faithfulness is called."""
        policy = NoDeception()
        assert policy._model is None
