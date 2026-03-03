"""Tests for the NormConflictResolution policy."""

import pytest

from normlayer.policies.norm_conflict_resolution import (
    NormConflictResolution,
    _DEFAULT_CONTRADICTION_PAIRS,
)
from normlayer.testing import MockMessage


def _msg(content: str, sender: str = "agent_a") -> MockMessage:
    return MockMessage(content=content, sender=sender)


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------


class TestNormConflictResolutionPasses:
    def test_passes_with_no_directives(self):
        """Policy should pass when no directives are provided."""
        policy = NormConflictResolution()
        msg = _msg("Hello world.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is True
        assert result.violation_score == 0.0

    def test_passes_with_empty_directives(self):
        """Policy should pass when directives list is empty."""
        policy = NormConflictResolution()
        msg = _msg("Hello world.").to_agent_message()
        result = policy.evaluate(msg, context={"directives": []})
        assert result.passed is True

    def test_passes_no_contradictions(self):
        """Policy should pass when directives have no contradictory pairs."""
        policy = NormConflictResolution()
        msg = _msg("Proceeding.").to_agent_message()
        result = policy.evaluate(msg, context={"directives": ["be kind", "be polite"]})
        assert result.passed is True

    def test_passes_one_side_of_pair_only(self):
        """Policy should pass when only one side of a contradiction pair appears."""
        policy = NormConflictResolution()
        msg = _msg("Okay.").to_agent_message()
        result = policy.evaluate(msg, context={"directives": ["be brief"]})
        assert result.passed is True

    def test_passes_below_threshold(self):
        """Policy should pass when contradictions are below threshold."""
        policy = NormConflictResolution(conflict_threshold=3)
        msg = _msg("Okay.").to_agent_message()
        result = policy.evaluate(
            msg,
            context={"directives": ["be brief and thorough", "be concise and detailed"]},
        )
        # 2 contradictions, threshold is 3 → pass
        assert result.passed is True

    def test_passes_custom_pairs_no_match(self):
        """Policy should pass when custom pairs don't match directives."""
        policy = NormConflictResolution(
            contradiction_pairs=[("alpha", "beta")]
        )
        msg = _msg("Sure.").to_agent_message()
        result = policy.evaluate(
            msg, context={"directives": ["be brief and thorough"]}
        )
        assert result.passed is True

    def test_passes_agent_not_in_agent_directives(self):
        """Policy should pass when sender is not in agent_directives and no fallback."""
        policy = NormConflictResolution()
        msg = _msg("Hello.", sender="agent_x").to_agent_message()
        result = policy.evaluate(
            msg,
            context={"agent_directives": {"agent_y": ["be brief and thorough"]}},
        )
        assert result.passed is True


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------


class TestNormConflictResolutionViolations:
    def test_single_contradiction(self):
        """Should violate when a single contradiction pair is found."""
        policy = NormConflictResolution()
        msg = _msg("Understood.").to_agent_message()
        result = policy.evaluate(
            msg, context={"directives": ["be brief and thorough"]}
        )
        assert result.passed is False
        assert result.violation_score == 1.0

    def test_multiple_contradictions(self):
        """Should detect multiple contradiction pairs."""
        policy = NormConflictResolution(conflict_threshold=2)
        msg = _msg("Will do.").to_agent_message()
        result = policy.evaluate(
            msg,
            context={
                "directives": [
                    "be brief and thorough",
                    "be concise and detailed",
                    "be fast and careful",
                ]
            },
        )
        assert result.passed is False
        assert result.violation_score == 1.0  # min(3/2, 1.0)

    def test_default_pairs_all_detected(self):
        """Each default pair should be independently detectable."""
        for a, b in _DEFAULT_CONTRADICTION_PAIRS:
            policy = NormConflictResolution()
            msg = _msg("Ok.").to_agent_message()
            result = policy.evaluate(
                msg, context={"directives": [f"be {a} and {b}"]}
            )
            assert result.passed is False, f"Pair ({a}, {b}) was not detected"

    def test_custom_pairs(self):
        """Should detect custom contradiction pairs."""
        policy = NormConflictResolution(
            contradiction_pairs=[("alpha", "omega")]
        )
        msg = _msg("Ready.").to_agent_message()
        result = policy.evaluate(
            msg, context={"directives": ["use alpha and omega strategies"]}
        )
        assert result.passed is False

    def test_severity_medium_at_threshold(self):
        """Severity should be 'medium' when count equals threshold."""
        policy = NormConflictResolution(conflict_threshold=2)
        msg = _msg("Ok.").to_agent_message()
        result = policy.evaluate(
            msg,
            context={"directives": ["brief thorough", "concise detailed"]},
        )
        assert result.passed is False
        assert result.severity == "medium"

    def test_severity_high_at_double_threshold(self):
        """Severity should be 'high' when count >= 2x threshold."""
        policy = NormConflictResolution(conflict_threshold=1)
        msg = _msg("Ok.").to_agent_message()
        result = policy.evaluate(
            msg,
            context={"directives": ["brief thorough", "concise detailed"]},
        )
        assert result.passed is False
        assert result.severity == "high"

    def test_score_capped_at_one(self):
        """Violation score should be capped at 1.0."""
        policy = NormConflictResolution(conflict_threshold=1)
        msg = _msg("Ok.").to_agent_message()
        result = policy.evaluate(
            msg,
            context={
                "directives": [
                    "brief thorough concise detailed fast careful"
                ]
            },
        )
        assert result.violation_score == 1.0

    def test_handler_forwarded(self):
        """Handler should be forwarded from constructor."""
        policy = NormConflictResolution(handler="block")
        msg = _msg("Ok.").to_agent_message()
        result = policy.evaluate(
            msg, context={"directives": ["brief and thorough"]}
        )
        assert result.handler == "block"


# ---------------------------------------------------------------------------
# Context handling
# ---------------------------------------------------------------------------


class TestNormConflictResolutionContext:
    def test_agent_directives_override(self):
        """agent_directives should take precedence over directives."""
        policy = NormConflictResolution()
        msg = _msg("Ok.", sender="agent_a").to_agent_message()
        result = policy.evaluate(
            msg,
            context={
                "directives": ["be kind"],  # no contradiction
                "agent_directives": {"agent_a": ["be brief and thorough"]},
            },
        )
        assert result.passed is False

    def test_fallback_to_directives(self):
        """Should fall back to directives when agent not in agent_directives."""
        policy = NormConflictResolution()
        msg = _msg("Ok.", sender="agent_a").to_agent_message()
        result = policy.evaluate(
            msg,
            context={
                "directives": ["be brief and thorough"],
                "agent_directives": {"agent_b": ["be kind"]},
            },
        )
        assert result.passed is False

    def test_case_insensitive(self):
        """Directive matching should be case-insensitive."""
        policy = NormConflictResolution()
        msg = _msg("Ok.").to_agent_message()
        result = policy.evaluate(
            msg, context={"directives": ["Be BRIEF and THOROUGH"]}
        )
        assert result.passed is False


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestNormConflictResolutionMetadata:
    def test_policy_name(self):
        """Result should carry the correct policy name."""
        policy = NormConflictResolution()
        msg = _msg("Ok.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.policy_name == "NormConflictResolution"

    def test_details_contain_pairs(self):
        """Violation details should list the triggered contradiction pairs."""
        policy = NormConflictResolution()
        msg = _msg("Ok.").to_agent_message()
        result = policy.evaluate(
            msg, context={"directives": ["brief and thorough"]}
        )
        assert "brief" in result.details
        assert "thorough" in result.details
