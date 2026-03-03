"""Tests for the NoUnsanctionedAction policy."""

import pytest

from normlayer.policies.no_unsanctioned_action import (
    NoUnsanctionedAction,
    _ACTION_KEYWORDS,
)
from normlayer.testing import MockMessage


def _msg(content: str, sender: str = "agent_a") -> MockMessage:
    return MockMessage(content=content, sender=sender)


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------


class TestNoUnsanctionedActionPasses:
    def test_passes_no_permissions(self):
        """Policy should pass when no permissions are configured."""
        policy = NoUnsanctionedAction()
        msg = _msg("I will deploy the service.").to_agent_message()
        result = policy.evaluate(msg, context={})
        # No permissions set + sender not in dict → fail-open
        assert result.passed is True

    def test_passes_no_action_keywords(self):
        """Policy should pass when message has no action keywords."""
        policy = NoUnsanctionedAction(
            permissions={"agent_a": ["deploy"]}
        )
        msg = _msg("Hello, how are you?").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is True
        assert result.violation_score == 0.0

    def test_passes_agent_not_in_permissions(self):
        """Policy should fail-open when sender is not in permissions dict."""
        policy = NoUnsanctionedAction(
            permissions={"agent_b": ["deploy"]}
        )
        msg = _msg("I will deploy now.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is True

    def test_passes_all_actions_allowed(self):
        """Policy should pass when all extracted actions are allowed."""
        policy = NoUnsanctionedAction(
            permissions={"agent_a": ["deploy", "restart"]}
        )
        msg = _msg("I will deploy and restart the service.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is True

    def test_passes_conversational_plus_allowed(self):
        """Policy should pass when message has both conversational text and allowed actions."""
        policy = NoUnsanctionedAction(
            permissions={"agent_a": ["create", "update"]}
        )
        msg = _msg("Let me create the resource and update the config.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is True

    def test_passes_empty_permissions_dict(self):
        """Policy should fail-open when permissions dict exists but sender not in it."""
        policy = NoUnsanctionedAction(permissions={})
        msg = _msg("I will deploy this.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is True

    def test_passes_context_override_grants_access(self):
        """Runtime context permissions should override constructor permissions."""
        policy = NoUnsanctionedAction(
            permissions={"agent_a": []}  # no actions allowed
        )
        msg = _msg("I will deploy.").to_agent_message()
        result = policy.evaluate(
            msg, context={"permissions": {"agent_a": ["deploy"]}}
        )
        assert result.passed is True


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------


class TestNoUnsanctionedActionViolations:
    def test_action_not_in_allowed(self):
        """Should violate when action keyword is not in allowed list."""
        policy = NoUnsanctionedAction(
            permissions={"agent_a": ["deploy"]}
        )
        msg = _msg("I will delete the database.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is False
        assert "delete" in result.details

    def test_multiple_unsanctioned(self):
        """Should detect multiple unsanctioned actions."""
        policy = NoUnsanctionedAction(
            permissions={"agent_a": ["deploy"]}
        )
        msg = _msg("I will delete and shutdown the server.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is False
        assert result.violation_score > 0.0

    def test_global_forbidden_single_word(self):
        """Should violate for globally forbidden single-word action."""
        policy = NoUnsanctionedAction(
            global_forbidden=["shutdown"]
        )
        msg = _msg("I will shutdown the system.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is False
        assert result.violation_score == 1.0
        assert result.severity == "high"

    def test_global_forbidden_multi_word(self):
        """Should violate for globally forbidden multi-word phrase."""
        policy = NoUnsanctionedAction(
            global_forbidden=["deploy to production"]
        )
        msg = _msg("I will deploy to production now.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is False
        assert result.violation_score == 1.0
        assert result.severity == "high"

    def test_severity_medium(self):
        """Severity should be 'medium' when <=50% actions are unsanctioned."""
        policy = NoUnsanctionedAction(
            permissions={"agent_a": ["deploy", "restart"]}
        )
        # 3 actions: deploy (allowed), restart (allowed), delete (unsanctioned)
        msg = _msg("Deploy, restart, and delete.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is False
        assert result.severity == "medium"

    def test_severity_high(self):
        """Severity should be 'high' when >50% actions are unsanctioned."""
        policy = NoUnsanctionedAction(
            permissions={"agent_a": ["deploy"]}
        )
        # 3 actions: deploy (allowed), delete (unsanctioned), shutdown (unsanctioned)
        msg = _msg("Deploy, delete, and shutdown.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is False
        assert result.severity == "high"

    def test_score_proportional(self):
        """Violation score should be proportional to unsanctioned/total ratio."""
        policy = NoUnsanctionedAction(
            permissions={"agent_a": ["deploy"]}
        )
        # 2 actions: deploy (allowed), delete (unsanctioned)
        msg = _msg("Deploy and delete.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is False
        assert result.violation_score == pytest.approx(0.5)

    def test_handler_default_block(self):
        """Default handler should be 'block'."""
        policy = NoUnsanctionedAction(
            permissions={"agent_a": []}
        )
        msg = _msg("I will deploy.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.handler == "block"


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------


class TestNoUnsanctionedActionKeywordExtraction:
    def test_punctuation_stripped(self):
        """Action keywords should be detected even with punctuation."""
        policy = NoUnsanctionedAction(
            permissions={"agent_a": []}
        )
        msg = _msg("deploy! delete? restart.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is False

    def test_case_insensitive(self):
        """Action keywords should be matched case-insensitively."""
        policy = NoUnsanctionedAction(
            permissions={"agent_a": []}
        )
        msg = _msg("I will DEPLOY the service.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is False

    def test_non_action_words_ignored(self):
        """Non-action words should not be extracted."""
        policy = NoUnsanctionedAction(
            permissions={"agent_a": ["deploy"]}
        )
        msg = _msg("I will deploy the service to the cloud.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is True

    def test_only_known_keywords_extracted(self):
        """Only the 20 predefined action keywords should be extracted."""
        policy = NoUnsanctionedAction(
            permissions={"agent_a": []}
        )
        # "run" is NOT in the keyword set
        msg = _msg("I will run the process.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert result.passed is True  # no recognized action keywords


# ---------------------------------------------------------------------------
# Context override
# ---------------------------------------------------------------------------


class TestNoUnsanctionedActionContext:
    def test_context_override_permissions(self):
        """Runtime context should override constructor permissions."""
        policy = NoUnsanctionedAction(
            permissions={"agent_a": ["deploy"]}
        )
        msg = _msg("I will delete the records.").to_agent_message()
        result = policy.evaluate(
            msg, context={"permissions": {"agent_a": ["deploy", "delete"]}}
        )
        assert result.passed is True

    def test_details_content(self):
        """Violation details should list unsanctioned and allowed actions."""
        policy = NoUnsanctionedAction(
            permissions={"agent_a": ["deploy"]}
        )
        msg = _msg("I will delete things.").to_agent_message()
        result = policy.evaluate(msg, context={})
        assert "delete" in result.details
        assert "deploy" in result.details
