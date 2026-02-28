"""Tests for the RoleRespect policy."""

import pytest

from normlayer.policies.role_respect import RoleRespect
from normlayer.testing import MockMessage

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

ROLE_DEFS = {
    "planner": ["plan", "assign", "schedule", "coordinate"],
    "executor": ["execute", "run", "complete", "deploy"],
    "reviewer": ["review", "approve", "reject", "audit"],
}

AGENT_ROLES = {
    "planner_agent": "planner",
    "executor_agent": "executor",
    "reviewer_agent": "reviewer",
}


def _msg(content: str, sender: str = "planner_agent") -> object:
    return MockMessage(content=content, sender=sender).to_agent_message()


# ---------------------------------------------------------------------------
# Passes — no role or no definition configured
# ---------------------------------------------------------------------------

class TestRoleRespectPasses:
    def test_passes_when_no_agent_role_assigned(self):
        """Agents with no assigned role should always pass."""
        policy = RoleRespect(role_definitions=ROLE_DEFS, agent_roles={})
        msg = _msg("I will do whatever I want.", sender="unknown_agent")
        result = policy.evaluate(msg, context={})
        assert result.passed is True
        assert result.violation_score == 0.0

    def test_passes_when_no_role_definition_for_role(self):
        """If a role has no keyword list, the policy should pass."""
        policy = RoleRespect(
            role_definitions={},  # empty — no definitions
            agent_roles={"planner_agent": "planner"},
        )
        msg = _msg("I am doing something unusual.")
        result = policy.evaluate(msg, context={})
        assert result.passed is True

    def test_passes_in_non_strict_mode_with_allowed_keyword(self):
        """Non-strict mode: passes whenever content contains an allowed keyword."""
        policy = RoleRespect(
            role_definitions=ROLE_DEFS,
            agent_roles=AGENT_ROLES,
            strict=False,
        )
        msg = _msg("I will plan the sprint for next week.", sender="planner_agent")
        result = policy.evaluate(msg, context={})
        assert result.passed is True

    def test_passes_in_non_strict_mode_without_allowed_keyword(self):
        """Non-strict mode only checks forbidden keywords — off-scope content passes."""
        policy = RoleRespect(
            role_definitions=ROLE_DEFS,
            agent_roles=AGENT_ROLES,
            strict=False,
        )
        msg = _msg("The weather is nice today.", sender="planner_agent")
        result = policy.evaluate(msg, context={})
        assert result.passed is True

    def test_passes_in_strict_mode_with_allowed_keyword(self):
        """Strict mode: passes when at least one allowed keyword is present."""
        policy = RoleRespect(
            role_definitions=ROLE_DEFS,
            agent_roles=AGENT_ROLES,
            strict=True,
        )
        msg = _msg("I will assign task A to the executor.", sender="planner_agent")
        result = policy.evaluate(msg, context={})
        assert result.passed is True

    def test_allowed_keyword_is_case_insensitive(self):
        """Keyword matching should be case-insensitive."""
        policy = RoleRespect(
            role_definitions=ROLE_DEFS,
            agent_roles=AGENT_ROLES,
            strict=True,
        )
        msg = _msg("I will PLAN the roadmap.", sender="planner_agent")
        result = policy.evaluate(msg, context={})
        assert result.passed is True


# ---------------------------------------------------------------------------
# Violations — strict mode
# ---------------------------------------------------------------------------

class TestRoleRespectStrictViolations:
    def test_fires_in_strict_mode_without_allowed_keyword(self):
        """Strict mode: message with no allowed keywords should violate."""
        policy = RoleRespect(
            role_definitions=ROLE_DEFS,
            agent_roles=AGENT_ROLES,
            strict=True,
        )
        msg = _msg("The weather is nice today.", sender="planner_agent")
        result = policy.evaluate(msg, context={})
        assert result.passed is False
        assert result.violation_score > 0.0
        assert "planner_agent" in result.details

    def test_violation_handler_forwarded(self):
        """Result handler should match the policy's configured handler."""
        policy = RoleRespect(
            role_definitions=ROLE_DEFS,
            agent_roles=AGENT_ROLES,
            strict=True,
            handler="block",
        )
        msg = _msg("Something off-topic.", sender="executor_agent")
        result = policy.evaluate(msg, context={})
        assert result.passed is False
        assert result.handler == "block"


# ---------------------------------------------------------------------------
# Violations — forbidden keywords
# ---------------------------------------------------------------------------

class TestRoleRespectForbiddenKeywords:
    def test_forbidden_keyword_fires_regardless_of_role(self):
        """A forbidden keyword should trigger for any agent, any role."""
        policy = RoleRespect(
            role_definitions=ROLE_DEFS,
            agent_roles=AGENT_ROLES,
            forbidden_keywords=["override", "bypass"],
        )
        msg = _msg("I will bypass the safety check.", sender="planner_agent")
        result = policy.evaluate(msg, context={})
        assert result.passed is False
        assert result.severity == "high"
        assert "bypass" in result.details

    def test_forbidden_keyword_fires_for_unassigned_agent(self):
        """Forbidden keywords apply even to agents with no defined role."""
        policy = RoleRespect(
            forbidden_keywords=["override"],
            agent_roles={},
        )
        msg = _msg("I will override the system.", sender="rogue_agent")
        result = policy.evaluate(msg, context={})
        assert result.passed is False

    def test_forbidden_keyword_is_case_insensitive(self):
        """Forbidden keyword detection should be case-insensitive."""
        policy = RoleRespect(forbidden_keywords=["override"])
        msg = _msg("I will OVERRIDE the system.", sender="agent")
        result = policy.evaluate(msg, context={})
        assert result.passed is False


# ---------------------------------------------------------------------------
# Context overrides
# ---------------------------------------------------------------------------

class TestRoleRespectContextOverrides:
    def test_context_role_definitions_override_instance(self):
        """role_definitions in context should take precedence over instance config."""
        policy = RoleRespect(
            role_definitions={"planner": ["plan"]},
            agent_roles={"planner_agent": "planner"},
            strict=True,
        )
        # Override at runtime: planner is now allowed to "review"
        ctx = {
            "role_definitions": {"planner": ["review"]},
            "agent_roles": {"planner_agent": "planner"},
        }
        msg = _msg("I will review this document.", sender="planner_agent")
        result = policy.evaluate(msg, context=ctx)
        assert result.passed is True

    def test_context_agent_roles_override_instance(self):
        """agent_roles in context should take precedence over instance config."""
        policy = RoleRespect(
            role_definitions=ROLE_DEFS,
            agent_roles={"planner_agent": "planner"},
            strict=True,
        )
        # Override: planner_agent is now an executor at runtime
        ctx = {
            "role_definitions": ROLE_DEFS,
            "agent_roles": {"planner_agent": "executor"},
        }
        msg = _msg("I will execute and deploy.", sender="planner_agent")
        result = policy.evaluate(msg, context=ctx)
        assert result.passed is True

    def test_policy_name_in_result(self):
        policy = RoleRespect()
        msg = _msg("hello")
        result = policy.evaluate(msg, context={})
        assert result.policy_name == "RoleRespect"
