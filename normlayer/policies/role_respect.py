"""RoleRespect policy — flags agents operating outside their defined role."""

from __future__ import annotations

from typing import Any

from normlayer.base_policy import AgentMessage, BasePolicy, HandlerType, PolicyResult


class RoleRespect(BasePolicy):
    """Flags agents that operate outside their defined role or scope.

    Roles are defined as a set of allowed keywords or topic tokens. In strict
    mode the message must contain at least one allowed keyword; in non-strict
    mode (default) only globally forbidden keywords are checked.

    Role definitions and agent assignments can be provided at construction time
    **or** injected at runtime via the ``context`` dict — context values take
    precedence, enabling dynamic role configurations per conversation turn.

    Args:
        role_definitions: Mapping of ``role_name → list[allowed_keyword]``.
            Example::

                {
                    "planner": ["plan", "assign", "schedule"],
                    "executor": ["execute", "run", "complete"],
                }

        agent_roles: Mapping of ``agent_id → role_name``.
            Example::

                {"planner_agent": "planner", "worker_agent": "executor"}

        strict: If ``True``, a message must contain at least one keyword from
            its role's allowed list. If ``False`` (default), only
            ``forbidden_keywords`` are enforced.
        forbidden_keywords: Keywords that **no** agent may ever use, regardless
            of role. Case-insensitive.
        handler: Action to take on violation (default ``"warn"``).

    Context keys:
        role_definitions (dict): Overrides instance-level ``role_definitions``.
        agent_roles (dict): Overrides instance-level ``agent_roles``.
    """

    name: str = "RoleRespect"

    def __init__(
        self,
        role_definitions: dict[str, list[str]] | None = None,
        agent_roles: dict[str, str] | None = None,
        strict: bool = False,
        forbidden_keywords: list[str] | None = None,
        handler: HandlerType = "warn",
    ) -> None:
        super().__init__(handler=handler)
        self.role_definitions: dict[str, list[str]] = role_definitions or {}
        self.agent_roles: dict[str, str] = agent_roles or {}
        self.strict = strict
        self.forbidden_keywords: list[str] = [
            kw.lower() for kw in (forbidden_keywords or [])
        ]

    def evaluate(self, message: AgentMessage, context: dict[str, Any]) -> PolicyResult:
        """Evaluate whether the sending agent respects its assigned role.

        Checks are applied in this order:

        1. **Forbidden keywords** — fires for any agent, always high-severity.
        2. **Role assignment** — if the agent has no assigned role, pass.
        3. **Role definition** — if the role has no keyword list, pass.
        4. **Strict mode check** — message must contain at least one allowed keyword.

        Args:
            message: The AgentMessage to evaluate.
            context: Optional overrides for ``role_definitions`` and ``agent_roles``.

        Returns:
            PolicyResult indicating pass or role-scope violation.
        """
        role_defs: dict[str, list[str]] = context.get(
            "role_definitions", self.role_definitions
        )
        agent_roles: dict[str, str] = context.get("agent_roles", self.agent_roles)

        agent_id = message.sender
        content_lower = message.content.lower()

        # 1. Forbidden keywords — applies to all agents regardless of role.
        for kw in self.forbidden_keywords:
            if kw in content_lower:
                return PolicyResult(
                    passed=False,
                    violation_score=1.0,
                    policy_name=self.name,
                    agent_id=agent_id,
                    handler=self.handler,
                    severity="high",
                    details=(
                        f"Agent '{agent_id}' used globally forbidden keyword '{kw}'."
                    ),
                )

        # 2. No role assigned → nothing to enforce.
        role = agent_roles.get(agent_id)
        if role is None:
            return self._pass(agent_id)

        # 3. No keyword list for this role → nothing to enforce.
        allowed_keywords = role_defs.get(role, [])
        if not allowed_keywords:
            return self._pass(agent_id)

        # 4. Strict mode: message must contain at least one allowed keyword.
        if self.strict:
            allowed_lower = [kw.lower() for kw in allowed_keywords]
            has_allowed = any(kw in content_lower for kw in allowed_lower)
            if not has_allowed:
                return PolicyResult(
                    passed=False,
                    violation_score=0.8,
                    policy_name=self.name,
                    agent_id=agent_id,
                    handler=self.handler,
                    severity="medium",
                    details=(
                        f"Agent '{agent_id}' (role: '{role}') sent a message with "
                        f"no keywords matching its allowed scope {allowed_keywords}."
                    ),
                )

        return self._pass(agent_id)

    def _pass(self, agent_id: str) -> PolicyResult:
        """Return a clean passing result for the given agent.

        Args:
            agent_id: The agent whose message passed the check.

        Returns:
            A PolicyResult with ``passed=True`` and ``violation_score=0.0``.
        """
        return PolicyResult(
            passed=True,
            violation_score=0.0,
            policy_name=self.name,
            agent_id=agent_id,
            handler=self.handler,
            severity="low",
            details="",
        )
