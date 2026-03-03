"""NoUnsanctionedAction policy — enforces action allowlists per agent."""

from __future__ import annotations

from normlayer.base_policy import AgentMessage, BasePolicy, HandlerType, PolicyResult

_ACTION_KEYWORDS: set[str] = {
    "deploy",
    "delete",
    "approve",
    "reject",
    "transfer",
    "execute",
    "shutdown",
    "restart",
    "modify",
    "create",
    "update",
    "install",
    "remove",
    "send",
    "publish",
    "revoke",
    "grant",
    "terminate",
    "override",
    "escalate",
}


class NoUnsanctionedAction(BasePolicy):
    """Enforces an allowlist of action keywords per agent.

    Flags agents attempting actions outside their explicitly granted
    permissions. Extracts known action verbs from the message and checks
    them against the sender's allowed action set.

    Args:
        permissions: Mapping of ``agent_id → list[allowed_action]``.
            Example::

                {
                    "deployer_agent": ["deploy", "restart"],
                    "reviewer_agent": ["approve", "reject"],
                }

        global_forbidden: Actions that **no** agent may ever perform,
            regardless of their permissions. Supports multi-word phrases
            (substring match) and single words (word match).
        handler: Action to take on violation (default ``"block"``).

    Context keys:
        permissions (dict[str, list[str]]): Runtime override for per-agent
            allowlists. Takes precedence over constructor ``permissions``.
    """

    name: str = "NoUnsanctionedAction"

    def __init__(
        self,
        permissions: dict[str, list[str]] | None = None,
        global_forbidden: list[str] | None = None,
        handler: HandlerType = "block",
    ) -> None:
        super().__init__(handler=handler)
        self.permissions: dict[str, list[str]] = permissions or {}
        self.global_forbidden: list[str] = [
            f.lower() for f in (global_forbidden or [])
        ]

    def evaluate(self, message: AgentMessage, context: dict) -> PolicyResult:
        """Evaluate whether the sender's actions are sanctioned.

        Args:
            message: The AgentMessage to evaluate.
            context: Optional ``permissions`` override.

        Returns:
            PolicyResult indicating pass or unsanctioned-action violation.
        """
        sender = message.sender
        content_lower = message.content.lower()

        # Tokenize and extract action keywords.
        words = {w.strip(".,!?;:\"'()[]{}") for w in content_lower.split()}
        extracted = words & _ACTION_KEYWORDS

        # No action keywords found → pass.
        if not extracted:
            return self._pass(sender)

        # Check global_forbidden first.
        for forbidden in self.global_forbidden:
            # Multi-word phrases: substring match on full content.
            # Single words: word-level match.
            if " " in forbidden:
                if forbidden in content_lower:
                    return PolicyResult(
                        passed=False,
                        violation_score=1.0,
                        policy_name=self.name,
                        agent_id=sender,
                        handler=self.handler,
                        severity="high",
                        details=(
                            f"Agent '{sender}' used globally forbidden action "
                            f"'{forbidden}'."
                        ),
                    )
            else:
                if forbidden in words:
                    return PolicyResult(
                        passed=False,
                        violation_score=1.0,
                        policy_name=self.name,
                        agent_id=sender,
                        handler=self.handler,
                        severity="high",
                        details=(
                            f"Agent '{sender}' used globally forbidden action "
                            f"'{forbidden}'."
                        ),
                    )

        # Resolve permissions.
        permissions: dict[str, list[str]] = context.get(
            "permissions", self.permissions
        )

        # Sender not in permissions dict → fail-open.
        if sender not in permissions:
            return self._pass(sender)

        allowed = {a.lower() for a in permissions[sender]}
        unsanctioned = extracted - allowed

        if not unsanctioned:
            return self._pass(sender)

        violation_score = min(len(unsanctioned) / len(extracted), 1.0)
        severity = "high" if len(unsanctioned) > len(extracted) / 2 else "medium"

        return PolicyResult(
            passed=False,
            violation_score=violation_score,
            policy_name=self.name,
            agent_id=sender,
            handler=self.handler,
            severity=severity,
            details=(
                f"Agent '{sender}' attempted unsanctioned actions: "
                f"{sorted(unsanctioned)}. Allowed: {sorted(allowed)}."
            ),
        )

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
