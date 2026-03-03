"""NormConflictResolution policy — detects contradictory directives given to an agent."""

from __future__ import annotations

from typing import Any

from normlayer.base_policy import AgentMessage, BasePolicy, HandlerType, PolicyResult, SeverityLevel

_DEFAULT_CONTRADICTION_PAIRS: list[tuple[str, str]] = [
    ("brief", "thorough"),
    ("concise", "detailed"),
    ("fast", "careful"),
    ("short", "comprehensive"),
    ("simple", "exhaustive"),
    ("minimal", "complete"),
    ("quick", "rigorous"),
    ("silent", "verbose"),
    ("autonomous", "supervised"),
    ("conservative", "aggressive"),
]


class NormConflictResolution(BasePolicy):
    """Detects when an agent has been given contradictory directives.

    Checks the sender's directives for pairs of contradictory terms (e.g.,
    "be brief" + "be thorough"). When the number of detected contradictions
    reaches ``conflict_threshold``, a violation fires — flagging a no-win
    situation that requires human or supervisor intervention.

    Args:
        contradiction_pairs: List of ``(term_a, term_b)`` tuples representing
            contradictions. Defaults to 10 built-in pairs.
        conflict_threshold: Number of contradictions required to trigger a
            violation (default ``1``).
        handler: Action to take on violation (default ``"warn"``).

    Context keys:
        directives (list[str]): Directive strings for the sender.
        agent_directives (dict[str, list[str]]): Per-agent directive overrides.
            Takes precedence over ``directives``.
    """

    name: str = "NormConflictResolution"

    def __init__(
        self,
        contradiction_pairs: list[tuple[str, str]] | None = None,
        conflict_threshold: int = 1,
        handler: HandlerType = "warn",
    ) -> None:
        super().__init__(handler=handler)
        self.contradiction_pairs = contradiction_pairs or _DEFAULT_CONTRADICTION_PAIRS
        self.conflict_threshold = conflict_threshold

    def evaluate(self, message: AgentMessage, context: dict[str, Any]) -> PolicyResult:
        """Evaluate whether the sender's directives contain contradictions.

        Args:
            message: The AgentMessage to evaluate.
            context: Should contain ``directives`` or ``agent_directives``.

        Returns:
            PolicyResult indicating pass or norm-conflict violation.
        """
        sender = message.sender

        # Resolve directives: agent_directives[sender] → fallback directives → pass
        agent_directives: dict[str, list[str]] = context.get("agent_directives", {})
        directives: list[str] = agent_directives.get(
            sender, context.get("directives", [])
        )

        if not directives:
            return self._pass(sender)

        # Concatenate all directives into a single lowercase string.
        combined = " ".join(directives).lower()

        # Count contradictions.
        conflict_count = 0
        triggered_pairs: list[tuple[str, str]] = []
        for a, b in self.contradiction_pairs:
            if a in combined and b in combined:
                conflict_count += 1
                triggered_pairs.append((a, b))

        if conflict_count < self.conflict_threshold:
            return self._pass(sender)

        violation_score = min(conflict_count / max(self.conflict_threshold, 1), 1.0)
        severity: SeverityLevel = "high" if conflict_count >= 2 * self.conflict_threshold else "medium"

        return PolicyResult(
            passed=False,
            violation_score=violation_score,
            policy_name=self.name,
            agent_id=sender,
            handler=self.handler,
            severity=severity,
            details=(
                f"Agent '{sender}' has {conflict_count} contradictory directive "
                f"pair(s): {triggered_pairs}."
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
