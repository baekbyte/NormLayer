"""ResponseProportionality policy — catches disproportionate responses."""

from __future__ import annotations

from typing import Any

from normlayer.base_policy import AgentMessage, BasePolicy, HandlerType, PolicyResult, SeverityLevel


class ResponseProportionality(BasePolicy):
    """Catches disproportionate responses relative to the triggering input.

    Compares the length of an agent's response against the message that
    prompted it.  Responses significantly longer or shorter than the
    acceptable ratio range are flagged.

    Args:
        max_ratio: Maximum allowed response-to-input length ratio (default ``5.0``).
        min_ratio: Minimum allowed response-to-input length ratio (default ``0.1``).
        handler: Action to take on violation (default ``"warn"``).

    Context keys:
        triggering_message (AgentMessage): The message this response is
            replying to.  If absent, the policy passes by default.
    """

    name: str = "ResponseProportionality"

    def __init__(
        self,
        max_ratio: float = 5.0,
        min_ratio: float = 0.1,
        handler: HandlerType = "warn",
    ) -> None:
        super().__init__(handler=handler)
        self.max_ratio = max_ratio
        self.min_ratio = min_ratio

    def evaluate(self, message: AgentMessage, context: dict[str, Any]) -> PolicyResult:
        """Evaluate whether the response length is proportionate to the input.

        Args:
            message: The AgentMessage to evaluate.
            context: Must contain ``triggering_message: AgentMessage`` for
                comparison. If absent, the policy passes.

        Returns:
            PolicyResult indicating pass or proportionality violation.
        """
        triggering = context.get("triggering_message")
        if triggering is None:
            return self._pass(message.sender)

        trigger_len = len(triggering.content)
        if trigger_len == 0:
            return self._pass(message.sender)

        response_len = len(message.content)
        ratio = response_len / trigger_len

        if ratio > self.max_ratio:
            score = min((ratio - self.max_ratio) / self.max_ratio, 1.0)
            severity: SeverityLevel = "high" if ratio > 2 * self.max_ratio else "medium"
            return PolicyResult(
                passed=False,
                violation_score=score,
                policy_name=self.name,
                agent_id=message.sender,
                handler=self.handler,
                severity=severity,
                details=(
                    f"Agent '{message.sender}' response is disproportionately long: "
                    f"ratio {ratio:.2f} exceeds max {self.max_ratio}."
                ),
            )

        if ratio < self.min_ratio:
            score = min((self.min_ratio - ratio) / self.min_ratio, 1.0)
            sev: SeverityLevel = "high" if ratio < self.min_ratio / 2 else "medium"
            return PolicyResult(
                passed=False,
                violation_score=score,
                policy_name=self.name,
                agent_id=message.sender,
                handler=self.handler,
                severity=sev,
                details=(
                    f"Agent '{message.sender}' response is disproportionately short: "
                    f"ratio {ratio:.2f} is below min {self.min_ratio}."
                ),
            )

        return self._pass(message.sender)

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
