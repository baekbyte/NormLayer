"""EscalateOnConflict policy — stub, planned for Week 2."""

from __future__ import annotations

from normlayer.base_policy import AgentMessage, BasePolicy, HandlerType, PolicyResult


class EscalateOnConflict(BasePolicy):
    """Triggers escalation to a supervisor agent when agents disagree past a threshold.

    Monitors divergence between agent positions across consecutive turns. When
    the number of conflicting exchanges exceeds `conflict_threshold`, the policy
    fires and routes to the supervisor.

    .. note::
        This policy is a stub. Full implementation is planned for Week 2.

    Args:
        conflict_threshold: Number of conflicting turns before escalating (default ``3``).
        to: agent_id of the supervisor to route the escalation to.
        handler: Action to take on violation (default ``"escalate"``).
    """

    name: str = "EscalateOnConflict"

    def __init__(
        self,
        conflict_threshold: int = 3,
        to: str | None = None,
        handler: HandlerType = "escalate",
    ) -> None:
        super().__init__(handler=handler)
        self.conflict_threshold = conflict_threshold
        self.to = to

    def evaluate(self, message: AgentMessage, context: dict) -> PolicyResult:
        """Evaluate whether an inter-agent conflict has exceeded the threshold.

        Args:
            message: The AgentMessage to evaluate.
            context: Contextual information including conversation history.

        Returns:
            PolicyResult.

        Raises:
            NotImplementedError: This policy is not yet implemented.
        """
        raise NotImplementedError(
            "EscalateOnConflict is not yet implemented. Planned for Week 2."
        )
