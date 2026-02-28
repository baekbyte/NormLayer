"""CoalitionConsistency policy — stub, planned for Week 2."""

from __future__ import annotations

from normlayer.base_policy import AgentMessage, BasePolicy, HandlerType, PolicyResult


class CoalitionConsistency(BasePolicy):
    """Checks whether agents apply norms consistently across in-group vs. out-group agents.

    Detects double standards: an agent applying stricter or more lenient norms
    to some agents than others based on coalition membership rather than merit.

    .. note::
        This policy is a stub. Full implementation is planned for Week 2.

    Args:
        coalitions: Mapping of coalition_name → list[agent_id].
            Example: ``{"team_a": ["planner", "executor"], "team_b": ["reviewer"]}``
        handler: Action to take on violation (default ``"warn"``).
    """

    name: str = "CoalitionConsistency"

    def __init__(
        self,
        coalitions: dict[str, list[str]] | None = None,
        handler: HandlerType = "warn",
    ) -> None:
        super().__init__(handler=handler)
        self.coalitions: dict[str, list[str]] = coalitions or {}

    def evaluate(self, message: AgentMessage, context: dict) -> PolicyResult:
        """Evaluate whether this agent's behavior is consistent across coalitions.

        Args:
            message: The AgentMessage to evaluate.
            context: Contextual information including behavioral history.

        Returns:
            PolicyResult.

        Raises:
            NotImplementedError: This policy is not yet implemented.
        """
        raise NotImplementedError(
            "CoalitionConsistency is not yet implemented. Planned for Week 2."
        )
