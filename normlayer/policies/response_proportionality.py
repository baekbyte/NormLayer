"""ResponseProportionality policy — stub, planned for Week 2."""

from __future__ import annotations

from normlayer.base_policy import AgentMessage, BasePolicy, HandlerType, PolicyResult


class ResponseProportionality(BasePolicy):
    """Catches disproportionate responses relative to the triggering input.

    Compares the length and complexity of an agent's response against the
    message that prompted it. Responses significantly longer or shorter than
    expected for the input type are flagged.

    .. note::
        This policy is a stub. Full implementation is planned for Week 2.

    Args:
        max_ratio: Maximum allowed response-to-input length ratio (default ``5.0``).
        min_ratio: Minimum allowed response-to-input length ratio (default ``0.1``).
        handler: Action to take on violation (default ``"warn"``).
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

    def evaluate(self, message: AgentMessage, context: dict) -> PolicyResult:
        """Evaluate whether the response length is proportionate to the input.

        Args:
            message: The AgentMessage to evaluate.
            context: Must contain ``triggering_message: AgentMessage`` for comparison.

        Returns:
            PolicyResult.

        Raises:
            NotImplementedError: This policy is not yet implemented.
        """
        raise NotImplementedError(
            "ResponseProportionality is not yet implemented. Planned for Week 2."
        )
