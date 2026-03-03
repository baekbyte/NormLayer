"""LoopDetection policy — flags agents stuck in repetitive exchanges."""

from __future__ import annotations

from difflib import SequenceMatcher

from typing import Any

from normlayer.base_policy import AgentMessage, BasePolicy, HandlerType, PolicyResult


class LoopDetection(BasePolicy):
    """Detects agents stuck in unproductive repetitive exchanges.

    Maintains a sliding window of the sender's recent messages (drawn from
    ``context["history"]``). If the current message is sufficiently similar
    to at least `max_repetitions` messages in that window, a violation fires.

    Similarity is measured with :class:`difflib.SequenceMatcher` (character
    n-gram ratio), which is fast and requires no external dependencies.

    Args:
        max_repetitions: Number of similar past messages required to trigger
            a violation (default ``3``).
        similarity_threshold: SequenceMatcher ratio at or above which two
            messages are considered "the same" (default ``0.85``).
        window_size: How many of the sender's most recent messages to inspect
            (default ``10``).
        handler: Action to take on violation (default ``"warn"``).

    Context keys:
        history (list[AgentMessage]): All past messages in the conversation.
            If absent, the policy passes by default — no history means no loop.
    """

    name: str = "LoopDetection"

    def __init__(
        self,
        max_repetitions: int = 3,
        similarity_threshold: float = 0.85,
        window_size: int = 10,
        handler: HandlerType = "warn",
    ) -> None:
        super().__init__(handler=handler)
        self.max_repetitions = max_repetitions
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size

    def evaluate(self, message: AgentMessage, context: dict[str, Any]) -> PolicyResult:
        """Check whether the sending agent is stuck in a repetitive loop.

        Args:
            message: The current AgentMessage to evaluate.
            context: Must contain ``history: list[AgentMessage]`` to be useful.
                If the key is absent or empty, the policy passes.

        Returns:
            PolicyResult indicating pass or loop violation.
        """
        history: list[AgentMessage] = context.get("history", [])

        # Filter to only this sender's recent messages.
        sender_history = [
            m for m in history if m.sender == message.sender
        ][-self.window_size :]

        similar_count = sum(
            1
            for past in sender_history
            if self._similarity(message.content, past.content)
            >= self.similarity_threshold
        )

        passed = similar_count < self.max_repetitions
        violation_score = min(similar_count / max(self.max_repetitions, 1), 1.0)

        return PolicyResult(
            passed=passed,
            violation_score=violation_score,
            policy_name=self.name,
            agent_id=message.sender,
            handler=self.handler,
            severity="medium",
            details=(
                ""
                if passed
                else (
                    f"Agent '{message.sender}' sent {similar_count} near-identical "
                    f"messages in the last {self.window_size} turns "
                    f"(threshold: ≥{self.max_repetitions} at similarity "
                    f"≥{self.similarity_threshold})."
                )
            ),
        )

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Compute the SequenceMatcher ratio between two strings.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Float in [0, 1]; 1.0 means identical.
        """
        return SequenceMatcher(None, a, b).ratio()
