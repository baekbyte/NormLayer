"""EscalateOnConflict policy — triggers escalation when agents disagree past a threshold."""

from __future__ import annotations

from typing import Any

from normlayer.base_policy import AgentMessage, BasePolicy, HandlerType, PolicyResult

_CONFLICT_KEYWORDS: set[str] = {
    "disagree",
    "incorrect",
    "wrong",
    "reject",
    "oppose",
    "objection",
    "mistaken",
    "refuse",
    "denied",
    "invalid",
}

_CONFLICT_PHRASES: list[str] = [
    "i disagree",
    "that is not",
    "that's not",
    "you are wrong",
    "not correct",
    "that is incorrect",
]


class EscalateOnConflict(BasePolicy):
    """Triggers escalation to a supervisor agent when agents disagree past a threshold.

    Monitors the sender's messages in ``context["history"]`` for conflict
    indicators (keywords and phrases).  When the cumulative count of
    conflicting messages reaches ``conflict_threshold``, a violation fires.

    Args:
        conflict_threshold: Number of conflicting messages from the sender
            required to trigger escalation (default ``3``).
        to: ``agent_id`` of the supervisor to route the escalation to.
        handler: Action to take on violation (default ``"escalate"``).

    Context keys:
        history (list[AgentMessage]): All past messages in the conversation.
            If absent or empty, the policy passes by default.
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

    def evaluate(self, message: AgentMessage, context: dict[str, Any]) -> PolicyResult:
        """Evaluate whether inter-agent conflict has exceeded the threshold.

        Args:
            message: The current AgentMessage to evaluate.
            context: Must contain ``history: list[AgentMessage]`` to be useful.

        Returns:
            PolicyResult indicating pass or conflict-escalation violation.
        """
        history: list[AgentMessage] = context.get("history", [])

        # Filter to sender's messages only.
        sender_history = [m for m in history if m.sender == message.sender]

        # Count conflicting messages from sender in history.
        conflict_count = sum(
            1 for m in sender_history if self._is_conflicting(m.content)
        )

        # Check current message too.
        if self._is_conflicting(message.content):
            conflict_count += 1

        passed = conflict_count < self.conflict_threshold
        violation_score = min(
            conflict_count / max(self.conflict_threshold, 1), 1.0
        )

        if passed:
            return self._pass(message.sender)

        details = (
            f"Agent '{message.sender}' has {conflict_count} conflicting "
            f"messages (threshold: {self.conflict_threshold})."
        )
        if self.to:
            details += f" Escalating to supervisor '{self.to}'."

        return PolicyResult(
            passed=False,
            violation_score=violation_score,
            policy_name=self.name,
            agent_id=message.sender,
            handler=self.handler,
            severity="high",
            details=details,
        )

    @staticmethod
    def _is_conflicting(content: str) -> bool:
        """Check whether a message contains conflict indicators.

        Args:
            content: Message text to inspect.

        Returns:
            True if at least one conflict keyword or phrase is found.
        """
        lower = content.lower()
        # Strip punctuation from each word for keyword matching.
        words = {w.strip(".,!?;:\"'()[]{}") for w in lower.split()}
        if words & _CONFLICT_KEYWORDS:
            return True
        return any(phrase in lower for phrase in _CONFLICT_PHRASES)

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
