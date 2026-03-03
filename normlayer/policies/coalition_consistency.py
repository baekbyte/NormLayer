"""CoalitionConsistency policy — detects in-group vs. out-group double standards."""

from __future__ import annotations

from normlayer.base_policy import AgentMessage, BasePolicy, HandlerType, PolicyResult

_POSITIVE_KEYWORDS: set[str] = {
    "agree",
    "good",
    "great",
    "correct",
    "well done",
    "thank",
    "approve",
}

_NEGATIVE_KEYWORDS: set[str] = {
    "wrong",
    "bad",
    "disagree",
    "reject",
    "incorrect",
    "fail",
}


class CoalitionConsistency(BasePolicy):
    """Checks whether agents apply norms consistently across in-group vs. out-group agents.

    Detects double standards: an agent applying stricter or more lenient
    treatment to some agents than others based on coalition membership.

    Compares message length and sentiment ratios between messages sent to
    in-group members versus out-group members.  A large discrepancy
    triggers a violation.

    Args:
        coalitions: Mapping of ``coalition_name → list[agent_id]``.
            Example: ``{"team_a": ["planner", "executor"], "team_b": ["reviewer"]}``
        consistency_threshold: Maximum acceptable discrepancy between
            in-group and out-group treatment (default ``0.5``).
        handler: Action to take on violation (default ``"warn"``).

    Context keys:
        coalitions (dict): Runtime override for coalition definitions.
        history (list[AgentMessage]): All past messages in the conversation.
    """

    name: str = "CoalitionConsistency"

    def __init__(
        self,
        coalitions: dict[str, list[str]] | None = None,
        consistency_threshold: float = 0.5,
        handler: HandlerType = "warn",
    ) -> None:
        super().__init__(handler=handler)
        self.coalitions: dict[str, list[str]] = coalitions or {}
        self.consistency_threshold = consistency_threshold

    def evaluate(self, message: AgentMessage, context: dict) -> PolicyResult:
        """Evaluate whether this agent's behavior is consistent across coalitions.

        Args:
            message: The AgentMessage to evaluate.
            context: Must contain ``history`` and optionally ``coalitions``.

        Returns:
            PolicyResult indicating pass or consistency violation.
        """
        coalitions: dict[str, list[str]] = context.get(
            "coalitions", self.coalitions
        )
        sender = message.sender

        if not coalitions:
            return self._pass(sender)

        # Build in-group set: all agents sharing a coalition with sender.
        in_group: set[str] = set()
        sender_in_coalition = False
        for members in coalitions.values():
            if sender in members:
                sender_in_coalition = True
                in_group.update(members)
        in_group.discard(sender)

        if not sender_in_coalition:
            return self._pass(sender)

        history: list[AgentMessage] = context.get("history", [])
        if not history:
            return self._pass(sender)

        # Partition sender's history messages by recipient group.
        in_group_messages: list[AgentMessage] = []
        out_group_messages: list[AgentMessage] = []
        for m in history:
            if m.sender != sender:
                continue
            if m.recipient is None:
                continue
            if m.recipient in in_group:
                in_group_messages.append(m)
            else:
                out_group_messages.append(m)

        # Need messages in both groups to compare.
        if not in_group_messages or not out_group_messages:
            return self._pass(sender)

        # Compute per-group metrics.
        avg_in_len = sum(len(m.content) for m in in_group_messages) / len(
            in_group_messages
        )
        avg_out_len = sum(len(m.content) for m in out_group_messages) / len(
            out_group_messages
        )

        pos_in = self._sentiment_ratio(in_group_messages, _POSITIVE_KEYWORDS)
        pos_out = self._sentiment_ratio(out_group_messages, _POSITIVE_KEYWORDS)

        # Discrepancy = max of length discrepancy and sentiment discrepancy.
        length_discrepancy = abs(avg_in_len - avg_out_len) / max(
            avg_in_len, avg_out_len, 1
        )
        sentiment_discrepancy = abs(pos_in - pos_out)
        discrepancy = max(length_discrepancy, sentiment_discrepancy)

        passed = discrepancy < self.consistency_threshold
        violation_score = min(
            discrepancy / max(self.consistency_threshold, 0.01), 1.0
        )

        if passed:
            return self._pass(sender)

        severity = (
            "high"
            if discrepancy > 2 * self.consistency_threshold
            else "medium"
        )
        return PolicyResult(
            passed=False,
            violation_score=violation_score,
            policy_name=self.name,
            agent_id=sender,
            handler=self.handler,
            severity=severity,
            details=(
                f"Agent '{sender}' shows inconsistent behavior: "
                f"discrepancy {discrepancy:.2f} exceeds threshold "
                f"{self.consistency_threshold} "
                f"(length: {length_discrepancy:.2f}, "
                f"sentiment: {sentiment_discrepancy:.2f})."
            ),
        )

    @staticmethod
    def _sentiment_ratio(
        messages: list[AgentMessage], keywords: set[str]
    ) -> float:
        """Compute the fraction of messages containing at least one keyword.

        Args:
            messages: Messages to analyze.
            keywords: Sentiment keywords to search for.

        Returns:
            Float in [0, 1] — fraction of messages containing a keyword.
        """
        if not messages:
            return 0.0
        count = sum(
            1
            for m in messages
            if any(kw in m.content.lower() for kw in keywords)
        )
        return count / len(messages)

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
