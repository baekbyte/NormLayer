"""LLMPolicy — standalone natural-language policy evaluated by an LLM judge."""

from __future__ import annotations

from typing import Any

from normlayer.base_policy import (
    AgentMessage,
    BasePolicy,
    HandlerType,
    PolicyResult,
    SeverityLevel,
)
from normlayer.llm.judge import LLMJudge
from normlayer.llm.prompts import STANDALONE_TEMPLATE


class LLMPolicy(BasePolicy):
    """A policy defined in plain English, evaluated by an LLM judge.

    The user describes the policy as a natural-language description. For each
    message, the LLM evaluates whether the message violates the policy and
    returns a structured judgment.

    Args:
        description: Plain-English description of the policy rule.
        judge: An ``LLMJudge`` instance to evaluate messages.
        name: Human-readable policy name (default ``"LLMPolicy"``).
        handler: Action to dispatch on violation (default ``"warn"``).
        fail_open: If ``True`` (default), LLM failures result in a pass.
            If ``False``, LLM failures result in a violation.
    """

    name: str = "LLMPolicy"

    def __init__(
        self,
        description: str,
        judge: LLMJudge,
        name: str = "LLMPolicy",
        handler: HandlerType = "warn",
        fail_open: bool = True,
    ) -> None:
        super().__init__(handler=handler)
        self.name = name
        self.description = description
        self.judge = judge
        self.fail_open = fail_open

    def _format_prompt(self, message: AgentMessage, context: dict[str, Any]) -> str:
        """Format the evaluation prompt for the LLM.

        Args:
            message: The agent message to evaluate.
            context: Contextual information.

        Returns:
            Formatted prompt string.
        """
        context_section = ""
        if context:
            context_lines = [f"- {k}: {v}" for k, v in context.items()]
            context_section = "**Context:**\n" + "\n".join(context_lines)

        return STANDALONE_TEMPLATE.format(
            policy_description=self.description,
            sender=message.sender,
            recipient=message.recipient or "N/A",
            content=message.content,
            context_section=context_section,
        )

    def _result_from_judgment(
        self, message: AgentMessage, violated: bool, score: float,
        severity: str, reasoning: str,
    ) -> PolicyResult:
        """Build a PolicyResult from LLM judgment fields.

        Args:
            message: The evaluated message.
            violated: Whether the policy was violated.
            score: Violation score in [0, 1].
            severity: Severity level string.
            reasoning: Explanation from the LLM.

        Returns:
            A PolicyResult.
        """
        severity_val: SeverityLevel
        if severity in ("low", "medium", "high"):
            severity_val = severity  # type: ignore[assignment]
        else:
            severity_val = "medium"
        return PolicyResult(
            passed=not violated,
            violation_score=score,
            policy_name=self.name,
            agent_id=message.sender,
            handler=self.handler,
            severity=severity_val,
            details=reasoning,
        )

    def evaluate(self, message: AgentMessage, context: dict[str, Any]) -> PolicyResult:
        """Evaluate a message against this natural-language policy.

        Args:
            message: The AgentMessage to evaluate.
            context: Contextual information.

        Returns:
            PolicyResult from the LLM judge evaluation.
        """
        prompt = self._format_prompt(message, context)
        response = self.judge.judge(prompt)

        # If the LLM call failed (empty raw), respect fail_open setting
        if not response.raw and self.fail_open:
            return self._result_from_judgment(
                message, violated=False, score=0.0,
                severity="low", reasoning=response.reasoning,
            )
        if not response.raw and not self.fail_open:
            return self._result_from_judgment(
                message, violated=True, score=1.0,
                severity="high", reasoning=response.reasoning,
            )

        return self._result_from_judgment(
            message,
            violated=response.violated,
            score=response.violation_score,
            severity=response.severity,
            reasoning=response.reasoning,
        )

    async def async_evaluate(
        self, message: AgentMessage, context: dict[str, Any],
    ) -> PolicyResult:
        """Evaluate a message asynchronously.

        Args:
            message: The AgentMessage to evaluate.
            context: Contextual information.

        Returns:
            PolicyResult from the LLM judge evaluation.
        """
        prompt = self._format_prompt(message, context)
        response = await self.judge.async_judge(prompt)

        if not response.raw and self.fail_open:
            return self._result_from_judgment(
                message, violated=False, score=0.0,
                severity="low", reasoning=response.reasoning,
            )
        if not response.raw and not self.fail_open:
            return self._result_from_judgment(
                message, violated=True, score=1.0,
                severity="high", reasoning=response.reasoning,
            )

        return self._result_from_judgment(
            message,
            violated=response.violated,
            score=response.violation_score,
            severity=response.severity,
            reasoning=response.reasoning,
        )
