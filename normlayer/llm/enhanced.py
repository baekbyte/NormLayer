"""llm_enhanced() — two-tier wrapper that adds LLM second-pass to heuristic policies."""

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
from normlayer.llm.prompts import ENHANCED_TEMPLATE


class _EnhancedPolicy(BasePolicy):
    """Internal wrapper that adds LLM verification to a heuristic policy.

    The heuristic runs first. If the violation score falls within the
    ``borderline_range``, the LLM judge is invoked for a second opinion.
    Outside the borderline range, the heuristic result is returned as-is.

    Args:
        base: The underlying heuristic policy.
        judge: An ``LLMJudge`` instance for second-pass evaluation.
        borderline_range: Tuple of (low, high) defining the borderline zone.
        handler: Override handler. If None, uses the base policy's handler.
    """

    name: str = "EnhancedPolicy"

    def __init__(
        self,
        base: BasePolicy,
        judge: LLMJudge,
        borderline_range: tuple[float, float] = (0.3, 0.8),
        handler: HandlerType | None = None,
    ) -> None:
        effective_handler: HandlerType = handler if handler is not None else base.handler
        super().__init__(handler=effective_handler)
        self.base = base
        self.judge = judge
        self.borderline_range = borderline_range
        self.name = f"{base.name}+LLM"

    def _is_borderline(self, score: float) -> bool:
        """Check if a violation score falls in the borderline range.

        Args:
            score: The heuristic violation score.

        Returns:
            True if the score is within [low, high].
        """
        low, high = self.borderline_range
        return low <= score <= high

    def _format_enhanced_prompt(
        self, message: AgentMessage, context: dict[str, Any],
        heuristic_result: PolicyResult,
    ) -> str:
        """Format the enhanced evaluation prompt.

        Args:
            message: The agent message.
            context: Contextual information.
            heuristic_result: The result from the heuristic policy.

        Returns:
            Formatted prompt string.
        """
        context_section = ""
        if context:
            context_lines = [f"- {k}: {v}" for k, v in context.items()]
            context_section = "**Context:**\n" + "\n".join(context_lines)

        return ENHANCED_TEMPLATE.format(
            policy_name=self.base.name,
            heuristic_score=heuristic_result.violation_score,
            heuristic_details=heuristic_result.details or "No details",
            sender=message.sender,
            recipient=message.recipient or "N/A",
            content=message.content,
            context_section=context_section,
        )

    def _merge_result(
        self, message: AgentMessage, heuristic_result: PolicyResult,
        violated: bool, score: float, severity: str, reasoning: str,
    ) -> PolicyResult:
        """Build a PolicyResult merging heuristic and LLM judgments.

        Args:
            message: The evaluated message.
            heuristic_result: Original heuristic result.
            violated: LLM's violation judgment.
            score: LLM's violation score.
            severity: LLM's severity assessment.
            reasoning: LLM's reasoning.

        Returns:
            A merged PolicyResult.
        """
        severity_val: SeverityLevel
        if severity in ("low", "medium", "high"):
            severity_val = severity  # type: ignore[assignment]
        else:
            severity_val = "medium"
        details = (
            f"Heuristic score: {heuristic_result.violation_score:.3f}. "
            f"LLM judgment: {reasoning}"
        )
        return PolicyResult(
            passed=not violated,
            violation_score=score,
            policy_name=self.name,
            agent_id=message.sender,
            handler=self.handler,
            severity=severity_val,
            details=details,
        )

    def evaluate(self, message: AgentMessage, context: dict[str, Any]) -> PolicyResult:
        """Evaluate with heuristic first, LLM second on borderline scores.

        Args:
            message: The AgentMessage to evaluate.
            context: Contextual information.

        Returns:
            PolicyResult — either from the heuristic alone or enhanced by LLM.
        """
        heuristic_result = self.base.evaluate(message, context)

        if not self._is_borderline(heuristic_result.violation_score):
            # Outside borderline range — return heuristic result with updated name
            return PolicyResult(
                passed=heuristic_result.passed,
                violation_score=heuristic_result.violation_score,
                policy_name=self.name,
                agent_id=heuristic_result.agent_id,
                handler=self.handler,
                severity=heuristic_result.severity,
                details=heuristic_result.details,
            )

        # Borderline — invoke LLM
        prompt = self._format_enhanced_prompt(message, context, heuristic_result)
        response = self.judge.judge(prompt)

        # On LLM failure, fall back to heuristic result
        if not response.raw:
            return PolicyResult(
                passed=heuristic_result.passed,
                violation_score=heuristic_result.violation_score,
                policy_name=self.name,
                agent_id=heuristic_result.agent_id,
                handler=self.handler,
                severity=heuristic_result.severity,
                details=f"{heuristic_result.details} (LLM fallback: {response.reasoning})",
            )

        return self._merge_result(
            message, heuristic_result,
            violated=response.violated,
            score=response.violation_score,
            severity=response.severity,
            reasoning=response.reasoning,
        )

    async def async_evaluate(
        self, message: AgentMessage, context: dict[str, Any],
    ) -> PolicyResult:
        """Evaluate asynchronously with heuristic first, LLM second.

        Args:
            message: The AgentMessage to evaluate.
            context: Contextual information.

        Returns:
            PolicyResult — either from the heuristic alone or enhanced by LLM.
        """
        heuristic_result = await self.base.async_evaluate(message, context)

        if not self._is_borderline(heuristic_result.violation_score):
            return PolicyResult(
                passed=heuristic_result.passed,
                violation_score=heuristic_result.violation_score,
                policy_name=self.name,
                agent_id=heuristic_result.agent_id,
                handler=self.handler,
                severity=heuristic_result.severity,
                details=heuristic_result.details,
            )

        prompt = self._format_enhanced_prompt(message, context, heuristic_result)
        response = await self.judge.async_judge(prompt)

        if not response.raw:
            return PolicyResult(
                passed=heuristic_result.passed,
                violation_score=heuristic_result.violation_score,
                policy_name=self.name,
                agent_id=heuristic_result.agent_id,
                handler=self.handler,
                severity=heuristic_result.severity,
                details=f"{heuristic_result.details} (LLM fallback: {response.reasoning})",
            )

        return self._merge_result(
            message, heuristic_result,
            violated=response.violated,
            score=response.violation_score,
            severity=response.severity,
            reasoning=response.reasoning,
        )


def llm_enhanced(
    policy: BasePolicy,
    judge: LLMJudge,
    borderline_range: tuple[float, float] = (0.3, 0.8),
    handler: HandlerType | None = None,
) -> _EnhancedPolicy:
    """Wrap a heuristic policy with LLM second-pass on borderline scores.

    The heuristic runs first (microseconds). The LLM only fires when the
    ``violation_score`` falls within the configurable ``borderline_range``.

    Args:
        policy: The heuristic BasePolicy to enhance.
        judge: An ``LLMJudge`` instance.
        borderline_range: Tuple of (low, high) for the borderline zone
            (default ``(0.3, 0.8)``).
        handler: Override handler. If None, uses the base policy's handler.

    Returns:
        An ``_EnhancedPolicy`` that wraps the original with LLM verification.

    Example::

        smart_deception = llm_enhanced(
            policies.NoDeception(threshold=0.8, handler="escalate"),
            judge=judge,
            borderline_range=(0.3, 0.8),
        )
    """
    return _EnhancedPolicy(
        base=policy,
        judge=judge,
        borderline_range=borderline_range,
        handler=handler,
    )
