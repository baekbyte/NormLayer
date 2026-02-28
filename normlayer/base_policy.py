"""Core abstractions for NormLayer: message schema, policy results, and BasePolicy."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

HandlerType = Literal["block", "warn", "escalate", "log"]
SeverityLevel = Literal["low", "medium", "high"]


class AgentMessage(BaseModel):
    """A message passed between agents in a multi-agent pipeline.

    Attributes:
        content: The full text body of the message.
        sender: The agent_id of the sending agent.
        recipient: The agent_id of the intended recipient (optional).
        metadata: Arbitrary key/value pairs for framework-specific data.
    """

    content: str
    sender: str
    recipient: str | None = None
    metadata: dict = Field(default_factory=dict)


class PolicyResult(BaseModel):
    """The outcome of evaluating one policy against one message.

    Attributes:
        passed: True if the policy was not violated.
        violation_score: Continuous score in [0, 1]. 0.0 = clean, 1.0 = definite violation.
        policy_name: Name of the policy that produced this result.
        agent_id: The agent whose message was evaluated.
        handler: The action to dispatch on violation (block/warn/escalate/log).
        severity: Severity of the violation if it occurred.
        details: Human-readable explanation of the violation, or "" if passed.
    """

    passed: bool
    violation_score: float = Field(ge=0.0, le=1.0)
    policy_name: str
    agent_id: str
    handler: HandlerType
    severity: SeverityLevel = "medium"
    details: str = ""


class ViolationEvent(BaseModel):
    """A structured violation record, suitable for shipping to S3 / CloudWatch.

    Attributes:
        timestamp: ISO-8601 UTC timestamp of when the violation was detected.
        agent_id: The agent whose message triggered the violation.
        policy_violated: Name of the policy that was violated.
        severity: Severity level of the violation.
        message_snippet: First 200 characters of the offending message.
        context_window_hash: Short SHA-256 of the serialized context dict,
            used for deduplication and correlation across log entries.
        handler_dispatched: The handler action that was triggered.
        details: Human-readable explanation from the policy.
    """

    timestamp: str
    agent_id: str
    policy_violated: str
    severity: SeverityLevel
    message_snippet: str
    context_window_hash: str
    handler_dispatched: HandlerType
    details: str = ""

    @classmethod
    def from_policy_result(
        cls,
        result: PolicyResult,
        message: AgentMessage,
        context: dict,
    ) -> ViolationEvent:
        """Construct a ViolationEvent from a failed PolicyResult.

        Args:
            result: The PolicyResult that indicated a violation.
            message: The AgentMessage that triggered the violation.
            context: The context dict passed to the policy.

        Returns:
            A fully populated ViolationEvent ready for logging.
        """
        ctx_str = str(sorted(context.items()))
        ctx_hash = hashlib.sha256(ctx_str.encode()).hexdigest()[:16]
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_id=result.agent_id,
            policy_violated=result.policy_name,
            severity=result.severity,
            message_snippet=message.content[:200],
            context_window_hash=ctx_hash,
            handler_dispatched=result.handler,
            details=result.details,
        )


class BasePolicy(ABC):
    """Abstract base class for all NormLayer behavioral policies.

    Every policy must implement `evaluate`. For pipelines that use async
    execution (LangGraph, AutoGen), override `async_evaluate` as well —
    by default it delegates to the synchronous `evaluate`.

    Attributes:
        name: Human-readable policy name, overridden by each subclass.
        handler: Default action dispatched on violation.
    """

    name: str = "base_policy"

    def __init__(self, handler: HandlerType = "warn") -> None:
        """Initialize the policy with a violation handler.

        Args:
            handler: One of "block", "warn", "escalate", "log".
                - block: Raises EnforcementError, stopping the message.
                - warn: Logs the violation but allows the message through.
                - escalate: Routes to a designated supervisor agent.
                - log: Records silently for audit; no user-visible action.
        """
        self.handler = handler

    @abstractmethod
    def evaluate(self, message: AgentMessage, context: dict) -> PolicyResult:
        """Evaluate a message against this policy synchronously.

        Args:
            message: The AgentMessage to evaluate.
            context: Contextual information such as conversation history,
                agent role assignments, or the original message being summarized.

        Returns:
            A PolicyResult with `passed=True` if the policy was not violated,
            or `passed=False` with a non-zero violation_score if it was.
        """
        ...

    async def async_evaluate(
        self,
        message: AgentMessage,
        context: dict,
    ) -> PolicyResult:
        """Evaluate a message against this policy asynchronously.

        Default implementation calls the synchronous `evaluate`. Override
        this in subclasses that perform async I/O (e.g. async embedding APIs).

        Args:
            message: The AgentMessage to evaluate.
            context: Contextual information.

        Returns:
            A PolicyResult.
        """
        return self.evaluate(message, context)
