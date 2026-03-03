"""Deterministic test utilities — MockAgent and MockMessage.

Use these in all tests instead of real LLM calls. All behavior is scripted,
synchronous, and fully deterministic.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from normlayer.base_policy import AgentMessage


class MockMessage(BaseModel):
    """A scripted message for use in deterministic tests.

    Attributes:
        content: The text body of the message.
        sender: The agent_id of the sending agent.
        recipient: The agent_id of the intended recipient (optional).
        metadata: Arbitrary key/value payload.
    """

    content: str
    sender: str
    recipient: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_agent_message(self) -> AgentMessage:
        """Convert to an AgentMessage for use with policies and the engine.

        Returns:
            An AgentMessage with identical field values.
        """
        return AgentMessage(
            content=self.content,
            sender=self.sender,
            recipient=self.recipient,
            metadata=self.metadata,
        )


class MockAgent:
    """A scripted agent that returns pre-defined responses in sequence.

    Simulates agent behavior without any LLM calls or network access.
    Responses are consumed in order; calling beyond the last response
    raises :class:`StopIteration`.

    Args:
        agent_id: Unique identifier for this agent.
        role: Role label for this agent (e.g. ``"planner"``, ``"executor"``).
        responses: Ordered list of :class:`MockMessage` instances to return
            on successive invocations.

    Example::

        agent = MockAgent(
            agent_id="planner_agent",
            role="planner",
            responses=[
                MockMessage(content="I will assign task A.", sender="planner_agent"),
                MockMessage(content="Task A is now assigned.", sender="planner_agent"),
            ],
        )
        msg = agent()           # returns first response as AgentMessage
        msg = agent()           # returns second response
        agent()                 # raises StopIteration
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        responses: list[MockMessage],
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self.responses = responses
        self._index: int = 0

    def __call__(
        self,
        message: AgentMessage | None = None,
        context: dict[str, Any] | None = None,
    ) -> AgentMessage:
        """Return the next scripted response as an AgentMessage.

        The incoming `message` and `context` are intentionally ignored —
        responses are fully scripted and not computed from inputs.

        Args:
            message: Incoming message (ignored).
            context: Context dict (ignored).

        Returns:
            The next :class:`AgentMessage` in the scripted sequence.

        Raises:
            StopIteration: When all scripted responses have been consumed.
        """
        if self._index >= len(self.responses):
            raise StopIteration(
                f"MockAgent '{self.agent_id}' has no more scripted responses "
                f"(exhausted {len(self.responses)} response(s))."
            )
        response = self.responses[self._index]
        self._index += 1
        return response.to_agent_message()

    def reset(self) -> None:
        """Reset the response index to the beginning of the scripted sequence."""
        self._index = 0
