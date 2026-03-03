"""AutoGen adapter for NormLayer policy enforcement."""

from __future__ import annotations

from typing import Any

from normlayer.base_policy import AgentMessage
from normlayer.engine import PolicyEngine


class AutoGenAdapter:
    """Thin adapter that wraps an AutoGen agent with NormLayer enforcement.

    Intercepts incoming and outgoing messages in ``agent.on_messages()``
    (async-only, matching AutoGen's async-first design). Does not modify
    AutoGen internals — only wraps the public ``on_messages`` method.

    Args:
        engine: The configured :class:`PolicyEngine` instance.
    """

    def __init__(self, engine: PolicyEngine) -> None:
        self.engine = engine

    def wrap(self, agent: Any) -> _WrappedAgent:
        """Wrap an AutoGen agent with policy enforcement.

        Args:
            agent: An AutoGen agent with an async ``on_messages`` method.

        Returns:
            A proxy object that delegates to the original agent but checks
            incoming and outgoing messages against the engine's policy stack.
        """
        return _WrappedAgent(agent, self.engine)

    @staticmethod
    def _to_agent_message(msg: Any) -> AgentMessage | None:
        """Convert an AutoGen message to an AgentMessage.

        Only converts messages that have both ``content`` (str) and ``source``
        attributes. Returns ``None`` for unsupported message types.

        Args:
            msg: An AutoGen message object.

        Returns:
            An :class:`AgentMessage`, or ``None`` if the message type is
            not supported.
        """
        content = getattr(msg, "content", None)
        source = getattr(msg, "source", None)
        if not isinstance(content, str) or source is None:
            return None
        return AgentMessage(
            content=content,
            sender=str(source),
        )


class _WrappedAgent:
    """Proxy around an AutoGen agent with NormLayer enforcement.

    Delegates all attribute access to the underlying agent. Overrides
    ``on_messages`` to check incoming messages before execution and
    outgoing messages after execution.

    Args:
        agent: The original AutoGen agent.
        engine: The NormLayer :class:`PolicyEngine`.
    """

    def __init__(self, agent: Any, engine: PolicyEngine) -> None:
        self._agent = agent
        self._engine = engine

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

    async def on_messages(
        self,
        messages: list[Any],
        cancellation_token: Any = None,
    ) -> Any:
        """Process messages with policy checks on both input and output.

        Args:
            messages: List of incoming AutoGen messages.
            cancellation_token: AutoGen cancellation token (forwarded).

        Returns:
            The agent's ``Response`` object.

        Raises:
            EnforcementError: If a policy with ``handler="block"`` fires.
        """
        # Check incoming messages
        for msg in messages:
            agent_msg = AutoGenAdapter._to_agent_message(msg)
            if agent_msg is not None:
                await self._engine.async_check(agent_msg)

        # Execute the agent
        response = await self._agent.on_messages(messages, cancellation_token)

        # Check outgoing message
        chat_message = getattr(response, "chat_message", None)
        if chat_message is not None:
            agent_msg = AutoGenAdapter._to_agent_message(chat_message)
            if agent_msg is not None:
                await self._engine.async_check(agent_msg)

        return response
