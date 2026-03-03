"""LangGraph adapter for NormLayer policy enforcement."""

from __future__ import annotations

from typing import Any

from normlayer.base_policy import AgentMessage
from normlayer.engine import PolicyEngine


class LangGraphAdapter:
    """Thin adapter that wraps a compiled LangGraph graph with NormLayer enforcement.

    Intercepts messages produced by ``graph.invoke()`` / ``graph.ainvoke()`` and
    runs each new message through the engine's policy stack. Does not modify
    LangGraph internals — only wraps the public invoke API.

    Args:
        engine: The configured :class:`PolicyEngine` instance.
        messages_key: The state dict key that holds the list of messages.
            Defaults to ``"messages"``.
    """

    def __init__(
        self,
        engine: PolicyEngine,
        messages_key: str = "messages",
    ) -> None:
        self.engine = engine
        self.messages_key = messages_key

    def wrap(self, graph: Any) -> _WrappedGraph:
        """Wrap a compiled LangGraph graph with policy enforcement.

        Args:
            graph: A compiled LangGraph ``StateGraph`` with ``invoke`` / ``ainvoke``.

        Returns:
            A proxy object that delegates to the original graph but checks
            new messages against the engine's policy stack.
        """
        return _WrappedGraph(graph, self.engine, self.messages_key)

    @staticmethod
    def _to_agent_message(msg: Any) -> AgentMessage:
        """Convert a LangGraph BaseMessage to an AgentMessage.

        Maps ``content`` to content, ``type`` or ``name`` to sender,
        and ``additional_kwargs`` to metadata.

        Args:
            msg: A LangGraph message object with ``content`` and ``type`` attrs.

        Returns:
            An :class:`AgentMessage`.
        """
        sender = getattr(msg, "name", None) or getattr(msg, "type", "unknown")
        metadata = getattr(msg, "additional_kwargs", {})
        return AgentMessage(
            content=getattr(msg, "content", ""),
            sender=sender,
            metadata=metadata if isinstance(metadata, dict) else {},
        )


class _WrappedGraph:
    """Proxy around a compiled LangGraph graph with NormLayer enforcement.

    Delegates all attribute access to the underlying graph. Overrides
    ``invoke`` and ``ainvoke`` to check new messages post-execution.

    Args:
        graph: The original compiled graph.
        engine: The NormLayer :class:`PolicyEngine`.
        messages_key: State dict key for the messages list.
    """

    def __init__(
        self,
        graph: Any,
        engine: PolicyEngine,
        messages_key: str,
    ) -> None:
        self._graph = graph
        self._engine = engine
        self._messages_key = messages_key

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)

    def invoke(self, state: dict, **kwargs: Any) -> dict:
        """Run the graph and check new messages against the policy stack.

        Args:
            state: The LangGraph state dict.
            **kwargs: Additional arguments forwarded to the graph.

        Returns:
            The result state dict from the graph.

        Raises:
            EnforcementError: If a policy with ``handler="block"`` fires.
        """
        original_count = len(state.get(self._messages_key, []))
        result_state = self._graph.invoke(state, **kwargs)
        self._check_new_messages(result_state, original_count)
        return result_state

    async def ainvoke(self, state: dict, **kwargs: Any) -> dict:
        """Async version of :meth:`invoke`.

        Args:
            state: The LangGraph state dict.
            **kwargs: Additional arguments forwarded to the graph.

        Returns:
            The result state dict from the graph.

        Raises:
            EnforcementError: If a policy with ``handler="block"`` fires.
        """
        original_count = len(state.get(self._messages_key, []))
        result_state = await self._graph.ainvoke(state, **kwargs)
        await self._async_check_new_messages(result_state, original_count)
        return result_state

    def _check_new_messages(self, result_state: dict, original_count: int) -> None:
        """Check each new message via the engine (sync)."""
        all_messages = result_state.get(self._messages_key, [])
        new_messages = all_messages[original_count:]
        for msg in new_messages:
            agent_msg = LangGraphAdapter._to_agent_message(msg)
            self._engine.check(agent_msg, context={"state": result_state})

    async def _async_check_new_messages(
        self, result_state: dict, original_count: int
    ) -> None:
        """Check each new message via the engine (async)."""
        all_messages = result_state.get(self._messages_key, [])
        new_messages = all_messages[original_count:]
        for msg in new_messages:
            agent_msg = LangGraphAdapter._to_agent_message(msg)
            await self._engine.async_check(
                agent_msg, context={"state": result_state}
            )
