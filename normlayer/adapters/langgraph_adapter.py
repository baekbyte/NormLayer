"""LangGraph adapter for NormLayer — stub, planned for Week 5-6."""

from __future__ import annotations

from typing import Any

from normlayer.engine import PolicyEngine


class LangGraphAdapter:
    """Thin adapter for hooking NormLayer into LangGraph node execution.

    Hooks into LangGraph's official pre/post node execution extension points
    to intercept agent messages without monkey-patching framework internals.

    .. note::
        This adapter is a stub. Full implementation is planned for Week 5-6.

    Args:
        engine: The configured :class:`PolicyEngine` instance.
    """

    def __init__(self, engine: PolicyEngine) -> None:
        self.engine = engine

    def wrap(self, graph: Any) -> Any:
        """Wrap a LangGraph StateGraph with NormLayer policy enforcement.

        Args:
            graph: A LangGraph ``StateGraph`` instance.

        Returns:
            The wrapped graph with enforcement hooks applied.

        Raises:
            NotImplementedError: This adapter is not yet implemented.
        """
        raise NotImplementedError(
            "LangGraphAdapter will be implemented in Week 5-6."
        )
