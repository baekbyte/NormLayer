"""AutoGen adapter for NormLayer — stub, planned for Week 5-6."""

from __future__ import annotations

from typing import Any

from normlayer.engine import PolicyEngine


class AutoGenAdapter:
    """Thin adapter for intercepting AutoGen conversation message passing.

    Hooks into AutoGen's message flow via official extension points to
    enforce NormLayer policies without monkey-patching internals.

    .. note::
        This adapter is a stub. Full implementation is planned for Week 5-6.

    Args:
        engine: The configured :class:`PolicyEngine` instance.
    """

    def __init__(self, engine: PolicyEngine) -> None:
        self.engine = engine

    def wrap(self, agent: Any) -> Any:
        """Wrap an AutoGen ConversableAgent with NormLayer policy enforcement.

        Args:
            agent: An AutoGen ``ConversableAgent`` instance.

        Returns:
            The wrapped agent with enforcement hooks applied.

        Raises:
            NotImplementedError: This adapter is not yet implemented.
        """
        raise NotImplementedError(
            "AutoGenAdapter will be implemented in Week 5-6."
        )
