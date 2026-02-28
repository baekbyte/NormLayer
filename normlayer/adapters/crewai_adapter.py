"""CrewAI adapter for NormLayer — stub, planned for Week 5-6."""

from __future__ import annotations

from typing import Any

from normlayer.engine import PolicyEngine


class CrewAIAdapter:
    """Thin adapter for hooking NormLayer into CrewAI task execution.

    Wraps CrewAI agent task execution via official extension points to
    intercept messages without monkey-patching framework internals.

    .. note::
        This adapter is a stub. Full implementation is planned for Week 5-6.

    Args:
        engine: The configured :class:`PolicyEngine` instance.
    """

    def __init__(self, engine: PolicyEngine) -> None:
        self.engine = engine

    def wrap(self, crew: Any) -> Any:
        """Wrap a CrewAI Crew with NormLayer policy enforcement.

        Args:
            crew: A CrewAI ``Crew`` instance.

        Returns:
            The wrapped crew with enforcement hooks applied.

        Raises:
            NotImplementedError: This adapter is not yet implemented.
        """
        raise NotImplementedError(
            "CrewAIAdapter will be implemented in Week 5-6."
        )
