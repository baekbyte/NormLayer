"""CrewAI adapter for NormLayer policy enforcement."""

from __future__ import annotations

from typing import Any

from normlayer.base_policy import AgentMessage
from normlayer.engine import PolicyEngine


class CrewAIAdapter:
    """Thin adapter that wraps a CrewAI Crew with NormLayer enforcement.

    Intercepts task outputs from ``crew.kickoff()`` / ``crew.kickoff_async()``
    and runs each through the engine's policy stack. Does not modify CrewAI
    internals — only wraps the public kickoff API.

    Args:
        engine: The configured :class:`PolicyEngine` instance.
    """

    def __init__(self, engine: PolicyEngine) -> None:
        self.engine = engine

    def wrap(self, crew: Any) -> _WrappedCrew:
        """Wrap a CrewAI Crew with policy enforcement.

        Args:
            crew: A CrewAI ``Crew`` instance with ``kickoff`` / ``kickoff_async``.

        Returns:
            A proxy object that delegates to the original crew but checks
            task outputs against the engine's policy stack.
        """
        return _WrappedCrew(crew, self.engine)

    @staticmethod
    def _to_agent_message(task_output: Any, agent_role: str) -> AgentMessage:
        """Convert a CrewAI TaskOutput to an AgentMessage.

        Args:
            task_output: A CrewAI ``TaskOutput`` with a ``raw`` attribute.
            agent_role: The role string of the agent that produced the output.

        Returns:
            An :class:`AgentMessage`.
        """
        content = getattr(task_output, "raw", "") or ""
        return AgentMessage(
            content=content,
            sender=agent_role,
        )

    @staticmethod
    def _resolve_agent_role(crew: Any, index: int) -> str:
        """Resolve the agent role for a task by index.

        Tries ``crew.tasks[index].agent.role``, falling back to
        ``"crew_agent_{index}"`` if the attribute chain is missing.

        Args:
            crew: The original CrewAI Crew.
            index: The task index.

        Returns:
            The resolved agent role string.
        """
        try:
            tasks = getattr(crew, "tasks", [])
            if index < len(tasks):
                agent = getattr(tasks[index], "agent", None)
                if agent is not None:
                    role = getattr(agent, "role", None)
                    if role is not None:
                        return str(role)
        except (IndexError, AttributeError):
            pass
        return f"crew_agent_{index}"


class _WrappedCrew:
    """Proxy around a CrewAI Crew with NormLayer enforcement.

    Delegates all attribute access to the underlying crew. Overrides
    ``kickoff`` and ``kickoff_async`` to check task outputs post-execution.

    Args:
        crew: The original CrewAI Crew.
        engine: The NormLayer :class:`PolicyEngine`.
    """

    def __init__(self, crew: Any, engine: PolicyEngine) -> None:
        self._crew = crew
        self._engine = engine

    def __getattr__(self, name: str) -> Any:
        return getattr(self._crew, name)

    def kickoff(self, **kwargs: Any) -> Any:
        """Run the crew and check task outputs against the policy stack.

        Args:
            **kwargs: Arguments forwarded to ``crew.kickoff()``.

        Returns:
            The ``CrewOutput`` from the crew.

        Raises:
            EnforcementError: If a policy with ``handler="block"`` fires.
        """
        crew_output = self._crew.kickoff(**kwargs)
        self._check_task_outputs(crew_output)
        return crew_output

    async def kickoff_async(self, **kwargs: Any) -> Any:
        """Async version of :meth:`kickoff`.

        Args:
            **kwargs: Arguments forwarded to ``crew.kickoff_async()``.

        Returns:
            The ``CrewOutput`` from the crew.

        Raises:
            EnforcementError: If a policy with ``handler="block"`` fires.
        """
        crew_output = await self._crew.kickoff_async(**kwargs)
        await self._async_check_task_outputs(crew_output)
        return crew_output

    def _check_task_outputs(self, crew_output: Any) -> None:
        """Check each task output via the engine (sync)."""
        tasks_output = getattr(crew_output, "tasks_output", []) or []
        for i, task_output in enumerate(tasks_output):
            role = CrewAIAdapter._resolve_agent_role(self._crew, i)
            agent_msg = CrewAIAdapter._to_agent_message(task_output, role)
            self._engine.check(
                agent_msg, context={"crew_output": crew_output}
            )

    async def _async_check_task_outputs(self, crew_output: Any) -> None:
        """Check each task output via the engine (async)."""
        tasks_output = getattr(crew_output, "tasks_output", []) or []
        for i, task_output in enumerate(tasks_output):
            role = CrewAIAdapter._resolve_agent_role(self._crew, i)
            agent_msg = CrewAIAdapter._to_agent_message(task_output, role)
            await self._engine.async_check(
                agent_msg, context={"crew_output": crew_output}
            )
