"""Tests for CrewAIAdapter — all mocked, no CrewAI install needed."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from normlayer.adapters.crewai_adapter import CrewAIAdapter, _WrappedCrew
from normlayer.base_policy import AgentMessage, BasePolicy, PolicyResult
from normlayer.engine import EnforcementError, PolicyEngine


# ---------------------------------------------------------------------------
# Minimal policy stubs
# ---------------------------------------------------------------------------


class AlwaysPassPolicy(BasePolicy):
    name = "AlwaysPass"

    def evaluate(self, message: AgentMessage, context: dict) -> PolicyResult:
        return PolicyResult(
            passed=True,
            violation_score=0.0,
            policy_name=self.name,
            agent_id=message.sender,
            handler=self.handler,
            severity="low",
        )


class AlwaysBlockPolicy(BasePolicy):
    name = "AlwaysBlock"

    def __init__(self) -> None:
        super().__init__(handler="block")

    def evaluate(self, message: AgentMessage, context: dict) -> PolicyResult:
        return PolicyResult(
            passed=False,
            violation_score=1.0,
            policy_name=self.name,
            agent_id=message.sender,
            handler=self.handler,
            severity="high",
            details="Always blocked.",
        )


class AlwaysWarnPolicy(BasePolicy):
    name = "AlwaysWarn"

    def __init__(self) -> None:
        super().__init__(handler="warn")

    def evaluate(self, message: AgentMessage, context: dict) -> PolicyResult:
        return PolicyResult(
            passed=False,
            violation_score=0.7,
            policy_name=self.name,
            agent_id=message.sender,
            handler=self.handler,
            severity="medium",
            details="Always warned.",
        )


# ---------------------------------------------------------------------------
# Mock CrewAI objects
# ---------------------------------------------------------------------------


class MockTaskOutput:
    """Mimics a CrewAI TaskOutput."""

    def __init__(self, raw: str, pydantic: Any = None) -> None:
        self.raw = raw
        self.pydantic = pydantic


class MockCrewOutput:
    """Mimics a CrewAI CrewOutput."""

    def __init__(self, raw: str, tasks_output: list[MockTaskOutput]) -> None:
        self.raw = raw
        self.tasks_output = tasks_output


class MockCrewAgent:
    """Mimics a CrewAI Agent."""

    def __init__(self, role: str) -> None:
        self.role = role


class MockTask:
    """Mimics a CrewAI Task."""

    def __init__(self, agent: MockCrewAgent | None = None) -> None:
        self.agent = agent


class MockCrew:
    """Mimics a CrewAI Crew."""

    def __init__(
        self,
        tasks: list[MockTask],
        output: MockCrewOutput,
    ) -> None:
        self.tasks = tasks
        self._output = output
        self.some_attr = "crew_attr"

    def kickoff(self, **kwargs: Any) -> MockCrewOutput:
        return self._output

    async def kickoff_async(self, **kwargs: Any) -> MockCrewOutput:
        return self._output


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCrewAIAdapterInit:
    def test_stores_engine(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = CrewAIAdapter(engine)
        assert adapter.engine is engine


class TestWrapReturnsProxy:
    def test_wrap_returns_wrapped_crew(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = CrewAIAdapter(engine)
        crew = MockCrew(tasks=[], output=MockCrewOutput("done", []))
        wrapped = adapter.wrap(crew)
        assert isinstance(wrapped, _WrappedCrew)

    def test_getattr_delegation(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = CrewAIAdapter(engine)
        crew = MockCrew(tasks=[], output=MockCrewOutput("done", []))
        wrapped = adapter.wrap(crew)
        assert wrapped.some_attr == "crew_attr"


class TestToAgentMessage:
    def test_basic_conversion(self) -> None:
        task_output = MockTaskOutput(raw="Task result")
        result = CrewAIAdapter._to_agent_message(task_output, "researcher")
        assert result.content == "Task result"
        assert result.sender == "researcher"

    def test_empty_raw(self) -> None:
        task_output = MockTaskOutput(raw="")
        result = CrewAIAdapter._to_agent_message(task_output, "writer")
        assert result.content == ""
        assert result.sender == "writer"


class TestResolveAgentRole:
    def test_resolves_from_task_agent(self) -> None:
        agent = MockCrewAgent(role="researcher")
        task = MockTask(agent=agent)
        crew = MockCrew(tasks=[task], output=MockCrewOutput("", []))
        assert CrewAIAdapter._resolve_agent_role(crew, 0) == "researcher"

    def test_fallback_when_no_agent(self) -> None:
        task = MockTask(agent=None)
        crew = MockCrew(tasks=[task], output=MockCrewOutput("", []))
        assert CrewAIAdapter._resolve_agent_role(crew, 0) == "crew_agent_0"

    def test_fallback_when_index_out_of_range(self) -> None:
        crew = MockCrew(tasks=[], output=MockCrewOutput("", []))
        assert CrewAIAdapter._resolve_agent_role(crew, 5) == "crew_agent_5"

    def test_fallback_when_no_tasks_attr(self) -> None:
        crew = MockCrew(tasks=[], output=MockCrewOutput("", []))
        delattr(crew, "tasks")
        assert CrewAIAdapter._resolve_agent_role(crew, 0) == "crew_agent_0"


class TestKickoffPassthrough:
    def test_passthrough_no_violations(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = CrewAIAdapter(engine)
        task_output = MockTaskOutput(raw="All good")
        crew_output = MockCrewOutput("done", [task_output])
        agent = MockCrewAgent(role="writer")
        crew = MockCrew(tasks=[MockTask(agent)], output=crew_output)
        wrapped = adapter.wrap(crew)

        result = wrapped.kickoff()
        assert result is crew_output
        assert len(engine.violations) == 0

    def test_empty_tasks_output(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = CrewAIAdapter(engine)
        crew_output = MockCrewOutput("done", [])
        crew = MockCrew(tasks=[], output=crew_output)
        wrapped = adapter.wrap(crew)

        result = wrapped.kickoff()
        assert result is crew_output


class TestKickoffWithViolations:
    def test_block_raises_enforcement_error(self) -> None:
        engine = PolicyEngine(policies=[AlwaysBlockPolicy()])
        adapter = CrewAIAdapter(engine)
        task_output = MockTaskOutput(raw="Bad output")
        crew_output = MockCrewOutput("done", [task_output])
        agent = MockCrewAgent(role="agent")
        crew = MockCrew(tasks=[MockTask(agent)], output=crew_output)
        wrapped = adapter.wrap(crew)

        with pytest.raises(EnforcementError):
            wrapped.kickoff()

    def test_warn_allows_through(self) -> None:
        engine = PolicyEngine(policies=[AlwaysWarnPolicy()])
        adapter = CrewAIAdapter(engine)
        task_output = MockTaskOutput(raw="Warn output")
        crew_output = MockCrewOutput("done", [task_output])
        agent = MockCrewAgent(role="agent")
        crew = MockCrew(tasks=[MockTask(agent)], output=crew_output)
        wrapped = adapter.wrap(crew)

        result = wrapped.kickoff()
        assert result is crew_output
        assert len(engine.violations) == 1

    def test_multiple_tasks_checked(self) -> None:
        engine = PolicyEngine(policies=[AlwaysWarnPolicy()])
        adapter = CrewAIAdapter(engine)
        t1 = MockTaskOutput(raw="Output 1")
        t2 = MockTaskOutput(raw="Output 2")
        crew_output = MockCrewOutput("done", [t1, t2])
        a1 = MockCrewAgent(role="researcher")
        a2 = MockCrewAgent(role="writer")
        crew = MockCrew(
            tasks=[MockTask(a1), MockTask(a2)],
            output=crew_output,
        )
        wrapped = adapter.wrap(crew)

        wrapped.kickoff()
        assert len(engine.violations) == 2

    def test_role_fallback_used_in_violation(self) -> None:
        engine = PolicyEngine(policies=[AlwaysWarnPolicy()])
        adapter = CrewAIAdapter(engine)
        task_output = MockTaskOutput(raw="Some output")
        crew_output = MockCrewOutput("done", [task_output])
        crew = MockCrew(tasks=[MockTask(agent=None)], output=crew_output)
        wrapped = adapter.wrap(crew)

        wrapped.kickoff()
        assert engine.violations[0].agent_id == "crew_agent_0"


class TestKickoffAsync:
    def test_kickoff_async_passthrough(self) -> None:
        engine = PolicyEngine(policies=[AlwaysPassPolicy()])
        adapter = CrewAIAdapter(engine)
        task_output = MockTaskOutput(raw="Async result")
        crew_output = MockCrewOutput("done", [task_output])
        agent = MockCrewAgent(role="writer")
        crew = MockCrew(tasks=[MockTask(agent)], output=crew_output)
        wrapped = adapter.wrap(crew)

        result = asyncio.get_event_loop().run_until_complete(
            wrapped.kickoff_async()
        )
        assert result is crew_output
        assert len(engine.violations) == 0

    def test_kickoff_async_block(self) -> None:
        engine = PolicyEngine(policies=[AlwaysBlockPolicy()])
        adapter = CrewAIAdapter(engine)
        task_output = MockTaskOutput(raw="Bad async")
        crew_output = MockCrewOutput("done", [task_output])
        agent = MockCrewAgent(role="agent")
        crew = MockCrew(tasks=[MockTask(agent)], output=crew_output)
        wrapped = adapter.wrap(crew)

        with pytest.raises(EnforcementError):
            asyncio.get_event_loop().run_until_complete(
                wrapped.kickoff_async()
            )
